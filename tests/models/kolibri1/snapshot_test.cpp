// A saved Kolibri 1 context has to answer like the conversation it came from.
//
// The rows of a snapshot are copied, not recomputed, and a copy on this machine
// is exact: the same bytes leave one device buffer and enter another. So the
// logits of a session restored after `Reset()` have to be bit-for-bit the
// logits of the session that was saved - not close, not the same winner.
// Anything else means the payload did not carry what the session held: a
// section whose length was worked out from the wrong geometry, a ring written
// back in another order while the saved `position` still indexes it as
// `position % ring`, or a layer that was skipped.
//
// Bit equality of the frontier cannot show that the *next* token is right, so
// the check goes on: a session restored from the object and a second one
// restored from the payload's bytes continue together through several greedy
// steps. A sliding-layer row landed in the wrong slot leaves the frontier
// untouched and shows up a few tokens later, and that is the failure this API
// would otherwise ship with.
//
// A payload of another version or another ring has to be refused with the
// target untouched, which is the other half of the contract: a store that
// half-reads a foreign context answers from a cache it shuffled itself.
//
// Without GUFO_KOLIBRI1_MODEL this skips with 77, and without a gfx1151 device
// likewise. The artifact is loaded once - its 45 GiB upload is most of the
// runtime - and the deep half fills a session to its 8192 position bound, which
// is the size a serve path actually moves and where the rings have wrapped.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/models/kolibri1/engine.hpp"

namespace k1 = gufo::models::kolibri1;

namespace {

int failures = 0;

void Check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
  }
  std::cout << (ok ? "PASS " : "FAIL ") << what << "\n";
}

[[nodiscard]] std::size_t Argmax(std::span<const float> logits) {
  return static_cast<std::size_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

/// Exact equality of two logit rows, which is what a copy of device memory
/// either is or is not.
[[nodiscard]] bool Identical(std::span<const float> a,
                             std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

[[nodiscard]] double MaxAbs(std::span<const float> a,
                            std::span<const float> b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, static_cast<double>(std::abs(a[i] - b[i])));
  }
  return worst;
}

/// Rewrites one header field of a copy of a payload. The field is found by the
/// value the engine that wrote the payload must have put there - both
/// `kSnapshotPayloadVersion` and `Executor::ring` are public - and not by a
/// byte position, because where the format keeps a field is none of this
/// check's business: the first four bytes holding the value are rewritten, and
/// the window is longer than any header the format can write, which is seven
/// 4-byte fields and three 8-byte ones. A payload that does not carry the value
/// at all fails on its own terms. The host is little-endian, the order a
/// payload is written in and the only order this target builds for.
bool RewriteField(std::vector<std::uint8_t>* payload, std::uint32_t from,
                  std::uint32_t to) {
  const std::size_t limit = std::min<std::size_t>(payload->size(), 64);
  for (std::size_t at = 0; at + 4 <= limit; ++at) {
    std::uint32_t value = 0;
    std::memcpy(&value, payload->data() + at, sizeof(value));
    if (value == from) {
      std::memcpy(payload->data() + at, &to, sizeof(to));
      return true;
    }
  }
  return false;
}

/// Everything a session can be compared by: what it holds, where its frontier
/// is, and what it says about its last position.
struct Held {
  std::vector<k1::TokenId> tokens;
  std::vector<float> logits;
  std::uint32_t position{0};
};

[[nodiscard]] Held Hold(const k1::Session& session) {
  return Held{
      std::vector<k1::TokenId>(session.tokens().begin(),
                               session.tokens().end()),
      std::vector<float>(session.logits().begin(), session.logits().end()),
      session.position()};
}

[[nodiscard]] bool Same(const Held& held, const k1::Session& session) {
  return held.position == session.position() &&
         held.tokens.size() == session.tokens().size() &&
         std::equal(held.tokens.begin(), held.tokens.end(),
                    session.tokens().begin()) &&
         Identical(held.logits, session.logits());
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_KOLIBRI1_MODEL");
  if (path == nullptr) {
    std::cout << "SKIP: set GUFO_KOLIBRI1_MODEL to the artifact\n";
    return 77;
  }
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::cout << "SKIP: no gfx1151 device\n";
    return 77;
  }

  // One load for every check. The ring the sliding layers keep follows the
  // chunk width the executor was built for, so a payload only fits a session
  // loaded the same way.
  auto model = k1::Model::Load(
      path, k1::ModelOptions{.max_context = 8192, .prefill_chunk = 16});
  if (!model) {
    std::cout << "FAIL cannot load the model on the device\n";
    return 1;
  }
  const std::uint32_t ring = model->executor()->ring();
  std::cout << "loaded " << model->resident_bytes() / (1024U * 1024U)
            << " MiB of weights, " << model->max_context()
            << " position context, rings of " << ring << " slots\n";

  const std::string prompt = "Hauptstadt von Deutschland ist";
  std::vector<k1::TokenId> tokens =
      model->tokenizer().Encode(prompt, false, true);
  while (tokens.size() < 40) {
    const std::vector<k1::TokenId> more = model->tokenizer().Encode(
        " Berlin liegt an der Spree und hat viele Museen.", false, true);
    tokens.insert(tokens.end(), more.begin(), more.end());
  }
  tokens.resize(41);

  auto session = model->CreateSession();
  if (!session) {
    std::cout << "FAIL cannot create a session\n";
    return 1;
  }
  session->Append(tokens);
  const std::vector<float> frontier(session->logits().begin(),
                                    session->logits().end());
  if (frontier.size() != model->device().vocab_size()) {
    std::cout << "FAIL the frontier row covers " << frontier.size()
              << " entries of a " << model->device().vocab_size()
              << " entry vocabulary\n";
    return 1;
  }

  // ----------------------------------------------------------------- save
  const std::uint64_t promised = session->SnapshotBytes();
  std::string error;
  const auto saved_at = std::chrono::steady_clock::now();
  auto snapshot = session->SaveSnapshot(&error);
  const double save_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - saved_at)
                             .count();
  Check(snapshot != nullptr, "a session with a frontier saves" +
                                 (error.empty() ? "" : ": " + error));
  if (snapshot == nullptr) {
    return 1;
  }
  Check(snapshot->SizeBytes() == promised,
        "the payload is the size SnapshotBytes promised: " +
            std::to_string(snapshot->SizeBytes()) + " bytes");

  std::vector<std::uint8_t> payload(snapshot->SizeBytes());
  Check(snapshot->CopyTo(payload), "CopyTo fills a buffer of SizeBytes");
  Check(!snapshot->CopyTo(
            std::span<std::uint8_t>(payload).first(payload.size() - 1)),
        "CopyTo refuses a buffer that is not the payload");

  // The pieces have to add up to the same bytes: a writer that persists a
  // context piece by piece gets the payload, not part of it.
  constexpr std::size_t kPieceBound = std::size_t{64} << 20;
  std::vector<std::uint8_t> streamed;
  std::size_t longest = 0;
  const bool pieces =
      snapshot->Stream([&](std::span<const std::uint8_t> piece) {
        longest = std::max(longest, piece.size());
        streamed.insert(streamed.end(), piece.begin(), piece.end());
      });
  Check(pieces && streamed == payload,
        "Stream hands out the same payload, in " +
            std::to_string(streamed.size()) + " bytes");
  Check(longest > 0 && longest <= kPieceBound,
        "the longest piece is " + std::to_string(longest) +
            " bytes, within the 64 MiB bound");

  // -------------------------------------------------------------- restore
  session->Reset();
  Check(session->position() == 0 && session->logits().empty(),
        "Reset leaves no frontier");
  const auto restored_at = std::chrono::steady_clock::now();
  const bool restored = session->RestoreSnapshot(*snapshot, &error);
  const double restore_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - restored_at)
                                .count();
  Check(restored, "the snapshot restores into the reset session" +
                      (error.empty() ? "" : ": " + error));
  Check(session->position() == tokens.size(),
        "the restored frontier is " + std::to_string(session->position()) +
            " positions");
  Check(session->tokens().size() == tokens.size() &&
            std::equal(tokens.begin(), tokens.end(), session->tokens().begin()),
        "the restored tokens are the saved ones");
  // Exact, because the rows were copied and not recomputed. A difference here
  // is a bug, and its size would only say which section went wrong.
  Check(Identical(frontier, session->logits()),
        "the restored logits are bit-equal to the saved ones");

  // A second session of the same model, from the payload's bytes rather than
  // the object, reaches the same frontier and goes on answering the same.
  auto peer = model->CreateSession();
  if (peer == nullptr) {
    std::cout << "FAIL cannot create the second session\n";
    return 1;
  }
  Check(peer->RestoreSnapshot(std::span<const std::uint8_t>(payload), &error),
        "the payload restores into a second session" +
            (error.empty() ? "" : ": " + error));
  Check(Identical(frontier, peer->logits()),
        "the second session's logits are bit-equal too");

  constexpr int kSteps = 6;
  std::vector<k1::TokenId> greedy;
  bool agrees = true;
  for (int step = 0; step < kSteps; ++step) {
    const std::size_t next = Argmax(session->logits());
    if (next != Argmax(peer->logits())) {
      agrees = false;
      break;
    }
    const k1::TokenId token = static_cast<k1::TokenId>(next);
    greedy.push_back(token);
    session->Append({&token, 1});
    peer->Append({&token, 1});
  }
  Check(agrees, "both sessions pick the same " + std::to_string(kSteps) +
                    " tokens: " + model->tokenizer().Decode(greedy, false));
  Check(Identical(session->logits(), peer->logits()),
        "and their frontiers stay bit-equal, max_abs=" +
            std::to_string(MaxAbs(session->logits(), peer->logits())));

  std::cout << "payload of " << tokens.size() << " positions: "
            << static_cast<double>(promised) / (1024.0 * 1024.0)
            << " MiB, save " << save_ms << " ms, restore " << restore_ms
            << " ms\n";

  // -------------------------------------------------------------- refusal
  // A foreign version and a foreign ring, each built by rewriting one field of
  // a copy. The target holds a conversation by now, so a refusal that wrote
  // anything at all would show in its tokens, its frontier or its logits.
  const Held before = Hold(*peer);
  std::vector<std::uint8_t> bad = payload;
  Check(RewriteField(&bad, k1::Session::kSnapshotPayloadVersion,
                     k1::Session::kSnapshotPayloadVersion + 1),
        "the payload says which version it is");
  error.clear();
  const bool version_rejected =
      !peer->RestoreSnapshot(std::span<const std::uint8_t>(bad), &error);
  Check(version_rejected && error.find("version") != std::string::npos,
        "a foreign version is refused, naming it" +
            (error.empty() ? "" : ": " + error));
  Check(Same(before, *peer),
        "and the target kept its tokens, frontier and logits");

  bad = payload;
  Check(RewriteField(&bad, ring, ring * 2),
        "the payload says how wide its rings are");
  error.clear();
  const bool ring_rejected =
      !peer->RestoreSnapshot(std::span<const std::uint8_t>(bad), &error);
  Check(ring_rejected && error.find("ring") != std::string::npos,
        "a foreign ring is refused, naming it" +
            (error.empty() ? "" : ": " + error));
  Check(Same(before, *peer),
        "and the target kept its tokens, frontier and logits");

  // --------------------------------------------------------- full context
  // The size a serve path pays: a context filled to the cache's bound, where
  // the sliding rings have wrapped so that one slot holds the key of several
  // positions and only `position` says which. Restoring that has to be as exact
  // as restoring the short one.
  std::vector<k1::TokenId> deep = tokens;
  while (deep.size() < model->max_context()) {
    const std::vector<k1::TokenId> more = model->tokenizer().Encode(
        " Die Stadt hat viele Brücken, und im Sommer ist es dort warm.", false,
        true);
    deep.insert(deep.end(), more.begin(), more.end());
  }
  deep.resize(model->max_context());
  session->Reset();
  session->Append(deep);
  const std::vector<float> deep_frontier(session->logits().begin(),
                                         session->logits().end());
  const std::uint64_t deep_promised = session->SnapshotBytes();
  error.clear();
  const auto deep_saved_at = std::chrono::steady_clock::now();
  auto deep_snapshot = session->SaveSnapshot(&error);
  const double deep_save_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - deep_saved_at)
          .count();
  Check(deep_snapshot != nullptr, "a context filled to the bound saves" +
                                      (error.empty() ? "" : ": " + error));
  if (deep_snapshot == nullptr) {
    return 1;
  }
  Check(
      deep_snapshot->SizeBytes() == deep_promised,
      "and is the size promised: " + std::to_string(deep_promised) + " bytes");
  const auto deep_restored_at = std::chrono::steady_clock::now();
  error.clear();
  const bool deep_restored = peer->RestoreSnapshot(*deep_snapshot, &error);
  const double deep_restore_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - deep_restored_at)
          .count();
  Check(deep_restored, "and restores into the other session" +
                           (error.empty() ? "" : ": " + error));
  Check(peer->position() == deep.size(), "the deep frontier is " +
                                             std::to_string(peer->position()) +
                                             " positions");
  Check(Identical(deep_frontier, peer->logits()),
        "and its logits are bit-equal at the bound");

  std::cout << "payload at " << deep.size() << " positions: "
            << static_cast<double>(deep_snapshot->SizeBytes()) /
                   (1024.0 * 1024.0)
            << " MiB, save " << deep_save_ms << " ms, restore "
            << deep_restore_ms << " ms\n";

  if (failures != 0) {
    std::cout << failures << " snapshot checks failed\n";
    return 1;
  }
  std::cout << "a saved kolibri1 context answers like the conversation it came "
               "from\n";
  return 0;
}
