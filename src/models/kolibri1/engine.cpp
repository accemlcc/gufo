#include "src/models/kolibri1/engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>

#include "src/models/kolibri1/chat_template.hpp"

namespace gufo::models::kolibri1 {
namespace {

/// 'K','L','S','1' in the byte order a payload is written in. A context of
/// another architecture has to be refused, not read as this one.
constexpr std::uint32_t kSnapshotMagic =
    static_cast<std::uint32_t>('K') | (static_cast<std::uint32_t>('L') << 8) |
    (static_cast<std::uint32_t>('S') << 16) |
    (static_cast<std::uint32_t>('1') << 24);

void Put32(std::uint8_t* at, std::uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i) {
    at[i] = static_cast<std::uint8_t>(value >> (8 * i));
  }
}

void Put64(std::uint8_t* at, std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i) {
    at[i] = static_cast<std::uint8_t>(value >> (8 * i));
  }
}

std::uint32_t Get32(const std::uint8_t* at) {
  std::uint32_t value = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    value |= static_cast<std::uint32_t>(at[i]) << (8 * i);
  }
  return value;
}

std::uint64_t Get64(const std::uint8_t* at) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    value |= static_cast<std::uint64_t>(at[i]) << (8 * i);
  }
  return value;
}

/// The header of a payload: what a restore needs to know before it can place
/// a row. Two fields are not redundant with the artifact - `position`, because
/// a sliding layer holds the key of position `p` in slot `p % ring`, so its
/// rows mean nothing without the frontier they were taken at, and `ring`,
/// because it is one of the two moduli those rows were written with.
///
/// The fields go on the wire one at a time, little-endian, in the order the
/// format names them and with no padding: a `u64` that follows seven `u32`
/// has no offset a compiler guarantees, so the record is written field by
/// field and its size is a number rather than a layout accident.
struct SnapshotHeader {
  /// Magic, version, position, ring, max_context, layers and vocab take four
  /// bytes each, the three sizes eight.
  static constexpr std::size_t kBytes = 52;

  std::uint32_t magic{0};
  std::uint32_t version{0};
  std::uint32_t position{0};
  std::uint32_t ring{0};
  std::uint32_t max_context{0};
  std::uint32_t layers{0};
  std::uint32_t vocab{0};
  std::uint64_t token_bytes{0};
  std::uint64_t kv_bytes{0};
  std::uint64_t logit_bytes{0};

  void Write(std::uint8_t* at) const {
    Put32(at, magic);
    Put32(at + 4, version);
    Put32(at + 8, position);
    Put32(at + 12, ring);
    Put32(at + 16, max_context);
    Put32(at + 20, layers);
    Put32(at + 24, vocab);
    Put64(at + 28, token_bytes);
    Put64(at + 36, kv_bytes);
    Put64(at + 44, logit_bytes);
  }

  void Read(const std::uint8_t* at) {
    magic = Get32(at);
    version = Get32(at + 4);
    position = Get32(at + 8);
    ring = Get32(at + 12);
    max_context = Get32(at + 16);
    layers = Get32(at + 20);
    vocab = Get32(at + 24);
    token_bytes = Get64(at + 28);
    kv_bytes = Get64(at + 36);
    logit_bytes = Get64(at + 44);
  }
};

bool Fail(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
  return false;
}

/// Turns a device call's status into the same kind of refusal. A snapshot is
/// taken and dropped beside a running conversation, so a failed copy reports
/// rather than throws: the session it came from keeps answering.
[[nodiscard]] bool DeviceOk(hipError_t status, const char* what,
                            std::uint64_t bytes, std::string* error_msg) {
  if (status == hipSuccess) {
    return true;
  }
  return Fail(error_msg, "kolibri1 snapshot " + std::string(what) + " of " +
                             std::to_string(bytes) +
                             " bytes failed: " + hipGetErrorString(status));
}

/// Queues one run of the payload. A zero-length run is skipped rather than
/// handed to the runtime: a context at position zero carries no tokens and no
/// rows of the ten layers that index by position, and its vectors are empty.
[[nodiscard]] bool Queue(hipStream_t stream, void* destination,
                         const void* source, std::uint64_t bytes,
                         hipMemcpyKind kind, const char* what,
                         std::string* error_msg) {
  if (bytes == 0) {
    return true;
  }
  return DeviceOk(hipMemcpyAsync(destination, source, bytes, kind, stream),
                  what, bytes, error_msg);
}

/// The same over host memory, where a copy through a null pointer is
/// undefined however few bytes it moves.
void CopyHost(void* destination, const void* source, std::uint64_t bytes) {
  if (bytes != 0) {
    std::memcpy(destination, source, bytes);
  }
}

/// Bytes of one cache row: every key or value head of the layer in binary16.
/// `Executor::CreateCache` pads between the K and the V of a layer and never
/// between rows, so a run of rows is contiguous on both sides of a copy.
[[nodiscard]] std::uint64_t KvRowBytes(const Config& config,
                                       std::uint32_t layer) noexcept {
  return static_cast<std::uint64_t>(config.KvDim(layer)) * 2;
}

/// The rows of `layer` a payload carries: rows [0, position) for a layer
/// without positional encoding, which is defined to read the whole history,
/// and the whole ring for a sliding one, copied as the kernel stored it.
[[nodiscard]] std::uint64_t KvRows(const Config& config, std::uint32_t layer,
                                   std::uint32_t position,
                                   std::uint32_t ring) noexcept {
  return config.IsSliding(layer) ? ring : position;
}

/// The bytes of `layer` in the kv section, both halves.
[[nodiscard]] std::uint64_t KvLayerBytes(const Config& config,
                                         std::uint32_t layer,
                                         std::uint32_t position,
                                         std::uint32_t ring) noexcept {
  return 2 * KvRows(config, layer, position, ring) * KvRowBytes(config, layer);
}

/// Every layer's kv bytes, in the order the payload carries them.
[[nodiscard]] std::uint64_t KvPayloadBytes(const Config& config,
                                           std::uint32_t position,
                                           std::uint32_t ring) noexcept {
  std::uint64_t bytes = 0;
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    bytes += KvLayerBytes(config, l, position, ring);
  }
  return bytes;
}

/// Calls `run` with the K and then the V of every layer, in payload order, and
/// stops at the first run that reports false. The save, the export and both
/// restore paths move exactly these runs; only their direction differs, so
/// this is the one place the kv section's layout is written down.
template<class Run>
[[nodiscard]] bool ForEachKvRun(const Config& config, std::uint32_t position,
                                std::uint32_t ring, const Run& run) {
  for (std::uint32_t l = 0; l < config.num_layers; ++l) {
    const std::uint64_t bytes =
        KvRows(config, l, position, ring) * KvRowBytes(config, l);
    if (!run(l, true, bytes) || !run(l, false, bytes)) {
      return false;
    }
  }
  return true;
}

/// Checks a payload against the destination before a byte of it is written.
/// The geometry has to agree rather than be adapted: a ring of another width
/// puts every sliding row in a slot the attention kernel will not look in,
/// and another `max_context` moves the rows of the ten layers that index by
/// position, so a restore that reindexed on the way in would answer from a
/// context it had shuffled itself. Every refusal names what disagreed.
[[nodiscard]] bool CheckHeader(const SnapshotHeader& header,
                               std::uint64_t payload_bytes,
                               const Config& config, std::uint32_t vocab,
                               const rocm::KvCache& cache,
                               std::string* error_msg) {
  if (header.magic != kSnapshotMagic) {
    return Fail(error_msg, "kolibri1 payload magic is " +
                               std::to_string(header.magic) + ", not the " +
                               std::to_string(kSnapshotMagic) +
                               " of a kolibri1 context");
  }
  if (header.version != Session::kSnapshotPayloadVersion) {
    return Fail(error_msg,
                "kolibri1 payload version is " +
                    std::to_string(header.version) +
                    ", this build "
                    "reads version " +
                    std::to_string(Session::kSnapshotPayloadVersion));
  }
  if (header.layers != config.num_layers) {
    return Fail(error_msg, "kolibri1 payload holds " +
                               std::to_string(header.layers) +
                               " layers, this model has " +
                               std::to_string(config.num_layers));
  }
  if (header.vocab != vocab) {
    return Fail(error_msg, "kolibri1 payload logits cover " +
                               std::to_string(header.vocab) +
                               " entries, this model " + std::to_string(vocab));
  }
  if (header.ring != cache.ring) {
    return Fail(error_msg,
                "kolibri1 payload ring is " + std::to_string(header.ring) +
                    " slots, this cache " + std::to_string(cache.ring));
  }
  if (header.max_context != cache.max_context) {
    return Fail(error_msg, "kolibri1 payload context is " +
                               std::to_string(header.max_context) +
                               " positions, this cache " +
                               std::to_string(cache.max_context));
  }
  if (header.position > cache.max_context) {
    return Fail(error_msg,
                "kolibri1 payload position " + std::to_string(header.position) +
                    " is past the " + std::to_string(cache.max_context) +
                    " positions of this cache");
  }
  const std::uint64_t token_bytes =
      static_cast<std::uint64_t>(header.position) * sizeof(TokenId);
  const std::uint64_t kv_bytes =
      KvPayloadBytes(config, header.position, cache.ring);
  const std::uint64_t logit_bytes =
      static_cast<std::uint64_t>(vocab) * sizeof(float);
  if (header.token_bytes != token_bytes) {
    return Fail(error_msg, "kolibri1 payload names " +
                               std::to_string(header.token_bytes) +
                               " token bytes, its position needs " +
                               std::to_string(token_bytes));
  }
  if (header.kv_bytes != kv_bytes) {
    return Fail(error_msg, "kolibri1 payload names " +
                               std::to_string(header.kv_bytes) +
                               " kv bytes, its geometry holds " +
                               std::to_string(kv_bytes));
  }
  if (header.logit_bytes != logit_bytes) {
    return Fail(error_msg, "kolibri1 payload names " +
                               std::to_string(header.logit_bytes) +
                               " logit bytes, one row of this vocabulary is " +
                               std::to_string(logit_bytes));
  }
  const std::uint64_t described =
      SnapshotHeader::kBytes + token_bytes + kv_bytes + logit_bytes;
  if (payload_bytes != described) {
    return Fail(error_msg, "kolibri1 payload is " +
                               std::to_string(payload_bytes) +
                               " bytes, its header describes " +
                               std::to_string(described));
  }
  return true;
}

/// The size of a header plus its body, which is what `SizeBytes` reports and
/// what the writer of a payload must reserve.
[[nodiscard]] constexpr std::uint64_t PayloadBytes(
    std::uint64_t body_bytes) noexcept {
  return SnapshotHeader::kBytes + body_bytes;
}

/// The body of a context of this geometry at `position`: its tokens, the rows
/// of every layer, and one logits row.
[[nodiscard]] std::uint64_t BodyBytes(const Config& config,
                                      std::uint32_t position,
                                      std::uint32_t ring,
                                      std::uint32_t vocab) noexcept {
  return static_cast<std::uint64_t>(position) * sizeof(TokenId) +
         KvPayloadBytes(config, position, ring) +
         static_cast<std::uint64_t>(vocab) * sizeof(float);
}

}  // namespace

/// What a saved context holds: its header on the host, and one device buffer
/// with the body. The rows stay on the device because the restore writes them
/// back into a cache, and a round trip through a host buffer of the payload's
/// size would pay for itself twice on a machine whose processor and matrix
/// cores share one pool - once to write the pages and once to fault them in.
/// `Stream` therefore stages one piece at a time instead.
struct SessionSnapshot::Impl {
  ~Impl() {
    if (body != nullptr) {
      (void)hipFree(body);
    }
  }

  SnapshotHeader header{};
  /// Tokens, then the K and the V of every layer, then the logits row.
  void* body{nullptr};
  std::uint64_t body_bytes{0};
  /// The stream of the executor that froze it. Every forward of that model is
  /// queued on it, so these copies are ordered against them - and they have
  /// to be on that stream and not on the legacy one, which
  /// `hipStreamNonBlocking` shares no ordering with.
  hipStream_t stream{nullptr};
  /// One piece of the body for `Stream`. A snapshot is exported by one writer
  /// at a time; the buffer is reused between pieces, not copied per call.
  mutable std::vector<std::uint8_t> piece;
};

std::unique_ptr<Model> Model::Load(const std::string& path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::unique_ptr<Model> m(new Model());
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  // The engine implements one prompt format, and a file whose template differs
  // would be answered in a dialect the model was never shown.
  if (!ChatTemplate::ValidateGgufTemplate(*m->reader_, error_msg)) {
    return nullptr;
  }
  m->weights_ = ModelWeights::Bind(*m->reader_, error_msg);
  if (!m->weights_.has_value()) {
    return nullptr;
  }
  m->tokenizer_ = Tokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_, error_msg);
  if (!m->device_) {
    return nullptr;
  }
  // The logits are as wide as the output layer, the ids as long as the
  // tokenizer's list, and constrained sampling refuses a mask that does not fit
  // the row it masks. A file whose output layer is padded past its token list
  // would fail only on the first structured request, with a complaint about a
  // mask - so the two widths are tied together here, where the answer is still
  // "that artifact is not for this engine".
  if (m->tokenizer_->VocabSize() != m->device_->vocab_size()) {
    Fail(error_msg, "kolibri1 vocabulary lists " +
                        std::to_string(m->tokenizer_->VocabSize()) +
                        " entries, the output layer scores " +
                        std::to_string(m->device_->vocab_size()));
    return nullptr;
  }
  // Positions past the trained length are nothing to try: the ten global layers
  // carry no positional encoding at all, so outside the trained range not even
  // the sliding layers' relative positions have a reference. Refuse before the
  // cache is allocated at a width the model was never shown.
  if (options.max_context > m->weights_->config.context_length) {
    Fail(error_msg, "kolibri1 context " + std::to_string(options.max_context) +
                        " exceeds the trained " +
                        std::to_string(m->weights_->config.context_length));
    return nullptr;
  }
  m->executor_ = std::make_shared<rocm::Executor>(
      *m->device_, options.prefill_chunk, 1, options.max_context);
  return m;
}

Model::~Model() = default;

std::unique_ptr<Session> Model::CreateSession(std::string* error_msg) const {
  auto cache = executor_->CreateCache(options_.max_context, error_msg);
  if (cache == nullptr) {
    return nullptr;
  }
  return std::make_unique<Session>(*this, std::move(cache));
}

Session::Session(const Model& model, std::unique_ptr<rocm::KvCache> cache)
    : model_(model), cache_(std::move(cache)) {}

void Session::Reset() noexcept {
  // Nothing to free and no marker to clear: the cache holds no length, the
  // attention kernels read exactly the positions a forward names, and the next
  // Append starts writing at zero again.
  tokens_.clear();
  logits_.clear();
}

void Session::Append(std::span<const TokenId> tokens) {
  if (tokens.empty()) {
    return;
  }
  if (position() + tokens.size() > model_.max_context()) {
    throw std::invalid_argument("kolibri1 session past its " +
                                std::to_string(model_.max_context()) +
                                " position context");
  }
  const std::uint32_t chunk =
      static_cast<std::uint32_t>(model_.executor()->max_rows());
  std::size_t done = 0;
  while (done < tokens.size()) {
    const std::size_t n = std::min<std::size_t>(chunk, tokens.size() - done);
    // Only the very last row of the whole append is asked for logits: the rows
    // before it feed nothing. The executor numbers the rows it gathers within
    // the forward it is running, not within the append, so the frontier of the
    // final chunk is that chunk's last row - `done + n - 1` would name a row
    // past a chunk shorter than the append, and the forward rejects that.
    const bool last = done + n == tokens.size();
    const std::uint32_t frontier = static_cast<std::uint32_t>(n) - 1;
    // DIAGNOSTIC ONLY, for the open race: asking for the frontier row of every
    // chunk makes the head run where an append would skip it, so one append
    // then does the per-chunk work an append split at that boundary does.
    const bool each = std::getenv("GUFO_KOLIBRI_LOGITS_EACH_CHUNK") != nullptr;
    const std::span<const std::uint32_t> rows =
        (last || each) ? std::span<const std::uint32_t>(&frontier, 1)
                       : std::span<const std::uint32_t>{};
    model_.executor()->Forward(*cache_, tokens.subspan(done, n), position(),
                               rows);
    tokens_.insert(tokens_.end(), tokens.begin() + done,
                   tokens.begin() + done + n);
    done += n;
  }
  model_.executor()->CopyLogits(1, &logits_);
}

SessionSnapshot::~SessionSnapshot() = default;

std::uint64_t SessionSnapshot::SizeBytes() const noexcept {
  return PayloadBytes(impl_->body_bytes);
}

bool SessionSnapshot::CopyTo(std::span<std::uint8_t> destination) const {
  // Exactly the payload or nothing: a caller whose own arithmetic disagrees
  // with `SizeBytes` is one field misread away from a truncated file, and a
  // half-written buffer is the worse way to find that out.
  if (destination.size() != SizeBytes()) {
    return false;
  }
  impl_->header.Write(destination.data());
  return Queue(impl_->stream, destination.data() + SnapshotHeader::kBytes,
               impl_->body, impl_->body_bytes, hipMemcpyDeviceToHost, "export",
               nullptr) &&
         DeviceOk(hipStreamSynchronize(impl_->stream), "export",
                  impl_->body_bytes, nullptr);
}

bool SessionSnapshot::Stream(const Sink& sink) const {
  // 64 MiB per piece: the ceiling a reader has to be ready for, and the reason
  // to stream at all. The body of a full 8192 position context is 241 MiB, and
  // handing it over in one piece would need host pages of that size touched
  // once and never read again.
  constexpr std::uint64_t kPieceBytes = std::uint64_t{64} << 20;
  std::array<std::uint8_t, SnapshotHeader::kBytes> head{};
  impl_->header.Write(head.data());
  sink({head.data(), head.size()});
  impl_->piece.resize(static_cast<std::size_t>(kPieceBytes));
  const auto* body = static_cast<const std::uint8_t*>(impl_->body);
  for (std::uint64_t done = 0; done < impl_->body_bytes;) {
    const std::uint64_t bytes = std::min(kPieceBytes, impl_->body_bytes - done);
    if (!Queue(impl_->stream, impl_->piece.data(), body + done, bytes,
               hipMemcpyDeviceToHost, "export", nullptr) ||
        !DeviceOk(hipStreamSynchronize(impl_->stream), "export", bytes,
                  nullptr)) {
      return false;
    }
    sink({impl_->piece.data(), static_cast<std::size_t>(bytes)});
    done += bytes;
  }
  return true;
}

std::uint64_t Session::SnapshotBytes() const noexcept {
  // The capacity of the cache is not in this: a sliding layer costs its ring
  // whether or not the context is full, and a layer without positional
  // encoding only carries the rows a token has written.
  return PayloadBytes(BodyBytes(model_.config(), position(), cache_->ring,
                                model_.device().vocab_size()));
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  const Config& c = model_.config();
  const std::uint32_t vocab = model_.device().vocab_size();
  if (logits_.size() != vocab) {
    Fail(error_msg,
         "a kolibri1 session that has not evaluated a forward has no frontier "
         "logits to save");
    return nullptr;
  }
  const std::uint32_t position = this->position();
  auto impl = std::make_unique<SessionSnapshot::Impl>();
  impl->header = SnapshotHeader{
      .magic = kSnapshotMagic,
      .version = Session::kSnapshotPayloadVersion,
      .position = position,
      .ring = cache_->ring,
      .max_context = cache_->max_context,
      .layers = c.num_layers,
      .vocab = vocab,
      .token_bytes = static_cast<std::uint64_t>(position) * sizeof(TokenId),
      .kv_bytes = KvPayloadBytes(c, position, cache_->ring),
      .logit_bytes = static_cast<std::uint64_t>(vocab) * sizeof(float)};
  impl->body_bytes = BodyBytes(c, position, cache_->ring, vocab);
  if (!DeviceOk(hipMalloc(&impl->body, impl->body_bytes), "allocation",
                impl->body_bytes, error_msg)) {
    return nullptr;
  }
  impl->stream = model_.executor()->stream();

  // Every run is queued on the executor's stream and the queue is waited for
  // once. Nothing else has to freeze the session: a forward a later `Append`
  // starts can only be queued behind these copies, so it cannot overwrite a
  // row still on its way into the snapshot.
  auto* at = static_cast<std::uint8_t*>(impl->body);
  if (!Queue(impl->stream, at, tokens_.data(), impl->header.token_bytes,
             hipMemcpyHostToDevice, "token copy", error_msg)) {
    return nullptr;
  }
  at += impl->header.token_bytes;
  const bool rows = ForEachKvRun(
      c, position, cache_->ring,
      [&](std::uint32_t l, bool key, std::uint64_t bytes) {
        const void* source = key ? static_cast<const void*>(cache_->k[l])
                                 : static_cast<const void*>(cache_->v[l]);
        const bool ok = Queue(impl->stream, at, source, bytes,
                              hipMemcpyDeviceToDevice, "cache copy", error_msg);
        at += bytes;
        return ok;
      });
  if (!rows ||
      !Queue(impl->stream, at, logits_.data(), impl->header.logit_bytes,
             hipMemcpyHostToDevice, "logit copy", error_msg) ||
      !DeviceOk(hipStreamSynchronize(impl->stream), "save", impl->body_bytes,
                error_msg)) {
    return nullptr;
  }
  auto snapshot = std::unique_ptr<SessionSnapshot>(new SessionSnapshot());
  snapshot->impl_ = std::move(impl);
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  const Config& c = model_.config();
  const std::uint32_t vocab = model_.device().vocab_size();
  const SnapshotHeader& header = snapshot.impl_->header;
  if (!CheckHeader(header, snapshot.SizeBytes(), c, vocab, *cache_,
                   error_msg)) {
    return false;
  }
  const hipStream_t stream = model_.executor()->stream();
  const auto* at = static_cast<const std::uint8_t*>(snapshot.impl_->body);
  // The rows move from one device buffer to the other, which is what keeping
  // them on the device buys; the two host sections land in locals and only
  // replace the session's own once every copy has been waited for, so a
  // failure cannot leave a session claiming a position its cache does not
  // hold.
  std::vector<TokenId> tokens(header.position);
  std::vector<float> logits(vocab);
  if (!Queue(stream, tokens.data(), at, header.token_bytes,
             hipMemcpyDeviceToHost, "token restore", error_msg)) {
    return false;
  }
  at += header.token_bytes;
  const bool rows =
      ForEachKvRun(c, header.position, cache_->ring,
                   [&](std::uint32_t l, bool key, std::uint64_t bytes) {
                     void* destination = key ? static_cast<void*>(cache_->k[l])
                                             : static_cast<void*>(cache_->v[l]);
                     const bool ok = Queue(stream, destination, at, bytes,
                                           hipMemcpyDeviceToDevice,
                                           "cache restore", error_msg);
                     at += bytes;
                     return ok;
                   });
  if (!rows ||
      !Queue(stream, logits.data(), at, header.logit_bytes,
             hipMemcpyDeviceToHost, "logit restore", error_msg) ||
      !DeviceOk(hipStreamSynchronize(stream), "restore", snapshot.SizeBytes(),
                error_msg)) {
    return false;
  }
  tokens_ = std::move(tokens);
  logits_ = std::move(logits);
  return true;
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  if (payload.size() < SnapshotHeader::kBytes) {
    Fail(error_msg, "kolibri1 payload is " + std::to_string(payload.size()) +
                        " bytes, shorter than its " +
                        std::to_string(SnapshotHeader::kBytes) +
                        " byte header");
    return false;
  }
  SnapshotHeader header{};
  header.Read(payload.data());
  const Config& c = model_.config();
  const std::uint32_t vocab = model_.device().vocab_size();
  if (!CheckHeader(header, payload.size(), c, vocab, *cache_, error_msg)) {
    return false;
  }
  const hipStream_t stream = model_.executor()->stream();
  const auto* at = payload.data() + SnapshotHeader::kBytes;
  std::vector<TokenId> tokens(header.position);
  std::vector<float> logits(vocab);
  CopyHost(tokens.data(), at, header.token_bytes);
  at += header.token_bytes;
  const bool rows =
      ForEachKvRun(c, header.position, cache_->ring,
                   [&](std::uint32_t l, bool key, std::uint64_t bytes) {
                     void* destination = key ? static_cast<void*>(cache_->k[l])
                                             : static_cast<void*>(cache_->v[l]);
                     const bool ok = Queue(stream, destination, at, bytes,
                                           hipMemcpyHostToDevice,
                                           "cache restore", error_msg);
                     at += bytes;
                     return ok;
                   });
  if (!rows) {
    return false;
  }
  CopyHost(logits.data(), at, header.logit_bytes);
  if (!DeviceOk(hipStreamSynchronize(stream), "restore", payload.size(),
                error_msg)) {
    return false;
  }
  tokens_ = std::move(tokens);
  logits_ = std::move(logits);
  return true;
}

}  // namespace gufo::models::kolibri1
