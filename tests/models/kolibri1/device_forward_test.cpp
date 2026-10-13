// Kolibri 1 on the device against the scalar oracle on the same artifact.
//
// This is the check that says the engine answers: the oracle walks the graph in
// double with the released weights and the kernels walk it on the matrix cores,
// so agreement means the operators, the routing rules, the cache layout and the
// order of a layer all mean the same thing in both. What it compares is the
// frontier logits over a short prefix and the token sampled from them, then one
// decode step from that token. Before that it pins two things bit for bit — the
// embedding row as the file packs it and as the device holds it, and the gather
// against the host dequantizer — because a wrong upload or a wrong row length
// misreads as a wrong kernel in every stage above them, and a prefix that only
// agrees until the first mixture layer shows as a second row that differs.
//
// The oracle decodes every weight it touches, so this stays at a handful of
// tokens and is labelled slow. Without GUFO_KOLIBRI1_MODEL it skips with 77, and
// without a gfx1151 device likewise.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_gemm.hpp"
#include "src/models/kolibri1/engine.hpp"
#include "src/models/kolibri1/kernels/rocm/moe.hpp"
#include "src/models/kolibri1/reference.hpp"

namespace k1 = gufo::models::kolibri1;

namespace {

int failures = 0;

/// The largest relative difference of two logit rows, and the entries the two
/// agree on being largest.
struct Diff {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  std::int32_t device_best = -1;
  std::int32_t oracle_best = -1;
};

Diff Compare(const std::vector<float>& device, const std::vector<float>& oracle) {
  Diff d;
  double total = 0.0;
  float device_peak = -std::numeric_limits<float>::infinity();
  float oracle_peak = device_peak;
  for (std::size_t i = 0; i < device.size(); ++i) {
    const double diff = std::abs(static_cast<double>(device[i]) -
                                 static_cast<double>(oracle[i]));
    total += diff;
    if (diff > d.max_abs) {
      d.max_abs = diff;
    }
    if (device[i] > device_peak) {
      device_peak = device[i];
      d.device_best = static_cast<std::int32_t>(i);
    }
    if (oracle[i] > oracle_peak) {
      oracle_peak = oracle[i];
      d.oracle_best = static_cast<std::int32_t>(i);
    }
  }
  d.mean_abs = total / static_cast<double>(device.size());
  return d;
}

void Report(const char* what, const Diff& d, double max_abs, double mean_abs) {
  const bool ok = d.device_best == d.oracle_best && d.max_abs <= max_abs &&
                  d.mean_abs <= mean_abs;
  if (!ok) {
    ++failures;
  }
  std::cout << (ok ? "PASS " : "FAIL ") << what << ": max_abs=" << d.max_abs
            << " mean_abs=" << d.mean_abs << " argmax device=" << d.device_best
            << " oracle=" << d.oracle_best << "\n";
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

  // The reader stays mapped while the oracle reads its weights, so both views
  // of the file are the same bytes.
  auto reader = gufo::core::GgufReader::OpenFile(path);
  if (!reader) {
    std::cout << "FAIL cannot open " << path << "\n";
    return 1;
  }
  const auto weights = k1::ModelWeights::Bind(*reader);
  if (!weights.has_value()) {
    std::cout << "FAIL cannot bind " << path << "\n";
    return 1;
  }

  const std::string prompt = "Hauptstadt von Deutschland ist";
  auto model = k1::Model::Load(path, k1::ModelOptions{.max_context = 1024,
                                                      .prefill_chunk = 16});
  if (!model) {
    std::cout << "FAIL cannot load the model on the device\n";
    return 1;
  }
  std::cout << "loaded " << model->resident_bytes() / (1024U * 1024U)
            << " MiB of weights\n";
  auto session = model->CreateSession();
  if (!session) {
    std::cout << "FAIL cannot create a session\n";
    return 1;
  }
  const std::vector<std::int32_t> tokens =
      model->tokenizer().Encode(prompt, false, true);
  if (tokens.size() < 3) {
    std::cout << "FAIL the prompt tokenized to " << tokens.size()
              << " tokens; the check needs a real sentence\n";
    return 1;
  }

  // Two things the forward depends on before its arithmetic means anything:
  // the table on the device has to be the file's own bytes, and the gather has
  // to read them the way the host dequantizer does. A staging offset that is
  // off by one shard, or a row length the format does not have, breaks every
  // stage below it and reads there like a wrong kernel.
  const auto& host_embd = weights->token_embd;
  const auto& device_embd = model->device().token_embd();
  const std::size_t row_bytes = host_embd.RowBytes();
  {
    std::vector<std::uint8_t> uploaded(row_bytes), packed(row_bytes);
    HIP_CHECK(hipMemcpy(uploaded.data(),
                        static_cast<const std::uint8_t*>(device_embd.data) +
                            static_cast<std::size_t>(tokens[0]) * row_bytes,
                        row_bytes, hipMemcpyDeviceToHost));
    std::memcpy(packed.data(),
                static_cast<const std::uint8_t*>(host_embd.data) +
                    static_cast<std::size_t>(tokens[0]) * row_bytes,
                row_bytes);
    const bool exact = std::equal(uploaded.begin(), uploaded.end(),
                                  packed.begin());
    if (!exact) {
      ++failures;
    }
    std::cout << (exact ? "PASS " : "FAIL ")
              << "the device holds an embedding row as the file packs it\n";
  }
  {
    // Rows the prompt visits and the last row of the table, which a prompt of
    // this length never reaches: the whole table is addressed, not its start.
    const std::size_t d = weights->config.hidden_size;
    const std::vector<std::int32_t> probe{
        0, 1, tokens[0],
        static_cast<std::int32_t>(weights->VocabSize() - 1)};
    std::int32_t* ids = nullptr;
    float* rows = nullptr;
    HIP_CHECK(hipMalloc(&ids, probe.size() * sizeof(std::int32_t)));
    HIP_CHECK(hipMalloc(&rows, probe.size() * d * sizeof(float)));
    HIP_CHECK(hipMemcpy(ids, probe.data(), probe.size() * sizeof(std::int32_t),
                        hipMemcpyHostToDevice));
    k1::rocm::EmbedRows({device_embd.data, device_embd.quant(), ids, rows,
                         static_cast<std::uint32_t>(probe.size()),
                         static_cast<std::uint32_t>(d)},
                        nullptr);
    std::vector<float> got(d), want(d);
    double worst = 0.0;
    for (std::size_t r = 0; r < probe.size(); ++r) {
      HIP_CHECK(hipMemcpy(got.data(), rows + r * d, d * sizeof(float),
                          hipMemcpyDeviceToHost));
      gufo::quant::Dequantize(
          host_embd.type,
          static_cast<const std::uint8_t*>(host_embd.data) +
              static_cast<std::size_t>(probe[r]) * row_bytes,
          want.data(), d);
      for (std::size_t i = 0; i < d; ++i) {
        worst = std::max(worst,
                         static_cast<double>(std::abs(got[i] - want[i])));
      }
    }
    if (worst != 0.0) {
      ++failures;
    }
    std::cout << (worst == 0.0 ? "PASS " : "FAIL ")
              << "the gather decodes the table like the host: max_abs=" << worst
              << "\n";
    (void)hipFree(ids);
    (void)hipFree(rows);
  }

  // Prefill on the device, then the same tokens through the oracle. The oracle
  // returns one logits row per token; the last is the frontier both compared.
  session->Append(tokens);

  const std::vector<float> device_first(session->logits().begin(),
                                        session->logits().end());
  k1::Reference reference(*weights);
  std::vector<float> oracle_logits;
  reference.Forward(tokens, &oracle_logits);
  const std::size_t vocab = weights->VocabSize();
  const std::vector<float> oracle_last(
      oracle_logits.end() - static_cast<std::ptrdiff_t>(vocab),
      oracle_logits.end());
  const Diff prefill = Compare(device_first, oracle_last);
  Report("prefill", prefill, 0.5, 0.02);

  // One decode step from the token both sides agree on, which is the state a
  // conversation is actually in: everything in the cache, one row to add.
  const std::int32_t next = prefill.device_best;
  session->Append({&next, 1});
  std::vector<float> device_next(session->logits().begin(),
                                 session->logits().end());
  std::vector<float> oracle_next;
  reference.Forward({&next, 1}, &oracle_next);
  Report("decode step", Compare(device_next, oracle_next), 0.5, 0.02);

  if (failures != 0) {
    std::cout << failures << " device checks failed\n";
    return 1;
  }
  std::cout << "kolibri1 on the device matches the scalar reference\n";
  return 0;
}
