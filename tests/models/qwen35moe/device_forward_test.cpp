// Qwen3.5-MoE's device forward against the scalar oracle on the same
// artifact: a short prefill plus one decode step, compared by top-1 and by
// absolute error. This is the contract the engine ships under: the same
// argmax as the reference at every frontier, within half a logit.
//
// Needs the artifact named by GUFO_QWEN35MOE_MODEL and a gfx1151 device;
// without either the check skips with 77.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen35moe/engine.hpp"
#include "src/models/qwen35moe/reference.hpp"
#include "src/models/qwen35moe/weights.hpp"

namespace m = gufo::models::qwen35moe;

namespace {

int failures = 0;

void Report(const char* name, bool ok, const std::string& detail) {
  std::cout << (ok ? "PASS " : "FAIL ") << name << ": " << detail << "\n";
  if (!ok) {
    ++failures;
  }
}

std::vector<m::TokenId> Tokens() {
  return {760, 6511, 314, 9564, 369, 19241, 13, 561};
}

void Forward(const std::string& path) {
  m::ModelOptions options;
  options.max_context = 256;
  options.prefill_chunk = 16;
  std::string error;
  auto model = m::Model::Load(path, options, &error);
  if (!model) {
    Report("device.forward.load", false, error);
    return;
  }
  auto session = model->CreateSession(&error);
  if (!session) {
    Report("device.forward.session", false, error);
    return;
  }
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (!reader) {
    Report("device.forward.reader", false, error);
    return;
  }
  auto weights = m::ModelWeights::LoadFromGguf(*reader, &error);
  if (!weights.has_value()) {
    Report("device.forward.weights", false, error);
    return;
  }
  m::Reference ref(*weights);
  const std::vector<m::TokenId> ids = Tokens();
  bool ok = true;
  std::string detail;
  // Prefill the first six, then decode two more one at a time: the second
  // phase reads the caches the first one wrote.
  const std::vector<std::size_t> splits = {6, 7, 8};
  std::size_t done = 0;
  for (std::size_t k = 0; k < splits.size() && ok; ++k) {
    std::vector<m::TokenId> chunk(ids.begin() + done, ids.begin() + splits[k]);
    session->Append(chunk);
    std::vector<float> ref_logits;
    std::vector<std::int32_t> prefix(ids.begin(), ids.begin() + splits[k]);
    // The oracle appends after its own frontier, so restart it: each
    // compared frontier is the same full prefix the session has seen.
    ref.Reset();
    ref.Forward(prefix, &ref_logits);
    // The session keeps only the requested (last) row; the oracle returns
    // every position, so compare against its last row.
    const std::size_t v = session->logits().size();
    const std::size_t n = splits[k];
    if (v == 0 || ref_logits.size() != n * v) {
      ok = false;
      detail = "logit width mismatch";
      break;
    }
    const float* ref_last = ref_logits.data() + (n - 1) * v;
    std::size_t got_top = 0, want_top = 0;
    for (std::size_t i = 0; i < v; ++i) {
      if (session->logits()[i] > session->logits()[got_top]) {
        got_top = i;
      }
      if (ref_last[i] > ref_last[want_top]) {
        want_top = i;
      }
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < v; ++i) {
      worst = std::max(
          worst,
          std::fabs(static_cast<double>(session->logits()[i]) - ref_last[i]));
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "k=%zu got=%zu want=%zu worst=%.3f",
                  splits[k], got_top, want_top, worst);
    if (got_top != want_top || worst > 0.5) {
      ok = false;
      detail = buf;
    }
    done = splits[k];
  }
  Report("device.forward", ok, ok ? "3 frontiers agree" : detail);
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::cout << "SKIP: no gfx1151 device\n";
    return 77;
  }
  const char* path = std::getenv("GUFO_QWEN35MOE_MODEL");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: no model\n";
    return 77;
  }
  Forward(path);
  if (failures != 0) {
    std::cout << "FAILURES: " << failures << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
