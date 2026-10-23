// Qwen3.5-MoE's mixture block on the device against a CPU oracle over the
// same mapped weights: RMSNorm, softmax top-8 router, grouped gate/up,
// activation, grouped down, sigmoid-gated shared expert and the finishing
// norm. The oracle decodes with the scalar CPU path and accumulates in
// double, so the only difference under test is the kernels' own arithmetic.
//
// Needs the artifact named by GUFO_QWEN35MOE_MODEL and a gfx1151 device;
// without either the check skips with 77.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen35moe/kernels/rocm/device_model.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"
#include "src/models/qwen35moe/weights.hpp"

namespace q = gufo::quant;
namespace m = gufo::models::qwen35moe;
namespace k = gufo::models::qwen35moe::rocm;

namespace {

int failures = 0;

void Report(const char* name, bool ok, const std::string& detail) {
  std::cout << (ok ? "PASS " : "FAIL ") << name << ": " << detail << "\n";
  if (!ok) {
    ++failures;
  }
}

template<class T>
T* Device(const std::vector<T>& host) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, host.size() * sizeof(T) + 16));
  HIP_CHECK(hipMemcpy(ptr, host.data(), host.size() * sizeof(T),
                      hipMemcpyHostToDevice));
  return ptr;
}

template<class T>
std::vector<T> Read(T* device, std::vector<T> host) {
  HIP_CHECK(hipMemcpy(host.data(), device, host.size() * sizeof(T),
                      hipMemcpyDeviceToHost));
  return host;
}

double MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return worst;
}

// One dequantized row of a projection tensor in [cols][rows][experts] order.
void DecodeRow(const m::TensorRef& t, std::uint32_t row, std::uint32_t expert,
               std::vector<float>& out) {
  const std::size_t rb =
      q::EncodedSizeBytes(t.type, static_cast<std::size_t>(t.cols));
  const auto* base = static_cast<const std::uint8_t*>(t.data) +
                     (static_cast<std::size_t>(expert) * t.rows + row) * rb;
  out.assign(t.cols, 0.0F);
  if (t.type == gufo::core::GgmlType::kF32) {
    std::memcpy(out.data(), base, t.cols * sizeof(float));
    return;
  }
  if (t.type == gufo::core::GgmlType::kQ4_K) {
    q::DequantizeQ4_K(base, out.data(), t.cols);
  } else if (t.type == gufo::core::GgmlType::kQ5_K) {
    q::DequantizeQ5_K(base, out.data(), t.cols);
  } else if (t.type == gufo::core::GgmlType::kQ6_K) {
    q::DequantizeQ6_K(base, out.data(), t.cols);
  } else {
    std::fprintf(stderr,
                 "moe.DecodeRow abort: type=%d cols=%llu rows=%llu experts=%llu"
                 " row=%u expert=%u name=%.*s\n",
                 static_cast<int>(t.type),
                 static_cast<unsigned long long>(t.cols),
                 static_cast<unsigned long long>(t.rows),
                 static_cast<unsigned long long>(t.experts), row, expert,
                 static_cast<int>(t.name.size()), t.name.data());
    std::abort();  // the binder rejects anything else already
  }
}

/// CPU mixture for one layer, in double, mirroring Reference::Mixture.
void CpuMoe(const m::ModelWeights& w, const m::LayerWeights& l,
            const std::vector<float>& h, std::uint32_t rows,
            std::vector<float>& out) {
  const std::size_t d = w.config.hidden_size;
  const std::size_t e = w.config.num_experts;
  const std::size_t f = w.config.expert_ffn_size;
  const std::size_t s = w.config.shared_ffn_size;
  const std::size_t u = w.config.experts_used;
  std::vector<double> xn(rows * d), logits(rows * e);
  for (std::size_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
      ss += static_cast<double>(h[r * d + i]) * h[r * d + i];
    }
    const double sc = 1.0 / std::sqrt(ss / d + w.config.rms_eps);
    const auto* pn = static_cast<const float*>(l.post_attention_norm.data);
    for (std::size_t i = 0; i < d; ++i) {
      xn[r * d + i] = h[r * d + i] * sc * pn[i];
    }
  }
  std::vector<float> row(d);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < e; ++j) {
      DecodeRow(l.ffn_gate_inp, static_cast<std::uint32_t>(j), 0, row);
      double acc = 0.0;
      for (std::size_t i = 0; i < d; ++i) {
        acc += row[i] * xn[r * d + i];
      }
      logits[r * e + j] = static_cast<float>(acc);
    }
  }
  out.assign(rows * d, 0.0F);
  std::vector<std::size_t> order(e);
  std::vector<double> probs(e);
  for (std::size_t r = 0; r < rows; ++r) {
    double best = -1e300;
    for (std::size_t j = 0; j < e; ++j) {
      best = std::max(best, static_cast<double>(logits[r * e + j]));
    }
    double total = 0.0;
    for (std::size_t j = 0; j < e; ++j) {
      probs[j] = std::exp(logits[r * e + j] - best);
      total += probs[j];
    }
    for (std::size_t j = 0; j < e; ++j) {
      probs[j] /= total;
      order[j] = j;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                       return probs[a] > probs[b];
                     });
    double wsum = 0.0;
    for (std::size_t j = 0; j < u; ++j) {
      wsum += probs[order[j]];
    }
    std::vector<double> mix(d, 0.0);
    std::vector<float> grow(d), urow(d), drow(f);
    for (std::size_t j = 0; j < u; ++j) {
      const std::size_t ex = order[j];
      const double wt = probs[ex] / wsum;
      // gate[e][i] / up[e][i]: row i of expert e; down[e][o] similarly.
      const std::size_t rbg =
          q::EncodedSizeBytes(l.ffn_gate_exps.type, d);
      const std::size_t rbu =
          q::EncodedSizeBytes(l.ffn_up_exps.type, d);
      const std::size_t rbd =
          q::EncodedSizeBytes(l.ffn_down_exps.type, f);
      std::vector<double> avt(f);
      for (std::size_t i = 0; i < f; ++i) {
        const auto* gb = static_cast<const std::uint8_t*>(
                             l.ffn_gate_exps.data) +
                         (ex * f + i) * rbg;
        const auto* ub =
            static_cast<const std::uint8_t*>(l.ffn_up_exps.data) +
            (ex * f + i) * rbu;
        if (l.ffn_gate_exps.type == gufo::core::GgmlType::kQ5_K) {
          q::DequantizeQ5_K(gb, grow.data(), d);
        } else if (l.ffn_gate_exps.type == gufo::core::GgmlType::kQ4_K) {
          q::DequantizeQ4_K(gb, grow.data(), d);
        } else if (l.ffn_gate_exps.type == gufo::core::GgmlType::kQ6_K) {
          q::DequantizeQ6_K(gb, grow.data(), d);
        } else if (l.ffn_gate_exps.type == gufo::core::GgmlType::kIQ4_XS) {
          q::DequantizeIQ4_XS(gb, grow.data(), d);
        } else {
          std::abort();
        }
        if (l.ffn_up_exps.type == gufo::core::GgmlType::kQ5_K) {
          q::DequantizeQ5_K(ub, urow.data(), d);
        } else if (l.ffn_up_exps.type == gufo::core::GgmlType::kQ4_K) {
          q::DequantizeQ4_K(ub, urow.data(), d);
        } else if (l.ffn_up_exps.type == gufo::core::GgmlType::kQ6_K) {
          q::DequantizeQ6_K(ub, urow.data(), d);
        } else if (l.ffn_up_exps.type == gufo::core::GgmlType::kIQ4_XS) {
          q::DequantizeIQ4_XS(ub, urow.data(), d);
        } else {
          std::abort();
        }
        double gg = 0.0, uu = 0.0;
        for (std::size_t kk = 0; kk < d; ++kk) {
          gg += grow[kk] * xn[r * d + kk];
          uu += urow[kk] * xn[r * d + kk];
        }
        avt[i] = gg / (1.0 + std::exp(-gg)) * uu;
      }
      for (std::size_t o = 0; o < d; ++o) {
        const auto* base =
            static_cast<const std::uint8_t*>(l.ffn_down_exps.data) +
            (ex * d + o) * rbd;
        if (l.ffn_down_exps.type == gufo::core::GgmlType::kQ6_K) {
          q::DequantizeQ6_K(base, drow.data(), f);
        } else if (l.ffn_down_exps.type == gufo::core::GgmlType::kQ4_K) {
          q::DequantizeQ4_K(base, drow.data(), f);
        } else if (l.ffn_down_exps.type == gufo::core::GgmlType::kQ5_K) {
          q::DequantizeQ5_K(base, drow.data(), f);
        } else {
          std::abort();
        }
        double dd = 0.0;
        for (std::size_t i = 0; i < f; ++i) {
          dd += drow[i] * avt[i];
        }
        mix[o] += wt * dd;
      }
    }
    // shared expert, sigmoid gated
    {
      // NOTE: shared gate/up/down are [s][d] / [d][s] dense matrices.
      std::vector<double> sa(s, 0.0);
      const std::size_t rbg = q::EncodedSizeBytes(l.ffn_gate_shexp.type, d);
      const std::size_t rbu =
          q::EncodedSizeBytes(l.ffn_up_shexp.type, d);
      std::vector<float> gr(d), ur(d);
      for (std::size_t i = 0; i < s; ++i) {
        const auto* gb = static_cast<const std::uint8_t*>(
                             l.ffn_gate_shexp.data) +
                         i * rbg;
        const auto* ub = static_cast<const std::uint8_t*>(l.ffn_up_shexp.data) +
                         i * rbu;
        if (l.ffn_gate_shexp.type == gufo::core::GgmlType::kQ5_K) {
          q::DequantizeQ5_K(gb, gr.data(), d);
        } else if (l.ffn_gate_shexp.type == gufo::core::GgmlType::kQ4_K) {
          q::DequantizeQ4_K(gb, gr.data(), d);
        } else if (l.ffn_gate_shexp.type == gufo::core::GgmlType::kQ6_K) {
          q::DequantizeQ6_K(gb, gr.data(), d);
        } else {
          std::abort();
        }
        if (l.ffn_up_shexp.type == gufo::core::GgmlType::kQ5_K) {
          q::DequantizeQ5_K(ub, ur.data(), d);
        } else if (l.ffn_up_shexp.type == gufo::core::GgmlType::kQ4_K) {
          q::DequantizeQ4_K(ub, ur.data(), d);
        } else if (l.ffn_up_shexp.type == gufo::core::GgmlType::kQ6_K) {
          q::DequantizeQ6_K(ub, ur.data(), d);
        } else {
          std::abort();
        }
        double gg = 0.0, uu = 0.0;
        for (std::size_t kk = 0; kk < d; ++kk) {
          gg += gr[kk] * xn[r * d + kk];
          uu += ur[kk] * xn[r * d + kk];
        }
        sa[i] = gg / (1.0 + std::exp(-gg)) * uu;
      }
      const auto* gv0 =
          static_cast<const float*>(l.ffn_gate_inp_shexp.data);
      double dot = 0.0;
      for (std::size_t i = 0; i < d; ++i) {
        dot += gv0[i] * xn[r * d + i];
      }
      const double gs = 1.0 / (1.0 + std::exp(-dot));
      for (std::size_t o = 0; o < d; ++o) {
        std::vector<float> dr(s);
        const std::size_t rb = q::EncodedSizeBytes(
            l.ffn_down_shexp.type, static_cast<std::size_t>(s));
        const auto* base = static_cast<const std::uint8_t*>(
                               l.ffn_down_shexp.data) +
                           o * rb;
        if (l.ffn_down_shexp.type == gufo::core::GgmlType::kQ6_K) {
          q::DequantizeQ6_K(base, dr.data(), s);
        } else if (l.ffn_down_shexp.type == gufo::core::GgmlType::kQ4_K) {
          q::DequantizeQ4_K(base, dr.data(), s);
        } else if (l.ffn_down_shexp.type == gufo::core::GgmlType::kQ5_K) {
          q::DequantizeQ5_K(base, dr.data(), s);
        } else {
          std::abort();
        }
        double dd = 0.0;
        for (std::size_t i = 0; i < s; ++i) {
          dd += dr[i] * sa[i];
        }
        mix[o] += gs * dd;
      }
    }
    // finish: the raw mixture adds straight onto the residual
    for (std::size_t i = 0; i < d; ++i) {
      out[r * d + i] = static_cast<float>(h[r * d + i] + mix[i]);
    }
  }
}

void MoeBlock(std::uint32_t layer_idx, std::uint32_t rows) {
  const char* path = std::getenv("GUFO_QWEN35MOE_MODEL");
  if (path == nullptr || path[0] == '\0') {
    std::cout << "SKIP: no model\n";
    return;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (!reader) {
    std::cout << "SKIP: cannot open model\n";
    return;
  }
  auto weights = m::ModelWeights::LoadFromGguf(*reader, &error);
  if (!weights.has_value()) {
    std::cout << "SKIP: cannot bind model\n";
    return;
  }
  auto device = k::DeviceModel::Upload(*weights, *reader, &error);
  if (!device) {
    Report("moe.upload", false, error);
    return;
  }
  const m::LayerWeights& l = weights->layers[layer_idx];
  const auto& dl = device->layers()[layer_idx];
  const std::size_t d = weights->config.hidden_size;
  const std::size_t u = weights->config.experts_used;
  std::mt19937 rng(1000 + layer_idx);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> h(rows * d);
  for (float& v : h) {
    v = normal(rng);
  }
  std::vector<float> want;
  CpuMoe(*weights, l, h, rows, want);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  float* dx = Device(h);
  std::vector<float> hnorm(rows * d);
  float* dhnorm = Device(hnorm);
  k::RmsNormRows({dx, dl.post_attention_norm.f32(), dhnorm,
                  static_cast<std::uint32_t>(rows),
                  static_cast<std::uint32_t>(d), weights->config.rms_eps},
                 stream);
  const std::size_t e = weights->config.num_experts;
  std::vector<std::int32_t> ids(rows * u, -1);
  std::vector<float> rweights(rows * u, 0.0F);
  std::vector<std::uint32_t> counts(e, 0);
  std::int32_t* dids = Device(ids);
  float* dweights = Device(rweights);
  std::uint32_t* dcounts = Device(counts);
  std::vector<std::int32_t> groups(k::GroupTableInts(e), -1);
  std::int32_t* dgroups = Device(groups);
  k::MoeRoute({dhnorm, dl.ffn_gate_inp.f32(), nullptr, dids, dweights,
               dcounts, dgroups, rows, static_cast<std::uint32_t>(d),
               static_cast<std::uint32_t>(e), static_cast<std::uint32_t>(u)},
              stream);
  const std::size_t f = weights->config.expert_ffn_size;
  const std::size_t s = weights->config.shared_ffn_size;
  std::vector<float> gate(rows * u * f), up(rows * u * f), act(rows * u * f),
      routed(rows * u * d);
  float* dgate = Device(gate);
  float* dup = Device(up);
  float* dact = Device(act);
  float* drouted = Device(routed);
  auto proj = [&](const k::DeviceTensor& w, const float* in, float* y,
                  std::uint32_t m, std::uint32_t kk, std::uint32_t spr) {
    if (!k::GroupedGemv({w.quant(), w.data, dgroups,
                         static_cast<std::uint32_t>(e), in, y, rows, spr, m,
                         kk},
                        stream)) {
      throw std::invalid_argument("no grouped kernel");
    }
  };
  proj(dl.ffn_gate_exps, dhnorm, dgate, static_cast<std::uint32_t>(f),
       static_cast<std::uint32_t>(d), static_cast<std::uint32_t>(u));
  proj(dl.ffn_up_exps, dhnorm, dup, static_cast<std::uint32_t>(f),
       static_cast<std::uint32_t>(d), static_cast<std::uint32_t>(u));
  k::GateUpAct({dgate, dup, dact, rows * u * f}, stream);
  proj(dl.ffn_down_exps, dact, drouted, static_cast<std::uint32_t>(d),
       static_cast<std::uint32_t>(f), 1);
  // shared expert, dense single-group table
  std::vector<float> shgate(rows * s), shup(rows * s), shact(rows * s),
      shout(rows * d);
  float* dshgate = Device(shgate);
  float* dshup = Device(shup);
  float* dshact = Device(shact);
  float* dshout = Device(shout);
  std::vector<std::int32_t> shgroups(k::GroupTableInts(1), -1);
  shgroups[0] = 1;
  shgroups[1] = 0;
  shgroups[2] = static_cast<std::int32_t>(rows);
  for (std::uint32_t r = 0; r < rows; ++r) {
    shgroups[3 + r] = static_cast<std::int32_t>(r);
  }
  std::int32_t* dshgroups = Device(shgroups);
  auto dense = [&](const k::DeviceTensor& w, const float* in, float* y,
                   std::uint32_t m, std::uint32_t kk) {
    if (!k::GroupedGemv({w.quant(), w.data, dshgroups, 1, in, y, rows, 1, m,
                         kk},
                        stream)) {
      throw std::invalid_argument("no grouped kernel");
    }
  };
  dense(dl.ffn_gate_shexp, dhnorm, dshgate, static_cast<std::uint32_t>(s),
        static_cast<std::uint32_t>(d));
  dense(dl.ffn_up_shexp, dhnorm, dshup, static_cast<std::uint32_t>(s),
        static_cast<std::uint32_t>(d));
  k::GateUpAct({dshgate, dshup, dshact, rows * s}, stream);
  dense(dl.ffn_down_shexp, dshact, dshout, static_cast<std::uint32_t>(d),
        static_cast<std::uint32_t>(s));
  // gate the shared output on the host (one dot per row)
  shout = Read(dshout, shout);
  HIP_CHECK(hipDeviceSynchronize());
  const auto* gv =
      static_cast<const float*>(l.ffn_gate_inp_shexp.data);
  std::vector<float> hnorm_host = Read(dhnorm, hnorm);
  for (std::uint32_t r = 0; r < rows; ++r) {
    double dot = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
      dot += gv[i] * hnorm_host[r * d + i];
    }
    const float gs = static_cast<float>(1.0 / (1.0 + std::exp(-dot)));
    for (std::size_t i = 0; i < d; ++i) {
      shout[r * d + i] *= gs;
    }
  }
  float* dshgated = Device(shout);
  std::vector<float> x = h;
  float* dx2 = Device(x);
  std::vector<float> combined(rows * d);
  float* dcombined = Device(combined);
  k::MoeCombineArgs cargs{};
  cargs.out = dcombined;
  cargs.routed = drouted;
  cargs.shared = dshgated;
  cargs.weights = dweights;
  cargs.rows = rows;
  cargs.hidden = static_cast<std::uint32_t>(d);
  cargs.used = static_cast<std::uint32_t>(u);
  k::MoeCombine(cargs, stream);
  k::AddRows({dx2, dcombined, rows * d}, stream);
  HIP_CHECK(hipDeviceSynchronize());
  x = Read(dx2, x);
  HIP_CHECK(hipStreamDestroy(stream));
  const double worst = MaxAbsDiff(x, want);
  Report(("moe.block_layer" + std::to_string(layer_idx)).c_str(),
         worst < 1e-3,
         "max_abs=" + std::to_string(worst));
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::cout << "SKIP: no gfx1151 device\n";
    return 77;
  }
  MoeBlock(0, 4);
  MoeBlock(3, 4);
  MoeBlock(39, 4);

  if (failures != 0) {
    std::cout << "FAILURES: " << failures << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
