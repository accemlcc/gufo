// Qwen3.5-MoE's full-attention block on the device against a CPU oracle over
// the same mapped weights: the packed q|gate projection, K/V projections,
// QK-norm/RoPE/split/cache write, online GQA attention, the sigmoid output
// gate and the output projection. The oracle decodes with the scalar CPU path
// and accumulates in double, so the only difference under test is the kernels'
// own arithmetic. The synthetic qkv_post/attention checks cover shapes; this
// one covers the composition on released weights, including the tensor
// binding the synthetics never touch.
//
// Needs the artifact named by GUFO_QWEN35MOE_MODEL and a gfx1151 device;
// without either the check skips with 77.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen35moe/kernels/rocm/attention.hpp"
#include "src/models/qwen35moe/kernels/rocm/device_model.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"
#include "src/models/qwen35moe/kernels/rocm/qkv_post.hpp"
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

void DecodeRow(const m::TensorRef& t, std::uint32_t row, std::uint32_t expert,
               float* out) {
  const std::size_t rb =
      q::EncodedSizeBytes(t.type, static_cast<std::size_t>(t.cols));
  const auto* p = static_cast<const std::uint8_t*>(t.data) +
                  (static_cast<std::size_t>(expert) * t.rows + row) * rb;
  if (t.type == gufo::core::GgmlType::kF32) {
    std::memcpy(out, p, t.cols * sizeof(float));
    return;
  }
  if (t.type == gufo::core::GgmlType::kQ4_K) {
    q::DequantizeQ4_K(p, out, t.cols);
  } else if (t.type == gufo::core::GgmlType::kQ5_K) {
    q::DequantizeQ5_K(p, out, t.cols);
  } else if (t.type == gufo::core::GgmlType::kQ6_K) {
    q::DequantizeQ6_K(p, out, t.cols);
  } else if (t.type == gufo::core::GgmlType::kQ8_0) {
    q::DequantizeQ8_0(p, out, t.cols);
  } else {
    std::abort();
  }
}

void RmsNorm(const float* x, const float* w, std::size_t n, float eps,
             float* y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(x[i]) * x[i];
  }
  const double scale = 1.0 / std::sqrt(sum / static_cast<double>(n) + eps);
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = static_cast<float>(static_cast<double>(x[i]) * scale * w[i]);
  }
}

void Rope(float* head, std::size_t dim, std::uint32_t position, float theta) {
  const std::size_t half = dim / 2;
  for (std::size_t i = 0; i < half; ++i) {
    const double angle =
        static_cast<double>(position) *
        std::pow(static_cast<double>(theta),
                 -2.0 * static_cast<double>(i) / static_cast<double>(dim));
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double a = head[i];
    const double b = head[i + half];
    head[i] = static_cast<float>(a * c - b * s);
    head[i + half] = static_cast<float>(a * s + b * c);
  }
}

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

/// CPU full-attention block, mirroring Reference::FullLayer at position zero
/// with a fresh cache.
void CpuAttn(const m::ModelWeights& w, const m::LayerWeights& l,
             const std::vector<float>& x, std::uint32_t rows,
             std::vector<float>& out) {
  const m::Config& c = w.config;
  const std::size_t d = c.hidden_size;
  const std::size_t heads = c.num_heads;
  const std::size_t kv_heads = c.num_kv_heads;
  const std::size_t dim = c.head_dim;
  const std::size_t group = heads / kv_heads;
  const std::size_t q_dim = heads * dim;
  const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
  const float* qn = static_cast<const float*>(l.attn_q_norm.data);
  const float* kn = static_cast<const float*>(l.attn_k_norm.data);
  std::vector<float> row(d);
  std::vector<float> qg(rows * 2 * q_dim);
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::size_t o = 0; o < 2 * q_dim; ++o) {
      DecodeRow(l.attn_q, static_cast<std::uint32_t>(o), 0, row.data());
      double acc = 0.0;
      for (std::size_t i = 0; i < d; ++i) {
        acc += row[i] * x[r * d + i];
      }
      qg[r * 2 * q_dim + o] = static_cast<float>(acc);
    }
  }
  std::vector<float> keys(rows * kv_heads * dim);
  std::vector<float> values(rows * kv_heads * dim);
  std::vector<float> kh(dim);
  std::vector<float> krow(kv_heads * dim);
  std::vector<float> vrow(kv_heads * dim);
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::size_t o = 0; o < kv_heads * dim; ++o) {
      DecodeRow(l.attn_k, static_cast<std::uint32_t>(o), 0, row.data());
      double acc = 0.0;
      for (std::size_t i = 0; i < d; ++i) {
        acc += row[i] * x[r * d + i];
      }
      krow[o] = static_cast<float>(acc);
      DecodeRow(l.attn_v, static_cast<std::uint32_t>(o), 0, row.data());
      acc = 0.0;
      for (std::size_t i = 0; i < d; ++i) {
        acc += row[i] * x[r * d + i];
      }
      vrow[o] = static_cast<float>(acc);
    }
    for (std::size_t h = 0; h < kv_heads; ++h) {
      for (std::size_t i = 0; i < dim; ++i) {
        kh[i] = krow[h * dim + i];
      }
      RmsNorm(kh.data(), kn, dim, c.rms_eps, kh.data());
      Rope(kh.data(), c.rotary_dim, r, c.rope_theta);
      for (std::size_t i = 0; i < dim; ++i) {
        keys[(r * kv_heads + h) * dim + i] = kh[i];
        values[(r * kv_heads + h) * dim + i] = vrow[h * dim + i];
      }
    }
  }
  std::vector<float> qh(dim);
  std::vector<float> gated(rows * heads * dim);
  for (std::uint32_t r = 0; r < rows; ++r) {
    const float* qgr = &qg[r * 2 * q_dim];
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t i = 0; i < dim; ++i) {
        qh[i] = qgr[h * 2 * dim + i];
      }
      RmsNorm(qh.data(), qn, dim, c.rms_eps, qh.data());
      Rope(qh.data(), c.rotary_dim, r, c.rope_theta);
      const std::size_t kvh = h / group;
      std::vector<double> scores(r + 1);
      double best = -INFINITY;
      for (std::uint32_t p = 0; p <= r; ++p) {
        const float* kk =
            keys.data() + (static_cast<std::size_t>(p) * kv_heads + kvh) * dim;
        double s = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
          s += static_cast<double>(qh[i]) * kk[i];
        }
        scores[p] = s * scale;
        best = std::max(best, scores[p]);
      }
      double total = 0.0;
      for (double& s : scores) {
        s = std::exp(s - best);
        total += s;
      }
      std::vector<double> acc(dim, 0.0);
      for (std::uint32_t p = 0; p <= r; ++p) {
        const float* vv = values.data() +
                                (static_cast<std::size_t>(p) * kv_heads + kvh) *
                                    dim;
        const double weight = scores[p] / total;
        for (std::size_t i = 0; i < dim; ++i) {
          acc[i] += weight * vv[i];
        }
      }
      const float* gh = qgr + h * 2 * dim + dim;
      for (std::size_t i = 0; i < dim; ++i) {
        gated[(r * heads + h) * dim + i] =
            static_cast<float>(acc[i] * Sigmoid(gh[i]));
      }
    }
  }
  out.assign(rows * d, 0.0F);
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::size_t o = 0; o < d; ++o) {
      DecodeRow(l.attn_output, static_cast<std::uint32_t>(o), 0, row.data());
      double acc = 0.0;
      for (std::size_t i = 0; i < heads * dim; ++i) {
        acc += row[i] * gated[r * heads * dim + i];
      }
      out[r * d + o] = static_cast<float>(acc);
    }
  }
}

}  // namespace

void AttnBlock(const m::ModelWeights& weights, k::DeviceModel& device,
               std::uint32_t layer_idx, std::uint32_t rows) {
  const m::LayerWeights& l = weights.layers[layer_idx];
  const k::DeviceLayer& dl = device.layers()[layer_idx];
  const m::Config& c = weights.config;
  const std::size_t d = c.hidden_size;
  const std::size_t qd = c.AttnOutDim();
  const std::size_t kvd = c.KvDim();
  constexpr std::uint32_t kCache = 16;
  std::mt19937 rng(300 + layer_idx);
  std::normal_distribution<float> normal(0.0F, 0.3F);
  std::vector<float> x(rows * d);
  for (float& v : x) {
    v = normal(rng);
  }
  std::vector<float> want;
  CpuAttn(weights, l, x, rows, want);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  float* dx = Device(x);
  std::vector<float> hqg(rows * 2 * qd, 0.0F);
  std::vector<float> hk(rows * kvd, 0.0F);
  std::vector<float> hv(rows * kvd, 0.0F);
  std::vector<float> hattn(rows * qd, 0.0F);
  std::vector<float> hgate(rows * qd, 0.0F);
  std::vector<float> hkc(c.num_kv_heads * kCache * c.head_dim, 0.0F);
  std::vector<float> hvc(c.num_kv_heads * kCache * c.head_dim, 0.0F);
  float* dqg = Device(hqg);
  float* dk = Device(hk);
  float* dv = Device(hv);
  float* dattn = Device(hattn);
  float* dgate = Device(hgate);
  float* dkc = Device(hkc);
  float* dvc = Device(hvc);
  std::vector<std::int32_t> groups(k::GroupTableInts(1), -1);
  groups[0] = 1;
  groups[1] = 0;
  groups[2] = static_cast<std::int32_t>(rows);
  for (std::uint32_t r = 0; r < rows; ++r) {
    groups[3 + r] = static_cast<std::int32_t>(r);
  }
  std::int32_t* dgroups = Device(groups);
  auto dense = [&](const k::DeviceTensor& w, const float* in, float* y,
                   std::uint32_t mm, std::uint32_t kk) {
    if (!k::GroupedGemv({w.quant(), w.data, dgroups, 1, in, y, rows, 1, mm,
                         kk},
                        stream)) {
      throw std::invalid_argument("no grouped kernel");
    }
  };
  dense(dl.attn_q, dx, dqg, static_cast<std::uint32_t>(2 * qd),
        static_cast<std::uint32_t>(d));
  dense(dl.attn_k, dx, dk, static_cast<std::uint32_t>(kvd),
        static_cast<std::uint32_t>(d));
  dense(dl.attn_v, dx, dv, static_cast<std::uint32_t>(kvd),
        static_cast<std::uint32_t>(d));
  k::QkvPostArgs qargs{};
  qargs.qg = dqg;
  qargs.k = dk;
  qargs.v = dv;
  qargs.q_norm = dl.attn_q_norm.f32();
  qargs.k_norm = dl.attn_k_norm.f32();
  qargs.q_out = dattn;
  qargs.gate_out = dgate;
  qargs.k_cache = dkc;
  qargs.v_cache = dvc;
  qargs.rows = rows;
  qargs.first_position = 0;
  qargs.max_context = kCache;
  qargs.heads = c.num_heads;
  qargs.kv_heads = c.num_kv_heads;
  qargs.dim = c.head_dim;
  qargs.rotary_dim = c.rotary_dim;
  qargs.theta_scale = device.theta_scale();
  qargs.eps = c.rms_eps;
  k::QkvPost(qargs, stream);
  k::AttentionArgs aargs{};
  aargs.q = dattn;
  aargs.k_cache = dkc;
  aargs.v_cache = dvc;
  aargs.out = dattn;
  aargs.rows = rows;
  aargs.first_position = 0;
  aargs.max_context = kCache;
  aargs.heads = c.num_heads;
  aargs.kv_heads = c.num_kv_heads;
  aargs.dim = c.head_dim;
  aargs.scale = 1.0F / std::sqrt(static_cast<float>(c.head_dim));
  k::Attention(aargs, stream);
  k::SigmoidGateAct({dattn, dgate, dattn, rows * qd}, stream);
  std::vector<float> got(rows * d, -1e30F);
  float* dgot = Device(got);
  dense(dl.attn_output, dattn, dgot, static_cast<std::uint32_t>(d),
        static_cast<std::uint32_t>(qd));
  HIP_CHECK(hipDeviceSynchronize());
  got = Read(dgot, got);
  HIP_CHECK(hipStreamDestroy(stream));
  const double worst = MaxAbsDiff(got, want);
  Report(("attn.block_layer" + std::to_string(layer_idx)).c_str(),
         worst < 1e-3, "worst=" + std::to_string(worst));
}

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
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (!reader) {
    std::cout << "SKIP: cannot open model\n";
    return 77;
  }
  auto weights = m::ModelWeights::LoadFromGguf(*reader, &error);
  if (!weights.has_value()) {
    std::cout << "SKIP: cannot bind model\n";
    return 77;
  }
  auto device = k::DeviceModel::Upload(*weights, *reader, &error);
  if (!device) {
    Report("attn.upload", false, error);
    return 1;
  }
  std::uint32_t tested = 0;
  for (std::uint32_t i = 0; i < weights->layers.size() && tested < 2; ++i) {
    if (!weights->layers[i].is_full_attention) {
      continue;
    }
    AttnBlock(*weights, *device, i, tested == 0 ? 2 : 4);
    ++tested;
  }
  if (tested == 0) {
    std::cout << "SKIP: no full-attention layer\n";
    return 77;
  }
  if (failures != 0) {
    std::cout << "FAILURES: " << failures << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
