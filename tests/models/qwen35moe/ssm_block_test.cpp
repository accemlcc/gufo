// Qwen3.5-MoE's Gated DeltaNet block on the device against a CPU oracle over
// the same mapped weights: the joint projection, the causal depthwise conv,
// the recurrent gated-delta update with per-head norms and SiLU gating, and
// the output projection. The oracle decodes with the scalar CPU path and
// accumulates the recurrence in double, so the only difference under test is
// the kernels' own arithmetic.
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
#include "src/models/qwen35moe/kernels/rocm/device_model.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"
#include "src/models/qwen35moe/kernels/rocm/ssm.hpp"
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

/// CPU Gated DeltaNet block, mirroring Reference::LinearLayer.
void CpuSsm(const m::ModelWeights& w, const m::LayerWeights& l,
            const std::vector<float>& x, std::uint32_t rows,
            std::vector<float>& out) {
  const std::size_t d = w.config.hidden_size;
  const std::size_t kh = w.config.ssm_group_count;
  const std::size_t vh = w.config.ssm_time_step_rank;
  const std::size_t kd = w.config.ssm_state_size;
  const std::size_t vd = w.config.ssm_inner_size;
  const std::size_t hk = vd / vh;
  const std::size_t qkv = w.config.SsmQkvDim();
  const std::size_t k4 = w.config.ssm_conv_kernel;
  std::vector<float> row(d);
  std::vector<double> qkv_r(qkv), gate_r(vd), alpha_r(vh), beta_r(vh);
  std::vector<float> filt(k4);
  std::vector<double> mixed(qkv);
  std::vector<double> core(vd, 0.0);
  std::vector<double> st(vh * kd * hk, 0.0);
  std::vector<double> conv(qkv * (k4 - 1), 0.0);
  out.assign(rows * d, 0.0F);
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::size_t o = 0; o < qkv; ++o) {
      DecodeRow(l.attn_qkv, static_cast<std::uint32_t>(o), 0, row.data());
      double acc = 0.0;
      for (std::size_t i = 0; i < d; ++i) {
        acc += row[i] * x[r * d + i];
      }
      qkv_r[o] = acc;
    }
    for (std::size_t o = 0; o < vd; ++o) {
      DecodeRow(l.attn_gate, static_cast<std::uint32_t>(o), 0, row.data());
      double acc = 0.0;
      for (std::size_t i = 0; i < d; ++i) {
        acc += row[i] * x[r * d + i];
      }
      gate_r[o] = acc;
    }
    for (std::size_t o = 0; o < vh; ++o) {
      DecodeRow(l.ssm_alpha, static_cast<std::uint32_t>(o), 0, row.data());
      double aa = 0.0, bb = 0.0;
      for (std::size_t i = 0; i < d; ++i) {
        aa += row[i] * x[r * d + i];
      }
      DecodeRow(l.ssm_beta, static_cast<std::uint32_t>(o), 0, row.data());
      for (std::size_t i = 0; i < d; ++i) {
        bb += row[i] * x[r * d + i];
      }
      alpha_r[o] = aa;
      beta_r[o] = bb;
    }
    const auto* avec = static_cast<const float*>(l.ssm_a.data);
    const auto* dvec = static_cast<const float*>(l.ssm_dt.data);
    for (std::size_t ch = 0; ch < qkv; ++ch) {
      DecodeRow(l.ssm_conv1d, static_cast<std::uint32_t>(ch), 0, row.data());
      for (std::size_t i = 0; i < k4; ++i) {
        filt[i] = row[i];
      }
      double acc = 0.0;
      for (std::size_t i = 0; i < k4 - 1; ++i) {
        acc += filt[i] * conv[ch * (k4 - 1) + i];
        conv[ch * (k4 - 1) + i] =
            i + 1 < k4 - 1 ? conv[ch * (k4 - 1) + i + 1] : qkv_r[ch];
      }
      acc += filt[k4 - 1] * qkv_r[ch];
      mixed[ch] = acc / (1.0 + std::exp(-acc));
    }
    std::vector<double> q(kh * kd), kv(kh * kd), vv(vd);
    for (std::size_t hh = 0; hh < kh; ++hh) {
      double nq = 0.0, nk = 0.0;
      for (std::size_t i = 0; i < kd; ++i) {
        const double a = mixed[hh * kd + i];
        const double b = mixed[kd * kh + hh * kd + i];
        nq += a * a;
        nk += b * b;
        q[hh * kd + i] = a;
        kv[hh * kd + i] = b;
      }
      const double sq = 1.0 / (std::sqrt(nq + 1e-6) * std::sqrt(kd));
      const double sk = 1.0 / std::sqrt(nk + 1e-6);
      for (std::size_t i = 0; i < kd; ++i) {
        q[hh * kd + i] *= sq;
        kv[hh * kd + i] *= sk;
      }
    }
    for (std::size_t i = 0; i < vd; ++i) {
      vv[i] = mixed[2 * kd * kh + i];
    }
    const std::size_t rep = vh / kh;
    for (std::size_t hh = 0; hh < vh; ++hh) {
      const std::size_t khh = hh / rep;
      const double g = avec[hh] * std::log1p(std::exp(alpha_r[hh] + dvec[hh]));
      const double b = 1.0 / (1.0 + std::exp(-beta_r[hh]));
      const double decay = std::exp(g);
      double* s = st.data() + (hh * kd) * hk;
      for (std::size_t i = 0; i < kd * hk; ++i) {
        s[i] *= decay;
      }
      std::vector<double> kvsum(hk, 0.0);
      for (std::size_t j = 0; j < hk; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < kd; ++i) {
          acc += s[i * hk + j] * kv[khh * kd + i];
        }
        kvsum[j] = acc;
      }
      for (std::size_t i = 0; i < kd; ++i) {
        for (std::size_t j = 0; j < hk; ++j) {
          s[i * hk + j] +=
              kv[khh * kd + i] * (vv[hh * hk + j] - kvsum[j]) * b;
        }
      }
      double* chh = core.data() + hh * hk;
      for (std::size_t j = 0; j < hk; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < kd; ++i) {
          acc += s[i * hk + j] * q[khh * kd + i];
        }
        chh[j] = acc;
      }
    }
    const auto* wn = static_cast<const float*>(l.ssm_norm.data);
    std::vector<float> normed(vd);
    for (std::size_t hh = 0; hh < vh; ++hh) {
      double ss = 0.0;
      for (std::size_t j = 0; j < hk; ++j) {
        ss += core[hh * hk + j] * core[hh * hk + j];
      }
      const double sc = 1.0 / std::sqrt(ss / hk + w.config.rms_eps);
      for (std::size_t j = 0; j < hk; ++j) {
        const double z = gate_r[hh * hk + j];
        normed[hh * hk + j] = static_cast<float>(core[hh * hk + j] * sc *
                                                 wn[j] * z / (1 + std::exp(-z)));
      }
    }
    for (std::size_t o = 0; o < d; ++o) {
      DecodeRow(l.ssm_out, static_cast<std::uint32_t>(o), 0, row.data());
      double acc = 0.0;
      for (std::size_t i = 0; i < vd; ++i) {
        acc += row[i] * normed[i];
      }
      out[r * d + o] = static_cast<float>(acc);
    }
  }
}

void SsmBlock(std::uint32_t layer_idx, std::uint32_t rows) {
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
  if (weights->layers[layer_idx].is_full_attention) {
    std::cout << "SKIP: not a linear layer\n";
    return;
  }
  auto device = k::DeviceModel::Upload(*weights, *reader, &error);
  if (!device) {
    Report("ssm.upload", false, error);
    return;
  }
  const m::LayerWeights& l = weights->layers[layer_idx];
  const auto& dl = device->layers()[layer_idx];
  const std::size_t d = weights->config.hidden_size;
  const std::size_t qkv = weights->config.SsmQkvDim();
  const std::size_t vd = weights->config.SsmValueDim();
  const std::size_t rk = weights->config.ssm_time_step_rank;
  std::mt19937 rng(200 + layer_idx);
  std::normal_distribution<float> normal(0.0F, 0.3F);
  std::vector<float> x(rows * d);
  for (float& v : x) {
    v = normal(rng);
  }
  std::vector<float> want;
  CpuSsm(*weights, l, x, rows, want);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  float* dx = Device(x);
  std::vector<float> hqkv(rows * qkv), hgate(rows * vd), halpha(rows * rk),
      hbeta(rows * rk), hconv(rows * qkv), hout(rows * vd),
      hstate(rk * 128 * 128, 0.0F), hconvst(qkv * 4, 0.0F);
  float *dqkv = Device(hqkv), *dgate = Device(hgate), *dalpha = Device(halpha),
        *dbeta = Device(hbeta), *dconv = Device(hconv), *dout = Device(hout),
        *dstate = Device(hstate), *dconvst = Device(hconvst);
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
  dense(dl.attn_qkv, dx, dqkv, static_cast<std::uint32_t>(qkv),
        static_cast<std::uint32_t>(d));
  dense(dl.attn_gate, dx, dgate, static_cast<std::uint32_t>(vd),
        static_cast<std::uint32_t>(d));
  dense(dl.ssm_alpha, dx, dalpha, static_cast<std::uint32_t>(rk),
        static_cast<std::uint32_t>(d));
  dense(dl.ssm_beta, dx, dbeta, static_cast<std::uint32_t>(rk),
        static_cast<std::uint32_t>(d));
  k::SsmArgs sa{};
  sa.qkv = dqkv;
  sa.conv_w = static_cast<const float*>(dl.ssm_conv1d.data);
  sa.conv_state = dconvst;
  sa.conv_out = dconv;
  sa.recurrent = dstate;
  sa.alpha = dalpha;
  sa.beta = dbeta;
  sa.ssm_a = dl.ssm_a.f32();
  sa.ssm_dt = dl.ssm_dt.f32();
  sa.ssm_norm = dl.ssm_norm.f32();
  sa.gate = dgate;
  sa.out = dout;
  sa.rows = rows;
  sa.qkv_dim = static_cast<std::uint32_t>(qkv);
  sa.num_key_heads = weights->config.ssm_group_count;
  sa.num_heads = static_cast<std::uint32_t>(rk);
  sa.key_dim = weights->config.ssm_state_size;
  sa.value_dim = static_cast<std::uint32_t>(vd);
  sa.head_dim = weights->config.SsmValueHeadDim();
  k::LaunchSsm(sa, stream);
  std::vector<float> got(rows * d, -1e30F);
  float* dgot = Device(got);
  dense(dl.ssm_out, dout, dgot, static_cast<std::uint32_t>(d),
        static_cast<std::uint32_t>(vd));
  HIP_CHECK(hipDeviceSynchronize());
  got = Read(dgot, got);
  HIP_CHECK(hipStreamDestroy(stream));
  const double worst = MaxAbsDiff(got, want);
  Report(("ssm.block_layer" + std::to_string(layer_idx)).c_str(),
         worst < 1e-3, "worst=" + std::to_string(worst));
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
      std::strstr(prop.gcnArchName, "gfx1151") == nullptr) {
    std::cout << "SKIP: no gfx1151 device\n";
    return 77;
  }

  SsmBlock(0, 2);
  SsmBlock(2, 4);

  if (failures != 0) {
    std::cout << "FAILURES: " << failures << "\n";
    return 1;
  }
  std::cout << "ALL PASS\n";
  return 0;
}
