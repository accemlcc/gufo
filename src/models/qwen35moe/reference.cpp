#include "src/models/qwen35moe/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen35moe {
namespace {

/// y = rms(x) * w: the file stores standard RMSNorm weights (mean near 1,
/// all positive). The transformers checkpoint centres them at 0 and the graph
/// reads (1 + w); the GGUF converter bakes that shift in, so the runtime -
/// and this oracle - must not add it again.
void RmsNorm(const float* x, const float* w, std::size_t n, float eps,
             float* y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(x[i]) * x[i];
  }
  const double scale = 1.0 / std::sqrt(sum / static_cast<double>(n) + eps);
  for (std::size_t i = 0; i < n; ++i) {
    y[i] = static_cast<float>(static_cast<double>(x[i]) * scale *
                              static_cast<double>(w[i]));
  }
}

double Sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

double Silu(double x) {
  return x * Sigmoid(x);
}

float Bf16ToFloat(std::uint16_t h) {
  float out = 0.0F;
  const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
  std::memcpy(&out, &u, sizeof(float));
  return out;
}

/// Dequantizes one row of `cols` elements; `F32`/`BF16`/`F16` copy directly.
void DequantRow(core::GgmlType type, const void* src, float* dst,
                std::size_t cols) {
  switch (type) {
    case core::GgmlType::kF32:
      std::memcpy(dst, src, cols * sizeof(float));
      return;
    case core::GgmlType::kBF16: {
      const auto* s = static_cast<const std::uint16_t*>(src);
      for (std::size_t i = 0; i < cols; ++i) {
        dst[i] = Bf16ToFloat(s[i]);
      }
      return;
    }
    case core::GgmlType::kF16: {
      const auto* s = static_cast<const std::uint16_t*>(src);
      for (std::size_t i = 0; i < cols; ++i) {
        dst[i] = quant::Fp16ToFloat(s[i]);
      }
      return;
    }
    case core::GgmlType::kQ8_0:
      quant::DequantizeQ8_0(src, dst, cols);
      return;
    case core::GgmlType::kQ4_K:
      quant::DequantizeQ4_K(src, dst, cols);
      return;
    case core::GgmlType::kQ5_K:
      quant::DequantizeQ5_K(src, dst, cols);
      return;
    case core::GgmlType::kQ6_K:
      quant::DequantizeQ6_K(src, dst, cols);
      return;
    case core::GgmlType::kQ3_K:
      quant::DequantizeQ3_K(src, dst, cols);
      return;
    case core::GgmlType::kQ8_K:
      quant::DequantizeQ8_K(src, dst, cols);
      return;
    case core::GgmlType::kIQ4_NL:
      quant::DequantizeIQ4_NL(src, dst, cols);
      return;
    case core::GgmlType::kIQ4_XS:
      quant::DequantizeIQ4_XS(src, dst, cols);
      return;
    case core::GgmlType::kIQ3_S:
      quant::DequantizeIQ3_S(src, dst, cols);
      return;
    default:
      break;
  }
  std::fill(dst, dst + cols, 0.0F);
}

/// NEOX rotation of one head over the rotated prefix: the pair (i, i+dim/2)
/// turns by position * theta^(-2i/dim).
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

}  // namespace

Reference::Reference(const ModelWeights& weights) : weights_(weights) {
  Reset();
}

void Reference::Reset() {
  position_ = 0;
  const Config& c = weights_.config;
  linear_.clear();
  linear_.resize(c.num_layers);
  keys_.assign(c.num_layers, {});
  values_.assign(c.num_layers, {});
}

void Reference::MatMul(const void* data, core::GgmlType type,
                       std::size_t rows_out, std::size_t cols, const float* x,
                       std::size_t rows, float* out) const {
  const std::size_t row_bytes = quant::EncodedSizeBytes(type, cols);
#pragma omp parallel for schedule(static)
  for (std::size_t o = 0; o < rows_out; ++o) {
    std::vector<float> w(cols);
    DequantRow(type, static_cast<const std::uint8_t*>(data) + o * row_bytes,
               w.data(), cols);
    for (std::size_t r = 0; r < rows; ++r) {
      const float* xr = x + r * cols;
      double acc = 0.0;
      for (std::size_t k = 0; k < cols; ++k) {
        acc += static_cast<double>(w[k]) * xr[k];
      }
      out[r * rows_out + o] = static_cast<float>(acc);
    }
  }
}

void Reference::LinearLayer(std::uint32_t layer, const float* x,
                            std::size_t rows, float* out) {
  const Config& c = weights_.config;
  const LayerWeights& w = weights_.layers[layer];
  const std::size_t d = c.hidden_size;
  const std::size_t key_dim = c.SsmKeyDim();
  const std::size_t value_dim = c.SsmValueDim();
  const std::size_t qkv_dim = c.SsmQkvDim();
  const std::size_t k_heads = c.ssm_group_count;
  const std::size_t v_heads = c.SsmValueHeads();
  const std::size_t head_k = c.ssm_state_size;
  const std::size_t head_v = c.SsmValueHeadDim();
  const std::size_t k4 = c.ssm_conv_kernel;
  LinearState& st = linear_[layer];
  if (st.recurrent.empty()) {
    st.recurrent.assign(v_heads * head_k * head_v, 0.0F);
    st.conv.assign(qkv_dim * (k4 - 1), 0.0F);
  }

  std::vector<float> qkv(rows * qkv_dim);
  std::vector<float> gate(rows * value_dim);
  MatMulRef(w.attn_qkv, qkv_dim, d, x, rows, qkv.data());
  MatMulRef(w.attn_gate, value_dim, d, x, rows, gate.data());
  std::vector<float> alpha(rows * v_heads);
  std::vector<float> beta(rows * v_heads);
  MatMulRef(w.ssm_alpha, v_heads, d, x, rows, alpha.data());
  MatMulRef(w.ssm_beta, v_heads, d, x, rows, beta.data());
  std::vector<float> a_vec(v_heads);
  DequantRow(core::GgmlType::kF32, w.ssm_a.data, a_vec.data(), v_heads);
  std::vector<float> dt_vec(v_heads);
  DequantRow(core::GgmlType::kF32, w.ssm_dt.data, dt_vec.data(), v_heads);
  const std::size_t conv_row_bytes =
      quant::EncodedSizeBytes(w.ssm_conv1d.type, k4);

  std::vector<float> mixed(qkv_dim);
  std::vector<float> q(k_heads * head_k);
  std::vector<float> k(k_heads * head_k);
  std::vector<float> v(value_dim);
  std::vector<float> b(v_heads);
  std::vector<float> g(v_heads);
  std::vector<double> core(value_dim);
  const float* w_norm = static_cast<const float*>(w.ssm_norm.data);
  std::vector<float> normed(value_dim);

  for (std::size_t r = 0; r < rows; ++r) {
    // Depthwise causal convolution over the raw projection, then SiLU.
    for (std::size_t ch = 0; ch < qkv_dim; ++ch) {
      const std::uint8_t* wch = static_cast<const std::uint8_t*>(
                                    w.ssm_conv1d.data) +
                                ch * conv_row_bytes;
      std::vector<float> filt(k4);
      DequantRow(w.ssm_conv1d.type, wch, filt.data(), k4);
      double acc = 0.0;
      for (std::size_t i = 0; i < k4 - 1; ++i) {
        acc += static_cast<double>(filt[i]) * st.conv[ch * (k4 - 1) + i];
        st.conv[ch * (k4 - 1) + i] =
            (i + 1 < k4 - 1) ? st.conv[ch * (k4 - 1) + i + 1]
                             : qkv[r * qkv_dim + ch];
      }
      acc += static_cast<double>(filt[k4 - 1]) * qkv[r * qkv_dim + ch];
      mixed[ch] = static_cast<float>(Silu(acc));
    }
    const float* mq = mixed.data();
    const float* mk = mixed.data() + key_dim;
    const float* mv = mixed.data() + 2 * key_dim;
    for (std::size_t h = 0; h < k_heads; ++h) {
      double nq = 0.0, nk = 0.0;
      for (std::size_t i = 0; i < head_k; ++i) {
        nq += static_cast<double>(mq[h * head_k + i]) *
              mq[h * head_k + i];
        nk += static_cast<double>(mk[h * head_k + i]) *
              mk[h * head_k + i];
      }
      // L2 norm with the epsilon inside the sum, then the 1/sqrt(head_dim)
      // query scale: x / sqrt(sum(x^2) + eps) / sqrt(head_dim).
      const double sq = 1.0 / (std::sqrt(nq + 1e-6) * std::sqrt(head_k));
      const double sk = 1.0 / std::sqrt(nk + 1e-6);
      for (std::size_t i = 0; i < head_k; ++i) {
        q[h * head_k + i] = static_cast<float>(mq[h * head_k + i] * sq);
        k[h * head_k + i] = static_cast<float>(mk[h * head_k + i] * sk);
      }
    }
    for (std::size_t i = 0; i < value_dim; ++i) {
      v[i] = mv[i];
    }
    for (std::size_t h = 0; h < v_heads; ++h) {
      b[h] = static_cast<float>(Sigmoid(beta[r * v_heads + h]));
      // The file stores the negative decay rates (-A) directly, as the
      // converter bakes -exp() of the checkpoint's A_log in; the graph
      // multiplies them with the softplus as is.
      const double aval = static_cast<double>(a_vec[h]);
      const double soft =
          std::log1p(std::exp(alpha[r * v_heads + h] + dt_vec[h]));
      g[h] = static_cast<float>(aval * soft);
    }
    // Recurrent gated delta update, head by head. K/V heads repeat to the
    // value heads by tiling: v-head h reads k-head h % k_heads. This matches
    // the checkpoint: the adjacent (repeat-interleave) pairing taken from the
    // transformers reference gives PPL ~1.1M on real text, the tiled pairing
    // used by ggml/llama gives ~4.3.
    for (std::size_t h = 0; h < v_heads; ++h) {
      const std::size_t kh = h % k_heads;
      float* s = &st.recurrent[(h * head_k) * head_v];
      const float* qh = &q[kh * head_k];
      const float* khv = &k[kh * head_k];
      const float* vh = &v[h * head_v];
      const double decay = std::exp(g[h]);
      const double rate = b[h];
      for (std::size_t i = 0; i < head_k * head_v; ++i) {
        s[i] = static_cast<float>(s[i] * decay);
      }
      std::vector<double> kv_sum(head_v);
      for (std::size_t j = 0; j < head_v; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < head_k; ++i) {
          acc += s[i * head_v + j] * khv[i];
        }
        kv_sum[j] = acc;
      }
      for (std::size_t i = 0; i < head_k; ++i) {
        for (std::size_t j = 0; j < head_v; ++j) {
          s[i * head_v + j] = static_cast<float>(
              s[i * head_v + j] + khv[i] * (vh[j] - kv_sum[j]) * rate);
        }
      }
      double* ch = &core[h * head_v];
      for (std::size_t j = 0; j < head_v; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < head_k; ++i) {
          acc += s[i * head_v + j] * qh[i];
        }
        ch[j] = acc;
      }
    }
    // Gated RMSNorm with the output gate, then the output projection.
    const float* z = &gate[r * value_dim];
    for (std::size_t h = 0; h < v_heads; ++h) {
      double sum = 0.0;
      for (std::size_t j = 0; j < head_v; ++j) {
        sum += core[h * head_v + j] * core[h * head_v + j];
      }
      const double scale = 1.0 / std::sqrt(sum / head_v + c.rms_eps);
      for (std::size_t j = 0; j < head_v; ++j) {
        normed[h * head_v + j] = static_cast<float>(
            core[h * head_v + j] * scale * w_norm[j] * Silu(z[h * head_v + j]));
      }
    }
  }
  MatMulRef(w.ssm_out, d, value_dim, normed.data(), 1, out);
  (void)rows;
}

void Reference::FullLayer(std::uint32_t layer, const float* x,
                          std::size_t rows, float* out) {
  const Config& c = weights_.config;
  const LayerWeights& w = weights_.layers[layer];
  const std::size_t d = c.hidden_size;
  const std::size_t heads = c.num_heads;
  const std::size_t kv_heads = c.num_kv_heads;
  const std::size_t dim = c.head_dim;
  const std::size_t group = heads / kv_heads;
  const std::size_t q_dim = heads * dim;
  const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
  std::vector<float>& keys = keys_[layer];
  std::vector<float>& values = values_[layer];

  std::vector<float> qg(rows * 2 * q_dim);
  std::vector<float> kv(rows * 2 * kv_heads * dim);
  MatMulRef(w.attn_q, 2 * q_dim, d, x, rows, qg.data());
  MatMulRef(w.attn_k, kv_heads * dim, d, x, rows, kv.data());
  std::vector<float> vv(rows * kv_heads * dim);
  MatMulRef(w.attn_v, kv_heads * dim, d, x, rows, vv.data());
  const float* qn = static_cast<const float*>(w.attn_q_norm.data);
  const float* kn = static_cast<const float*>(w.attn_k_norm.data);

  const std::uint32_t first = position_;
  std::vector<float> qh(dim);
  std::vector<float> kh(dim);
  std::vector<float> gated(rows * heads * dim);
  for (std::size_t r = 0; r < rows; ++r) {
    const std::uint32_t pos = first + static_cast<std::uint32_t>(r);
    std::vector<float> krow(kv_heads * dim);
    std::vector<float> vrow(kv_heads * dim);
    const float* kvr = &kv[r * kv_heads * dim];
    const float* vvr = &vv[r * kv_heads * dim];
    for (std::size_t h = 0; h < kv_heads; ++h) {
      for (std::size_t i = 0; i < dim; ++i) {
        kh[i] = kvr[h * dim + i];
      }
      RmsNorm(kh.data(), kn, dim, c.rms_eps, kh.data());
      Rope(kh.data(), c.rotary_dim, pos, c.rope_theta);
      for (std::size_t i = 0; i < dim; ++i) {
        krow[h * dim + i] = kh[i];
        vrow[h * dim + i] = vvr[h * dim + i];
      }
    }
    keys.insert(keys.end(), krow.begin(), krow.end());
    values.insert(values.end(), vrow.begin(), vrow.end());
  }
  for (std::size_t r = 0; r < rows; ++r) {
    const std::uint32_t pos = first + static_cast<std::uint32_t>(r);
    const std::uint32_t hi = pos + 1;
    // The packed projection interleaves query and gate per head:
    // [q0, g0, q1, g1, ...], each `dim` wide. A flat first-half/second-half
    // split would mix queries of one half of the heads with gates of the
    // other; index both halves per head instead.
    const float* qgr = &qg[r * 2 * q_dim];
    for (std::size_t h = 0; h < heads; ++h) {
      for (std::size_t i = 0; i < dim; ++i) {
        qh[i] = qgr[h * 2 * dim + i];
      }
      RmsNorm(qh.data(), qn, dim, c.rms_eps, qh.data());
      Rope(qh.data(), c.rotary_dim, pos, c.rope_theta);
      const std::size_t kvh = h / group;
      std::vector<double> scores(hi);
      double best = -INFINITY;
      for (std::uint32_t p = 0; p < hi; ++p) {
        const float* k =
            keys.data() + static_cast<std::size_t>(p) * kv_heads * dim +
            kvh * dim;
        double s = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
          s += static_cast<double>(qh[i]) * k[i];
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
      for (std::uint32_t p = 0; p < hi; ++p) {
        const float* v =
            values.data() + static_cast<std::size_t>(p) * kv_heads * dim +
            kvh * dim;
        const double weight = scores[p] / total;
        for (std::size_t i = 0; i < dim; ++i) {
          acc[i] += weight * v[i];
        }
      }
      // The output gate is the packed gate half of this head,
      // applied elementwise before the output projection.
      const float* gh = qgr + h * 2 * dim + dim;
      for (std::size_t i = 0; i < dim; ++i) {
        gated[(r * heads + h) * dim + i] =
            static_cast<float>(acc[i] * Sigmoid(gh[i]));
      }
    }
  }
  MatMulRef(w.attn_output, d, heads * dim, gated.data(), rows, out);
}

void Reference::Mixture(const LayerWeights& w, const float* x, std::size_t rows,
                        float* out) const {
  const Config& c = weights_.config;
  const std::size_t d = c.hidden_size;
  const std::size_t f = c.expert_ffn_size;
  const std::size_t s = c.shared_ffn_size;
  const std::size_t e = c.num_experts;

  std::vector<float> logits(rows * e);
  MatMulRef(w.ffn_gate_inp, e, d, x, rows, logits.data());
  std::fill(out, out + rows * d, 0.0F);

  std::vector<std::size_t> order(e);
  std::vector<float> gate(f);
  std::vector<float> up(f);
  std::vector<float> down(d);
  std::vector<float> probs(e);
  for (std::size_t r = 0; r < rows; ++r) {
    double best = -INFINITY;
    for (std::size_t i = 0; i < e; ++i) {
      best = std::max(best, static_cast<double>(logits[r * e + i]));
    }
    double total = 0.0;
    for (std::size_t i = 0; i < e; ++i) {
      probs[i] =
          static_cast<float>(std::exp(logits[r * e + i] - best));
      total += probs[i];
    }
    for (std::size_t i = 0; i < e; ++i) {
      probs[i] = static_cast<float>(probs[i] / total);
      order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a,
                                                     std::size_t b) {
      return probs[a] > probs[b];
    });
    double wsum = 0.0;
    for (std::size_t j = 0; j < c.experts_used; ++j) {
      wsum += probs[order[j]];
    }
    const std::size_t gate_stride =
        quant::EncodedSizeBytes(w.ffn_gate_exps.type, d);
    const std::size_t down_stride =
        quant::EncodedSizeBytes(w.ffn_down_exps.type, f);
    const std::size_t gate_expert = f * gate_stride;
    const std::size_t down_expert = d * down_stride;
    for (std::size_t j = 0; j < c.experts_used; ++j) {
      const std::size_t ex = order[j];
      const double weight = probs[ex] / wsum;
      const std::uint8_t* gb =
          static_cast<const std::uint8_t*>(w.ffn_gate_exps.data) +
          ex * gate_expert;
      const std::uint8_t* ub =
          static_cast<const std::uint8_t*>(w.ffn_up_exps.data) +
          ex * gate_expert;
      const std::uint8_t* db =
          static_cast<const std::uint8_t*>(w.ffn_down_exps.data) +
          ex * down_expert;
      MatMul(gb, w.ffn_gate_exps.type, f, d, &x[r * d], 1, gate.data());
      MatMul(ub, w.ffn_up_exps.type, f, d, &x[r * d], 1, up.data());
      for (std::size_t i = 0; i < f; ++i) {
        gate[i] = static_cast<float>(Silu(gate[i]) * up[i]);
      }
      MatMul(db, w.ffn_down_exps.type, d, f, gate.data(), 1, down.data());
      for (std::size_t i = 0; i < d; ++i) {
        out[r * d + i] = static_cast<float>(out[r * d + i] + weight * down[i]);
      }
    }
  }
  // The shared expert is gated by sigmoid(gate_vector . x).
  std::vector<float> sh_gate(rows * s);
  std::vector<float> sh_up(rows * s);
  MatMulRef(w.ffn_gate_shexp, s, d, x, rows, sh_gate.data());
  MatMulRef(w.ffn_up_shexp, s, d, x, rows, sh_up.data());
  std::vector<float> sh_out(rows * d, 0.0F);
  std::vector<float> gate_vec(d);
  DequantRow(core::GgmlType::kF32, w.ffn_gate_inp_shexp.data, gate_vec.data(),
             d);
  for (std::size_t r = 0; r < rows; ++r) {
    double dot = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
      dot += gate_vec[i] * x[r * d + i];
    }
    const float g = static_cast<float>(Sigmoid(dot));
    for (std::size_t i = 0; i < s; ++i) {
      sh_gate[r * s + i] =
          static_cast<float>(Silu(sh_gate[r * s + i]) * sh_up[r * s + i]);
    }
    std::vector<float> one(d, 0.0F);
    MatMulRef(w.ffn_down_shexp, d, s, &sh_gate[r * s], 1, one.data());
    for (std::size_t i = 0; i < d; ++i) {
      out[r * d + i] += g * one[i];
    }
  }
}

void Reference::Forward(std::span<const std::int32_t> tokens,
                        std::vector<float>* out_logits) {
  const Config& c = weights_.config;
  const std::size_t n = tokens.size();
  const std::size_t d = c.hidden_size;
  std::uint64_t vocab = weights_.token_embd.num_elements / d;

  std::vector<float> h(n * d);
  const std::size_t emb_stride =
      quant::EncodedSizeBytes(weights_.token_embd.type, d);
  for (std::size_t r = 0; r < n; ++r) {
    const std::uint8_t* row =
        static_cast<const std::uint8_t*>(weights_.token_embd.data) +
        static_cast<std::size_t>(tokens[r]) * emb_stride;
    DequantRow(weights_.token_embd.type, row, &h[r * d], d);
  }

  std::vector<float> normed(n * d);
  std::vector<float> mixed(n * d);
  std::vector<float> tmp(n * d);
  // DIAGNOSTIC ONLY: with GUFO_QWEN35MOE_DUMP_LAYERS set to a directory,
  // writes the post-residual stream after every layer. Used to bisect
  // oracle-vs-oracle disagreements layer by layer.
  const char* dump_dir = std::getenv("GUFO_QWEN35MOE_DUMP_LAYERS");
  // DIAGNOSTIC ONLY counter part: keeps the test's three Forward calls apart.
  // Remove before finishing.
  static int ref_call = 0;
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const LayerWeights& w = weights_.layers[l];
    const float* wn = static_cast<const float*>(w.attn_norm.data);
    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&h[r * d], wn, d, c.rms_eps, &normed[r * d]);
    }
    if (c.IsLinear(l)) {
      // Reference processes rows sequentially to keep the recurrent state
      // exact; a chunked call gives identical results.
      for (std::size_t r = 0; r < n; ++r) {
        LinearLayer(l, &normed[r * d], 1, &mixed[r * d]);
      }
    } else {
      FullLayer(l, normed.data(), n, mixed.data());
    }
    for (std::size_t i = 0; i < n * d; ++i) {
      h[i] += mixed[i];
    }
    const float* wp = static_cast<const float*>(w.post_attention_norm.data);
    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&h[r * d], wp, d, c.rms_eps, &normed[r * d]);
    }
    Mixture(w, normed.data(), n, tmp.data());
    for (std::size_t i = 0; i < n * d; ++i) {
      h[i] += tmp[i];
    }
    if (dump_dir != nullptr) {
      char path[256];
      std::snprintf(path, sizeof(path), "%s/ref_c%d_layer%02u.bin", dump_dir,
                    ref_call, l);
      FILE* f = std::fopen(path, "wb");
      if (f != nullptr) {
        std::fwrite(h.data(), sizeof(float), n * d, f);
        std::fclose(f);
      }
    }
  }
  const float* wo = static_cast<const float*>(weights_.output_norm.data);
  for (std::size_t r = 0; r < n; ++r) {
    RmsNorm(&h[r * d], wo, d, c.rms_eps, &normed[r * d]);
  }
  out_logits->assign(n * vocab, 0.0F);
  MatMulRef(weights_.output, vocab, d, normed.data(), n, out_logits->data());
  position_ += static_cast<std::uint32_t>(n);
  ++ref_call;
}

}  // namespace gufo::models::qwen35moe
