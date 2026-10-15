#include "src/models/kolibri1/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include "src/core/quant/ggml_gemm.hpp"

namespace gufo::models::kolibri1 {
namespace {

/// y = x * rsqrt(mean(x^2) + eps) * w; `w` null means an unweighted norm.
void RmsNorm(const float* x, const float* w, std::size_t n, float eps,
             float* y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(x[i]) * x[i];
  }
  const double scale = 1.0 / std::sqrt(sum / static_cast<double>(n) + eps);
  for (std::size_t i = 0; i < n; ++i) {
    const double v = static_cast<double>(x[i]) * scale;
    y[i] = static_cast<float>(w != nullptr ? v * w[i] : v);
  }
}

const float* Vec(const TensorRef& t) {
  return static_cast<const float*>(t.data);
}

/// NEOX rotation of one head: the pair (i, i + dim/2) turns by
/// position * theta^(-2i/dim).
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

double Sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

/// SwiGLU's inner activation: x * sigmoid(x).
double Silu(double x) {
  return x * Sigmoid(x);
}

/// One [cols, rows] matrix out of a stack along ne[2].
[[nodiscard]] TensorRef Slice(const TensorRef& t, std::size_t expert) {
  TensorRef s = t;
  s.experts = 1;
  s.data = static_cast<const std::uint8_t*>(t.data) + expert * t.ExpertBytes();
  return s;
}

}  // namespace

Reference::Reference(const ModelWeights& weights) : weights_(weights) {
  Reset();
}

void Reference::Reset() {
  position_ = 0;
  keys_.assign(weights_.config.num_layers, {});
  values_.assign(weights_.config.num_layers, {});
  layer_trace_.clear();
  stage_trace_.clear();
}

void Reference::MatMul(const TensorRef& weight, const float* x_in,
                       std::size_t rows, float* out) const {
  const std::size_t cols = weight.cols;
  const float* x = x_in;
  const std::size_t outputs = weight.rows;
  const std::size_t row_bytes = weight.RowBytes();
  std::vector<float> w(cols);
  for (std::size_t o = 0; o < outputs; ++o) {
    quant::Dequantize(
        weight.type,
        static_cast<const std::uint8_t*>(weight.data) + o * row_bytes, w.data(),
        cols);
    for (std::size_t r = 0; r < rows; ++r) {
      const float* xr = x + r * cols;
      double acc = 0.0;
      for (std::size_t c = 0; c < cols; ++c) {
        acc += static_cast<double>(w[c]) * xr[c];
      }
      out[r * outputs + o] = static_cast<float>(acc);
    }
  }
}

void Reference::Attention(std::uint32_t layer, const float* q, std::size_t rows,
                          float* out) const {
  const Config& c = weights_.config;
  const std::size_t dim = c.head_dim;
  const std::size_t heads = c.num_heads;
  const std::size_t kv_heads = c.kv_heads[layer];
  const std::size_t group = heads / kv_heads;
  const std::size_t kv_stride = kv_heads * dim;
  const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
  const auto& keys = keys_[layer];
  const auto& values = values_[layer];
  const std::uint32_t first = position_;  // position of row 0
  for (std::size_t r = 0; r < rows; ++r) {
    const std::uint32_t pos = first + static_cast<std::uint32_t>(r);
    // A sliding layer sees `sliding_window` keys including its own query; the
    // global layers are unwindowed and un-roped.
    const std::uint32_t lo = c.IsSliding(layer) && pos + 1 > c.sliding_window
                                 ? pos + 1 - c.sliding_window
                                 : 0;
    const std::uint32_t hi = pos + 1;
    for (std::size_t h = 0; h < heads; ++h) {
      const float* qh = q + (r * heads + h) * dim;
      const std::size_t kvh = h / group;
      std::vector<double> scores(hi - lo);
      double max_score = -INFINITY;
      for (std::uint32_t p = lo; p < hi; ++p) {
        const float* k =
            keys.data() + static_cast<std::size_t>(p) * kv_stride + kvh * dim;
        double s = 0.0;
        for (std::size_t d = 0; d < dim; ++d) {
          s += static_cast<double>(qh[d]) * k[d];
        }
        scores[p - lo] = s * scale;
        max_score = std::max(max_score, scores[p - lo]);
      }
      double total = 0.0;
      for (double& s : scores) {
        s = std::exp(s - max_score);
        total += s;
      }
      std::vector<double> acc(dim, 0.0);
      for (std::uint32_t p = lo; p < hi; ++p) {
        const float* v =
            values.data() + static_cast<std::size_t>(p) * kv_stride + kvh * dim;
        const double weight = scores[p - lo] / total;
        for (std::size_t d = 0; d < dim; ++d) {
          acc[d] += weight * v[d];
        }
      }
      float* o = out + (r * heads + h) * dim;
      for (std::size_t d = 0; d < dim; ++d) {
        o[d] = static_cast<float>(acc[d]);
      }
    }
  }
}

void Reference::Stage(const std::string& name, const float* data,
                      std::size_t count) {
  if (!stage_trace_enabled_) {
    return;
  }
  stage_trace_[name].assign(data, data + count);
}

void Reference::Mixture(const LayerWeights& w, const float* x, std::size_t rows,
                        float* routed, float* shared) const {
  const Config& c = weights_.config;
  const std::size_t d = c.hidden_size;
  const std::size_t width = c.expert_ffn_size;
  const std::size_t experts = c.num_experts;

  // The router scores every expert in F32. Its bias moves the selection only;
  // the weight a chosen expert contributes is the unbiased sigmoid.
  std::vector<float> logits(rows * experts);
  MatMul(w.router, x, rows, logits.data());
  const float* bias = Vec(w.router_bias);

  // Either output may be dropped; a null pointer gets a scratch buffer so the
  // arithmetic below stays one path.
  std::vector<float> scratch(rows * d);
  if (routed == nullptr) {
    routed = scratch.data();
  }
  std::fill(routed, routed + rows * d, 0.0F);
  std::vector<std::size_t> order(experts);
  std::vector<float> gate(width);
  std::vector<float> up(width);
  std::vector<float> act(width);
  std::vector<float> down(d);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* l = &logits[r * experts];
    for (std::size_t e = 0; e < experts; ++e) {
      order[e] = e;
    }
    // top-k over `logits + bias`; equal scores keep the lower expert index.
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                       return static_cast<double>(l[a]) + bias[a] >
                              static_cast<double>(l[b]) + bias[b];
                     });
    for (std::size_t j = 0; j < c.experts_used; ++j) {
      const std::size_t e = order[j];
      const double weight = Sigmoid(static_cast<double>(l[e]));
      MatMul(Slice(w.gate_exps, e), &x[r * d], 1, gate.data());
      MatMul(Slice(w.up_exps, e), &x[r * d], 1, up.data());
      for (std::size_t i = 0; i < width; ++i) {
        act[i] = static_cast<float>(Silu(gate[i]) * up[i]);
      }
      MatMul(Slice(w.down_exps, e), act.data(), 1, down.data());
      for (std::size_t i = 0; i < d; ++i) {
        routed[r * d + i] = static_cast<float>(
            static_cast<double>(routed[r * d + i]) + weight * down[i]);
      }
    }
  }
  // The one ungated shared expert reads the same input and adds with weight 1.
  const std::size_t sh_width = c.shared_expert_ffn_size;
  std::vector<float> sh_gate(rows * sh_width);
  std::vector<float> sh_up(rows * sh_width);
  std::vector<float> shared_out(rows * d);
  MatMul(w.gate_shexp, x, rows, sh_gate.data());
  MatMul(w.up_shexp, x, rows, sh_up.data());
  for (std::size_t i = 0; i < rows * sh_width; ++i) {
    sh_gate[i] = static_cast<float>(Silu(sh_gate[i]) * sh_up[i]);
  }
  MatMul(w.down_shexp, sh_gate.data(), rows, shared_out.data());
  if (shared != nullptr) {
    std::copy(shared_out.begin(), shared_out.end(), shared);
  }
}

void Reference::Forward(std::span<const std::int32_t> tokens,
                        std::vector<float>* logits,
                        std::vector<float>* hidden) {
  const Config& c = weights_.config;
  const std::size_t n = tokens.size();
  const std::size_t d = c.hidden_size;
  const std::size_t dim = c.head_dim;
  const std::size_t heads = c.num_heads;
  const float eps = c.rms_eps;
  std::vector<float> x(n * d);
  std::vector<float> h(n * d);

  // Kolibri scales its embeddings by neither sqrt(hidden) nor a learned
  // factor: the row is the residual stream.
  const std::size_t embed_row = weights_.token_embd.RowBytes();
  for (std::size_t r = 0; r < n; ++r) {
    quant::Dequantize(
        weights_.token_embd.type,
        static_cast<const std::uint8_t*>(weights_.token_embd.data) +
            static_cast<std::size_t>(tokens[r]) * embed_row,
        x.data() + r * d, d);
  }
  if (trace_) {
    layer_trace_.assign(static_cast<std::size_t>(c.num_layers) * n * d, 0.0F);
  }

  std::vector<float> q(n * c.QDim());
  std::vector<float> k(n * c.KvDim(0));
  std::vector<float> v(n * c.KvDim(0));
  std::vector<float> attn(n * c.QDim());
  std::vector<float> branch(n * d);
  std::vector<float> routed(n * d);
  std::vector<float> shared(n * d);
  for (std::uint32_t l = 0; l < c.num_layers; ++l) {
    const std::string at = "-" + std::to_string(l);
    const LayerWeights& w = weights_.layers[l];
    const std::size_t kv_heads = c.kv_heads[l];
    q.assign(n * c.QDim(), 0.0F);
    k.assign(n * c.KvDim(l), 0.0F);
    v.assign(n * c.KvDim(l), 0.0F);

    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&x[r * d], Vec(w.attn_norm), d, eps, &h[r * d]);
    }
    Stage("attn_norm" + at, h.data(), h.size());
    MatMul(w.attn_q, h.data(), n, q.data());
    MatMul(w.attn_k, h.data(), n, k.data());
    MatMul(w.attn_v, h.data(), n, v.data());
    Stage("Vcur" + at, v.data(), v.size());

    // Per-head RMS norms over the head width, then RoPE where the layer has a
    // window. V is neither normed nor rotated. The two passes are kept apart
    // because the reference traces the normed and the rotated tensor apart.
    for (std::size_t r = 0; r < n; ++r) {
      for (std::size_t hh = 0; hh < heads; ++hh) {
        float* head = &q[(r * heads + hh) * dim];
        RmsNorm(head, Vec(w.attn_q_norm), dim, eps, head);
      }
      for (std::size_t hh = 0; hh < kv_heads; ++hh) {
        float* kh = &k[(r * kv_heads + hh) * dim];
        RmsNorm(kh, Vec(w.attn_k_norm), dim, eps, kh);
      }
    }
    Stage("Qcur_normed" + at, q.data(), q.size());
    Stage("Kcur_normed" + at, k.data(), k.size());
    if (c.UsesRope(l)) {
      for (std::size_t r = 0; r < n; ++r) {
        const auto pos = static_cast<std::uint32_t>(position_ + r);
        for (std::size_t hh = 0; hh < heads; ++hh) {
          Rope(&q[(r * heads + hh) * dim], dim, pos, c.rope_theta);
        }
        for (std::size_t hh = 0; hh < kv_heads; ++hh) {
          Rope(&k[(r * kv_heads + hh) * dim], dim, pos, c.rope_theta);
        }
      }
      Stage("Qcur_rope" + at, q.data(), q.size());
      Stage("Kcur_rope" + at, k.data(), k.size());
    }
    keys_[l].insert(keys_[l].end(), k.begin(), k.end());
    values_[l].insert(values_[l].end(), v.begin(), v.end());

    Attention(l, q.data(), n, attn.data());
    MatMul(w.attn_output, attn.data(), n, branch.data());
    Stage("attn_out" + at, branch.data(), branch.size());
    // The sandwich norm sits on the branch output; the residual is added
    // after it and becomes the input of the mixture.
    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&branch[r * d], Vec(w.post_attn_norm), d, eps, &branch[r * d]);
    }
    Stage("attn_post_norm" + at, branch.data(), branch.size());
    for (std::size_t r = 0; r < n; ++r) {
      for (std::size_t i = 0; i < d; ++i) {
        x[r * d + i] += branch[r * d + i];
      }
      RmsNorm(&x[r * d], Vec(w.ffn_norm), d, eps, &h[r * d]);
    }
    Stage("ffn_inp" + at, x.data(), x.size());
    Stage("ffn_norm" + at, h.data(), h.size());

    Mixture(w, h.data(), n, routed.data(), shared.data());
    Stage("ffn_moe_out" + at, routed.data(), routed.size());
    Stage("ffn_shexp" + at, shared.data(), shared.size());
    for (std::size_t i = 0; i < n * d; ++i) {
      branch[i] =
          static_cast<float>(static_cast<double>(routed[i]) + shared[i]);
    }
    Stage("ffn_out" + at, branch.data(), branch.size());
    for (std::size_t r = 0; r < n; ++r) {
      RmsNorm(&branch[r * d], Vec(w.post_ffn_norm), d, eps, &branch[r * d]);
    }
    Stage("ffn_post_norm" + at, branch.data(), branch.size());
    for (std::size_t r = 0; r < n; ++r) {
      for (std::size_t i = 0; i < d; ++i) {
        x[r * d + i] += branch[r * d + i];
      }
    }
    Stage("l_out" + at, x.data(), x.size());
    if (trace_) {
      std::copy(x.begin(), x.end(),
                layer_trace_.begin() + static_cast<std::size_t>(l) * n * d);
    }
  }

  for (std::size_t r = 0; r < n; ++r) {
    RmsNorm(&x[r * d], Vec(weights_.output_norm), d, eps, &h[r * d]);
  }
  Stage("result_norm", h.data(), h.size());
  if (hidden != nullptr) {
    *hidden = h;
  }
  if (logits != nullptr) {
    logits->assign(n * weights_.VocabSize(), 0.0F);
    MatMul(weights_.output, h.data(), n, logits->data());
  }
  position_ += static_cast<std::uint32_t>(n);
}

}  // namespace gufo::models::kolibri1
