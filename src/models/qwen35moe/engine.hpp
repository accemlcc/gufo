#ifndef GUFO_MODELS_QWEN35MOE_ENGINE_HPP_
#define GUFO_MODELS_QWEN35MOE_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen35moe/config.hpp"
#include "src/models/qwen35moe/kernels/rocm/device_model.hpp"
#include "src/models/qwen35moe/kernels/rocm/executor.hpp"
#include "src/models/qwen35moe/weights.hpp"

namespace gufo::models::qwen35moe {

using TokenId = std::int32_t;

/// Buffer sizes. `max_context` bounds one session's cache and `prefill_chunk`
/// the rows of one forward, which together decide how much weight traffic a
/// long prompt pays for: a forward re-reads the projections and the experts
/// it touches once per chunk.
struct ModelOptions {
  std::uint32_t max_context = 8192;
  std::uint32_t prefill_chunk = 16;
};

class Session;

/// A loaded model: the artifact mapped, its weights resident on the device.
/// One executor serves every session; a forward is serialized on its stream.
class Model {
 public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::unique_ptr<Model> Load(
      const std::string& path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept {
    return weights_->config;
  }
  [[nodiscard]] const rocm::DeviceModel& device() const noexcept {
    return *device_;
  }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return options_.max_context;
  }
  /// Bytes of weight the device holds, for a startup line that can be checked
  /// against the file.
  [[nodiscard]] std::size_t resident_bytes() const noexcept {
    return device_->resident_bytes();
  }
  /// A session with an empty context and its own cache.
  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::string* error_msg = nullptr) const;
  /// The device driver every session of this model shares.
  [[nodiscard]] const std::shared_ptr<rocm::Executor>& executor()
      const noexcept {
    return executor_;
  }

 private:
  Model() = default;

  ModelOptions options_;
  std::unique_ptr<core::GgufReader> reader_;
  std::optional<ModelWeights> weights_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::shared_ptr<rocm::Executor> executor_;
};

class Session {
 public:
  Session(const Model& model, std::unique_ptr<rocm::KvCache> cache,
          std::unique_ptr<rocm::SsmState> ssm);
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Evaluates `tokens` at the positions after the current frontier and leaves
  /// the logits of the last of them in `logits()`.
  void Append(std::span<const TokenId> tokens);
  /// Forgets the conversation: back to position zero, so the next `Append`
  /// writes the first positions again.
  void Reset() noexcept;
  /// The logits of `logits().size()` values, one per vocabulary entry, of the
  /// last evaluated position.
  [[nodiscard]] std::span<const float> logits() const noexcept {
    return logits_;
  }
  [[nodiscard]] std::span<const TokenId> tokens() const noexcept {
    return tokens_;
  }
  /// Positions filled so far, which is where the next token lands.
  [[nodiscard]] std::uint32_t position() const noexcept {
    return static_cast<std::uint32_t>(tokens_.size());
  }

 private:
  const Model& model_;
  std::unique_ptr<rocm::KvCache> cache_;
  std::unique_ptr<rocm::SsmState> ssm_;
  std::vector<TokenId> tokens_;
  std::vector<float> logits_;
  std::vector<std::uint32_t> logit_rows_;
};

}  // namespace gufo::models::qwen35moe

#endif  // GUFO_MODELS_QWEN35MOE_ENGINE_HPP_
