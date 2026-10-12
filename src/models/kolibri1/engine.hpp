#ifndef GUFO_MODELS_KOLIBRI1_ENGINE_HPP_
#define GUFO_MODELS_KOLIBRI1_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/kolibri1/config.hpp"
#include "src/models/kolibri1/kernels/rocm/device_model.hpp"
#include "src/models/kolibri1/kernels/rocm/executor.hpp"
#include "src/models/kolibri1/tokenizer.hpp"
#include "src/models/kolibri1/weights.hpp"

namespace gufo::models::kolibri1 {

class Session;

/// Buffer sizes. `max_context` bounds one session's cache and `prefill_chunk`
/// the rows of one forward, which together decide how much weight traffic a
/// long prompt pays: a forward re-reads the projections and the experts it
/// touches once per chunk.
struct ModelOptions {
  std::uint32_t max_context = 8192;
  std::uint32_t prefill_chunk = 16;
};

/// A loaded model: the artifact mapped, its weights resident on the device, and
/// the tokenizer of the same file. One executor serves every session; a forward
/// is serialized on its stream.
class Model {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::unique_ptr<Model> Load(
      const std::string& path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept { return weights_->config; }
  [[nodiscard]] const Tokenizer& tokenizer() const noexcept {
    return *tokenizer_;
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
  /// The device driver every session of this model shares. Shared ownership
  /// rather than a bare reference, so a session may drive it through a const
  /// model: the executor is the model's machinery, not its state.
  [[nodiscard]] const std::shared_ptr<rocm::Executor>& executor() const noexcept {
    return executor_;
  }

private:
  Model() = default;

  ModelOptions options_;
  std::unique_ptr<core::GgufReader> reader_;
  std::optional<ModelWeights> weights_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::shared_ptr<rocm::Executor> executor_;
  std::unique_ptr<Tokenizer> tokenizer_;
};

/// One conversation: its tokens, its attention state, and the logits of its
/// last position. Sampling is the caller's business - this class knows the
/// graph, not the product's decoding rules.
class Session {
public:
  Session(const Model& model, std::unique_ptr<rocm::KvCache> cache);

  /// Evaluates `tokens` at the positions after the current frontier and leaves
  /// the logits of the last of them in `logits()`.
  void Append(std::span<const TokenId> tokens);
  /// Forgets the conversation: back to position zero, so the next `Append`
  /// writes the first positions again. A client that re-renders a whole
  /// transcript per turn needs this, and without it the second turn would
  /// append a copy of the first instead of replacing it.
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
  std::vector<TokenId> tokens_;
  std::vector<float> logits_;
};

}  // namespace gufo::models::kolibri1

#endif  // GUFO_MODELS_KOLIBRI1_ENGINE_HPP_
