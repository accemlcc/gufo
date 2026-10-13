#ifndef GUFO_MODELS_KOLIBRI1_ENGINE_HPP_
#define GUFO_MODELS_KOLIBRI1_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
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
/// One image of a context: its tokens, the key/value rows a later token can
/// still attend, and the logits of its last position. The rows stay on the
/// device: on this unified-memory APU a device copy runs near memory bandwidth
/// while a fresh host buffer pays first-touch page faults. A patternless layer
/// contributes rows [0, position) and a sliding layer its whole ring exactly as
/// the attention kernel stored it, so a restore reorders nothing and the
/// destination must have the same geometry. Payload order: header, tokens, then
/// K and V of each layer in layer order, then the logits.
class SessionSnapshot final {
public:
  /// Hands the payload to `sink` in order, in pieces of at most 64 MiB.
  using Sink = std::function<void(std::span<const std::uint8_t>)>;

  ~SessionSnapshot();
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;

  [[nodiscard]] std::uint64_t SizeBytes() const noexcept;
  /// Writes the payload; false on a size mismatch or a failed device copy.
  [[nodiscard]] bool CopyTo(std::span<std::uint8_t> destination) const;
  [[nodiscard]] bool Stream(const Sink& sink) const;

private:
  struct Impl;

  SessionSnapshot() = default;

  std::unique_ptr<Impl> impl_;
  friend class Session;
};

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

  /// Payloads of this architecture. A store must reject another version rather
  /// than read it as this one.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 1;
  /// Bytes the payload of the current state takes: header, tokens, the rows of
  /// every layer, logits. It does not depend on the cache's capacity, only on
  /// the rows a token has written.
  [[nodiscard]] std::uint64_t SnapshotBytes() const noexcept;
  /// Freezes the context. Nullptr with `error_msg` set when a device copy fails.
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  /// Writes a snapshot over this context, whose geometry must match the one it
  /// was taken from. False, leaving the context as it was, when the payload is
  /// of another version or geometry.
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  /// The same from a payload read back, which is checked before anything is
  /// written.
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);

private:
  const Model& model_;
  std::unique_ptr<rocm::KvCache> cache_;
  std::vector<TokenId> tokens_;
  std::vector<float> logits_;
};

}  // namespace gufo::models::kolibri1

#endif  // GUFO_MODELS_KOLIBRI1_ENGINE_HPP_
