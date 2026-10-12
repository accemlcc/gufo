#include "src/models/kolibri1/engine.hpp"

#include <algorithm>
#include <utility>

#include "src/models/kolibri1/chat_template.hpp"

namespace gufo::models::kolibri1 {

std::unique_ptr<Model> Model::Load(const std::string& path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::unique_ptr<Model> m(new Model());
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  // The engine implements one prompt format, and a file whose template differs
  // would be answered in a dialect the model was never shown.
  if (!ChatTemplate::ValidateGgufTemplate(*m->reader_, error_msg)) {
    return nullptr;
  }
  m->weights_ = ModelWeights::Bind(*m->reader_, error_msg);
  if (!m->weights_.has_value()) {
    return nullptr;
  }
  m->tokenizer_ = Tokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_, error_msg);
  if (!m->device_) {
    return nullptr;
  }
  m->executor_ = std::make_shared<rocm::Executor>(*m->device_,
                                                   options.prefill_chunk, 1,
                                                   options.max_context);
  return m;
}

Model::~Model() = default;

std::unique_ptr<Session> Model::CreateSession(std::string* error_msg) const {
  auto cache = executor_->CreateCache(options_.max_context, error_msg);
  if (cache == nullptr) {
    return nullptr;
  }
  return std::make_unique<Session>(*this, std::move(cache));
}

Session::Session(const Model& model, std::unique_ptr<rocm::KvCache> cache)
    : model_(model), cache_(std::move(cache)) {}

void Session::Reset() noexcept {
  // Nothing to free and no marker to clear: the cache holds no length, the
  // attention kernels read exactly the positions a forward names, and the next
  // Append starts writing at zero again.
  tokens_.clear();
  logits_.clear();
}

void Session::Append(std::span<const TokenId> tokens) {
  if (tokens.empty()) {
    return;
  }
  if (position() + tokens.size() > model_.max_context()) {
    throw std::invalid_argument("kolibri1 session past its " +
                                std::to_string(model_.max_context()) +
                                " position context");
  }
  const std::uint32_t chunk =
      static_cast<std::uint32_t>(model_.executor()->max_rows());
  std::size_t done = 0;
  while (done < tokens.size()) {
    const std::size_t n = std::min<std::size_t>(chunk, tokens.size() - done);
    // Only the very last row of the whole append is asked for logits: the rows
    // before it feed nothing.
    const bool last = done + n == tokens.size();
    const std::uint32_t frontier = static_cast<std::uint32_t>(done + n) - 1;
    const std::span<const std::uint32_t> rows =
        last ? std::span<const std::uint32_t>(&frontier, 1)
             : std::span<const std::uint32_t>{};
    model_.executor()->Forward(*cache_, tokens.subspan(done, n), position(),
                              rows);
    tokens_.insert(tokens_.end(), tokens.begin() + done,
                   tokens.begin() + done + n);
    done += n;
  }
  model_.executor()->CopyLogits(1, &logits_);
}

}  // namespace gufo::models::kolibri1
