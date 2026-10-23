#include "src/models/qwen35moe/engine.hpp"

#include <string>
#include <utility>

namespace gufo::models::qwen35moe {

Model::~Model() = default;

std::unique_ptr<Model> Model::Load(const std::string& path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::unique_ptr<Model> m(new Model());
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  m->weights_ = ModelWeights::LoadFromGguf(*m->reader_, error_msg);
  if (!m->weights_.has_value()) {
    return nullptr;
  }
  m->device_ = rocm::DeviceModel::Upload(*m->weights_, *m->reader_, error_msg);
  if (!m->device_) {
    return nullptr;
  }
  const std::size_t vocab = m->device_->vocab_size();
  if (vocab == 0) {
    if (error_msg != nullptr) {
      *error_msg = "qwen35moe artifact has no vocabulary";
    }
    return nullptr;
  }
  m->executor_ = std::make_shared<rocm::Executor>(
      *m->device_, options.prefill_chunk, /*max_logit_rows=*/1,
      options.max_context);
  return m;
}

std::unique_ptr<Session> Model::CreateSession(std::string* error_msg) const {
  auto cache = executor_->CreateCache(options_.max_context, error_msg);
  if (!cache) {
    return nullptr;
  }
  auto ssm = executor_->CreateSsmState(error_msg);
  if (!ssm) {
    return nullptr;
  }
  return std::unique_ptr<Session>(new Session(*this, std::move(cache),
                                              std::move(ssm)));
}

Session::Session(const Model& model, std::unique_ptr<rocm::KvCache> cache,
                 std::unique_ptr<rocm::SsmState> ssm)
    : model_(model), cache_(std::move(cache)), ssm_(std::move(ssm)) {}

void Session::Append(std::span<const TokenId> tokens) {
  const std::uint32_t chunk = model_.executor()->max_rows();
  std::size_t done = 0;
  while (done < tokens.size()) {
    const std::size_t n = std::min<std::size_t>(chunk, tokens.size() - done);
    const bool last = done + n == tokens.size();
    logit_rows_.clear();
    if (last) {
      logit_rows_.push_back(static_cast<std::uint32_t>(n - 1));
    }
    model_.executor()->Forward(
        *cache_, *ssm_, tokens.subspan(done, n),
        static_cast<std::uint32_t>(tokens_.size() + done), logit_rows_);
    if (last) {
      model_.executor()->CopyLogits(logit_rows_.size(), &logits_);
    }
    done += n;
  }
  tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
}

void Session::Reset() noexcept {
  tokens_.clear();
  logits_.clear();
  logit_rows_.clear();
  auto fresh_cache = model_.executor()->CreateCache(model_.max_context(),
                                                    nullptr);
  if (fresh_cache) {
    cache_ = std::move(fresh_cache);
  }
  auto fresh_ssm = model_.executor()->CreateSsmState(nullptr);
  if (fresh_ssm) {
    ssm_ = std::move(fresh_ssm);
  }
}

}  // namespace gufo::models::qwen35moe
