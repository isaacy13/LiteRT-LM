// Copyright 2026 GuideAI contributors.
// Licensed under the Apache License, Version 2.0.

#include "runtime/components/constrained_decoding/constraint_provider_cache.h"

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "runtime/components/constrained_decoding/constraint_provider_factory.h"

namespace litert::lm {

absl::StatusOr<std::shared_ptr<const ConstraintProvider>>
ConstraintProviderCache::Get(
    const ConstraintProviderConfig& config, const support::Tokenizer& tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids) {
  const auto* llg_config = std::get_if<LlGuidanceConfig>(&config);
  if (llg_config == nullptr) {
    // External providers may own mutable resources. Preserve their existing
    // per-conversation creation, without evicting the reusable vocabulary.
    ABSL_ASSIGN_OR_RETURN(auto provider,
                          CreateConstraintProvider(config, tokenizer,
                                                   stop_token_ids));
    return std::shared_ptr<const ConstraintProvider>(std::move(provider));
  }

  absl::MutexLock lock(&mutex_);
  if (provider_ != nullptr && tokenizer_ == &tokenizer &&
      config_.eos_id == llg_config->eos_id &&
      config_.special_token_ids == llg_config->special_token_ids &&
      stop_token_ids_ == stop_token_ids) {
    return provider_;
  }

  // Serialize initialization as well as publication so concurrent sessions
  // never build duplicate vocabularies for the same key. Failure preserves the
  // previous valid entry. Stop order is significant when choosing a default EOS.
  ABSL_ASSIGN_OR_RETURN(auto provider,
                        CreateConstraintProvider(config, tokenizer,
                                                 stop_token_ids));
  provider_ = std::shared_ptr<const ConstraintProvider>(std::move(provider));
  tokenizer_ = &tokenizer;
  config_ = *llg_config;
  stop_token_ids_ = stop_token_ids;
  return provider_;
}

void ConstraintProviderCache::Clear() {
  absl::MutexLock lock(&mutex_);
  provider_.reset();
  tokenizer_ = nullptr;
  config_ = LlGuidanceConfig();
  stop_token_ids_.clear();
}

}  // namespace litert::lm
