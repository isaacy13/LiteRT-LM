// Copyright 2026 GuideAI contributors.
// Licensed under the Apache License, Version 2.0.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_PROVIDER_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_PROVIDER_CACHE_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// One immutable LLGuidance vocabulary per engine. A changed configuration
// replaces the cached entry; existing conversations retain their own owner.
// Grammars and parser states are never cached here. The tokenizer must remain
// alive while providers and their constraints are used, as with a native session.
class ConstraintProviderCache {
 public:
  absl::StatusOr<std::shared_ptr<const ConstraintProvider>> Get(
      const ConstraintProviderConfig& config, const support::Tokenizer& tokenizer,
      const std::vector<std::vector<int>>& stop_token_ids);

  // Release the engine's cache before tearing down its tokenizer.
  void Clear();

 private:
  absl::Mutex mutex_;
  const support::Tokenizer* tokenizer_ ABSL_GUARDED_BY(mutex_) = nullptr;
  LlGuidanceConfig config_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::vector<int>> stop_token_ids_ ABSL_GUARDED_BY(mutex_);
  std::shared_ptr<const ConstraintProvider> provider_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_PROVIDER_CACHE_H_
