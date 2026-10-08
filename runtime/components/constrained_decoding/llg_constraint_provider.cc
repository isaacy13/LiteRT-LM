// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/constrained_decoding/llg_constraint_provider.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/llg_constraint.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "llguidance.h"

namespace litert::lm {
namespace {

::LlgConstraint* CreateLlgConstraint(LlgConstraintInit* llg_constraint_init,
                                     absl::string_view constraint_string,
                                     LlgConstraintType constraint_type) {
  switch (constraint_type) {
    case LlgConstraintType::kRegex:
      return llg_new_constraint_regex(llg_constraint_init,
                                      constraint_string.data());
    case LlgConstraintType::kJsonSchema:
      return llg_new_constraint_json(llg_constraint_init,
                                     constraint_string.data());
    case LlgConstraintType::kLark:
      return llg_new_constraint_lark(llg_constraint_init,
                                     constraint_string.data());
    case LlgConstraintType::kLlGuidanceInternal:
      return llg_new_constraint(llg_constraint_init, constraint_string.data());
  }
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<ConstraintProvider>>
LlgConstraintProvider::Create(const Tokenizer& tokenizer,
                              LlGuidanceConfig llg_config) {
  if (!llg_config.eos_id.has_value()) {
    return absl::InvalidArgumentError("LlGuidanceConfig::eos_id must be set.");
  }

  auto vocabulary = tokenizer.GetConstraintVocabulary();
  auto& tokens = vocabulary.token_bytes;
  if (*llg_config.eos_id >= tokens.size()) {
    return absl::InvalidArgumentError("LLGuidance EOS is outside vocabulary.");
  }
  std::vector<bool> special(tokens.size(), false);
  vocabulary.special_token_ids.insert(vocabulary.special_token_ids.end(),
                                     llg_config.special_token_ids.begin(),
                                     llg_config.special_token_ids.end());
  if (!llg_config.special_tokens.empty()) {
    const auto raw_tokens = tokenizer.GetTokens();
    if (raw_tokens.size() != tokens.size()) {
      return absl::InvalidArgumentError("Raw and constraint vocabularies differ in size.");
    }
    const absl::flat_hash_set<absl::string_view> configured(
        llg_config.special_tokens.begin(), llg_config.special_tokens.end());
    for (int id = 0; id < raw_tokens.size(); ++id) {
      if (configured.contains(raw_tokens[id])) {
        vocabulary.special_token_ids.push_back(id);
      }
    }
  }
  vocabulary.special_token_ids.push_back(*llg_config.eos_id);
  for (int id : vocabulary.special_token_ids) {
    if (id < 0 || id >= tokens.size()) {
      return absl::InvalidArgumentError(
          "LLGuidance special token is outside vocabulary.");
    }
    special[id] = true;
  }

  std::string tokenizer_json;
  if (vocabulary.tokenizer_json.has_value()) {
    auto json = nlohmann::json::parse(*vocabulary.tokenizer_json,
                                     /*cb=*/nullptr, /*allow_exceptions=*/false);
    if (!json.is_object()) {
      return absl::InvalidArgumentError("Invalid HF constraint tokenizer JSON.");
    }
    auto& added = json["added_tokens"];
    if (added.is_null()) added = nlohmann::json::array();
    if (!added.is_array()) {
      return absl::InvalidArgumentError("HF added_tokens must be an array.");
    }
    for (int id = 0; id < special.size(); ++id) {
      if (!special[id]) continue;
      bool found = false;
      for (auto& token : added) {
        if (token.is_object() && token.contains("id") && token["id"] == id) {
          token["special"] = true;
          found = true;
          break;
        }
      }
      if (!found) {
        added.push_back({{"id", id}, {"content", tokens[id]}, {"special", true}});
      }
    }
    tokenizer_json = json.dump();
  } else {
    for (int id = 0; id < tokens.size(); ++id) {
      if (special[id] && (tokens[id].empty() ||
                         static_cast<uint8_t>(tokens[id].front()) != 0xff)) {
        // llguidance's native marker distinguishes control IDs from text.
        tokens[id].insert(tokens[id].begin(), static_cast<char>(0xff));
      }
    }
  }

  std::vector<uint32_t> token_lens;
  std::vector<uint8_t> token_bytes;
  size_t total_size = 0;
  token_lens.reserve(tokens.size());
  for (const auto& token : tokens) {
    token_lens.push_back(token.size());
    total_size += token.size();
  }
  token_bytes.reserve(total_size);
  for (const auto& token : tokens) {
    token_bytes.insert(token_bytes.end(), token.begin(), token.end());
  }

  auto context = std::make_shared<const TokenizationContext>(TokenizationContext{
      const_cast<Tokenizer&>(tokenizer), vocabulary.special_token_ids});
  auto tokenize_fn = [](const void* user_data, const uint8_t* bytes,
                        size_t bytes_len, uint32_t* output_tokens,
                        size_t output_tokens_len) -> size_t {
    absl::string_view text(reinterpret_cast<const char*>(bytes), bytes_len);

    const auto& context = *static_cast<const TokenizationContext*>(user_data);
    auto token_ids = context.tokenizer.BytesToTokenIdsForConstraint(
        text, context.special_token_ids);
    if (!token_ids.ok()) {
      return 0;
    }
    if (output_tokens_len > 0) {
      memcpy(output_tokens, token_ids->data(),
             std::min(output_tokens_len, token_ids->size()) * sizeof(uint32_t));
    }
    return token_ids->size();
  };

  LlgTokenizerInit tok_init = {
      .vocab_size = static_cast<uint32_t>(tokens.size()),
      .tok_eos = *llg_config.eos_id,
      .token_lens = token_lens.data(),
      .token_bytes = token_bytes.data(),
      .tokenizer_json = tokenizer_json.empty() ? nullptr : tokenizer_json.c_str(),
      .tokenize_assumes_string = true,
      .tokenize_fn = tokenize_fn,
      .tokenize_user_data = context.get(),
  };

  char error_buf[128];
  LlgTokenizer* llg_tokenizer =
      llg_new_tokenizer(&tok_init, error_buf, sizeof(error_buf));
  if (llg_tokenizer == nullptr) {
    return absl::InternalError(error_buf);
  }

  return std::make_unique<LlgConstraintProvider>(
      std::move(token_lens), std::move(token_bytes), llg_tokenizer, llg_config,
      std::move(context));
}

LlgConstraintProvider::~LlgConstraintProvider() {
  llg_free_tokenizer(llg_tokenizer_);
}

absl::StatusOr<std::unique_ptr<Constraint>>
LlgConstraintProvider::CreateConstraint(ConstraintArg constraint_arg) const {
  if (!std::holds_alternative<LlGuidanceConstraintArg>(constraint_arg)) {
    return absl::InvalidArgumentError(
        "LlgConstraintProvider only supports LlGuidanceConstraintArg.");
  }
  const auto& llg_arg = std::get<LlGuidanceConstraintArg>(constraint_arg);

  LlgConstraintInit llg_constraint_init;
  llg_constraint_init_set_defaults(&llg_constraint_init, llg_tokenizer_);
  ::LlgConstraint* llg_constraint = CreateLlgConstraint(
      &llg_constraint_init, llg_arg.constraint_string, llg_arg.constraint_type);

  if (llg_get_error(llg_constraint)) {
    std::string error_message = llg_get_error(llg_constraint);
    llg_free_constraint(llg_constraint);
    return absl::InternalError(absl::StrCat(
        "Failed to create LLGuidance constraint: ", error_message));
  }

  return std::make_unique<LlgConstraint>(llg_constraint,
                                         static_cast<int>(token_lens_.size()),
                                         *llg_config_.eos_id,
                                         tokenization_context_);
}

}  // namespace litert::lm
