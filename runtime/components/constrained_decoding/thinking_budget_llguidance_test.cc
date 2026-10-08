// Copyright 2026 The ODML Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

#include "runtime/components/constrained_decoding/thinking_budget_constraint.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_provider.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/util/test_utils.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

constexpr int kVocab = 11;
constexpr int kEOS = 1;
constexpr int kA = 2;
constexpr int kB = 3;
constexpr int kStart = 4;
constexpr int kStartTail = 5;
constexpr int kThought = 6;
constexpr int kEnd = 7;
constexpr int kEndTail = 8;

// Identity vocabulary only; grammar parsing/masking/commit use real LLGuidance.
class IdentityTokenizer final : public ::litert::support::Tokenizer {
 public:
  ::litert::support::TokenizerType GetTokenizerType() const override {
    return ::litert::support::TokenizerType::kSentencePiece;
  }
  absl::StatusOr<::litert::support::TokenIds> TextToTokenIds(
      absl::string_view text) override {
    ::litert::support::TokenIds ids;
    for (char c : text) {
      if (c == 'a') ids.push_back(kA);
      else if (c == 'b') ids.push_back(kB);
      else if (c == 'x') ids.push_back(kThought);
      else if (c == '\n') ids.push_back(9);
      else if (c == '"') ids.push_back(10);
      else return absl::InvalidArgumentError("Unsupported identity text.");
    }
    return ids;
  }
  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    const auto tokens = GetTokens();
    for (int i = 0; i < tokens.size(); ++i) {
      if (token == tokens[i]) return i;
    }
    return absl::NotFoundError("Unknown identity token.");
  }
  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> ids, bool skip_special_tokens) override {
    std::string text;
    const auto tokens = GetTokens();
    for (int id : ids) {
      if (id < 0 || id >= tokens.size()) return absl::InvalidArgumentError("Bad ID.");
      if (skip_special_tokens && (id == 0 || id == kEOS || id == kStart ||
          id == kStartTail || id == kEnd || id == kEndTail)) continue;
      text += tokens[id];
    }
    return text;
  }
  std::vector<std::string> GetTokens() const override {
    return {"<pad>", "<eos>", "a", "b", "<think>", "<thought>",
            "x", "</think>", "<end>", "\n", "\""};
  }
  int GetVocabSize() const override { return kVocab; }
};

class ThinkingLlgTest : public ::testing::Test {
 protected:
  absl::StatusOr<std::unique_ptr<Constraint>> Grammar() {
    LlGuidanceConfig config;
    config.eos_id = kEOS;
    config.special_token_ids = {0, kStart, kStartTail, kEnd, kEndTail};
    auto provider = LlgConstraintProvider::Create(tokenizer_, config);
    if (!provider.ok()) return provider.status();
    return (*provider)->CreateConstraint(LlGuidanceConstraintArg{
        .constraint_type = LlgConstraintType::kRegex, .constraint_string = "ab"});
  }
  IdentityTokenizer tokenizer_;
};

TEST_F(ThinkingLlgTest, InitialChoiceIsContentOrThinkingNotUnrelatedText) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  ThinkingBudgetConstraint c(grammar.get(), 32, {kStart, kStartTail},
                             {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK_AND_ASSIGN(auto mask, c.ComputeMask(*state));
  ASSERT_EQ(mask->GetType(), MaskType::kBitmap);
  const auto& bitmap = static_cast<const BitmapLogitMask&>(*mask);
  EXPECT_TRUE(bitmap.IsAllowed(kA));
  EXPECT_TRUE(bitmap.IsAllowed(kStart));
  EXPECT_FALSE(bitmap.IsAllowed(kB));
  EXPECT_FALSE(bitmap.IsAllowed(kThought));
  EXPECT_FALSE(bitmap.IsAllowed(kEOS));
  std::vector<float> logits(kVocab, 3.0f);
  ASSERT_OK(mask->Apply(absl::MakeSpan(logits)));
  EXPECT_EQ(logits[kA], 3.0f);
  EXPECT_EQ(logits[kStart], 3.0f);
  EXPECT_TRUE(std::isinf(logits[kThought]) && logits[kThought] < 0);
}

TEST_F(ThinkingLlgTest, SkippedThinkingCommitsFirstRealGrammarTokenExactlyOnce) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  ThinkingBudgetConstraint c(grammar.get(), 32, {kStart, kStartTail},
                             {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK(c.ComputeMask(*state));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kA));
  ASSERT_OK_AND_ASSIGN(auto mask, c.ComputeMask(*state));
  const auto& bitmap = static_cast<const BitmapLogitMask&>(*mask);
  EXPECT_TRUE(bitmap.IsAllowed(kB));
  EXPECT_FALSE(bitmap.IsAllowed(kA));
  EXPECT_FALSE(bitmap.IsAllowed(kStart));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kB));
  ASSERT_OK_AND_ASSIGN(mask, c.ComputeMask(*state));
  EXPECT_TRUE(static_cast<const BitmapLogitMask&>(*mask).IsAllowed(kEOS));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kEOS));
  EXPECT_TRUE(c.IsEnded(*state));
}

TEST_F(ThinkingLlgTest, DirectSkipInitializesMaskBeforeRealGrammarCommit) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  ThinkingBudgetConstraint c(grammar.get(), 32, {kStart, kStartTail},
                             {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kA));
  ASSERT_OK_AND_ASSIGN(auto bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kB));
  EXPECT_FALSE(bitmap->Get(kA));
}

TEST_F(ThinkingLlgTest, InvalidSkipRefusesBeforeCommitAndKeepsInitialGrammar) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  ThinkingBudgetConstraint c(grammar.get(), 32, {kStart, kStartTail},
                             {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  auto invalid = c.ComputeNext(*state, kThought);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(invalid.status().message(), "Skipped-thinking token violates the user mask.");
  ASSERT_OK_AND_ASSIGN(auto bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kA));
  EXPECT_TRUE(bitmap->Get(kStart));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kA));
  ASSERT_OK_AND_ASSIGN(bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kB));
}

TEST_F(ThinkingLlgTest, PartialStartMustFinishBeforeThinkingOrContent) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  ThinkingBudgetConstraint c(grammar.get(), 32, {kStart, kStartTail},
                             {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK(c.ComputeMask(*state));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kStart));
  ASSERT_OK_AND_ASSIGN(auto bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kStartTail));
  EXPECT_FALSE(bitmap->Get(kA));
  EXPECT_FALSE(bitmap->Get(kEOS));
  auto mismatch = c.ComputeNext(*state, kA);
  ASSERT_FALSE(mismatch.ok());
  EXPECT_EQ(mismatch.status().code(), absl::StatusCode::kInvalidArgument);
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kStartTail));
  ASSERT_OK_AND_ASSIGN(bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kThought));
  EXPECT_TRUE(bitmap->Get(kEOS));  // The outer runtime retains EOS stop ownership.
}

TEST_F(ThinkingLlgTest, NaturalEndRemainsOutsideTheRealContentGrammar) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  ThinkingBudgetConstraint c(grammar.get(), 32, {kStart, kStartTail},
                             {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK(c.ComputeMask(*state));
  for (int token : {kStart, kStartTail, kThought, kEnd, kEndTail}) {
    ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, token));
  }
  ASSERT_OK_AND_ASSIGN(auto mask, c.ComputeMask(*state));
  const auto& bitmap = static_cast<const BitmapLogitMask&>(*mask);
  EXPECT_TRUE(bitmap.IsAllowed(kA));
  EXPECT_FALSE(bitmap.IsAllowed(kEnd));
  EXPECT_FALSE(bitmap.IsAllowed(kB));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kA));
  ASSERT_OK(c.ComputeMask(*state));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kB));
  ASSERT_OK(c.ComputeMask(*state));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kEOS));
  EXPECT_TRUE(c.IsEnded(*state));
}

TEST_F(ThinkingLlgTest, ForcedEndRejectsMismatchAndThenOpensTheContentGrammar) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  ThinkingBudgetConstraint c(grammar.get(), 1, {kStart, kStartTail},
                             {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  for (int token : {kStart, kStartTail, kThought}) {
    ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, token));
  }
  ASSERT_OK_AND_ASSIGN(auto mask, c.ComputeMask(*state));
  EXPECT_TRUE(static_cast<const BitmapLogitMask&>(*mask).IsAllowed(kEnd));
  EXPECT_FALSE(static_cast<const BitmapLogitMask&>(*mask).IsAllowed(kA));
  auto mismatch = c.ComputeNext(*state, kA);
  ASSERT_FALSE(mismatch.ok());
  EXPECT_EQ(mismatch.status().message(), "Token violates forced thinking end delimiter.");
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kEnd));
  ASSERT_OK_AND_ASSIGN(auto bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kEndTail));
  EXPECT_FALSE(bitmap->Get(kEnd));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kEndTail));
  ASSERT_OK_AND_ASSIGN(mask, c.ComputeMask(*state));
  EXPECT_TRUE(static_cast<const BitmapLogitMask&>(*mask).IsAllowed(kA));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kA));
}

TEST_F(ThinkingLlgTest, PrefilledThinkingWaitsForEndBeforeGrammarContent) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  ThinkingBudgetConstraint c(grammar.get(), 32, {}, {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK_AND_ASSIGN(auto bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kThought));
  EXPECT_TRUE(bitmap->Get(kEOS));
  for (int token : {kThought, kEnd, kEndTail}) {
    ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, token));
  }
  ASSERT_OK_AND_ASSIGN(bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kA));
  EXPECT_FALSE(bitmap->Get(kThought));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kA));
}

TEST_F(ThinkingLlgTest, UnlimitedThinkingStillRequiresNaturalEndBeforeContent) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  ThinkingBudgetConstraint c(grammar.get(), -1, {kStart, kStartTail},
                             {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  for (int token : {kStart, kStartTail}) {
    ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, token));
  }
  for (int i = 0; i < 40; ++i) {
    ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kThought));
  }
  ASSERT_OK_AND_ASSIGN(auto bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kThought));
  EXPECT_TRUE(bitmap->Get(kA));
  for (int token : {kEnd, kEndTail}) {
    ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, token));
  }
  ASSERT_OK_AND_ASSIGN(bitmap, c.ComputeBitmap(*state));
  EXPECT_FALSE(bitmap->Get(kThought));
  EXPECT_TRUE(bitmap->Get(kA));
}

TEST_F(ThinkingLlgTest, BitmapChoiceMatchesTheActualLogitMask) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  ThinkingBudgetConstraint c(grammar.get(), 32, {kStart, kStartTail},
                             {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK_AND_ASSIGN(auto bitmap, c.ComputeBitmap(*state));
  ASSERT_OK_AND_ASSIGN(auto mask, c.ComputeMask(*state));
  const auto& bits = static_cast<const BitmapLogitMask&>(*mask);
  for (int i = 0; i < kVocab; ++i) EXPECT_EQ(bitmap->Get(i), bits.IsAllowed(i));
}

TEST_F(ThinkingLlgTest, UnwrappedNoThinkingGrammarStillHasTheOriginalExactMask) {
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar());
  auto state = grammar->Start();
  ASSERT_OK_AND_ASSIGN(auto bitmap, grammar->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kA));
  EXPECT_FALSE(bitmap->Get(kStart));
  EXPECT_FALSE(bitmap->Get(kThought));
  ASSERT_OK_AND_ASSIGN(state, grammar->ComputeNext(*state, kA));
  ASSERT_OK_AND_ASSIGN(bitmap, grammar->ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kB));
}

// Non-bitmap masks retain their actual Apply operation. They also make mask
// initialization before commit observable, without pretending to be a grammar.
class CustomMask final : public LogitMask {
 public:
  absl::Status Apply(absl::Span<float> logits) const override {
    for (int i = 0; i < logits.size(); ++i) {
      if (i == kA) logits[i] += 2;
      else logits[i] = -std::numeric_limits<float>::infinity();
    }
    return absl::OkStatus();
  }
  absl::Status Apply(absl::Span<tflite::half> logits) const override {
    for (int i = 0; i < logits.size(); ++i) {
      if (i == kA) logits[i] = tflite::half(static_cast<float>(logits[i]) + 2);
      else logits[i] = std::numeric_limits<tflite::half>::lowest();
    }
    return absl::OkStatus();
  }
};

class MaskOnlyConstraint final : public Constraint {
 public:
  struct UserState final : State { mutable bool mask_computed = false; };
  bool null_mask = false;
  absl::Status mask_failure;
  std::unique_ptr<State> Start() const override { return std::make_unique<UserState>(); }
  bool IsEnded(const State&) const override { return false; }
  int GetVocabularySize() const override { return kVocab; }
  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(const State& state) const override {
    if (!mask_failure.ok()) return mask_failure;
    static_cast<const UserState&>(state).mask_computed = true;
    if (null_mask) return std::unique_ptr<LogitMask>{};
    return std::make_unique<CustomMask>();
  }
  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state, int token) const override {
    if (!static_cast<const UserState&>(state).mask_computed) {
      return absl::FailedPreconditionError("Mask was never computed.");
    }
    if (!null_mask && token != kA) {
      return absl::InvalidArgumentError("Custom user token is disallowed.");
    }
    return std::make_unique<UserState>();
  }
};

TEST(ThinkingGenericMaskTest, CustomChoicePreservesFloatAndHalfAndRejectsBadSkip) {
  MaskOnlyConstraint user;
  ThinkingBudgetConstraint c(&user, 32, {kStart, kStartTail}, {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK_AND_ASSIGN(auto mask, c.ComputeMask(*state));
  EXPECT_EQ(mask->GetType(), MaskType::kCustom);
  std::vector<float> logits(kVocab, 3);
  logits[kStart] = 11;
  ASSERT_OK(mask->Apply(absl::MakeSpan(logits)));
  EXPECT_EQ(logits[kA], 5);
  EXPECT_EQ(logits[kStart], 11);
  EXPECT_TRUE(std::isinf(logits[kThought]) && logits[kThought] < 0);
  std::vector<tflite::half> half_logits(kVocab, tflite::half(3.0f));
  half_logits[kStart] = tflite::half(11.0f);
  ASSERT_OK(mask->Apply(absl::MakeSpan(half_logits)));
  EXPECT_EQ(static_cast<float>(half_logits[kA]), 5);
  EXPECT_EQ(static_cast<float>(half_logits[kStart]), 11);
  EXPECT_EQ(half_logits[kThought], std::numeric_limits<tflite::half>::lowest());
  auto bitmap = c.ComputeBitmap(*state);
  ASSERT_FALSE(bitmap.ok());  // A custom-only API cannot be silently made AllAllowed.
  EXPECT_EQ(bitmap.status().code(), absl::StatusCode::kUnimplemented);
  auto invalid = c.ComputeNext(*state, kThought);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kA));
}

TEST(ThinkingGenericMaskTest, OtherConstraintsStillSuppressTheThinkingStart) {
  MaskOnlyConstraint user;
  ThinkingBudgetConstraint c(&user, 32, {kStart, kStartTail}, {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK_AND_ASSIGN(auto choice, c.ComputeMask(*state));
  CompositeLogitMask composite;
  const std::vector<int> suppressed = {kStart};
  composite.AddMask(BitmapLogitMask::CreateFromDisallowedTokens(kVocab, suppressed));
  composite.AddMask(std::move(choice));
  std::vector<float> logits(kVocab, 3);
  ASSERT_OK(composite.Apply(absl::MakeSpan(logits)));
  EXPECT_EQ(logits[kA], 5);
  EXPECT_TRUE(std::isinf(logits[kStart]) && logits[kStart] < 0);
}

TEST(ThinkingGenericMaskTest, NullUserRetainsUnconstrainedSkipAndBudgetedThinking) {
  ThinkingBudgetConstraint c(nullptr, 1, {kStart, kStartTail}, {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK_AND_ASSIGN(auto bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kThought));
  EXPECT_TRUE(bitmap->Get(kEOS));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kThought));
  EXPECT_FALSE(c.IsEnded(*state));
  state = c.Start();
  for (int token : {kStart, kStartTail, kThought}) {
    ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, token));
  }
  ASSERT_OK_AND_ASSIGN(bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kEnd));
  EXPECT_FALSE(bitmap->Get(kThought));
  for (int token : {kEnd, kEndTail}) {
    ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, token));
  }
  ASSERT_OK_AND_ASSIGN(bitmap, c.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(kThought));
}

TEST(ThinkingGenericMaskTest, NullUserMaskStillInitializesBeforeUnconstrainedSkip) {
  MaskOnlyConstraint user;
  user.null_mask = true;
  ThinkingBudgetConstraint c(&user, 32, {kStart, kStartTail}, {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  ASSERT_OK_AND_ASSIGN(auto mask, c.ComputeMask(*state));
  EXPECT_TRUE(static_cast<const BitmapLogitMask&>(*mask).IsAllowed(kThought));
  ASSERT_OK_AND_ASSIGN(state, c.ComputeNext(*state, kThought));
}

TEST(ThinkingGenericMaskTest, UserMaskRefusalKeepsExactOriginalStatus) {
  MaskOnlyConstraint user;
  user.mask_failure = absl::FailedPreconditionError("Grammar owner refused.");
  ThinkingBudgetConstraint c(&user, 32, {kStart, kStartTail}, {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  auto mask = c.ComputeMask(*state);
  ASSERT_FALSE(mask.ok());
  EXPECT_EQ(mask.status(), user.mask_failure);
  auto next = c.ComputeNext(*state, kA);
  ASSERT_FALSE(next.ok());
  EXPECT_EQ(next.status(), user.mask_failure);
}

TEST(ThinkingGenericMaskTest, OutOfRangeTokensAndMalformedDelimitersRefuse) {
  ThinkingBudgetConstraint c(nullptr, 32, {kStart, kStartTail}, {kEnd, kEndTail}, kVocab);
  auto state = c.Start();
  for (int token : {-1, kVocab}) {
    auto next = c.ComputeNext(*state, token);
    ASSERT_FALSE(next.ok());
    EXPECT_EQ(next.status().code(), absl::StatusCode::kInvalidArgument);
  }
  ThinkingBudgetConstraint bad_start(nullptr, 32, {kVocab}, {kEnd}, kVocab);
  EXPECT_FALSE(bad_start.ComputeMask(*bad_start.Start()).ok());
  ThinkingBudgetConstraint bad_end(nullptr, 32, {kStart}, {}, kVocab);
  EXPECT_FALSE(bad_end.ComputeBitmap(*bad_end.Start()).ok());
  EXPECT_FALSE(bad_end.ComputeNext(*bad_end.Start(), kStart).ok());
}

}  // namespace
}  // namespace litert::lm
