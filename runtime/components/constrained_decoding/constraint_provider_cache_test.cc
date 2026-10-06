// Copyright 2026 GuideAI contributors.
// Licensed under the Apache License, Version 2.0.

#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/external_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/engine/engine.h"
#include "runtime/util/test_utils.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

using ::litert::support::ConstraintVocabulary;
using ::litert::support::TokenIds;
using ::litert::support::Tokenizer;
using ::testing::Return;
using ::testing::ReturnRef;

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(::litert::support::TokenizerType, GetTokenizerType, (),
              (const, override));
  MOCK_METHOD(absl::StatusOr<TokenIds>, TextToTokenIds, (absl::string_view),
              (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int>, bool), (override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(ConstraintVocabulary, GetConstraintVocabulary, (),
              (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));

  absl::StatusOr<TokenIds> BytesToTokenIdsForConstraint(
      absl::string_view bytes, absl::Span<const int> excluded) override {
    TokenIds result;
    for (char byte : bytes) {
      const int id = byte == 'a' ? 3 : byte == 'b' ? 4 : -1;
      if (id < 0) return absl::InvalidArgumentError("Unknown test byte");
      for (int reserved : excluded) {
        if (id == reserved) {
          return absl::InvalidArgumentError("Reserved test token");
        }
      }
      result.push_back(id);
    }
    return result;
  }
};

class MockEngine : public Engine {
 public:
  MOCK_METHOD(const EngineSettings&, GetEngineSettings, (), (const, override));
  MOCK_METHOD(const Tokenizer&, GetTokenizer, (), (const, override));
  MOCK_METHOD(absl::StatusOr<AudioExecutorProperties>, GetAudioExecutorProperties,
              (), (const, override));
  MOCK_METHOD(absl::StatusOr<VisionExecutorProperties>, GetVisionExecutorProperties,
              (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<SessionInterface>>, CreateSession,
              (const SessionConfig&), (override));

  using Engine::ClearConstraintProviderCache;
};

class ConstraintProviderCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EXPECT_CALL(engine_, GetTokenizer()).WillRepeatedly(ReturnRef(tokenizer_));
  }

  void ExpectVocabulary(int count) {
    EXPECT_CALL(tokenizer_, GetConstraintVocabulary())
        .Times(count)
        .WillRepeatedly(Return(vocabulary_));
  }

  absl::StatusOr<std::unique_ptr<Constraint>> Grammar(
      const ConstraintProvider& provider, std::string expression = "[ab]+") {
    return provider.CreateConstraint(LlGuidanceConstraintArg{
        .constraint_type = LlgConstraintType::kRegex,
        .constraint_string = std::move(expression)});
  }

  static bool Allowed(const LogitMask& mask, int id) {
    EXPECT_EQ(mask.GetType(), MaskType::kBitmap);
    return static_cast<const BitmapLogitMask&>(mask).IsAllowed(id);
  }

  const ConstraintVocabulary vocabulary_{
      .token_bytes = {"<pad>", "<eos>", "<end>", "a", "b"},
      .special_token_ids = {0, 1, 2}};
  testing::StrictMock<MockTokenizer> tokenizer_;
  testing::StrictMock<MockEngine> engine_;
};

TEST_F(ConstraintProviderCacheTest, RepeatedSessionsBuildTheVocabularyOnce) {
  ExpectVocabulary(1);
  ASSERT_OK_AND_ASSIGN(auto first,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  for (int i = 0; i < 64; ++i) {
    ASSERT_OK_AND_ASSIGN(auto next,
                        engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
    EXPECT_EQ(first.get(), next.get());
    ASSERT_OK_AND_ASSIGN(auto grammar, Grammar(*next));
    auto state = grammar->Start();
    ASSERT_OK_AND_ASSIGN(auto mask, grammar->ComputeMask(*state));
    EXPECT_TRUE(Allowed(*mask, 3));
    EXPECT_TRUE(Allowed(*mask, 4));
    EXPECT_FALSE(Allowed(*mask, 1));
  }
}

TEST_F(ConstraintProviderCacheTest, DifferentStopConfigurationsKeepTheirOwnEos) {
  ExpectVocabulary(2);
  ASSERT_OK_AND_ASSIGN(auto first,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  ASSERT_OK_AND_ASSIGN(auto second,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{2}}));
  EXPECT_NE(first.get(), second.get());
  for (int eos : {1, 2}) {
    const auto& provider = eos == 1 ? first : second;
    ASSERT_OK_AND_ASSIGN(auto grammar, Grammar(*provider));
    auto state = grammar->Start();
    ASSERT_OK_AND_ASSIGN(state, grammar->ComputeNext(*state, 3));
    ASSERT_OK_AND_ASSIGN(auto mask, grammar->ComputeMask(*state));
    EXPECT_TRUE(Allowed(*mask, eos));
    EXPECT_FALSE(Allowed(*mask, eos == 1 ? 2 : 1));
  }
}

TEST_F(ConstraintProviderCacheTest, ExplicitEosAndSpecialIdsInvalidateTheEntry) {
  ExpectVocabulary(3);
  ASSERT_OK_AND_ASSIGN(auto first, engine_.GetConstraintProvider(
      LlGuidanceConfig{.eos_id = 1}, {{1}, {2}}));
  ASSERT_OK_AND_ASSIGN(auto second, engine_.GetConstraintProvider(
      LlGuidanceConfig{.eos_id = 2}, {{1}, {2}}));
  ASSERT_OK_AND_ASSIGN(auto third, engine_.GetConstraintProvider(
      LlGuidanceConfig{.eos_id = 2, .special_token_ids = {4}}, {{1}, {2}}));
  EXPECT_NE(first.get(), second.get());
  EXPECT_NE(second.get(), third.get());
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar(*third));
  auto state = grammar->Start();
  ASSERT_OK_AND_ASSIGN(auto mask, grammar->ComputeMask(*state));
  EXPECT_TRUE(Allowed(*mask, 3));
  EXPECT_FALSE(Allowed(*mask, 4));
}

TEST_F(ConstraintProviderCacheTest, StopOrderPreservesDefaultEosSelection) {
  ExpectVocabulary(2);
  ASSERT_OK_AND_ASSIGN(auto first,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}, {2}}));
  ASSERT_OK_AND_ASSIGN(auto second,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{2}, {1}}));
  EXPECT_NE(first.get(), second.get());
  for (int eos : {1, 2}) {
    ASSERT_OK_AND_ASSIGN(auto grammar, Grammar(*(eos == 1 ? first : second)));
    auto state = grammar->Start();
    ASSERT_OK_AND_ASSIGN(state, grammar->ComputeNext(*state, 3));
    ASSERT_OK_AND_ASSIGN(auto mask, grammar->ComputeMask(*state));
    EXPECT_TRUE(Allowed(*mask, eos));
    EXPECT_FALSE(Allowed(*mask, eos == 1 ? 2 : 1));
  }
}

TEST_F(ConstraintProviderCacheTest, TokenizerIdentityAndEngineOwnershipAreDistinct) {
  ExpectVocabulary(2);
  ASSERT_OK_AND_ASSIGN(auto first,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  testing::StrictMock<MockEngine> other_engine;
  EXPECT_CALL(other_engine, GetTokenizer()).WillRepeatedly(ReturnRef(tokenizer_));
  ASSERT_OK_AND_ASSIGN(auto other,
                      other_engine.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  EXPECT_NE(first.get(), other.get());

  testing::StrictMock<MockTokenizer> other_tokenizer;
  EXPECT_CALL(other_tokenizer, GetConstraintVocabulary()).WillOnce(Return(vocabulary_));
  EXPECT_CALL(engine_, GetTokenizer()).WillRepeatedly(ReturnRef(other_tokenizer));
  ASSERT_OK_AND_ASSIGN(auto replacement,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  EXPECT_NE(first.get(), replacement.get());
  // Clear while the replacement's borrowed tokenizer is still alive.
  engine_.ClearConstraintProviderCache();
}

TEST_F(ConstraintProviderCacheTest, InitializationFailurePreservesTheValidEntry) {
  ExpectVocabulary(2);
  ASSERT_OK_AND_ASSIGN(auto valid,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  auto failed = engine_.GetConstraintProvider(LlGuidanceConfig{.eos_id = 99}, {{1}});
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kInvalidArgument);
  ASSERT_OK_AND_ASSIGN(auto retained,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  EXPECT_EQ(valid.get(), retained.get());
}

TEST_F(ConstraintProviderCacheTest, ExternalProvidersDoNotReadOrEvictTheVocabulary) {
  ExpectVocabulary(1);
  ASSERT_OK_AND_ASSIGN(auto cached,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  ASSERT_OK_AND_ASSIGN(auto first,
                      engine_.GetConstraintProvider(ExternalConstraintConfig(), {{1}}));
  ASSERT_OK_AND_ASSIGN(auto second,
                      engine_.GetConstraintProvider(ExternalConstraintConfig(), {{1}}));
  EXPECT_NE(first.get(), second.get());
  ASSERT_OK_AND_ASSIGN(auto retained,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  EXPECT_EQ(cached.get(), retained.get());
}

TEST_F(ConstraintProviderCacheTest, EvictedProviderAndItsParserRemainUsable) {
  ExpectVocabulary(2);
  ASSERT_OK_AND_ASSIGN(auto provider,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  std::weak_ptr<const ConstraintProvider> old = provider;
  ASSERT_OK_AND_ASSIGN(auto grammar, Grammar(*provider, "ab|ba"));
  auto initial = grammar->Start();
  ASSERT_OK_AND_ASSIGN(auto next, grammar->ComputeNext(*initial, 3));
  ASSERT_OK_AND_ASSIGN(auto replacement,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{2}}));
  EXPECT_FALSE(old.expired());
  provider.reset();
  EXPECT_TRUE(old.expired());
  ASSERT_OK_AND_ASSIGN(auto mask, grammar->ComputeMask(*next));
  EXPECT_TRUE(Allowed(*mask, 4));
  EXPECT_FALSE(Allowed(*mask, 3));
  ASSERT_OK_AND_ASSIGN(auto original_mask, grammar->ComputeMask(*initial));
  EXPECT_TRUE(Allowed(*original_mask, 3));
  EXPECT_TRUE(Allowed(*original_mask, 4));
}

TEST_F(ConstraintProviderCacheTest, ClearingReleasesTheEntryAndKeepsLiveGrammars) {
  ExpectVocabulary(2);
  ASSERT_OK_AND_ASSIGN(auto provider,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  ASSERT_OK_AND_ASSIGN(auto a, Grammar(*provider, "a+"));
  ASSERT_OK_AND_ASSIGN(auto b, Grammar(*provider, "b+"));
  auto a_state = a->Start();
  auto b_state = b->Start();
  engine_.ClearConstraintProviderCache();
  ASSERT_OK_AND_ASSIGN(auto fresh,
                      engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
  EXPECT_NE(provider.get(), fresh.get());
  provider.reset();
  ASSERT_OK_AND_ASSIGN(auto a_mask, a->ComputeMask(*a_state));
  ASSERT_OK_AND_ASSIGN(auto b_mask, b->ComputeMask(*b_state));
  EXPECT_TRUE(Allowed(*a_mask, 3));
  EXPECT_FALSE(Allowed(*a_mask, 4));
  EXPECT_TRUE(Allowed(*b_mask, 4));
  EXPECT_FALSE(Allowed(*b_mask, 3));
}

TEST_F(ConstraintProviderCacheTest, ConcurrentRequestsShareOnlyTheVocabulary) {
  ExpectVocabulary(1);
  std::vector<std::shared_ptr<const ConstraintProvider>> providers(8);
  std::vector<std::thread> threads;
  for (int i = 0; i < providers.size(); ++i) {
    threads.emplace_back([&, i] {
      ASSERT_OK_AND_ASSIGN(providers[i],
                          engine_.GetConstraintProvider(LlGuidanceConfig(), {{1}}));
      ASSERT_OK_AND_ASSIGN(auto grammar, Grammar(*providers[i], "ab|ba"));
      auto initial = grammar->Start();
      const int first = i % 2 == 0 ? 3 : 4;
      ASSERT_OK_AND_ASSIGN(auto next, grammar->ComputeNext(*initial, first));
      ASSERT_OK_AND_ASSIGN(auto mask, grammar->ComputeMask(*next));
      EXPECT_TRUE(Allowed(*mask, first == 3 ? 4 : 3));
      EXPECT_FALSE(Allowed(*mask, first));
      EXPECT_FALSE(Allowed(*mask, 1));
    });
  }
  for (auto& thread : threads) thread.join();
  for (const auto& provider : providers) {
    ASSERT_NE(provider, nullptr);
    EXPECT_EQ(provider.get(), providers.front().get());
  }
}

}  // namespace
}  // namespace litert::lm
