// Copyright 2026 GuideAI contributors.
// Licensed under the Apache License, Version 2.0.

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider_factory.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_provider.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/util/test_utils.h"
#include "support/tokenizer/sentencepiece_tokenizer.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

using ::litert::support::ConstraintVocabulary;
using ::litert::support::SentencePieceTokenizer;
using ::litert::support::TokenIds;
using ::testing::Return;

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
};

std::string GemmaTokenizerPath() {
  return (std::filesystem::path(::testing::SrcDir()) /
          "litert_lm/support/tokenizer/testdata/gemma3_sentencepiece.model")
      .string();
}

void ConsumeAndCheckEOS(const Constraint& constraint,
                       absl::Span<const int> tokens, int eos) {
  auto state = constraint.Start();
  for (int token : tokens) {
    ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
    ASSERT_EQ(mask->GetType(), MaskType::kBitmap);
    ASSERT_TRUE(static_cast<const BitmapLogitMask&>(*mask).IsAllowed(token))
        << "Rejected token " << token;
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, token));
  }
  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
  EXPECT_TRUE(static_cast<const BitmapLogitMask&>(*mask).IsAllowed(eos));
}

TEST(ConstraintVocabularyTest, SentencePieceKeepsSpacesUnicodeAndLiteralAngles) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                      SentencePieceTokenizer::CreateFromFile(GemmaTokenizerPath()));
  ASSERT_OK_AND_ASSIGN(auto provider, LlgConstraintProvider::Create(
      *tokenizer, LlGuidanceConfig{.eos_id = 1}));
  const std::vector<std::pair<std::string, std::string>> cases = {
      {R"("making every day outdoor")",
       R"({"type":"string","enum":["making every day outdoor"]})"},
      {R"("a  b")", R"({"type":"string","enum":["a  b"]})"},
      {R"("literal <angle> brackets")",
       R"({"type":"string","enum":["literal <angle> brackets"]})"},
      {R"("café 東京 🧭")", R"({"type":"string","enum":["café 東京 🧭"]})"},
  };
  for (const auto& [text, schema] : cases) {
    SCOPED_TRACE(text);
    ASSERT_OK_AND_ASSIGN(auto constraint, provider->CreateConstraint(
        LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kJsonSchema,
                               .constraint_string = schema}));
    ASSERT_OK_AND_ASSIGN(auto tokens, tokenizer->TextToTokenIds(text));
    ConsumeAndCheckEOS(*constraint, tokens, 1);
  }
}

void SampleAndCheckNativeText(const Constraint& constraint, Tokenizer& tokenizer,
                              const std::string& expected, int eos) {
  auto state = constraint.Start();
  std::vector<int> tokens;
  bool completed = false;
  for (int step = 0; step < 128; ++step) {
    ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
    ASSERT_EQ(mask->GetType(), MaskType::kBitmap);
    const auto& bitmap = static_cast<const BitmapLogitMask&>(*mask);
    if (bitmap.IsAllowed(eos)) {
      completed = true;
      break;
    }
    int selected = -1;
    for (int id = 0; id < constraint.GetVocabularySize(); ++id) {
      if (bitmap.IsAllowed(id)) {
        selected = id;
        break;
      }
    }
    ASSERT_GE(selected, 0);
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, selected));
    tokens.push_back(selected);
  }
  EXPECT_TRUE(completed);
  ASSERT_OK_AND_ASSIGN(auto output, tokenizer.TokenIdsToText(tokens));
  EXPECT_EQ(output, expected);
}

TEST(ConstraintVocabularyTest, ByteFallbackPreservesLiteralSpaceSymbol) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                      SentencePieceTokenizer::CreateFromFile(GemmaTokenizerPath()));
  ASSERT_OK_AND_ASSIGN(auto provider, LlgConstraintProvider::Create(
      *tokenizer, LlGuidanceConfig{.eos_id = 1}));
  ASSERT_OK_AND_ASSIGN(auto constraint, provider->CreateConstraint(
      LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kJsonSchema,
                             .constraint_string = R"({"type":"string","enum":["▁🧭"]})"}));
  SampleAndCheckNativeText(*constraint, *tokenizer, "\"▁🧭\"", 1);
}

TEST(ConstraintVocabularyTest, LiteralStopSpellingsUseTextInsteadOfControlIDs) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                      SentencePieceTokenizer::CreateFromFile(GemmaTokenizerPath()));
  ASSERT_OK_AND_ASSIGN(auto provider, CreateConstraintProvider(
      LlGuidanceConfig{.eos_id = 1}, *tokenizer, {{1}, {50}, {106}}));
  for (const auto& value : {"<end_of_turn>", "<unused44>",
                           "literal <end_of_turn> marker", "café 東京 🧭"}) {
    SCOPED_TRACE(value);
    const std::string quoted = "\"" + std::string(value) + "\"";
    ASSERT_OK_AND_ASSIGN(auto constraint, provider->CreateConstraint(
        LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kJsonSchema,
            .constraint_string = "{\"type\":\"string\",\"enum\":[" + quoted + "]}"}));
    SampleAndCheckNativeText(*constraint, *tokenizer, quoted, 1);
  }
}

TEST(ConstraintVocabularyTest, ConstraintRetainsCallbackAfterProviderDestruction) {
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                      SentencePieceTokenizer::CreateFromFile(GemmaTokenizerPath()));
  ASSERT_OK_AND_ASSIGN(auto provider, CreateConstraintProvider(
      LlGuidanceConfig{.eos_id = 1}, *tokenizer, {{1}, {106}}));
  ASSERT_OK_AND_ASSIGN(auto constraint, provider->CreateConstraint(
      LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kJsonSchema,
          .constraint_string = R"({"type":"string","enum":["<end_of_turn>"]})"}));
  provider.reset();
  SampleAndCheckNativeText(*constraint, *tokenizer, "\"<end_of_turn>\"", 1);
}

TEST(ConstraintVocabularyTest, EverySingleTokenStopIsBlockedInOpenJSON) {
  MockTokenizer tokenizer;
  ConstraintVocabulary vocabulary{
      .token_bytes = {"<pad>", "<eos>", "<bos>", "<tool_response>", "\"", "a", "<turn>"},
      .special_token_ids = {0, 2}};
  EXPECT_CALL(tokenizer, GetConstraintVocabulary()).WillOnce(Return(vocabulary));
  EXPECT_CALL(tokenizer, TextToTokenIds(::testing::_)).WillRepeatedly(
      [](absl::string_view text) -> absl::StatusOr<TokenIds> {
        TokenIds ids;
        for (char c : text) {
          if (c == '"') ids.push_back(4);
          else if (c == 'a') ids.push_back(5);
        }
        return ids;
      });
  ASSERT_OK_AND_ASSIGN(auto provider, CreateConstraintProvider(
      LlGuidanceConfig{.eos_id = 1}, tokenizer, {{1}, {3}, {6}, {5, 5}}));
  ASSERT_OK_AND_ASSIGN(auto constraint, provider->CreateConstraint(
      LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kJsonSchema,
                             .constraint_string = R"({"type":"string"})"}));
  auto state = constraint->Start();
  ASSERT_OK(constraint->ComputeMask(*state));
  ASSERT_OK_AND_ASSIGN(state, constraint->ComputeNext(*state, 4));
  ASSERT_OK_AND_ASSIGN(auto mask, constraint->ComputeMask(*state));
  const auto& bitmap = static_cast<const BitmapLogitMask&>(*mask);
  for (int id : {0, 1, 2, 3, 6}) EXPECT_FALSE(bitmap.IsAllowed(id)) << id;
  EXPECT_TRUE(bitmap.IsAllowed(5));  // Multi-token stop components remain text.
}

TEST(ConstraintVocabularyTest, SpecialTokenStillWorksInExplicitToolGrammar) {
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, GetConstraintVocabulary()).WillOnce(Return(
      ConstraintVocabulary{.token_bytes = {"<pad>", "<eos>", "<call>", "a"},
                           .special_token_ids = {0, 2}}));
  EXPECT_CALL(tokenizer, TextToTokenIds(::testing::_)).WillRepeatedly(
      [](absl::string_view text) -> absl::StatusOr<TokenIds> {
        return text == "a" ? TokenIds{3} : TokenIds{};
      });
  ASSERT_OK_AND_ASSIGN(auto provider, LlgConstraintProvider::Create(
      tokenizer, LlGuidanceConfig{.eos_id = 1}));
  ASSERT_OK_AND_ASSIGN(auto constraint, provider->CreateConstraint(
      LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kLark,
                             .constraint_string = "start: <call> \"a\""}));
  ConsumeAndCheckEOS(*constraint, std::vector<int>{2, 3}, 1);
}

TEST(ConstraintVocabularyTest, HuggingFaceUsesByteLevelDecoderMetadata) {
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, GetConstraintVocabulary()).WillOnce(Return(
      ConstraintVocabulary{.token_bytes = {"<pad>", "<eos>", "\"", "a", "Ġ"},
        .tokenizer_json = R"({"decoder":{"type":"ByteLevel"},"added_tokens":[
          {"id":0,"content":"<pad>","special":true}],
          "model":{"vocab":{"<pad>":0,"<eos>":1,"\"":2,"a":3,"Ġ":4}}})"}));
  EXPECT_CALL(tokenizer, TextToTokenIds(::testing::_)).WillRepeatedly(
      [](absl::string_view text) -> absl::StatusOr<TokenIds> {
        TokenIds ids;
        for (char c : text) {
          if (c == '"') ids.push_back(2);
          else if (c == 'a') ids.push_back(3);
          else if (c == ' ') ids.push_back(4);
        }
        return ids;
      });
  ASSERT_OK_AND_ASSIGN(auto provider, LlgConstraintProvider::Create(
      tokenizer, LlGuidanceConfig{.eos_id = 1}));
  ASSERT_OK_AND_ASSIGN(auto constraint, provider->CreateConstraint(
      LlGuidanceConstraintArg{.constraint_type = LlgConstraintType::kJsonSchema,
                             .constraint_string = R"({"type":"string","enum":["a a"]})"}));
  ConsumeAndCheckEOS(*constraint, std::vector<int>{2, 3, 4, 3, 2}, 1);
}

TEST(ConstraintVocabularyTest, RejectsInvalidControlIDsBeforeNativeAllocation) {
  for (int id : {-1, 4}) {
    MockTokenizer tokenizer;
    EXPECT_CALL(tokenizer, GetConstraintVocabulary()).WillOnce(Return(
        ConstraintVocabulary{.token_bytes = {"<pad>", "<eos>", "a", "b"}}));
    EXPECT_FALSE(LlgConstraintProvider::Create(
        tokenizer, LlGuidanceConfig{.eos_id = 1, .special_token_ids = {id}}).ok());
  }
}

}  // namespace
}  // namespace litert::lm
