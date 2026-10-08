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

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "litert/cc/litert_layout.h"
#include "runtime/components/constrained_decoding/composite_constraint.h"
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/gemma_model_constraint_provider.h"
#include "runtime/util/test_utils.h"
#include "support/tokenizer/sentencepiece_tokenizer.h"
#include "tflite/types/half.h"

namespace litert::lm {
namespace {

// The prebuilt provider returns a C++ Constraint across an opaque C boundary.
// Creating it alone cannot establish compatibility: execute the virtual mask
// and state methods through the same composite decoder used by inference.
class GemmaToolConstraintAbiTest
    : public ::testing::TestWithParam<LiteRtLmGemmaFuncallFormat> {
 protected:
  template <typename Scalar>
  void ExerciseDecoder() {
    const bool fc = GetParam() == kLiteRtLmGemmaFuncallFormatFcStyle;
    const auto path = std::filesystem::path(::testing::SrcDir()) /
                      "litert_lm/runtime/components/testdata" /
                      (fc ? "function_gemma_sentencepiece.model"
                          : "gemma3_sentencepiece.model");
    ASSERT_OK_AND_ASSIGN(
        auto tokenizer,
        ::litert::support::SentencePieceTokenizer::CreateFromFile(path.string()));
    const auto model = tokenizer->GetProcessor().model_proto().SerializeAsString();
    const int eos[] = {1};
    const int* stops[] = {eos};
    const size_t lengths[] = {1};
    std::unique_ptr<LiteRtLmGemmaModelConstraintProvider,
                    decltype(&LiteRtLmGemmaModelConstraintProvider_Destroy)>
        provider(LiteRtLmGemmaModelConstraintProvider_Create(
                     model.data(), model.size(), stops, lengths, 1),
                 &LiteRtLmGemmaModelConstraintProvider_Destroy);
    ASSERT_NE(provider, nullptr);

    const LiteRtLmGemmaModelConstraintOptions options = {
        .funcall_format = GetParam(),
        .constraint_mode = kLiteRtLmGemmaConstraintModeFunctionCallOnly,
        .code_fence_start = "<start_function_call>",
        .code_fence_end = "<end_function_call>",
        .open_quote = fc ? "<escape>" : "\"",
        .close_quote = fc ? "<escape>" : "\"",
        .function_response_start = "<start_function_response>",
    };
    constexpr char tools[] = R"([{"name":"read_facts","parameters":{"type":"object","properties":{"count":{"type":"integer"}},"required":["count"]}}])";
    auto* raw = LiteRtLmGemmaModelConstraintProvider_CreateConstraintFromTools(
        provider.get(), tools, &options);
    ASSERT_NE(raw, nullptr);
    // This is the production model processor's ownership boundary.
    std::unique_ptr<Constraint> constraint(reinterpret_cast<Constraint*>(raw));
    const int vocab = constraint->GetVocabularySize();
    ASSERT_EQ(vocab, tokenizer->GetVocabSize());
    ASSERT_GT(vocab, 0);
    ASSERT_OK_AND_ASSIGN(auto composite, CompositeConstraint::Create(vocab));
    ASSERT_OK(composite->AddConstraint(std::move(constraint)));
    ConstrainedDecoder decoder(composite.get(), 1);
    const std::vector<::litert::Layout::Dim> dimensions = {1, 1, vocab};
    std::vector<Scalar> logits(vocab);
    for (int position = 0; position < 2; ++position) {
      std::fill(logits.begin(), logits.end(), static_cast<Scalar>(1.0f));
      ASSERT_OK(decoder.ProcessLogits(absl::MakeSpan(logits), dimensions));
      const auto allowed = std::find_if(logits.begin(), logits.end(),
                                       [](Scalar value) {
                                         return static_cast<float>(value) == 1.0f;
                                       });
      ASSERT_NE(allowed, logits.end());
      EXPECT_NE(std::count(logits.begin(), logits.end(),
                           static_cast<Scalar>(1.0f)), vocab);
      std::vector<int> selected = {
          static_cast<int>(std::distance(logits.begin(), allowed))};
      ASSERT_OK(decoder.UpdateState(absl::MakeSpan(selected)));
    }
  }
};

TEST_P(GemmaToolConstraintAbiTest, MasksFloat32AndAdvancesState) {
  ASSERT_NO_FATAL_FAILURE(ExerciseDecoder<float>());
}

TEST_P(GemmaToolConstraintAbiTest, MasksFloat16AndAdvancesState) {
  ASSERT_NO_FATAL_FAILURE(ExerciseDecoder<tflite::half>());
}

INSTANTIATE_TEST_SUITE_P(
    NativeToolFormats, GemmaToolConstraintAbiTest,
    ::testing::Values(kLiteRtLmGemmaFuncallFormatPythonStyle,
                      kLiteRtLmGemmaFuncallFormatFcStyle));

}  // namespace
}  // namespace litert::lm
