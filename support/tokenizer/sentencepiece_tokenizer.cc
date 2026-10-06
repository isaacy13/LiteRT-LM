// Copyright 2025 The ODML Authors.
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

#include "support/tokenizer/sentencepiece_tokenizer.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "sentencepiece_model.pb.h"  // from @sentencepiece
#include "model_interface.h"  // from @sentencepiece
#include "sentencepiece_processor.h"  // from @sentencepiece

namespace litert::support {
namespace {

bool IsConstraintControl(const sentencepiece::SentencePieceProcessor& processor,
                         int id) {
  return processor.IsControl(id) || processor.IsUnknown(id) ||
         processor.IsUnused(id);
}

std::string ConstraintTokenBytes(
    const sentencepiece::SentencePieceProcessor& processor, int id) {
  const std::string& piece = processor.IdToPiece(id);
  if (processor.IsByte(id)) {
    return std::string(1, static_cast<char>(sentencepiece::PieceToByte(piece)));
  }
  if (IsConstraintControl(processor, id)) return piece;
  return absl::StrReplaceAll(piece, {{"\xe2\x96\x81", " "}});
}

}  // namespace

SentencePieceTokenizer::SentencePieceTokenizer(
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor)
    : processor_(std::move(processor)), vocab_size_(processor_->GetPieceSize()) {
  byte_token_ids_.fill(-1);
  for (int id = 0; id < vocab_size_; ++id) {
    if (processor_->IsByte(id)) {
      int byte = sentencepiece::PieceToByte(processor_->IdToPiece(id));
      if (byte >= 0 && byte < byte_token_ids_.size()) byte_token_ids_[byte] = id;
    }
  }
}

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromFile(absl::string_view model_path) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->Load(model_path);
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromBuffer(absl::string_view model_buffer) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->LoadFromSerializedProto(model_buffer);
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

absl::StatusOr<std::unique_ptr<SentencePieceTokenizer>>
SentencePieceTokenizer::CreateFromProto(
    std::unique_ptr<sentencepiece::ModelProto> model_proto) {
  auto processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
  auto status = processor->Load(std::move(model_proto));
  if (!status.ok()) {
    return status;
  }
  return absl::WrapUnique(new SentencePieceTokenizer(std::move(processor)));
}

// Encodes the given text into a TensorBuffer of token ids.
absl::StatusOr<std::vector<int>> SentencePieceTokenizer::TextToTokenIds(
    absl::string_view text) {
  std::vector<int> ids;
  auto status = processor_->Encode(text, &ids);
  if (!status.ok()) {
    return status;
  }
  return ids;
}

absl::StatusOr<int> SentencePieceTokenizer::TokenToId(absl::string_view token) {
  int id = processor_->PieceToId(token);
  if (id == processor_->unk_id()) {
    return absl::NotFoundError(absl::StrCat("Unknown token: ", token));
  }
  return id;
}

// Decodes the given TensorBuffer of token ids into a string.
absl::StatusOr<std::string> SentencePieceTokenizer::TokenIdsToText(
    absl::Span<const int> token_ids, bool skip_special_tokens) {
  if (skip_special_tokens) {
    return absl::InvalidArgumentError(
        "SentencePieceTokenizer does not support skipping special tokens. "
        "Special tokens are handled by the SentencePiece buildenormalizer.");
  }

  for (const auto& token_id : token_ids) {
    if (token_id >= vocab_size_ || token_id < 0) {
      return absl::NotFoundError(
          absl::StrCat("Token id ", token_id,
                       " is out of range. Vocab size is ", vocab_size_));
    }
  }

  // We need special handling for control tokens like BOS and EOS.
  if (token_ids.size() == 1 && processor_->IsControl(token_ids[0])) {
    return processor_->IdToPiece(token_ids[0]);
  }

  std::string text;
  auto status = processor_->Decode(token_ids, &text);
  if (!status.ok()) {
    return status;
  }
  return text;
}

std::vector<std::string> SentencePieceTokenizer::GetTokens() const {
  std::vector<std::string> tokens;
  for (const auto& piece : processor_->model_proto().pieces()) {
    tokens.push_back(piece.piece());
  }
  return tokens;
}

int SentencePieceTokenizer::GetVocabSize() const {
  return processor_->GetPieceSize();
}

ConstraintVocabulary SentencePieceTokenizer::GetConstraintVocabulary() const {
  ConstraintVocabulary vocabulary;
  vocabulary.token_bytes.reserve(vocab_size_);
  for (int id = 0; id < vocab_size_; ++id) {
    vocabulary.token_bytes.push_back(ConstraintTokenBytes(*processor_, id));
    if (IsConstraintControl(*processor_, id)) {
      vocabulary.special_token_ids.push_back(id);
    }
  }
  return vocabulary;
}

absl::StatusOr<TokenIds> SentencePieceTokenizer::BytesToTokenIdsForConstraint(
    absl::string_view bytes, absl::Span<const int> excluded_token_ids) {
  auto excluded = [&](int id) {
    return std::find(excluded_token_ids.begin(), excluded_token_ids.end(), id) !=
           excluded_token_ids.end();
  };
  auto encoded = TextToTokenIds(bytes);
  if (encoded.ok()) {
    std::string reconstructed;
    reconstructed.reserve(bytes.size());
    bool ordinary = true;
    for (int id : *encoded) {
      if (IsConstraintControl(*processor_, id) || excluded(id)) {
        ordinary = false;
        break;
      }
      reconstructed += ConstraintTokenBytes(*processor_, id);
    }
    if (ordinary && reconstructed == bytes) return encoded;
  }
  // Byte pieces preserve source characters the normal encoder would change,
  // as well as partial UTF8 at grammar boundaries. Faithful ordinary encodings
  // retain their original model IDs.
  TokenIds exact;
  exact.reserve(bytes.size());
  for (unsigned char byte : bytes) {
    int id = byte_token_ids_[byte];
    if (id < 0 || excluded(id)) {
      return absl::InvalidArgumentError(
          "SentencePiece cannot encode constraint bytes faithfully.");
    }
    exact.push_back(id);
  }
  return exact;
}

}  // namespace litert::support
