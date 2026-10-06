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

#include "runtime/framework/resource_management/execution_manager.h"

#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {

ExecutionManager::ExecutionManager(DecodeInputBufferFactory buffer_factory)
    : decode_input_buffer_factory_(std::move(buffer_factory)) {
  if (!decode_input_buffer_factory_) {
    decode_input_buffer_factory_ = AllocateDecodeInputBuffer;
  }
}

absl::StatusOr<::litert::TensorBuffer>
ExecutionManager::AllocateDecodeInputBuffer(absl::Span<const int> decoded_ids) {
  if (decoded_ids.empty() ||
      decoded_ids.size() > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError("Invalid decoded ID batch size.");
  }
  auto buffer = CopyToTensorBuffer<int>(
      decoded_ids, {static_cast<int>(decoded_ids.size()), 1});
  if (!buffer.HasValue()) {
    return absl::InternalError(buffer.Error().Message());
  }
  return std::move(buffer.Value());
}

absl::StatusOr<::litert::TensorBuffer>
ExecutionManager::CreateDecodeInputBuffer(
    absl::Span<const int> decoded_ids) const {
  return decode_input_buffer_factory_(decoded_ids);
}

}  // namespace litert::lm
