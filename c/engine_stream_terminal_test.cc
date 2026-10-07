// Copyright 2026 The ODML Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0

#include "c/engine.h"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "c/engine_internal.h"
#include "runtime/core/session_advanced.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/test_utils.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

const absl::Duration kWatchdog = absl::Seconds(5);
constexpr int kVocabSize = 10;

void NotifyOnce(absl::Notification& value) {
  if (!value.HasBeenNotified()) value.Notify();
}

class JoinOnExit {
 public:
  JoinOnExit(std::thread& thread, absl::Notification& release)
      : thread_(thread), release_(release) {}
  ~JoinOnExit() {
    NotifyOnce(release_);
    if (thread_.joinable()) thread_.join();
  }
 private:
  std::thread& thread_;
  absl::Notification& release_;
};

class MockTokenizer : public ::litert::support::Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int>, bool), (override));
  MOCK_METHOD(::litert::support::TokenizerType, GetTokenizerType, (),
              (const, override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));
};

struct Effects {
  bool hold_prefill = false;
  bool hold_decode = false;
  std::atomic<int> prefill_calls = 0;
  std::atomic<int> decode_calls = 0;
  absl::Notification prefill_entered;
  absl::Notification prefill_release;
  absl::Notification decode_entered;
  absl::Notification decode_release;
};

// The repository's real FakeLlmExecutor owns validation, context and tensors.
// Only input-boundary barriers/counters are added; native owners are not faked.
class HeldFakeLlmExecutor : public FakeLlmExecutor {
 public:
  using FakeLlmExecutor::Prefill;
  using FakeLlmExecutor::Decode;
  explicit HeldFakeLlmExecutor(std::shared_ptr<Effects> effects)
      : FakeLlmExecutor(kVocabSize, {{1, 2, 3}}, {{4}, {5}, {6}}),
        effects_(std::move(effects)) {}
  absl::Status Prefill(const ExecutorInputs& inputs) override {
    return FakeLlmExecutor::Prefill(inputs);
  }
  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& params) override {
    // Hold before the real fake executor's configurable status check, so an
    // actual accepted failing Prefill can have a waiting native continuation.
    ++effects_->prefill_calls;
    effects_->prefill_entered.Notify();
    if (effects_->hold_prefill) effects_->prefill_release.WaitForNotification();
    return FakeLlmExecutor::Prefill(inputs, params);
  }
  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams& params) override {
    ++effects_->decode_calls;
    NotifyOnce(effects_->decode_entered);
    if (effects_->hold_decode) effects_->decode_release.WaitForNotification();
    return FakeLlmExecutor::Decode(params);
  }
 private:
  std::shared_ptr<Effects> effects_;
};

struct NativeProbe {
  absl::Notification terminal;
  std::atomic<int> terminal_calls = 0;
  void OnResponse(absl::StatusOr<Responses> response) {
    if (response.ok() && !IsTaskEndState(response->GetTaskState())) return;
    if (terminal_calls.fetch_add(1) == 0) terminal.Notify();
  }
};

struct CProbe {
  bool hold = false;
  std::atomic<int> finals = 0;
  std::atomic<int> chunks = 0;
  absl::Notification terminal;
  absl::Notification release;
  absl::Mutex mutex;
  std::optional<std::string> final_text ABSL_GUARDED_BY(mutex);
  std::optional<std::string> final_error ABSL_GUARDED_BY(mutex);

  static void OnChunk(void* data, const LiteRtLmStreamChunk* chunk) {
    auto& probe = *static_cast<CProbe*>(data);
    ++probe.chunks;
    if (!litert_lm_stream_chunk_is_final(chunk)) return;
    // C chunk/text/error pointers are borrowed only during this callback.
    // Copy them before returning; never retain a borrowed address.
    const char* text = litert_lm_stream_chunk_get_text(chunk);
    const char* error = litert_lm_stream_chunk_get_error(chunk);
    {
      absl::MutexLock lock(&probe.mutex);
      probe.final_text = text ? std::optional<std::string>(text) : std::nullopt;
      probe.final_error = error ? std::optional<std::string>(error) : std::nullopt;
    }
    if (probe.finals.fetch_add(1) == 0) probe.terminal.Notify();
    if (probe.hold) probe.release.WaitForNotification();
  }
};

class EngineStreamTerminalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EXPECT_CALL(tokenizer_, GetVocabSize())
        .WillRepeatedly(::testing::Return(kVocabSize));
    EXPECT_CALL(tokenizer_, TokenIdsToText(::testing::_, ::testing::_))
        .WillRepeatedly([](absl::Span<const int> ids, bool) {
          std::string text;
          for (int id : ids) text += std::to_string(id);
          return text;
        });
  }
  void TearDown() override {
    if (effects_) {
      NotifyOnce(effects_->prefill_release);
      NotifyOnce(effects_->decode_release);
    }
    for (auto& probe : probes_) NotifyOnce(probe->release);
    if (owner_) EXPECT_OK(owner_->WaitUntilAllDone(kWatchdog));
    // Actual SessionAdvanced releases its exact registered native session.
    // Retain all raw callback-data probes until real tasks/pools have drained.
    session_.reset();
    owner_.reset();
    probes_.clear();
  }
  void CreateSession(bool hold_prefill = false,
                     absl::Status prefill_status = absl::OkStatus(),
                     bool hold_decode = false,
                     absl::Status decode_status = absl::OkStatus(),
                     std::optional<int> max_context = std::nullopt) {
    effects_ = std::make_shared<Effects>();
    effects_->hold_prefill = hold_prefill;
    effects_->hold_decode = hold_decode;
    auto executor = std::make_unique<HeldFakeLlmExecutor>(effects_);
    executor->SetPrefillStatus(prefill_status);
    executor->SetDecodeStatus(decode_status);
    if (max_context) {
      ASSERT_OK_AND_ASSIGN(auto settings, executor->GetMutableExecutorSettings());
      settings->SetMaxNumTokens(*max_context);
    }
    ASSERT_OK_AND_ASSIGN(
        auto created_owner, ThreadedExecutionManager::Create(
            &tokenizer_, nullptr, std::move(executor), nullptr, nullptr,
            nullptr, nullptr, nullptr));
    owner_ = std::move(created_owner);
    ASSERT_OK_AND_ASSIGN(auto assets, ModelAssets::Create("fixture_model"));
    ASSERT_OK_AND_ASSIGN(auto settings, EngineSettings::CreateDefault(assets));
    proto::LlmMetadata metadata;
    metadata.mutable_stop_tokens()->Add()->mutable_token_ids()->mutable_ids()->Add(6);
    metadata.mutable_llm_model_type()->mutable_gemma3n();
    ASSERT_OK(settings.MaybeUpdateAndValidate(&tokenizer_, &metadata));
    auto config = SessionConfig::CreateDefault();
    ASSERT_OK(config.MaybeUpdateAndValidate(settings));
    config.SetUseExternalSampler(false);
    config.SetApplyPromptTemplateInSession(false);
    config.SetMaxOutputTokens(1);
    ASSERT_OK_AND_ASSIGN(auto created_session,
        SessionAdvanced::Create(owner_, &tokenizer_, config, std::nullopt));
    advanced_ = created_session.get();
    // A legitimate actual native session wrapped in the existing internal C
    // owner type. No invented opaque pointer or mock SessionInterface.
    session_ = std::make_unique<LiteRtLmSession>();
    session_->session = std::move(created_session);
  }
  absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>> StartPrefill() {
    ABSL_ASSIGN_OR_RETURN(auto tensor,
        tokenizer_.TokenIdsToTensorBuffer({1, 2, 3}));
    std::vector<InputData> inputs;
    inputs.emplace_back(InputText(std::move(tensor)));
    auto probe = std::make_shared<NativeProbe>();
    return advanced_->PrefillPreprocessedContents(std::move(inputs),
        [probe](absl::StatusOr<Responses> response) {
          probe->OnResponse(std::move(response));
        });
  }
  std::shared_ptr<CProbe> Probe(bool hold = false) {
    auto probe = std::make_shared<CProbe>();
    probe->hold = hold;
    probes_.push_back(probe);
    return probe;
  }
  void ExpectFinal(CProbe& probe, std::optional<std::string> error) {
    EXPECT_EQ(probe.finals.load(), 1);
    absl::MutexLock lock(&probe.mutex);
    EXPECT_EQ(probe.final_text, std::nullopt);
    EXPECT_EQ(probe.final_error, error);
  }
  void DependentControl(bool cancelled, bool immediate) {
    CreateSession(true, cancelled ? absl::OkStatus()
                                 : absl::InternalError("fixture: prefill failure"));
    ASSERT_NE(session_, nullptr);
    ASSERT_OK_AND_ASSIGN(auto prefill, StartPrefill());
    ASSERT_TRUE(effects_->prefill_entered.WaitForNotificationWithTimeout(kWatchdog));
    if (cancelled) ASSERT_OK(prefill->Cancel());
    auto probe = Probe(immediate);
    if (!immediate) {
      // Accepted while the actual predecessor is still Processing. This
      // exercises real FollowingWaitingTasks/UpdateAllTasksToState delivery.
      EXPECT_EQ(litert_lm_session_run_decode_async(
          session_.get(), &CProbe::OnChunk, probe.get()), 0);
      NotifyOnce(effects_->prefill_release);
      ASSERT_TRUE(probe->terminal.WaitForNotificationWithTimeout(kWatchdog));
      ASSERT_OK(owner_->WaitUntilAllDone(kWatchdog));
    } else {
      NotifyOnce(effects_->prefill_release);
      ASSERT_OK(prefill->WaitUntilDone(kWatchdog));
      ASSERT_OK(owner_->WaitUntilAllDone(kWatchdog));
      std::atomic<int> start_status = -1;
      std::atomic<bool> returned = false;
      std::thread submit([&] {
        start_status = litert_lm_session_run_decode_async(
            session_.get(), &CProbe::OnChunk, probe.get());
        returned = true;
      });
      JoinOnExit cleanup(submit, probe->release);
      ASSERT_TRUE(probe->terminal.WaitForNotificationWithTimeout(kWatchdog));
      EXPECT_FALSE(returned.load());
      // Both pools drained before submission. This is a caller-owned accepted
      // immediate callback, fenced by v4's authoritative native active set.
      EXPECT_EQ(owner_->WaitUntilAllDone(absl::ZeroDuration()).code(),
                absl::StatusCode::kDeadlineExceeded);
      NotifyOnce(probe->release);
      submit.join();
      EXPECT_TRUE(returned.load());
      EXPECT_EQ(start_status.load(), 0);
      ASSERT_OK(owner_->WaitUntilAllDone(kWatchdog));
    }
    ExpectFinal(*probe, cancelled ? "CANCELLED." : "Task failed.");
    EXPECT_EQ(probe->chunks.load(), 1);
    EXPECT_EQ(effects_->prefill_calls.load(), 1);
    EXPECT_EQ(effects_->decode_calls.load(), 0);
  }
  ::testing::NiceMock<MockTokenizer> tokenizer_;
  std::shared_ptr<Effects> effects_;
  std::shared_ptr<ThreadedExecutionManager> owner_;
  SessionAdvanced* advanced_ = nullptr;
  std::unique_ptr<LiteRtLmSession> session_;
  std::vector<std::shared_ptr<CProbe>> probes_;
};

TEST_F(EngineStreamTerminalTest, QueuedDecodeAfterFailedPrefillGetsOneFinalError) {
  DependentControl(false, false);
}
TEST_F(EngineStreamTerminalTest, QueuedDecodeAfterCancelledPrefillGetsOneFinalError) {
  DependentControl(true, false);
}
TEST_F(EngineStreamTerminalTest, ImmediateFailedDecodeFencesActualCHandlerReturn) {
  DependentControl(false, true);
}
TEST_F(EngineStreamTerminalTest, ImmediateCancelledDecodeFencesActualCHandlerReturn) {
  DependentControl(true, true);
}
TEST_F(EngineStreamTerminalTest, SuccessfulDecodePreservesFinalNullErrorPayload) {
  CreateSession();
  ASSERT_NE(session_, nullptr);
  ASSERT_OK_AND_ASSIGN(auto prefill, StartPrefill());
  ASSERT_OK(prefill->WaitUntilDone(kWatchdog));
  auto probe = Probe();
  EXPECT_EQ(litert_lm_session_run_decode_async(
      session_.get(), &CProbe::OnChunk, probe.get()), 0);
  ASSERT_TRUE(probe->terminal.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(owner_->WaitUntilAllDone(kWatchdog));
  ExpectFinal(*probe, std::nullopt);
  EXPECT_EQ(effects_->prefill_calls.load(), 1);
  EXPECT_EQ(effects_->decode_calls.load(), 1);
}
TEST_F(EngineStreamTerminalTest, MaxContextPreservesExistingFinalErrorPayload) {
  CreateSession(false, absl::OkStatus(), false, absl::OkStatus(), 4);
  ASSERT_NE(session_, nullptr);
  ASSERT_OK_AND_ASSIGN(auto prefill, StartPrefill());
  ASSERT_OK(prefill->WaitUntilDone(kWatchdog));
  auto probe = Probe();
  EXPECT_EQ(litert_lm_session_run_decode_async(
      session_.get(), &CProbe::OnChunk, probe.get()), 0);
  ASSERT_TRUE(probe->terminal.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(owner_->WaitUntilAllDone(kWatchdog));
  ExpectFinal(*probe, "Max number of tokens reached.");
  EXPECT_EQ(effects_->decode_calls.load(), 1);
}
TEST_F(EngineStreamTerminalTest, DirectDecodeErrorPreservesExactStatusString) {
  const auto failure = absl::ResourceExhaustedError("fixture: decode failure");
  CreateSession(false, absl::OkStatus(), false, failure);
  ASSERT_NE(session_, nullptr);
  ASSERT_OK_AND_ASSIGN(auto prefill, StartPrefill());
  ASSERT_OK(prefill->WaitUntilDone(kWatchdog));
  auto probe = Probe();
  EXPECT_EQ(litert_lm_session_run_decode_async(
      session_.get(), &CProbe::OnChunk, probe.get()), 0);
  ASSERT_TRUE(probe->terminal.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(owner_->WaitUntilAllDone(kWatchdog));
  ExpectFinal(*probe, failure.ToString());
  EXPECT_EQ(effects_->decode_calls.load(), 1);
}
TEST_F(EngineStreamTerminalTest, DirectCancellationPreservesExistingFinalErrorPayload) {
  CreateSession(false, absl::OkStatus(), true);
  ASSERT_NE(session_, nullptr);
  ASSERT_OK_AND_ASSIGN(auto prefill, StartPrefill());
  ASSERT_OK(prefill->WaitUntilDone(kWatchdog));
  auto probe = Probe();
  EXPECT_EQ(litert_lm_session_run_decode_async(
      session_.get(), &CProbe::OnChunk, probe.get()), 0);
  ASSERT_TRUE(effects_->decode_entered.WaitForNotificationWithTimeout(kWatchdog));
  litert_lm_session_cancel_process(session_.get());
  NotifyOnce(effects_->decode_release);
  ASSERT_TRUE(probe->terminal.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(owner_->WaitUntilAllDone(kWatchdog));
  ExpectFinal(*probe, "CANCELLED.");
  EXPECT_EQ(effects_->decode_calls.load(), 1);
  // The cancelled session is drained/deleted; it is never reused here.
}

}  // namespace
}  // namespace litert::lm
