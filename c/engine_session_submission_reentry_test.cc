// Copyright 2026 The ODML Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0

#include "c/engine.h"

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

constexpr int kVocabSize = 10;
const absl::Duration kWatchdog = absl::Seconds(5);

bool Check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "%s\n", message);
  return condition;
}

void NotifyOnce(absl::Notification& notification) {
  if (!notification.HasBeenNotified()) notification.Notify();
}

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
  std::atomic<int> prefill_calls = 0;
  std::atomic<int> decode_calls = 0;
  absl::Notification prefill_entered;
  absl::Notification prefill_release;
};

class CountedExecutor : public FakeLlmExecutor {
 public:
  using FakeLlmExecutor::Prefill;
  using FakeLlmExecutor::Decode;
  explicit CountedExecutor(std::shared_ptr<Effects> effects)
      : FakeLlmExecutor(kVocabSize, {{1, 2, 3}, {1, 2, 3}},
                        {{4}, {5}, {6}}),
        effects_(std::move(effects)) {}
  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& params) override {
    ++effects_->prefill_calls;
    NotifyOnce(effects_->prefill_entered);
    if (effects_->hold_prefill) effects_->prefill_release.WaitForNotification();
    return FakeLlmExecutor::Prefill(inputs, params);
  }
  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams& params) override {
    ++effects_->decode_calls;
    return FakeLlmExecutor::Decode(params);
  }
 private:
  std::shared_ptr<Effects> effects_;
};

struct CProbe {
  std::atomic<int> chunks = 0;
  std::atomic<int> finals = 0;
  std::atomic<bool> action_ok = true;
  std::optional<std::string> text;
  std::optional<std::string> error;
  std::function<bool()> action;
  absl::Notification entered;
  absl::Notification finished;
  absl::Notification release;

  static void OnChunk(void* data, const LiteRtLmStreamChunk* chunk) {
    auto& probe = *static_cast<CProbe*>(data);
    ++probe.chunks;
    if (!litert_lm_stream_chunk_is_final(chunk)) return;
    const char* text = litert_lm_stream_chunk_get_text(chunk);
    const char* error = litert_lm_stream_chunk_get_error(chunk);
    probe.text = text ? std::optional<std::string>(text) : std::nullopt;
    probe.error = error ? std::optional<std::string>(error) : std::nullopt;
    ++probe.finals;
    NotifyOnce(probe.entered);
    if (probe.action && !probe.action()) probe.action_ok = false;
    NotifyOnce(probe.finished);
  }
};

class Harness {
 public:
  ~Harness() {
    if (effects) NotifyOnce(effects->prefill_release);
    for (const auto& probe : probes_) NotifyOnce(probe->release);
    if (owner && !owner->WaitUntilAllDone(kWatchdog).ok()) {
      // Fail the child before releasing C callback data or native resources.
      _exit(2);
    }
    sessions_.clear();
    owner.reset();
    probes_.clear();
  }

  bool Initialize(absl::Status prefill_status = absl::OkStatus(),
                  bool hold_prefill = false,
                  std::optional<int> max_context = std::nullopt) {
    ON_CALL(tokenizer_, GetVocabSize())
        .WillByDefault(::testing::Return(kVocabSize));
    ON_CALL(tokenizer_, TokenIdsToText(::testing::_, ::testing::_))
        .WillByDefault([](absl::Span<const int> ids, bool) {
          std::string text;
          for (int id : ids) text += std::to_string(id);
          return text;
        });
    effects = std::make_shared<Effects>();
    effects->hold_prefill = hold_prefill;
    auto executor = std::make_unique<CountedExecutor>(effects);
    executor->SetPrefillStatus(prefill_status);
    if (max_context) {
      auto settings = executor->GetMutableExecutorSettings();
      if (!settings.ok()) return false;
      (*settings)->SetMaxNumTokens(*max_context);
    }
    auto created = ThreadedExecutionManager::Create(
        &tokenizer_, nullptr, std::move(executor), nullptr, nullptr,
        nullptr, nullptr, nullptr);
    if (!created.ok()) return false;
    owner = std::move(*created);
    return true;
  }

  LiteRtLmSession* AddSession() {
    auto assets = ModelAssets::Create("fixture_model");
    if (!assets.ok()) return nullptr;
    auto settings = EngineSettings::CreateDefault(*assets);
    if (!settings.ok()) return nullptr;
    proto::LlmMetadata metadata;
    metadata.mutable_stop_tokens()->Add()->mutable_token_ids()->mutable_ids()->Add(6);
    metadata.mutable_llm_model_type()->mutable_gemma3n();
    if (!settings->MaybeUpdateAndValidate(&tokenizer_, &metadata).ok()) return nullptr;
    auto config = SessionConfig::CreateDefault();
    if (!config.MaybeUpdateAndValidate(*settings).ok()) return nullptr;
    config.SetUseExternalSampler(false);
    config.SetApplyPromptTemplateInSession(false);
    config.SetMaxOutputTokens(1);
    auto advanced = SessionAdvanced::Create(owner, &tokenizer_, config, std::nullopt);
    if (!advanced.ok()) return nullptr;
    auto session = std::make_unique<LiteRtLmSession>();
    session->session = std::move(*advanced);
    auto* result = session.get();
    sessions_.push_back(std::move(session));
    return result;
  }

  SessionAdvanced& Advanced(LiteRtLmSession* session) {
    return *static_cast<SessionAdvanced*>(session->session.get());
  }

  absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>> Prefill(
      LiteRtLmSession* session,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback =
          [](absl::StatusOr<Responses>) {}) {
    ABSL_ASSIGN_OR_RETURN(auto tensor,
        tokenizer_.TokenIdsToTensorBuffer({1, 2, 3}));
    std::vector<InputData> inputs;
    inputs.emplace_back(InputText(std::move(tensor)));
    return Advanced(session).PrefillPreprocessedContents(std::move(inputs),
                                                        std::move(callback));
  }

  std::shared_ptr<CProbe> Probe() {
    auto result = std::make_shared<CProbe>();
    probes_.push_back(result);
    return result;
  }

  bool Drain() { return owner->WaitUntilAllDone(kWatchdog).ok(); }
  std::shared_ptr<Effects> effects;
  std::shared_ptr<ThreadedExecutionManager> owner;

 private:
  ::testing::NiceMock<MockTokenizer> tokenizer_;
  std::vector<std::shared_ptr<CProbe>> probes_;
  std::vector<std::unique_ptr<LiteRtLmSession>> sessions_;
};

bool FailedPrecondition(const absl::Status& status) {
  return status.code() == absl::StatusCode::kFailedPrecondition;
}

bool RefusesEveryLockTakingEntry(Harness& harness, LiteRtLmSession* session) {
  auto& advanced = harness.Advanced(session);
  auto callbacks = std::make_shared<std::atomic<int>>(0);
  auto callback = [callbacks](absl::StatusOr<Responses>) { ++*callbacks; };
  std::vector<InputData> input;
  input.emplace_back(InputText("unused"));
  bool ok = Check(litert_lm_session_save_checkpoint(session, "raw-nested") != 0,
                  "C checkpoint refusal");
  ok = Check(FailedPrecondition(advanced.SaveCheckpoint("nested")), "SaveCheckpoint") && ok;
  ok = Check(FailedPrecondition(advanced.GetBenchmarkInfo().status()), "GetBenchmarkInfo") && ok;
  ok = Check(FailedPrecondition(advanced.RewindToCheckpoint("missing")), "RewindToCheckpoint") && ok;
  ok = Check(FailedPrecondition(advanced.RewindToStep(0)), "RewindToStep") && ok;
  ok = Check(FailedPrecondition(advanced.Clone().status()), "Clone") && ok;
  ok = Check(FailedPrecondition(advanced.CloneAsync(callback).status()), "CloneAsync") && ok;
  ok = Check(FailedPrecondition(advanced.RunPrefillAsync(input, callback).status()), "RunPrefillAsync") && ok;
  std::vector<InputData> preprocessed;
  preprocessed.emplace_back(InputText("unused"));
  ok = Check(FailedPrecondition(advanced.PrefillPreprocessedContents(
      std::move(preprocessed), callback).status()), "PrefillPreprocessedContents") && ok;
  ok = Check(FailedPrecondition(advanced.RunDecodeAsync(callback).status()), "RunDecodeAsync") && ok;
  ok = Check(FailedPrecondition(advanced.RunTextScoringAsync(
      {"unused"}, callback, false).status()), "RunTextScoringAsync") && ok;
  ok = Check(FailedPrecondition(advanced.RunPrefill(input)), "RunPrefill wrapper") && ok;
  ok = Check(FailedPrecondition(advanced.RunDecode().status()), "RunDecode wrapper") && ok;
  ok = Check(FailedPrecondition(advanced.RunTextScoring(
      {"unused"}, false).status()), "RunTextScoring wrapper") && ok;
  ok = Check(litert_lm_session_rewind_to_checkpoint(session, "missing") != 0 &&
      litert_lm_session_rewind_to_step(session, 0) != 0, "C rewind refusal") && ok;
  auto nested = harness.Probe();
  ok = Check(litert_lm_session_run_decode_async(
      session, &CProbe::OnChunk, nested.get()) ==
      static_cast<int>(absl::StatusCode::kFailedPrecondition), "C decode status") && ok;
  ok = Check(nested->chunks.load() == 0 && callbacks->load() == 0,
             "Refused operations emitted callbacks") && ok;
  return ok;
}

enum class Seed { kFailed, kCancelled, kMaxTokens };

bool SeedTerminal(Harness& harness, LiteRtLmSession* session, Seed seed) {
  auto prefill = harness.Prefill(session);
  if (!prefill.ok()) return false;
  if (seed == Seed::kCancelled) {
    if (!harness.effects->prefill_entered.WaitForNotificationWithTimeout(kWatchdog)) return false;
    if (!(*prefill)->Cancel().ok()) return false;
    NotifyOnce(harness.effects->prefill_release);
  }
  if (!(*prefill)->WaitUntilDone(kWatchdog).ok() || !harness.Drain()) return false;
  if (seed == Seed::kMaxTokens) {
    auto max_probe = harness.Probe();
    if (litert_lm_session_run_decode_async(session, &CProbe::OnChunk, max_probe.get()) != 0) return false;
    if (!max_probe->finished.WaitForNotificationWithTimeout(kWatchdog) || !harness.Drain()) return false;
    if (max_probe->finals.load() != 1 || max_probe->error != "Max number of tokens reached.") return false;
    // The next accepted prefill inherits max-token termination and restores
    // the Prefilled state needed by the actual C decode entry point.
    auto inherited = harness.Prefill(session);
    if (!inherited.ok() || !(*inherited)->WaitUntilDone(kWatchdog).ok() || !harness.Drain()) return false;
  }
  return true;
}

bool ImmediateControl(Seed seed) {
  Harness harness;
  const auto status = seed == Seed::kFailed
      ? absl::InternalError("fixture: prefill failure") : absl::OkStatus();
  if (!harness.Initialize(status, seed == Seed::kCancelled,
      seed == Seed::kMaxTokens ? std::optional<int>(4) : std::nullopt)) return false;
  auto* session = harness.AddSession();
  if (!session || !SeedTerminal(harness, session, seed)) return false;
  const int prefills = harness.effects->prefill_calls.load();
  const int decodes = harness.effects->decode_calls.load();
  auto probe = harness.Probe();
  probe->action = [&harness, session] {
    return RefusesEveryLockTakingEntry(harness, session);
  };
  const int started = litert_lm_session_run_decode_async(session, &CProbe::OnChunk, probe.get());
  if (!harness.Drain()) return false;
  const std::string error = seed == Seed::kFailed ? "Task failed."
      : seed == Seed::kCancelled ? "CANCELLED." : "Max number of tokens reached.";
  bool ok = Check(started == 0 && probe->action_ok.load(), "Accepted C terminal start or nested refusal");
  ok = Check(probe->chunks.load() == 1 && probe->finals.load() == 1 &&
      !probe->text.has_value() && probe->error == error, "Exact one terminal payload") && ok;
  ok = Check(litert_lm_session_save_checkpoint(session, "after-return") == 0 &&
      harness.Advanced(session).SaveCheckpoint("native-after-return").ok(), "Checkpoint after return") && ok;
  ok = Check(harness.effects->prefill_calls.load() == prefills &&
      harness.effects->decode_calls.load() == decodes, "Refusal caused native execution") && ok;
  return ok;
}

bool OtherThreadControl() {
  Harness harness;
  if (!harness.Initialize(absl::InternalError("fixture: prefill failure"))) return false;
  auto* session = harness.AddSession();
  if (!session || !SeedTerminal(harness, session, Seed::kFailed)) return false;
  auto probe = harness.Probe();
  probe->action = [value = probe.get()] {
    value->release.WaitForNotification();
    return true;
  };
  std::atomic<int> started = -1;
  std::atomic<int> checkpoint = -1;
  absl::Notification checkpoint_started;
  absl::Notification checkpoint_finished;
  std::thread submit([&] {
    started = litert_lm_session_run_decode_async(session, &CProbe::OnChunk, probe.get());
  });
  if (!probe->entered.WaitForNotificationWithTimeout(kWatchdog)) {
    NotifyOnce(probe->release);
    submit.join();
    return false;
  }
  std::thread other([&] {
    checkpoint_started.Notify();
    checkpoint = litert_lm_session_save_checkpoint(session, "other-thread");
    checkpoint_finished.Notify();
  });
  bool ok = checkpoint_started.WaitForNotificationWithTimeout(kWatchdog);
  ok = Check(!checkpoint_finished.WaitForNotificationWithTimeout(absl::Milliseconds(50)),
             "Other-thread call bypassed held mutex") && ok;
  ok = Check(harness.owner->WaitUntilAllDone(absl::ZeroDuration()).code() ==
      absl::StatusCode::kDeadlineExceeded, "Native return fence released before C final return") && ok;
  NotifyOnce(probe->release);
  submit.join();
  other.join();
  ok = Check(started.load() == 0 && checkpoint.load() == 0 &&
      probe->finals.load() == 1 && probe->chunks.load() == 1 &&
      !probe->text.has_value() && probe->error == "Task failed." && harness.Drain(),
      "Other-thread serialization, exact final payload or drain") && ok;
  ok = Check(harness.effects->prefill_calls.load() == 1 &&
      harness.effects->decode_calls.load() == 0, "Held final caused executor work") && ok;
  return ok;
}

bool NormalCallbackControl() {
  Harness harness;
  if (!harness.Initialize()) return false;
  auto* session = harness.AddSession();
  if (!session) return false;
  auto prefill = harness.Prefill(session);
  if (!prefill.ok() || !(*prefill)->WaitUntilDone(kWatchdog).ok() || !harness.Drain()) return false;
  auto probe = harness.Probe();
  auto next_terminals = std::make_shared<std::atomic<int>>(0);
  auto next_success = std::make_shared<std::atomic<bool>>(false);
  probe->action = [&harness, session, next_terminals, next_success] {
    if (litert_lm_session_save_checkpoint(session, "normal-terminal") != 0) return false;
    auto next = harness.Prefill(session, [next_terminals, next_success](absl::StatusOr<Responses> response) {
      if (!response.ok() || IsTaskEndState(response->GetTaskState())) {
        ++*next_terminals;
        *next_success = response.ok() && response->GetTaskState() == TaskState::kDone;
      }
    });
    return next.ok();
  };
  if (litert_lm_session_run_decode_async(session, &CProbe::OnChunk, probe.get()) != 0) return false;
  if (!probe->finished.WaitForNotificationWithTimeout(kWatchdog) || !harness.Drain()) return false;
  return Check(probe->action_ok.load() && probe->finals.load() == 1 &&
      !probe->text.has_value() && !probe->error.has_value() &&
      next_terminals->load() == 1 && next_success->load() &&
      harness.effects->prefill_calls.load() == 2 &&
      harness.effects->decode_calls.load() == 1, "Normal callback continuation refused or altered");
}

bool NestedSessionsControl() {
  Harness harness;
  if (!harness.Initialize(absl::InternalError("fixture: prefill failure"))) return false;
  auto* first = harness.AddSession();
  auto* second = harness.AddSession();
  if (!first || !second || !SeedTerminal(harness, first, Seed::kFailed) ||
      !SeedTerminal(harness, second, Seed::kFailed)) return false;
  auto a = harness.Probe();
  auto b = harness.Probe();
  b->action = [&harness, first, second] {
    return FailedPrecondition(harness.Advanced(first).SaveCheckpoint("outer-held")) &&
        FailedPrecondition(harness.Advanced(second).SaveCheckpoint("inner-held"));
  };
  a->action = [&harness, first, second, b] {
    if (litert_lm_session_save_checkpoint(second, "other-session") != 0) return false;
    if (litert_lm_session_run_decode_async(second, &CProbe::OnChunk, b.get()) != 0) return false;
    return b->action_ok.load() && litert_lm_session_save_checkpoint(second, "inner-returned") == 0 &&
        FailedPrecondition(harness.Advanced(first).SaveCheckpoint("outer-still-held"));
  };
  const int started = litert_lm_session_run_decode_async(first, &CProbe::OnChunk, a.get());
  if (!harness.Drain()) return false;
  return Check(started == 0 && a->action_ok.load() && a->finals.load() == 1 &&
      b->finals.load() == 1 && a->error == "Task failed." && b->error == "Task failed." &&
      litert_lm_session_save_checkpoint(first, "a-returned") == 0 &&
      litert_lm_session_save_checkpoint(second, "b-returned") == 0 &&
      harness.effects->prefill_calls.load() == 2 && harness.effects->decode_calls.load() == 0,
      "Nested different-session ownership stack or return behavior");
}

// Construct all native owners inside the death-test child. The parent accepts
// only exit 0; old production self-deadlocks/aborts and cannot pass. The alarm
// bounds that reproduction without freeing resources still owned by a callback.
#define REENTRY_CONTROL(expression) \
  ASSERT_EXIT({ \
    alarm(15); \
    const bool ok = (expression); \
    alarm(0); \
    _exit(ok ? 0 : 1); \
  }, ::testing::ExitedWithCode(0), ".*")

TEST(EngineSessionSubmissionReentryTest,
     ImmediateFailedCFinalRefusesSameSessionReentryThenAllowsCheckpoint) {
  REENTRY_CONTROL(ImmediateControl(Seed::kFailed));
}
TEST(EngineSessionSubmissionReentryTest,
     ImmediateCancelledCFinalRefusesSameSessionReentryThenAllowsCheckpoint) {
  REENTRY_CONTROL(ImmediateControl(Seed::kCancelled));
}
TEST(EngineSessionSubmissionReentryTest,
     ImmediateMaxTokenCFinalRefusesSameSessionReentryThenAllowsCheckpoint) {
  REENTRY_CONTROL(ImmediateControl(Seed::kMaxTokens));
}
TEST(EngineSessionSubmissionReentryTest, ImmediateFailedCFinalKeepsOtherThreadSerialized) {
  REENTRY_CONTROL(OtherThreadControl());
}
TEST(EngineSessionSubmissionReentryTest, NormalQueuedTerminalCFinalCanCheckpointAndSubmitNextPrefill) {
  REENTRY_CONTROL(NormalCallbackControl());
}
TEST(EngineSessionSubmissionReentryTest, DifferentSessionNestedCallbacksRetainOuterReentryGuard) {
  REENTRY_CONTROL(NestedSessionsControl());
}

#undef REENTRY_CONTROL

}  // namespace
}  // namespace litert::lm
