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

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/test_utils.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// Same existing production friend; only bounded observations of real ownership.
// A baseline locked terminal handler reports a failed assertion and returns,
// rather than making the test destructor wait ten minutes on a deadlocked owner.
class SchedulingLifetimeTestPeer {
 public:
  static bool LookupAvailable(ThreadedExecutionManager& owner) {
    if (!owner.session_and_task_lookup_mutex_.TryLock()) return false;
    owner.session_and_task_lookup_mutex_.Unlock();
    return true;
  }
  static std::optional<SessionId> OnlySession(ThreadedExecutionManager& owner) {
    absl::MutexLock lock(&owner.session_and_task_lookup_mutex_);
    if (owner.session_lookup_.size() != 1) return std::nullopt;
    return owner.session_lookup_.begin()->first;
  }
  static std::optional<TaskState> State(ThreadedExecutionManager& owner,
                                        TaskId task) {
    absl::MutexLock lock(&owner.session_and_task_lookup_mutex_);
    auto found = owner.task_lookup_.find(task);
    if (found == owner.task_lookup_.end()) return std::nullopt;
    return found->second.task_state;
  }
  static bool IsActive(ThreadedExecutionManager& owner, SessionId session,
                       TaskId task) {
    absl::MutexLock lock(&owner.session_and_task_lookup_mutex_);
    return owner.session_lookup_.at(session)->active_tasks.contains(task);
  }
  static size_t Active(ThreadedExecutionManager& owner, SessionId session) {
    absl::MutexLock lock(&owner.session_and_task_lookup_mutex_);
    return owner.session_lookup_.at(session)->active_tasks.size();
  }
};

namespace {

constexpr int kVocabSize = 10;
const absl::Duration kWatchdog = absl::Seconds(5);

void NotifyOnce(absl::Notification& value) {
  if (!value.HasBeenNotified()) value.Notify();
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

struct EffectControl {
  explicit EffectControl(bool hold) : hold_first(hold) {}
  const bool hold_first;
  std::atomic<int> prefill_calls = 0;
  absl::Notification entered;
  absl::Notification release;
};

// This is the repository's real FakeLlmExecutor. Only the first Prefill input
// boundary gains a counter/barrier; its validation, context, tensors, processed
// tokens, executor resource owner, and real threaded scheduling are unchanged.
class HeldFakeLlmExecutor : public FakeLlmExecutor {
 public:
  using FakeLlmExecutor::Prefill;
  explicit HeldFakeLlmExecutor(std::shared_ptr<EffectControl> control)
      : FakeLlmExecutor(kVocabSize,
                        {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {1, 2, 3}},
                        {{4}, {5}, {6}}),
        control_(std::move(control)) {}
  absl::Status Prefill(const ExecutorInputs& inputs) override {
    return FakeLlmExecutor::Prefill(inputs);
  }
  absl::Status Prefill(const ExecutorInputs& inputs,
                       const ExecutorPrefillParams& params) override {
    // Barrier precedes the real configurable status check.
    if (control_->prefill_calls.fetch_add(1) == 0 && control_->hold_first) {
      control_->entered.Notify();
      control_->release.WaitForNotification();
    }
    return FakeLlmExecutor::Prefill(inputs, params);
  }
 private:
  std::shared_ptr<EffectControl> control_;
};

struct TerminalProbe {
  explicit TerminalProbe(bool hold = false) : hold(hold) {}
  const bool hold;
  std::atomic<int> calls = 0;
  absl::Notification entered;
  absl::Notification release;
  absl::Mutex mutex;
  absl::Status terminal_status ABSL_GUARDED_BY(mutex);
  std::optional<TaskState> terminal_state ABSL_GUARDED_BY(mutex);
  absl::Status lookup_status ABSL_GUARDED_BY(mutex);
  absl::Status self_wait_status ABSL_GUARDED_BY(mutex);
  // Only a terminal handler uses these safe reads, outside the lookup lock.
  ThreadedExecutionManager* owner = nullptr;
  SessionId session_id = -1;

  void OnResponse(absl::StatusOr<Responses> result) {
    if (result.ok() && !IsTaskEndState(result->GetTaskState())) return;
    const int invocation = calls.fetch_add(1);
    absl::Status lookup = absl::FailedPreconditionError("fixture: lookup lock held");
    absl::Status self_wait = lookup;
    if (owner != nullptr &&
        SchedulingLifetimeTestPeer::LookupAvailable(*owner)) {
      lookup = owner->GetSessionInfo(session_id).status();
      // Bounded observation only. An indefinite self-wait cannot complete
      // while this callback owns the task's return fence.
      self_wait = owner->WaitUntilSessionDone(session_id, absl::ZeroDuration());
    }
    {
      absl::MutexLock lock(&mutex);
      terminal_status = result.status();
      terminal_state = result.ok()
                           ? std::optional<TaskState>(result->GetTaskState())
                           : std::nullopt;
      lookup_status = lookup;
      self_wait_status = self_wait;
    }
    if (invocation == 0) entered.Notify();
    if (hold) release.WaitForNotification();
  }
  absl::Status Status() {
    absl::MutexLock lock(&mutex);
    return terminal_status;
  }
  std::optional<TaskState> State() {
    absl::MutexLock lock(&mutex);
    return terminal_state;
  }
  absl::Status LookupStatus() {
    absl::MutexLock lock(&mutex);
    return lookup_status;
  }
  absl::Status SelfWaitStatus() {
    absl::MutexLock lock(&mutex);
    return self_wait_status;
  }
};


struct RetirementObservation {
  std::atomic<bool> retired = false;
  std::atomic<bool> lookup_available = false;
  std::atomic<bool> active_during_retirement = false;
  std::atomic<bool> last_callback_during_retirement = false;
  std::atomic<int> wait_code = -1;
};

// This object is owned only by the real native callback's callable. Its real
// destructor is the capture-retirement oracle; no manual cleanup manufactures it.
struct RetiredCapture {
  RetiredCapture(ThreadedExecutionManager* owner, SessionId session, TaskId task,
                 std::shared_ptr<RetirementObservation> observation)
      : owner(owner), session(session), task(task),
        observation(std::move(observation)) {}
  ThreadedExecutionManager* owner;
  SessionId session;
  TaskId task;
  std::shared_ptr<RetirementObservation> observation;
  ~RetiredCapture() {
    observation->lookup_available =
        SchedulingLifetimeTestPeer::LookupAvailable(*owner);
    if (observation->lookup_available.load()) {
      observation->active_during_retirement =
          SchedulingLifetimeTestPeer::IsActive(*owner, session, task);
      observation->last_callback_during_retirement =
          SchedulingLifetimeTestPeer::State(*owner, task) ==
          TaskState::kLastCallbackQueued;
      observation->wait_code = static_cast<int>(
          owner->WaitUntilDone(task, absl::ZeroDuration()).code());
    }
    observation->retired = true;
  }
};

void ExpectReturnFence(ThreadedExecutionManager& owner, SessionId session,
                       TaskId task) {
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(owner, task),
            TaskState::kLastCallbackQueued);
  EXPECT_TRUE(SchedulingLifetimeTestPeer::IsActive(owner, session, task));
  EXPECT_EQ(owner.WaitUntilDone(task, absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(owner.WaitUntilSessionDone(session, absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
}

class FanoutLifetimeManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EXPECT_CALL(tokenizer_, GetVocabSize())
        .WillRepeatedly(::testing::Return(kVocabSize));
    EXPECT_CALL(tokenizer_, TokenIdsToText(::testing::_, false))
        .WillRepeatedly([](absl::Span<const int> ids, bool) {
          std::string result;
          for (int id : ids) result += std::to_string(id);
          return result;
        });
  }
  void TearDown() override {
    if (effect_ != nullptr) NotifyOnce(effect_->release);
    for (auto& probe : probes_) NotifyOnce(probe->release);
    if (owner_) EXPECT_OK(owner_->WaitUntilAllDone(kWatchdog));
    owner_.reset();  // Actual destructor drains/joins; no detached fallback.
  }
  void CreateOwner(bool hold_first, bool cancelled) {
    effect_ = std::make_shared<EffectControl>(hold_first);
    auto executor = std::make_unique<HeldFakeLlmExecutor>(effect_);
    if (!cancelled) executor->SetPrefillStatus(
        absl::InternalError("fixture: prefill failure"));
    ASSERT_OK_AND_ASSIGN(
        owner_, ThreadedExecutionManager::Create(
                    &tokenizer_, nullptr,
                    std::move(executor), nullptr,
                    nullptr, nullptr, nullptr, nullptr));
  }
  absl::StatusOr<SessionId> RegisterSession() {
    ABSL_ASSIGN_OR_RETURN(auto assets, ModelAssets::Create("fixture_model"));
    ABSL_ASSIGN_OR_RETURN(auto settings, EngineSettings::CreateDefault(assets));
    proto::LlmMetadata metadata;
    metadata.mutable_stop_tokens()->Add()->mutable_token_ids()->mutable_ids()
        ->Add(0);
    metadata.mutable_stop_tokens()->Add()->mutable_token_ids()->mutable_ids()
        ->Add(6);
    metadata.mutable_llm_model_type()->mutable_gemma3n();
    ABSL_RETURN_IF_ERROR(settings.MaybeUpdateAndValidate(&tokenizer_, &metadata));
    auto config = SessionConfig::CreateDefault();
    ABSL_RETURN_IF_ERROR(config.MaybeUpdateAndValidate(settings));
    config.SetUseExternalSampler(false);
    return owner_->RegisterNewSession(std::move(config), std::nullopt);
  }
  absl::Status AddPrefill(SessionId session, TaskId task,
                          absl::flat_hash_set<TaskId> dependencies,
                          std::shared_ptr<TerminalProbe> probe) {
    ABSL_ASSIGN_OR_RETURN(auto buffer,
                          tokenizer_.TokenIdsToTensorBuffer({1, 2, 3}));
    std::vector<InputData> inputs;
    inputs.emplace_back(InputText(std::move(buffer)));
    return owner_->AddPrefillTask(
        session, task, std::move(inputs), std::move(dependencies),
        std::make_shared<std::atomic<bool>>(false),
        [probe = std::move(probe)](absl::StatusOr<Responses> result) {
          probe->OnResponse(std::move(result));
        });
  }
  std::shared_ptr<TerminalProbe> Probe(SessionId session, bool hold = false) {
    auto probe = std::make_shared<TerminalProbe>(hold);
    probe->owner = owner_.get();
    probe->session_id = session;
    probes_.push_back(probe);
    return probe;
  }
  absl::Status DrainSession(SessionId session) {
    ABSL_RETURN_IF_ERROR(owner_->WaitUntilAllDone(kWatchdog));
    ABSL_RETURN_IF_ERROR(
        owner_->WaitUntilSessionDone(session, absl::ZeroDuration()));
    if (SchedulingLifetimeTestPeer::Active(*owner_, session) != 0) {
      return absl::InternalError("fixture: native active tasks remain after drain");
    }
    return absl::OkStatus();
  }
  std::vector<std::shared_ptr<TerminalProbe>> probes_;
  void HeldControl(bool cancelled) {
    CreateOwner(true, cancelled);
    ASSERT_NE(owner_, nullptr);
    ASSERT_OK_AND_ASSIGN(auto session, RegisterSession());
    ASSERT_OK_AND_ASSIGN(auto root, owner_->GetNewTaskId());
    ASSERT_OK_AND_ASSIGN(auto child, owner_->GetNewTaskId());
    auto parent_probe = Probe(session);
    auto child_probe = Probe(session, true);
    ASSERT_OK(AddPrefill(session, root, {}, parent_probe));
    ASSERT_TRUE(effect_->entered.WaitForNotificationWithTimeout(kWatchdog));
    ASSERT_OK(AddPrefill(session, child, {root}, child_probe));
    if (cancelled) ASSERT_OK(owner_->CancelAllTasksInSession(session));
    NotifyOnce(effect_->release);
    ASSERT_TRUE(child_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
    EXPECT_OK(child_probe->Status());
    EXPECT_EQ(child_probe->State(), cancelled ? TaskState::kDependentTaskCancelled
                                             : TaskState::kDependentTaskFailed);
    ASSERT_OK(child_probe->LookupStatus());
    EXPECT_EQ(child_probe->SelfWaitStatus().code(),
              absl::StatusCode::kDeadlineExceeded);
    ExpectReturnFence(*owner_, session, child);
    EXPECT_EQ(parent_probe->calls.load(), 0);
    EXPECT_EQ(effect_->prefill_calls.load(), 1);
    NotifyOnce(child_probe->release);
    ASSERT_OK(DrainSession(session));
    EXPECT_EQ(child_probe->calls.load(), 1);
    EXPECT_EQ(parent_probe->calls.load(), 1);
    EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, child),
              cancelled ? TaskState::kDependentTaskCancelled
                        : TaskState::kDependentTaskFailed);
    ASSERT_OK(owner_->ReleaseSession(session));
  }
  void DiamondControl(bool cancelled) {
    CreateOwner(true, cancelled);
    ASSERT_NE(owner_, nullptr);
    ASSERT_OK_AND_ASSIGN(auto session, RegisterSession());
    ASSERT_OK_AND_ASSIGN(auto a, owner_->GetNewTaskId());
    ASSERT_OK_AND_ASSIGN(auto b, owner_->GetNewTaskId());
    ASSERT_OK_AND_ASSIGN(auto c, owner_->GetNewTaskId());
    ASSERT_OK_AND_ASSIGN(auto d, owner_->GetNewTaskId());
    auto ap = Probe(session);
    auto bp = Probe(session);
    auto cp = Probe(session);
    auto dp = Probe(session, true);
    ASSERT_OK(AddPrefill(session, a, {}, ap));
    ASSERT_TRUE(effect_->entered.WaitForNotificationWithTimeout(kWatchdog));
    ASSERT_OK(AddPrefill(session, b, {a}, bp));
    ASSERT_OK(AddPrefill(session, c, {a}, cp));
    ASSERT_OK(AddPrefill(session, d, {b, c}, dp));
    if (cancelled) ASSERT_OK(owner_->CancelAllTasksInSession(session));
    NotifyOnce(effect_->release);
    ASSERT_TRUE(dp->entered.WaitForNotificationWithTimeout(kWatchdog));
    ASSERT_OK(dp->LookupStatus());
    ExpectReturnFence(*owner_, session, d);
    EXPECT_EQ(ap->calls.load(), 0);
    EXPECT_EQ(effect_->prefill_calls.load(), 1);
    NotifyOnce(dp->release);
    ASSERT_OK(DrainSession(session));
    const TaskState expected = cancelled ? TaskState::kDependentTaskCancelled
                                         : TaskState::kDependentTaskFailed;
    for (auto& probe : {bp, cp, dp}) {
      EXPECT_EQ(probe->calls.load(), 1);
      EXPECT_OK(probe->Status());
      EXPECT_EQ(probe->State(), expected);
      EXPECT_OK(probe->LookupStatus());
    }
    EXPECT_EQ(ap->calls.load(), 1);
    EXPECT_EQ(effect_->prefill_calls.load(), 1);
    ASSERT_OK(owner_->ReleaseSession(session));
  }
  void CaptureControl(bool cancelled) {
    CreateOwner(true, cancelled);
    ASSERT_NE(owner_, nullptr);
    ASSERT_OK_AND_ASSIGN(auto session, RegisterSession());
    ASSERT_OK_AND_ASSIGN(auto root, owner_->GetNewTaskId());
    ASSERT_OK_AND_ASSIGN(auto child, owner_->GetNewTaskId());
    auto parent_probe = Probe(session);
    auto child_probe = Probe(session);
    ASSERT_OK(AddPrefill(session, root, {}, parent_probe));
    ASSERT_TRUE(effect_->entered.WaitForNotificationWithTimeout(kWatchdog));
    auto observation = std::make_shared<RetirementObservation>();
    auto capture = std::make_unique<RetiredCapture>(
        owner_.get(), session, child, observation);
    ASSERT_OK_AND_ASSIGN(auto buffer,
                        tokenizer_.TokenIdsToTensorBuffer({1, 2, 3}));
    std::vector<InputData> inputs;
    inputs.emplace_back(InputText(std::move(buffer)));
    ASSERT_OK(owner_->AddPrefillTask(session, child, std::move(inputs), {root},
        std::make_shared<std::atomic<bool>>(false),
        [capture = std::move(capture), child_probe](absl::StatusOr<Responses> result) {
          child_probe->OnResponse(std::move(result));
        }));
    if (cancelled) ASSERT_OK(owner_->CancelAllTasksInSession(session));
    NotifyOnce(effect_->release);
    ASSERT_OK(DrainSession(session));
    EXPECT_EQ(child_probe->calls.load(), 1);
    EXPECT_OK(child_probe->Status());
    EXPECT_EQ(child_probe->State(), cancelled ? TaskState::kDependentTaskCancelled
                                             : TaskState::kDependentTaskFailed);
    EXPECT_EQ(parent_probe->calls.load(), 1);
    EXPECT_EQ(effect_->prefill_calls.load(), 1);
    EXPECT_TRUE(observation->retired.load());
    EXPECT_TRUE(observation->lookup_available.load());
    EXPECT_TRUE(observation->active_during_retirement.load());
    EXPECT_TRUE(observation->last_callback_during_retirement.load());
    EXPECT_EQ(observation->wait_code.load(),
              static_cast<int>(absl::StatusCode::kDeadlineExceeded));
    ASSERT_OK(owner_->ReleaseSession(session));
  }
  MockTokenizer tokenizer_;
  std::shared_ptr<EffectControl> effect_;
  std::unique_ptr<ThreadedExecutionManager> owner_;
};


TEST_F(FanoutLifetimeManagerTest, FailedDiamondTransfersEveryAcceptedOwnerOnce) {
  DiamondControl(false);
}
TEST_F(FanoutLifetimeManagerTest, CancelledDiamondTransfersEveryAcceptedOwnerOnce) {
  DiamondControl(true);
}
TEST_F(FanoutLifetimeManagerTest, FailedFanoutHandlerKeepsNativeReturnFence) {
  HeldControl(false);
}
TEST_F(FanoutLifetimeManagerTest, CancelledFanoutHandlerKeepsNativeReturnFence) {
  HeldControl(true);
}
TEST_F(FanoutLifetimeManagerTest, FailedCaptureRetiresBeforeNativeEndPublication) {
  CaptureControl(false);
}
TEST_F(FanoutLifetimeManagerTest, CancelledCaptureRetiresBeforeNativeEndPublication) {
  CaptureControl(true);
}

}  // namespace
}  // namespace litert::lm
