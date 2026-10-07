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
#include "runtime/framework/threadpool.h"
#include "runtime/framework/worker_thread.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/test_utils.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// This peer only installs per-instance worker-creation faults or inspects
// ownership under the real owners' mutexes. It never replaces Schedule,
// TrySchedule, RunWorker, QueueTask, FinishTask, or the manager implementation.
class SchedulingLifetimeTestPeer {
 public:
  using WorkerFactory = absl::AnyInvocable<
      absl::StatusOr<std::unique_ptr<WorkerThread>>(ThreadPool*,
                                                  const std::string&)>;
  static void InstallWorkerFactory(ThreadPool& pool, WorkerFactory factory) {
    absl::MutexLock lock(&pool.mutex_);
    pool.worker_factory_for_test_ = std::move(factory);
  }
  static void Stop(ThreadPool& pool) {
    absl::MutexLock lock(&pool.mutex_);
    pool.stopped_ = true;
  }
  static size_t Queued(ThreadPool& pool) {
    absl::MutexLock lock(&pool.mutex_);
    return pool.tasks_.size();
  }
  static bool WaitForStopped(ThreadPool& pool) {
    absl::MutexLock lock(&pool.mutex_);
    return pool.mutex_.AwaitWithTimeout(absl::Condition(&pool.stopped_),
                                        absl::Seconds(5));
  }
  static ThreadPool& ExecutionPool(ThreadedExecutionManager& owner) {
    return *owner.execution_thread_pool_;
  }
  static ThreadPool& CallbackPool(ThreadedExecutionManager& owner) {
    return *owner.callback_thread_pool_;
  }
  static bool HasTask(ThreadedExecutionManager& owner, TaskId id) {
    absl::MutexLock lock(&owner.session_and_task_lookup_mutex_);
    return owner.task_lookup_.contains(id);
  }
  static std::optional<TaskState> State(ThreadedExecutionManager& owner,
                                        TaskId id) {
    absl::MutexLock lock(&owner.session_and_task_lookup_mutex_);
    auto found = owner.task_lookup_.find(id);
    if (found == owner.task_lookup_.end()) return std::nullopt;
    return found->second.task_state;
  }
  static size_t Active(ThreadedExecutionManager& owner, SessionId id) {
    absl::MutexLock lock(&owner.session_and_task_lookup_mutex_);
    return owner.session_lookup_.at(id)->active_tasks.size();
  }
};

namespace {

constexpr int kVocabSize = 10;
const absl::Duration kWatchdog = absl::Seconds(5);

absl::Status RefusedWorker() {
  return absl::ResourceExhaustedError("fixture: native worker creation refused");
}

// No sleeps assert ordering. These notifications hold actual callbacks/workers.
// Cleanup releases a barrier on assertion failure; it does not make it pass.
void NotifyOnce(absl::Notification& notification) {
  if (!notification.HasBeenNotified()) notification.Notify();
}

class ReleaseOnExit {
 public:
  explicit ReleaseOnExit(absl::Notification& release) : release_(release) {}
  ~ReleaseOnExit() { NotifyOnce(release_); }
 private:
  absl::Notification& release_;
};

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

void FailFirstWorker(ThreadPool& pool) {
  SchedulingLifetimeTestPeer::InstallWorkerFactory(
      pool, [attempt = 0](ThreadPool* actual_pool,
                          const std::string& name) mutable
                -> absl::StatusOr<std::unique_ptr<WorkerThread>> {
        if (attempt++ == 0) return RefusedWorker();
        return WorkerThread::Create(actual_pool, name);
      });
}

TEST(SchedulingLifetimeThreadPoolTest,
     FirstWorkerRefusalRetainsCallableAndNeverEnqueuesIt) {
  ThreadPool pool("first-refusal", 1);
  FailFirstWorker(pool);
  auto refused_calls = std::make_shared<std::atomic<int>>(0);
  absl::AnyInvocable<void() &&> refused = [refused_calls] { ++*refused_calls; };
  EXPECT_EQ(pool.TrySchedule(refused), RefusedWorker());
  EXPECT_TRUE(refused != nullptr);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Queued(pool), 0);
  EXPECT_EQ(pool.num_threads(), 0);
  EXPECT_EQ(refused_calls->load(), 0);

  auto accepted_calls = std::make_shared<std::atomic<int>>(0);
  absl::AnyInvocable<void() &&> accepted = [accepted_calls] { ++*accepted_calls; };
  ASSERT_OK(pool.TrySchedule(accepted));
  EXPECT_TRUE(accepted == nullptr);
  ASSERT_OK(pool.WaitUntilDone(kWatchdog));
  EXPECT_EQ(accepted_calls->load(), 1);
  EXPECT_EQ(refused_calls->load(), 0);
  // A retained refusal remains caller-owned and is independently disposable.
  refused = nullptr;
}

TEST(SchedulingLifetimeThreadPoolTest, StoppedRefusalRetainsCallable) {
  ThreadPool pool("stopped-refusal", 1);
  SchedulingLifetimeTestPeer::Stop(pool);
  std::atomic<int> calls = 0;
  absl::AnyInvocable<void() &&> callback = [&] { ++calls; };
  EXPECT_EQ(pool.TrySchedule(callback).code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(callback != nullptr);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Queued(pool), 0);
  EXPECT_EQ(pool.num_threads(), 0);
  EXPECT_EQ(calls.load(), 0);
}

TEST(SchedulingLifetimeThreadPoolTest,
     ExtraWorkerRefusalStillAcceptsOnceOnExistingWorker) {
  ThreadPool pool("extra-refusal", 2);
  SchedulingLifetimeTestPeer::InstallWorkerFactory(
      pool, [attempt = 0](ThreadPool* actual_pool,
                          const std::string& name) mutable
                -> absl::StatusOr<std::unique_ptr<WorkerThread>> {
        if (attempt++ == 1) return RefusedWorker();
        return WorkerThread::Create(actual_pool, name);
      });
  struct HeldCalls {
    absl::Notification entered;
    absl::Notification release;
    std::atomic<int> calls = 0;
  };
  auto control = std::make_shared<HeldCalls>();
  ReleaseOnExit cleanup(control->release);
  ASSERT_OK(pool.Schedule([control] {
    ++control->calls;
    control->entered.Notify();
    control->release.WaitForNotification();
  }));
  ASSERT_TRUE(control->entered.WaitForNotificationWithTimeout(kWatchdog));
  absl::AnyInvocable<void() &&> accepted = [control] { ++control->calls; };
  ASSERT_OK(pool.TrySchedule(accepted));
  EXPECT_TRUE(accepted == nullptr);
  EXPECT_EQ(pool.num_threads(), 1);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Queued(pool), 1);
  EXPECT_EQ(control->calls.load(), 1);
  NotifyOnce(control->release);
  ASSERT_OK(pool.WaitUntilDone(kWatchdog));
  EXPECT_EQ(control->calls.load(), 2);
}

TEST(SchedulingLifetimeThreadPoolTest, DestructorJoinsHeldRealWorker) {
  auto pool = std::make_unique<ThreadPool>("join-held-worker", 1);
  absl::Notification entered;
  absl::Notification release;
  absl::Notification destroyed;
  ASSERT_OK(pool->Schedule([&] {
    entered.Notify();
    release.WaitForNotification();
  }));
  // Install cleanup before assertions so the real owner can always finish.
  ReleaseOnExit callback_cleanup(release);
  ASSERT_TRUE(entered.WaitForNotificationWithTimeout(kWatchdog));
  ThreadPool* actual_pool = pool.get();
  std::thread teardown([owned = std::move(pool), &destroyed]() mutable {
    owned.reset();
    destroyed.Notify();
  });
  JoinOnExit teardown_cleanup(teardown, release);
  ASSERT_TRUE(SchedulingLifetimeTestPeer::WaitForStopped(*actual_pool));
  // Destructor has entered its real stop/join path; the held worker still
  // owns the pool, so no memory is accessed after worker release.
  EXPECT_FALSE(destroyed.HasBeenNotified());
  NotifyOnce(release);
  teardown.join();
  EXPECT_TRUE(destroyed.HasBeenNotified());
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
    if (control_->prefill_calls.fetch_add(1) == 0 && control_->hold_first) {
      control_->entered.Notify();
      control_->release.WaitForNotification();
    }
    return FakeLlmExecutor::Prefill(inputs);
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
    absl::Status lookup;
    absl::Status self_wait;
    if (owner != nullptr) {
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

class SchedulingLifetimeManagerTest : public ::testing::Test {
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
    owner_.reset();  // Actual destructor drains/joins; no detached fallback.
  }
  void CreateOwner(bool hold_first = false) {
    effect_ = std::make_shared<EffectControl>(hold_first);
    ASSERT_OK_AND_ASSIGN(
        owner_, ThreadedExecutionManager::Create(
                    &tokenizer_, nullptr,
                    std::make_unique<HeldFakeLlmExecutor>(effect_), nullptr,
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
  absl::Status DrainSession(SessionId session) {
    ABSL_RETURN_IF_ERROR(owner_->WaitUntilAllDone(kWatchdog));
    ABSL_RETURN_IF_ERROR(
        owner_->WaitUntilSessionDone(session, absl::ZeroDuration()));
    if (SchedulingLifetimeTestPeer::Active(*owner_, session) != 0) {
      return absl::InternalError("fixture: native active tasks remain after drain");
    }
    return absl::OkStatus();
  }
  MockTokenizer tokenizer_;
  std::shared_ptr<EffectControl> effect_;
  std::unique_ptr<ThreadedExecutionManager> owner_;
};

TEST_F(SchedulingLifetimeManagerTest,
       InitialExecutionRefusalRollsBackWithoutTerminalCallback) {
  CreateOwner();
  ASSERT_NE(owner_, nullptr);
  FailFirstWorker(SchedulingLifetimeTestPeer::ExecutionPool(*owner_));
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId task, owner_->GetNewTaskId());
  auto rejected = std::make_shared<TerminalProbe>();
  EXPECT_EQ(AddPrefill(session, task, {}, rejected), RefusedWorker());
  EXPECT_EQ(rejected->calls.load(), 0);
  EXPECT_EQ(effect_->prefill_calls.load(), 0);
  EXPECT_FALSE(SchedulingLifetimeTestPeer::HasTask(*owner_, task));
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 0);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Queued(
                SchedulingLifetimeTestPeer::ExecutionPool(*owner_)), 0);
  ASSERT_OK(owner_->WaitUntilSessionDone(session, absl::ZeroDuration()));
  ASSERT_OK(owner_->ReleaseSession(session));

  // Same manager can create a real worker after the one-shot refusal. The
  // rejected task's callable/callback is never revived by that worker.
  ASSERT_OK_AND_ASSIGN(const SessionId next_session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId next_task, owner_->GetNewTaskId());
  auto accepted = std::make_shared<TerminalProbe>();
  ASSERT_OK(AddPrefill(next_session, next_task, {}, accepted));
  ASSERT_TRUE(accepted->entered.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(DrainSession(next_session));
  EXPECT_OK(accepted->Status());
  EXPECT_EQ(accepted->State(), TaskState::kDone);
  EXPECT_EQ(accepted->calls.load(), 1);
  EXPECT_EQ(rejected->calls.load(), 0);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  ASSERT_OK(owner_->ReleaseSession(next_session));
}

TEST_F(SchedulingLifetimeManagerTest,
       TerminalWorkerRefusalCallsOnceOutsideLockAndFencesReturn) {
  CreateOwner();
  ASSERT_NE(owner_, nullptr);
  FailFirstWorker(SchedulingLifetimeTestPeer::CallbackPool(*owner_));
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId task, owner_->GetNewTaskId());
  auto probe = std::make_shared<TerminalProbe>(true);
  probe->owner = owner_.get();
  probe->session_id = session;
  ReleaseOnExit cleanup(probe->release);
  ASSERT_OK(AddPrefill(session, task, {}, probe));
  ASSERT_TRUE(probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_EQ(probe->Status(), RefusedWorker());
  EXPECT_EQ(probe->calls.load(), 1);
  EXPECT_OK(probe->LookupStatus());
  EXPECT_EQ(probe->SelfWaitStatus().code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, task),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 1);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Queued(
                SchedulingLifetimeTestPeer::CallbackPool(*owner_)), 0);
  EXPECT_EQ(SchedulingLifetimeTestPeer::CallbackPool(*owner_).num_threads(), 0);
  EXPECT_EQ(owner_->WaitUntilSessionDone(session, absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
  NotifyOnce(probe->release);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, task), TaskState::kFailed);
  EXPECT_EQ(probe->calls.load(), 1);
  ASSERT_OK(owner_->ReleaseSession(session));

  ASSERT_OK_AND_ASSIGN(const SessionId next_session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId next_task, owner_->GetNewTaskId());
  auto recovered = std::make_shared<TerminalProbe>();
  ASSERT_OK(AddPrefill(next_session, next_task, {}, recovered));
  ASSERT_TRUE(recovered->entered.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(DrainSession(next_session));
  EXPECT_OK(recovered->Status());
  EXPECT_EQ(recovered->calls.load(), 1);
  EXPECT_EQ(probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 2);
  ASSERT_OK(owner_->ReleaseSession(next_session));
}

TEST_F(SchedulingLifetimeManagerTest,
       TerminalRefusalFencesQueuedDescendantsWithoutNativeEffects) {
  CreateOwner(true);
  ASSERT_NE(owner_, nullptr);
  FailFirstWorker(SchedulingLifetimeTestPeer::CallbackPool(*owner_));
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId a, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId b, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId c, owner_->GetNewTaskId());
  auto a_probe = std::make_shared<TerminalProbe>(true);
  auto b_probe = std::make_shared<TerminalProbe>(true);
  auto c_probe = std::make_shared<TerminalProbe>();
  b_probe->owner = owner_.get();
  b_probe->session_id = session;
  ReleaseOnExit release_a(a_probe->release);
  ReleaseOnExit release_b(b_probe->release);
  ReleaseOnExit release_effect(effect_->release);
  ASSERT_OK(AddPrefill(session, a, {}, a_probe));
  ASSERT_TRUE(effect_->entered.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(AddPrefill(session, b, {a}, b_probe));
  ASSERT_OK(AddPrefill(session, c, {b}, c_probe));
  NotifyOnce(effect_->release);
  ASSERT_TRUE(b_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_EQ(b_probe->Status(), RefusedWorker());
  EXPECT_OK(b_probe->LookupStatus());
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  EXPECT_EQ(owner_->WaitUntilSessionDone(session, absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
  NotifyOnce(b_probe->release);
  ASSERT_TRUE(a_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kDependentTaskFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, c),
            TaskState::kDependentTaskFailed);
  EXPECT_EQ(c_probe->Status(), RefusedWorker());
  EXPECT_EQ(a_probe->Status(), RefusedWorker());
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  NotifyOnce(a_probe->release);
  // Draining actual queued closures must preserve the refused end states and
  // never invoke their now-transferred callbacks or model/tool effects again.
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a), TaskState::kFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kDependentTaskFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, c),
            TaskState::kDependentTaskFailed);
  EXPECT_EQ(a_probe->calls.load(), 1);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(c_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  ASSERT_OK(owner_->ReleaseSession(session));
}

TEST_F(SchedulingLifetimeManagerTest,
       NewDependencyDuringRefusedCallbackCannotRunAfterItsFailure) {
  CreateOwner();
  ASSERT_NE(owner_, nullptr);
  FailFirstWorker(SchedulingLifetimeTestPeer::CallbackPool(*owner_));
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId a, owner_->GetNewTaskId());
  auto a_probe = std::make_shared<TerminalProbe>(true);
  ReleaseOnExit release_a(a_probe->release);
  ASSERT_OK(AddPrefill(session, a, {}, a_probe));
  ASSERT_TRUE(a_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_EQ(a_probe->Status(), RefusedWorker());
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a),
            TaskState::kLastCallbackQueued);

  // Submission occurs after a has reported its scheduling failure but before
  // a's handler returns. It must observe failure intent, not provisional
  // success from kLastCallbackQueued; no new model/tool effect may execute.
  ASSERT_OK_AND_ASSIGN(const TaskId b, owner_->GetNewTaskId());
  auto b_probe = std::make_shared<TerminalProbe>();
  ASSERT_OK(AddPrefill(session, b, {a}, b_probe));
  ASSERT_TRUE(b_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(b_probe->Status());
  EXPECT_EQ(b_probe->State(), TaskState::kDependentTaskFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kDependentTaskFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 1);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Queued(
                SchedulingLifetimeTestPeer::ExecutionPool(*owner_)), 0);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  NotifyOnce(a_probe->release);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(a_probe->calls.load(), 1);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a), TaskState::kFailed);
  ASSERT_OK(owner_->ReleaseSession(session));
}

TEST_F(SchedulingLifetimeManagerTest,
       DeferredExecutionRefusalCompletesAcceptedChainBeforePredecessor) {
  CreateOwner(true);
  ASSERT_NE(owner_, nullptr);
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId a, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId b, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId c, owner_->GetNewTaskId());
  auto a_probe = std::make_shared<TerminalProbe>(true);
  auto b_probe = std::make_shared<TerminalProbe>();
  auto c_probe = std::make_shared<TerminalProbe>();
  b_probe->owner = owner_.get();
  b_probe->session_id = session;
  ReleaseOnExit release_a(a_probe->release);
  ReleaseOnExit release_effect(effect_->release);
  ASSERT_OK(AddPrefill(session, a, {}, a_probe));
  ASSERT_TRUE(effect_->entered.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(AddPrefill(session, b, {a}, b_probe));
  ASSERT_OK(AddPrefill(session, c, {b}, c_probe));
  // Fault the actual stopped-before-enqueue path while a owns the sole real
  // execution worker. Do not restart this stopped pool or fake QueueTask.
  SchedulingLifetimeTestPeer::Stop(
      SchedulingLifetimeTestPeer::ExecutionPool(*owner_));
  NotifyOnce(effect_->release);
  ASSERT_TRUE(a_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(a_probe->Status());
  EXPECT_EQ(a_probe->State(), TaskState::kDone);
  EXPECT_EQ(b_probe->Status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(c_probe->Status(), b_probe->Status());
  EXPECT_OK(b_probe->LookupStatus());
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b), TaskState::kFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, c),
            TaskState::kDependentTaskFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(c_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  NotifyOnce(a_probe->release);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(a_probe->calls.load(), 1);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a), TaskState::kDone);
  ASSERT_OK(owner_->ReleaseSession(session));
}

TEST_F(SchedulingLifetimeManagerTest,
       CancellationDuringRefusedTerminalReturnDoesNotDuplicateCompletion) {
  CreateOwner();
  ASSERT_NE(owner_, nullptr);
  FailFirstWorker(SchedulingLifetimeTestPeer::CallbackPool(*owner_));
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId task, owner_->GetNewTaskId());
  auto probe = std::make_shared<TerminalProbe>(true);
  ReleaseOnExit release(probe->release);
  ASSERT_OK(AddPrefill(session, task, {}, probe));
  ASSERT_TRUE(probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(owner_->CancelAllTasksInSession(session));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, task),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(probe->Status(), RefusedWorker());
  EXPECT_EQ(probe->calls.load(), 1);
  NotifyOnce(probe->release);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(probe->calls.load(), 1);
  EXPECT_EQ(probe->Status(), RefusedWorker());
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, task), TaskState::kFailed);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  ASSERT_OK(owner_->ReleaseSession(session));
}

TEST_F(SchedulingLifetimeManagerTest,
       NormalHeldCallbackPreservesSuccessfulDependencyContinuation) {
  CreateOwner();
  ASSERT_NE(owner_, nullptr);
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId a, owner_->GetNewTaskId());
  auto a_probe = std::make_shared<TerminalProbe>(true);
  ReleaseOnExit release_a(a_probe->release);
  ASSERT_OK(AddPrefill(session, a, {}, a_probe));
  ASSERT_TRUE(a_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(a_probe->Status());
  EXPECT_EQ(a_probe->State(), TaskState::kDone);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a),
            TaskState::kLastCallbackQueued);
  ASSERT_OK_AND_ASSIGN(const TaskId b, owner_->GetNewTaskId());
  auto b_probe = std::make_shared<TerminalProbe>();
  ASSERT_OK(AddPrefill(session, b, {a}, b_probe));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b), TaskState::kQueued);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  NotifyOnce(a_probe->release);
  ASSERT_TRUE(b_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(DrainSession(session));
  EXPECT_OK(b_probe->Status());
  EXPECT_EQ(b_probe->State(), TaskState::kDone);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a), TaskState::kDone);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b), TaskState::kDone);
  EXPECT_EQ(a_probe->calls.load(), 1);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 2);
  ASSERT_OK(owner_->ReleaseSession(session));
}

TEST_F(SchedulingLifetimeManagerTest,
       NormalTerminalReturnFencesSessionAndAllowsSafeLookup) {
  CreateOwner();
  ASSERT_NE(owner_, nullptr);
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId task, owner_->GetNewTaskId());
  auto probe = std::make_shared<TerminalProbe>(true);
  probe->owner = owner_.get();
  probe->session_id = session;
  ReleaseOnExit release(probe->release);
  ASSERT_OK(AddPrefill(session, task, {}, probe));
  ASSERT_TRUE(probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(probe->Status());
  EXPECT_OK(probe->LookupStatus());
  EXPECT_EQ(probe->SelfWaitStatus().code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, task),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 1);
  EXPECT_EQ(owner_->WaitUntilSessionDone(session, absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
  NotifyOnce(probe->release);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, task), TaskState::kDone);
  EXPECT_EQ(probe->calls.load(), 1);
  ASSERT_OK(owner_->ReleaseSession(session));
}

TEST_F(SchedulingLifetimeManagerTest,
       DeferredRefusalDiamondCompletesEveryAcceptedOwnerOnce) {
  CreateOwner(true);
  ASSERT_NE(owner_, nullptr);
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId a, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId b, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId c, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId d, owner_->GetNewTaskId());
  auto a_probe = std::make_shared<TerminalProbe>(true);
  auto b_probe = std::make_shared<TerminalProbe>();
  auto c_probe = std::make_shared<TerminalProbe>();
  auto d_probe = std::make_shared<TerminalProbe>(true);
  d_probe->owner = owner_.get();
  d_probe->session_id = session;
  ReleaseOnExit release_a(a_probe->release);
  ReleaseOnExit release_d(d_probe->release);
  ReleaseOnExit release_effect(effect_->release);
  ASSERT_OK(AddPrefill(session, a, {}, a_probe));
  ASSERT_TRUE(effect_->entered.WaitForNotificationWithTimeout(kWatchdog));
  ASSERT_OK(AddPrefill(session, b, {a}, b_probe));
  ASSERT_OK(AddPrefill(session, c, {a}, c_probe));
  ASSERT_OK(AddPrefill(session, d, {b, c}, d_probe));
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 4);
  SchedulingLifetimeTestPeer::Stop(
      SchedulingLifetimeTestPeer::ExecutionPool(*owner_));
  NotifyOnce(effect_->release);

  // Both branch queue attempts refuse before enqueue. The shared descendant
  // must retain one owner as traversal reaches it from the second branch.
  // No assumption depends on flat_hash_set's B/C iteration order.
  ASSERT_TRUE(d_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_EQ(d_probe->Status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_OK(d_probe->LookupStatus());
  EXPECT_EQ(d_probe->calls.load(), 1);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, d),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, c),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 4);
  EXPECT_EQ(owner_->WaitUntilSessionDone(session, absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  NotifyOnce(d_probe->release);

  ASSERT_TRUE(a_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(a_probe->Status());
  EXPECT_EQ(a_probe->State(), TaskState::kDone);
  EXPECT_EQ(b_probe->Status(), d_probe->Status());
  EXPECT_EQ(c_probe->Status(), d_probe->Status());
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b), TaskState::kFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, c), TaskState::kFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, d),
            TaskState::kDependentTaskFailed);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(c_probe->calls.load(), 1);
  EXPECT_EQ(d_probe->calls.load(), 1);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  NotifyOnce(a_probe->release);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(a_probe->calls.load(), 1);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(c_probe->calls.load(), 1);
  EXPECT_EQ(d_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a), TaskState::kDone);
  ASSERT_OK(owner_->ReleaseSession(session));
}

TEST_F(SchedulingLifetimeManagerTest,
       PriorCancelledDescendantDoesNotOrphanLaterRefusedBranch) {
  CreateOwner(true);
  ASSERT_NE(owner_, nullptr);
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId c, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId a, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId b, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId d, owner_->GetNewTaskId());
  auto c_probe = std::make_shared<TerminalProbe>();
  auto a_probe = std::make_shared<TerminalProbe>(true);
  auto b_probe = std::make_shared<TerminalProbe>();
  auto d_probe = std::make_shared<TerminalProbe>();
  a_probe->owner = owner_.get();
  a_probe->session_id = session;
  auto cancel_c = std::make_shared<std::atomic<bool>>(false);
  ReleaseOnExit release_a(a_probe->release);
  ReleaseOnExit release_effect(effect_->release);
  ASSERT_OK_AND_ASSIGN(auto c_buffer,
                       tokenizer_.TokenIdsToTensorBuffer({1, 2, 3}));
  std::vector<InputData> c_inputs;
  c_inputs.emplace_back(InputText(std::move(c_buffer)));
  ASSERT_OK(owner_->AddPrefillTask(
      session, c, std::move(c_inputs), {}, cancel_c,
      [c_probe](absl::StatusOr<Responses> result) {
        c_probe->OnResponse(std::move(result));
      }));
  ASSERT_TRUE(effect_->entered.WaitForNotificationWithTimeout(kWatchdog));
  // A has already queued behind held C. B waits on A; D waits on B and C.
  ASSERT_OK(AddPrefill(session, a, {}, a_probe));
  ASSERT_OK(AddPrefill(session, b, {a}, b_probe));
  ASSERT_OK(AddPrefill(session, d, {b, c}, d_probe));
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 4);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Queued(
                SchedulingLifetimeTestPeer::ExecutionPool(*owner_)), 1);
  // Cancellation finishes D through the real prior completion owner. The
  // stopped pool still drains already queued A, then refuses B before enqueue.
  cancel_c->store(true);
  SchedulingLifetimeTestPeer::Stop(
      SchedulingLifetimeTestPeer::ExecutionPool(*owner_));
  NotifyOnce(effect_->release);

  ASSERT_TRUE(a_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(a_probe->Status());
  EXPECT_EQ(a_probe->State(), TaskState::kDone);
  EXPECT_OK(a_probe->LookupStatus());
  EXPECT_OK(c_probe->Status());
  EXPECT_EQ(c_probe->State(), TaskState::kCancelled);
  EXPECT_OK(d_probe->Status());
  EXPECT_EQ(d_probe->State(), TaskState::kDependentTaskCancelled);
  EXPECT_EQ(b_probe->Status(), absl::FailedPreconditionError(
      "ThreadPool 'execution_thread_pool' is stopped."));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, c),
            TaskState::kCancelled);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b), TaskState::kFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, d),
            TaskState::kDependentTaskCancelled);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 1);
  EXPECT_EQ(owner_->WaitUntilSessionDone(session, absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(c_probe->calls.load(), 1);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(d_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 2);
  NotifyOnce(a_probe->release);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(a_probe->calls.load(), 1);
  EXPECT_EQ(c_probe->calls.load(), 1);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(d_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 2);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a), TaskState::kDone);
  ASSERT_OK(owner_->ReleaseSession(session));
}

TEST_F(SchedulingLifetimeManagerTest,
       ImmediateFailedContinuationLookupAndReentrantSubmissionFenceReturn) {
  CreateOwner();
  ASSERT_NE(owner_, nullptr);
  FailFirstWorker(SchedulingLifetimeTestPeer::CallbackPool(*owner_));
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId a, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId b, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId c, owner_->GetNewTaskId());
  auto a_probe = std::make_shared<TerminalProbe>(true);
  auto b_probe = std::make_shared<TerminalProbe>(true);
  auto c_probe = std::make_shared<TerminalProbe>();
  b_probe->owner = owner_.get();
  c_probe->owner = owner_.get();
  b_probe->session_id = c_probe->session_id = session;
  ReleaseOnExit release_a(a_probe->release);
  ASSERT_OK(AddPrefill(session, a, {}, a_probe));
  ASSERT_TRUE(a_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_EQ(a_probe->Status(), RefusedWorker());
  ASSERT_OK_AND_ASSIGN(auto b_buffer,
                       tokenizer_.TokenIdsToTensorBuffer({1, 2, 3}));
  std::vector<InputData> b_inputs;
  b_inputs.emplace_back(InputText(std::move(b_buffer)));
  absl::Status submit_status;
  absl::Status reentrant_status;
  absl::Notification submit_returned;
  std::thread submit_b([&, inputs = std::move(b_inputs)]() mutable {
    submit_status = owner_->AddPrefillTask(
        session, b, std::move(inputs), {a},
        std::make_shared<std::atomic<bool>>(false),
        [&](absl::StatusOr<Responses> result) {
          if (result.ok() && IsTaskEndState(result->GetTaskState())) {
            // A real continuation is submitted by B's accepted terminal
            // handler. C's own immediate handler performs safe native lookup.
            reentrant_status = AddPrefill(session, c, {b}, c_probe);
          }
          b_probe->OnResponse(std::move(result));
        });
    submit_returned.Notify();
  });
  JoinOnExit join_submit(submit_b, b_probe->release);
  ASSERT_TRUE(b_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(reentrant_status);
  EXPECT_OK(b_probe->LookupStatus());
  EXPECT_EQ(b_probe->SelfWaitStatus().code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_OK(c_probe->LookupStatus());
  EXPECT_EQ(b_probe->State(), TaskState::kDependentTaskFailed);
  EXPECT_EQ(c_probe->State(), TaskState::kDependentTaskFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, c),
            TaskState::kDependentTaskFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 2);
  EXPECT_FALSE(submit_returned.HasBeenNotified());
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Queued(
                SchedulingLifetimeTestPeer::ExecutionPool(*owner_)), 0);
  NotifyOnce(b_probe->release);
  ASSERT_TRUE(submit_returned.WaitForNotificationWithTimeout(kWatchdog));
  submit_b.join();
  EXPECT_OK(submit_status);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kDependentTaskFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 1);
  NotifyOnce(a_probe->release);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(a_probe->calls.load(), 1);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(c_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  ASSERT_OK(owner_->ReleaseSession(session));
}

// Real FakeLlmExecutor mechanics remain intact. The second native Prefill
// boundary alone is held so a live predecessor can finish during B's handler.
class SecondHeldFakeLlmExecutor : public FakeLlmExecutor {
 public:
  using FakeLlmExecutor::Prefill;
  explicit SecondHeldFakeLlmExecutor(std::shared_ptr<EffectControl> control)
      : FakeLlmExecutor(kVocabSize,
                        {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}, {1, 2, 3}},
                        {{4}, {5}, {6}}),
        control_(std::move(control)) {}
  absl::Status Prefill(const ExecutorInputs& inputs) override {
    if (control_->prefill_calls.fetch_add(1) == 1) {
      control_->entered.Notify();
      control_->release.WaitForNotification();
    }
    return FakeLlmExecutor::Prefill(inputs);
  }
 private:
  std::shared_ptr<EffectControl> control_;
};

TEST_F(SchedulingLifetimeManagerTest,
       ImmediateTerminalChildDoesNotRemainLivePredecessorsWaitingFollower) {
  effect_ = std::make_shared<EffectControl>(false);
  ASSERT_OK_AND_ASSIGN(
      owner_, ThreadedExecutionManager::Create(
                  &tokenizer_, nullptr,
                  std::make_unique<SecondHeldFakeLlmExecutor>(effect_), nullptr,
                  nullptr, nullptr, nullptr, nullptr));
  FailFirstWorker(SchedulingLifetimeTestPeer::CallbackPool(*owner_));
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId a, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId p, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId b, owner_->GetNewTaskId());
  auto a_probe = std::make_shared<TerminalProbe>();
  auto p_probe = std::make_shared<TerminalProbe>();
  auto b_probe = std::make_shared<TerminalProbe>(true);
  b_probe->owner = owner_.get();
  b_probe->session_id = session;
  ReleaseOnExit release_effect(effect_->release);
  ASSERT_OK(AddPrefill(session, a, {}, a_probe));
  ASSERT_OK(owner_->WaitUntilDone(a, kWatchdog));
  EXPECT_EQ(a_probe->Status(), RefusedWorker());
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a), TaskState::kFailed);
  ASSERT_OK(AddPrefill(session, p, {}, p_probe));
  ASSERT_TRUE(effect_->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, p), TaskState::kProcessing);
  absl::Status submit_status;
  absl::Notification submit_returned;
  std::thread submit_b([&] {
    submit_status = AddPrefill(session, b, {a, p}, b_probe);
    submit_returned.Notify();
  });
  JoinOnExit join_submit(submit_b, b_probe->release);
  ASSERT_TRUE(b_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(b_probe->LookupStatus());
  EXPECT_EQ(b_probe->State(), TaskState::kDependentTaskFailed);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 2);
  // Finish actual P while B's own immediate terminal handler is still held.
  // P must retain its success instead of rejecting B's valid return fence.
  NotifyOnce(effect_->release);
  ASSERT_OK(owner_->WaitUntilDone(p, kWatchdog));
  EXPECT_OK(p_probe->Status());
  EXPECT_EQ(p_probe->State(), TaskState::kDone);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, p), TaskState::kDone);
  ASSERT_OK(SchedulingLifetimeTestPeer::ExecutionPool(*owner_)
                .WaitUntilDone(kWatchdog));
  ASSERT_OK(SchedulingLifetimeTestPeer::CallbackPool(*owner_)
                .WaitUntilDone(kWatchdog));
  EXPECT_EQ(owner_->WaitUntilAllDone(absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 1);
  EXPECT_EQ(owner_->WaitUntilSessionDone(session, absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
  EXPECT_FALSE(submit_returned.HasBeenNotified());
  EXPECT_EQ(effect_->prefill_calls.load(), 2);
  NotifyOnce(b_probe->release);
  ASSERT_TRUE(submit_returned.WaitForNotificationWithTimeout(kWatchdog));
  submit_b.join();
  EXPECT_OK(submit_status);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kDependentTaskFailed);
  EXPECT_EQ(a_probe->calls.load(), 1);
  EXPECT_EQ(p_probe->calls.load(), 1);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 2);
  ASSERT_OK(owner_->ReleaseSession(session));
}

TEST_F(SchedulingLifetimeManagerTest,
       ImmediateCancelledContinuationPreservesIntentUntilCallbackReturns) {
  CreateOwner(true);
  ASSERT_NE(owner_, nullptr);
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId a, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId b, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId c, owner_->GetNewTaskId());
  auto a_probe = std::make_shared<TerminalProbe>(true);
  auto b_probe = std::make_shared<TerminalProbe>(true);
  auto c_probe = std::make_shared<TerminalProbe>();
  a_probe->owner = b_probe->owner = c_probe->owner = owner_.get();
  a_probe->session_id = b_probe->session_id = c_probe->session_id = session;
  ReleaseOnExit release_a(a_probe->release);
  auto cancel_a = std::make_shared<std::atomic<bool>>(false);
  ReleaseOnExit release_effect(effect_->release);
  ASSERT_OK_AND_ASSIGN(auto buffer,
                       tokenizer_.TokenIdsToTensorBuffer({1, 2, 3}));
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText(std::move(buffer)));
  ASSERT_OK(owner_->AddPrefillTask(
      session, a, std::move(inputs), {}, cancel_a,
      [a_probe](absl::StatusOr<Responses> result) {
        a_probe->OnResponse(std::move(result));
      }));
  ASSERT_TRUE(effect_->entered.WaitForNotificationWithTimeout(kWatchdog));
  cancel_a->store(true);
  NotifyOnce(effect_->release);
  ASSERT_TRUE(a_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_EQ(a_probe->State(), TaskState::kCancelled);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a),
            TaskState::kLastCallbackQueued);
  absl::Status submit_status;
  absl::Notification submit_returned;
  std::thread submit_b([&] {
    submit_status = AddPrefill(session, b, {a}, b_probe);
    submit_returned.Notify();
  });
  JoinOnExit join_submit(submit_b, b_probe->release);
  ASSERT_TRUE(b_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(b_probe->LookupStatus());
  EXPECT_EQ(b_probe->State(), TaskState::kDependentTaskCancelled);
  ASSERT_OK(AddPrefill(session, c, {b}, c_probe));
  EXPECT_OK(c_probe->LookupStatus());
  EXPECT_EQ(c_probe->State(), TaskState::kDependentTaskCancelled);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, c),
            TaskState::kDependentTaskCancelled);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 2);
  NotifyOnce(a_probe->release);
  ASSERT_OK(owner_->WaitUntilDone(a, kWatchdog));
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 1);
  ASSERT_OK(SchedulingLifetimeTestPeer::ExecutionPool(*owner_)
                .WaitUntilDone(kWatchdog));
  ASSERT_OK(SchedulingLifetimeTestPeer::CallbackPool(*owner_)
                .WaitUntilDone(kWatchdog));
  EXPECT_EQ(owner_->WaitUntilAllDone(absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  NotifyOnce(b_probe->release);
  ASSERT_TRUE(submit_returned.WaitForNotificationWithTimeout(kWatchdog));
  submit_b.join();
  EXPECT_OK(submit_status);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kDependentTaskCancelled);
  EXPECT_EQ(a_probe->calls.load(), 1);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(c_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  ASSERT_OK(owner_->ReleaseSession(session));
}

TEST_F(SchedulingLifetimeManagerTest,
       ImmediateMaxTokenContinuationPreservesIntentUntilCallbackReturns) {
  effect_ = std::make_shared<EffectControl>(false);
  auto executor = std::make_unique<HeldFakeLlmExecutor>(effect_);
  auto settings = executor->GetMutableExecutorSettings();
  ASSERT_OK(settings);
  settings.value()->SetMaxNumTokens(4);
  ASSERT_OK_AND_ASSIGN(
      owner_, ThreadedExecutionManager::Create(
                  &tokenizer_, nullptr, std::move(executor), nullptr,
                  nullptr, nullptr, nullptr, nullptr));
  ASSERT_OK_AND_ASSIGN(const SessionId session, RegisterSession());
  ASSERT_OK_AND_ASSIGN(const TaskId prefill, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId a, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId b, owner_->GetNewTaskId());
  ASSERT_OK_AND_ASSIGN(const TaskId c, owner_->GetNewTaskId());
  auto prefill_probe = std::make_shared<TerminalProbe>();
  auto a_probe = std::make_shared<TerminalProbe>(true);
  auto b_probe = std::make_shared<TerminalProbe>(true);
  auto c_probe = std::make_shared<TerminalProbe>();
  a_probe->owner = b_probe->owner = c_probe->owner = owner_.get();
  a_probe->session_id = b_probe->session_id = c_probe->session_id = session;
  ReleaseOnExit release_a(a_probe->release);
  ASSERT_OK(AddPrefill(session, prefill, {}, prefill_probe));
  ASSERT_OK(owner_->WaitUntilDone(prefill, kWatchdog));
  ASSERT_OK(owner_->AddDecodeTask(
      session, a, {}, RepetitionPenaltyConfig::Default(),
      NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(), nullptr,
      std::make_shared<std::atomic<bool>>(false),
      [a_probe](absl::StatusOr<Responses> result) {
        a_probe->OnResponse(std::move(result));
      }, 1));
  ASSERT_TRUE(a_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(a_probe->Status());
  EXPECT_EQ(a_probe->State(), TaskState::kMaxNumTokensReached);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, a),
            TaskState::kLastCallbackQueued);
  absl::Status submit_status;
  absl::Notification submit_returned;
  std::thread submit_b([&] {
    submit_status = AddPrefill(session, b, {a}, b_probe);
    submit_returned.Notify();
  });
  JoinOnExit join_submit(submit_b, b_probe->release);
  ASSERT_TRUE(b_probe->entered.WaitForNotificationWithTimeout(kWatchdog));
  EXPECT_OK(b_probe->LookupStatus());
  EXPECT_EQ(b_probe->State(), TaskState::kMaxNumTokensReached);
  ASSERT_OK(AddPrefill(session, c, {b}, c_probe));
  EXPECT_OK(c_probe->LookupStatus());
  EXPECT_EQ(c_probe->State(), TaskState::kMaxNumTokensReached);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kLastCallbackQueued);
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, c),
            TaskState::kMaxNumTokensReached);
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 2);
  NotifyOnce(a_probe->release);
  ASSERT_OK(owner_->WaitUntilDone(a, kWatchdog));
  EXPECT_EQ(SchedulingLifetimeTestPeer::Active(*owner_, session), 1);
  ASSERT_OK(SchedulingLifetimeTestPeer::ExecutionPool(*owner_)
                .WaitUntilDone(kWatchdog));
  ASSERT_OK(SchedulingLifetimeTestPeer::CallbackPool(*owner_)
                .WaitUntilDone(kWatchdog));
  EXPECT_EQ(owner_->WaitUntilAllDone(absl::ZeroDuration()).code(),
            absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  NotifyOnce(b_probe->release);
  ASSERT_TRUE(submit_returned.WaitForNotificationWithTimeout(kWatchdog));
  submit_b.join();
  EXPECT_OK(submit_status);
  ASSERT_OK(DrainSession(session));
  EXPECT_EQ(SchedulingLifetimeTestPeer::State(*owner_, b),
            TaskState::kMaxNumTokensReached);
  EXPECT_EQ(prefill_probe->calls.load(), 1);
  EXPECT_EQ(a_probe->calls.load(), 1);
  EXPECT_EQ(b_probe->calls.load(), 1);
  EXPECT_EQ(c_probe->calls.load(), 1);
  EXPECT_EQ(effect_->prefill_calls.load(), 1);
  ASSERT_OK(owner_->ReleaseSession(session));
}

}  // namespace
}  // namespace litert::lm
