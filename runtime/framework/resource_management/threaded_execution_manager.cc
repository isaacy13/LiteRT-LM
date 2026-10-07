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

#include "runtime/framework/resource_management/threaded_execution_manager.h"

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/core/tasks.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/resource_manager.h"
#include "runtime/framework/threadpool.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/executor_data_util.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/tensor_buffer_util.h"

#if defined(LITERT_LM_DEBUGGER_ENABLED)
#include "runtime/util/runtime_debugger.h"
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)

namespace litert::lm {

ThreadedExecutionManager::ThreadedExecutionManager(
    Tokenizer* absl_nonnull tokenizer,
    std::unique_ptr<ResourceManager> absl_nonnull resource_manager,
    ::litert::Environment* absl_nullable litert_env,
    std::shared_ptr<RuntimeDebugger> absl_nullable runtime_debugger,
    DecodeInputBufferFactory buffer_factory)
    : ExecutionManager(std::move(buffer_factory)),
      tokenizer_(std::move(tokenizer)),
      resource_manager_(std::move(resource_manager)),
      litert_env_(litert_env),
      runtime_debugger_(std::move(runtime_debugger)) {
  execution_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"execution_thread_pool",
                                   /*max_num_threads=*/1);
  callback_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"callback_thread_pool",
                                   /*max_num_threads=*/1);
}

ThreadedExecutionManager::~ThreadedExecutionManager() {
  WaitUntilAllDone(Engine::kDefaultTimeout).IgnoreError();
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);

    if (!session_lookup_.empty()) {
      ABSL_LOG(ERROR) << "Not all sessions are released before the execution "
                         "manager is destroyed.";
    }

    absl::erase_if(task_lookup_, [](const auto& kv) {
      return IsTaskEndState(kv.second.task_state);
    });

    if (!task_lookup_.empty()) {
      ABSL_LOG(ERROR)
          << "Not all tasks are done before the execution manager is "
             "destroyed.";
      for (const auto& [task_id, task_info] : task_lookup_) {
        ABSL_LOG(ERROR) << "Task " << task_id
                        << " is not done. State: " << task_info.task_state;
      }
    }
  }

  // Workaround to avoid nonnull warning when releasing the resources.
  std::unique_ptr<ThreadPool> execution_thread_pool =
      std::move(execution_thread_pool_);
  execution_thread_pool.reset();

  std::unique_ptr<ThreadPool> callback_thread_pool =
      std::move(callback_thread_pool_);
  callback_thread_pool.reset();

  std::unique_ptr<ResourceManager> resource_manager =
      std::move(resource_manager_);
  resource_manager.reset();
}

absl::StatusOr<SessionId> ThreadedExecutionManager::RegisterNewSession(
    SessionConfig session_config, std::optional<BenchmarkInfo> benchmark_info) {
  ABSL_ASSIGN_OR_RETURN(
      auto context_handler,
      resource_manager_->CreateContextHandler(session_config));
  std::unique_ptr<Sampler> sampler;
  if (session_config.UseExternalSampler()) {
    if (session_config.GetSamplerBackend() != Backend::CPU) {
      return absl::InvalidArgumentError(
          "External sampler currently only supports CPU backend.");
    }
    ABSL_ASSIGN_OR_RETURN(
        sampler,
        CreateSampler(
            session_config.GetSamplerBackend(),
            session_config.GetNumOutputCandidates(),
            session_config.GetSamplerParams(),
            litert_env_ == nullptr
                ? std::nullopt
                : std::optional<
                      std::reference_wrapper<const ::litert::Environment>>(
                      *litert_env_)));
  }
  auto stop_token_detector = std::make_unique<StopTokenDetector>(1);
  for (const auto& stop_token_sequence : session_config.GetStopTokenIds()) {
    auto status =
        stop_token_detector->AddStopTokenSequence(stop_token_sequence);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to add stop token sequence: " << status;
    }
  }
  SessionId session_id = next_session_id_.fetch_add(1);
  auto session_info = std::make_shared<SessionInfo>(SessionInfo{
      .session_config = std::move(session_config),
      .context_handler = std::move(context_handler),
      .sampler = std::move(sampler),
      .stop_token_detector = std::move(stop_token_detector),
      .benchmark_info = std::move(benchmark_info),
  });
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (session_lookup_.contains(session_id)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Session ", session_id, " already exists in session list."));
    }
    if (session_info->session_config.AudioModalityEnabled()) {
      ABSL_RETURN_IF_ERROR(resource_manager_->TryLoadingAudioExecutor());
    }
    if (session_info->session_config.VisionModalityEnabled()) {
      ABSL_RETURN_IF_ERROR(resource_manager_->TryLoadingVisionExecutor());
    }
    session_lookup_.insert({session_id, std::move(session_info)});
#if defined(LITERT_LM_DEBUGGER_ENABLED)
    if (runtime_debugger_ != nullptr) {
      runtime_debugger_->RegisterDebugSession(session_id);
    }
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)
  }
  return session_id;
}

absl::Status ThreadedExecutionManager::ReleaseSession(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  // If the session is the only session and it is audio modality enabled, we
  // need to reset the audio executor, so the streaming audio executor would
  // have a clean state for the next usage.
  if (session_lookup_.at(session_id)->session_config.AudioModalityEnabled() &&
      session_lookup_.size() == 1) {
    ABSL_ASSIGN_OR_RETURN(auto audio_executor,
                          resource_manager_->AcquireAudioExecutor());
    audio_executor->Reset().IgnoreError();
  }
  absl::erase_if(task_lookup_, [session_id](const auto& kv) {
    return kv.second.session_id == session_id;
  });
#if defined(LITERT_LM_DEBUGGER_ENABLED)
  if (runtime_debugger_ != nullptr) {
    runtime_debugger_->UnregisterDebugSession(session_id);
  }
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)
  session_lookup_.erase(session_id);
  if (session_lookup_.empty()) {
    resource_manager_->ResetCurrentHandler();
  }
  return absl::OkStatus();
}

absl::Status ThreadedExecutionManager::UpdateGpuEnableMetalResidencySet(
    bool enable_metal_residency_set) {
  return resource_manager_->UpdateGpuEnableMetalResidencySet(
      enable_metal_residency_set);
}

absl::Status ThreadedExecutionManager::CancelAllTasksInSession(
    SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  for (TaskId task_id : session_lookup_.at(session_id)->active_tasks) {
    task_lookup_.at(task_id).cancelled->store(true);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<const SessionInfo>>
ThreadedExecutionManager::GetSessionInfo(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  return session_lookup_.at(session_id);
}

absl::StatusOr<BenchmarkInfo*>
ThreadedExecutionManager::GetMutableBenchmarkInfo(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  if (!session_lookup_.at(session_id)->benchmark_info.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " does not have benchmark info."));
  }
  return &session_lookup_.at(session_id)->benchmark_info.value();
}

absl::StatusOr<TaskId> ThreadedExecutionManager::GetNewTaskId() {
  return next_task_id_.fetch_add(1);
}

absl::Status ThreadedExecutionManager::CreateTask(
    SessionId session_id, TaskId task_id,
    absl::AnyInvocable<void()> absl_nonnull task,
    absl::flat_hash_set<TaskId> dependent_tasks,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  TaskState task_state = TaskState::kCreated;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> immediate_callback;
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (!session_lookup_.contains(session_id)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Session ", session_id, " not found in session list. Task ", task_id,
          " cannot be created."));
    }
    if (task_lookup_.contains(task_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Task ", task_id, " already exists in task list."));
    }

    for (auto it = dependent_tasks.begin(); it != dependent_tasks.end();) {
      TaskId dep_task_id = *it;
      bool erase_dependency = false;
      auto task_it = task_lookup_.find(dep_task_id);
      if (task_it == task_lookup_.end()) {
        if (dep_task_id >= next_task_id_.load()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Dependency task ", dep_task_id, " is invalid."));
        }
        erase_dependency = true;
      } else {
        const TaskInfo& dep_task_info = task_it->second;
        TaskState dependency_state = dep_task_info.task_state;
        if (dependency_state == TaskState::kLastCallbackQueued &&
            dep_task_info.completion_state_after_callback.has_value()) {
          dependency_state = *dep_task_info.completion_state_after_callback;
        }
        if (IsTaskEndState(dependency_state)) {
          switch (dependency_state) {
            case TaskState::kFailed:
              ABSL_FALLTHROUGH_INTENDED;
            case TaskState::kDependentTaskFailed:
              task_state = TaskState::kDependentTaskFailed;
              break;
            case TaskState::kCancelled:
              ABSL_FALLTHROUGH_INTENDED;
            case TaskState::kDependentTaskCancelled:
              if (task_state != TaskState::kDependentTaskFailed) {
                task_state = TaskState::kDependentTaskCancelled;
              }
              break;
            case TaskState::kDone:
              break;
            case TaskState::kMaxNumTokensReached:
              if (task_state == TaskState::kCreated) {
                task_state = TaskState::kMaxNumTokensReached;
              }
              break;
            default:
              return absl::InvalidArgumentError(absl::StrCat(
                  "Dependency task ", dep_task_id, " is in end state ",
                  dependency_state, " but not in Done or Cancelled or Failed "
                                    "state."));
          }
          erase_dependency = true;
        } else if (dependency_state == TaskState::kLastCallbackQueued) {
          // Preserve normal successful callback continuations.
          erase_dependency = true;
        }
      }
      auto copy_it = it++;
      if (erase_dependency) dependent_tasks.erase(copy_it);
    }

    const bool immediate_terminal = IsTaskEndState(task_state);
    if (immediate_terminal) {
      // This completion has its own accepted callback owner. A live second
      // predecessor must not retain a waiting edge to its return fence.
      dependent_tasks.clear();
    } else {
      // Install reverse links only after all dependencies have validated and
      // only for a genuinely waiting Created task.
      for (TaskId dependency_id : dependent_tasks) {
        task_lookup_.at(dependency_id).following_tasks.insert(task_id);
      }
    }
    session_lookup_.at(session_id)->active_tasks.insert(task_id);
    TaskInfo task_info;
    task_info.session_id = session_id;
    task_info.task_state = immediate_terminal ? TaskState::kLastCallbackQueued
                                             : task_state;
    if (immediate_terminal) {
      task_info.completion_state_after_callback = task_state;
    }
    task_info.task = std::move(task);
    task_info.dependent_tasks = std::move(dependent_tasks);
    task_info.cancelled = cancelled;
    task_info.callback = std::move(callback);
    task_lookup_.insert({task_id, std::move(task_info)});

    if (immediate_terminal) {
      immediate_callback = std::move(task_lookup_.at(task_id).callback);
    } else {
      task_lookup_.at(task_id).callback(Responses(task_state));
      if (task_lookup_.at(task_id).dependent_tasks.empty()) {
        auto status = QueueTask(task_id);
        if (!status.ok()) {
          // Rejected initial start: no terminal callback or queued work; the
          // C caller still owns callback-data cleanup.
          session_lookup_.at(session_id)->active_tasks.erase(task_id);
          task_lookup_.erase(task_id);
          return status;
        }
      }
      return absl::OkStatus();
    }
  }

  // Accepted immediate-terminal callbacks run on this caller, outside the
  // lookup mutex, exactly once. Native waits retain the active return fence.
  immediate_callback(Responses(task_state));
  // Retire the caller-owned callable/captures before publishing its return
  // fence; unlike pooled callbacks, no worker join covers this local owner.
  immediate_callback = nullptr;
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    auto state_status = UpdateTaskState(task_id, task_state);
    if (!state_status.ok()) {
      ABSL_LOG(ERROR) << "Failed to finish accepted immediate-terminal task: "
                      << state_status << " with task id: " << task_id;
    }
  }
  // A delivered callback transferred callback-data ownership. A bookkeeping
  // failure cannot now be reported as a rejected start to the C caller.
  return absl::OkStatus();
}

absl::Status ThreadedExecutionManager::QueueTask(TaskId task_id) {
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
  if (task_lookup_.at(task_id).task_state != TaskState::kCreated) {
    auto error_status = absl::FailedPreconditionError(
        absl::StrCat("Task ", task_id, " is not in Created state."));
    task_lookup_.at(task_id).callback(error_status);
    return error_status;
  }
  if (!task_lookup_.at(task_id).dependent_tasks.empty()) {
    auto error_status = absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " has dependent tasks not finished."));
    task_lookup_.at(task_id).callback(error_status);
    return error_status;
  }

  auto task = std::move(task_lookup_.at(task_id).task);

  if (execution_thread_pool_ != nullptr) {
    ABSL_RETURN_IF_ERROR(execution_thread_pool_->Schedule(std::move(task)));
  } else {
    return absl::FailedPreconditionError("Execution thread pool is null.");
  }

  task_lookup_.at(task_id).callback(Responses(TaskState::kQueued));
  ABSL_RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kQueued));

  return absl::OkStatus();
}

absl::StatusOr<
    std::tuple<std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
ThreadedExecutionManager::StartTask(TaskId task_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
  // Refused dependent tasks may already have queued closures. Their
  // completion owner has finished them; dequeue those closures as no-ops.
  if (IsTaskEndState(task_lookup_.at(task_id).task_state)) {
    return std::make_tuple(nullptr, nullptr, nullptr);
  }
  if (task_lookup_.at(task_id).callback == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " has no callback."));
  }
  if (task_lookup_.at(task_id).task_state != TaskState::kQueued) {
    auto error_status = absl::FailedPreconditionError(
        absl::StrCat("Task ", task_id, " is not in Queued state."));
    task_lookup_.at(task_id).callback(error_status);
    return error_status;
  }
  task_lookup_.at(task_id).callback(Responses(TaskState::kProcessing));
  ABSL_RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kProcessing));

  if (!session_lookup_.contains(task_lookup_.at(task_id).session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", task_lookup_.at(task_id).session_id,
                     " not found in session list."));
  }
  std::shared_ptr<SessionInfo> session_info =
      session_lookup_.at(task_lookup_.at(task_id).session_id);
  return std::make_tuple(session_info, task_lookup_.at(task_id).cancelled,
                         std::move(task_lookup_.at(task_id).callback));
}

absl::Status ThreadedExecutionManager::FinishTask(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  absl::AnyInvocable<void() &&> terminal_callback;
  bool terminal_callback_scheduled = false;
  absl::Status terminal_schedule_status;
  std::vector<absl::AnyInvocable<void() &&>> failed_dependent_callbacks;
  absl::flat_hash_set<TaskId> deferred_failed_task_ids;
  std::shared_ptr<absl::Notification> failed_dependencies_drained;
  struct FailureDrain {
    std::vector<absl::AnyInvocable<void() &&>>& callbacks;
    std::shared_ptr<absl::Notification>& signal;
    bool drained = false;
    void Run() {
      if (drained) return;
      drained = true;
      for (auto& callback : callbacks) std::move(callback)();
      if (signal != nullptr) signal->Notify();
    }
    // Every return leaves the lookup-lock scope before this owner is destroyed.
    ~FailureDrain() { Run(); }
  } failure_drain{failed_dependent_callbacks, failed_dependencies_drained};
  auto defer_failed_task =
      [&](TaskId failed_id, absl::Status status, TaskState end_state)
          ABSL_EXCLUSIVE_LOCKS_REQUIRED(session_and_task_lookup_mutex_)
              -> absl::Status {
    if (deferred_failed_task_ids.contains(failed_id)) {
      return absl::OkStatus();
    }
    auto found = task_lookup_.find(failed_id);
    if (found == task_lookup_.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Refused dependent task ", failed_id, " not found."));
    }
    auto& failed = found->second;
    if ((failed.task_state != TaskState::kCreated &&
         failed.task_state != TaskState::kQueued) ||
        failed.callback == nullptr) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Refused dependent task ", failed_id,
          " does not own an unstarted callback."));
    }
    // Perform the only fallible state transition before moving the callable.
    // The lookup mutex keeps this exact record present throughout the transfer.
    ABSL_RETURN_IF_ERROR(
        UpdateTaskState(failed_id, TaskState::kLastCallbackQueued));
    failed.cancelled->store(true);
    failed.dependent_tasks.clear();
    failed.completion_state_after_callback = end_state;
    auto failed_callback = std::move(failed.callback);
    failed_dependent_callbacks.emplace_back(
        [failed_callback = std::move(failed_callback), failed_id, status,
         end_state, this]() mutable {
          failed_callback(status);
          absl::MutexLock lock(session_and_task_lookup_mutex_);
          auto state_status = UpdateTaskState(failed_id, end_state);
          if (!state_status.ok()) {
            ABSL_LOG(ERROR) << "Failed to finish refused dependent task: "
                            << state_status;
          }
        });
    // This ID now denotes a callable retained by this completion owner's
    // FailureDrain, not merely a visited graph node.
    deferred_failed_task_ids.insert(failed_id);
    return absl::OkStatus();
  };
  auto invoke_callback_and_return =
      [&](absl::Status status) ABSL_EXCLUSIVE_LOCKS_REQUIRED(
          session_and_task_lookup_mutex_) -> absl::Status {
    callback(status);
    ABSL_RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kFailed));
    return status;
  };
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (!task_lookup_.contains(task_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Task ", task_id, " not found in task list."));
    }
    if (task_lookup_.at(task_id).task_state != TaskState::kProcessing) {
      auto error_status = absl::FailedPreconditionError(
          absl::StrCat("Task ", task_id, " is not in Processing state."));
      return invoke_callback_and_return(error_status);
    }
    if (!responses.ok() || responses->GetTaskState() == TaskState::kCancelled) {
      auto following_waiting_tasks = FollowingWaitingTasks(task_id);
      if (!following_waiting_tasks.ok()) {
        return invoke_callback_and_return(following_waiting_tasks.status());
      }
      auto status = UpdateAllTasksToState(
          following_waiting_tasks.value(),
          responses.ok() ? TaskState::kDependentTaskCancelled
                         : TaskState::kDependentTaskFailed);
      if (!status.ok()) {
        return invoke_callback_and_return(status);
      }
    } else if (responses->GetTaskState() == TaskState::kDone ||
               responses->GetTaskState() == TaskState::kMaxNumTokensReached) {
      for (TaskId following_task_id :
           task_lookup_.at(task_id).following_tasks) {
        if (!task_lookup_.contains(following_task_id)) {
          auto error_status = absl::InvalidArgumentError(
              absl::StrCat("Following task ", following_task_id,
                           " not found in task list."));
          return invoke_callback_and_return(error_status);
        }
        if (IsTaskEndState(task_lookup_.at(following_task_id).task_state) ||
            deferred_failed_task_ids.contains(following_task_id)) {
          continue;
        }
        if (task_lookup_.at(following_task_id).task_state !=
            TaskState::kCreated) {
          auto error_status = absl::InvalidArgumentError(
              absl::StrCat("Following task ", following_task_id,
                           " is not in Created state. Task state: ",
                           task_lookup_.at(following_task_id).task_state));
          return invoke_callback_and_return(error_status);
        }
        if (!task_lookup_.at(following_task_id)
                 .dependent_tasks.contains(task_id)) {
          auto error_status = absl::InvalidArgumentError(
              absl::StrCat("Following task ", following_task_id,
                           " does not depend on task ", task_id));
          return invoke_callback_and_return(error_status);
        }
        task_lookup_.at(following_task_id).dependent_tasks.erase(task_id);
        if (task_lookup_.at(following_task_id).dependent_tasks.empty()) {
          auto status = QueueTask(following_task_id);
          if (!status.ok()) {
            // This task's start was already accepted while it waited on a
            // dependency. Refusal must finish that callback chain once rather
            // than abandon it or report failure only to its predecessor.
            ABSL_ASSIGN_OR_RETURN(
                auto dependent_tasks,
                FollowingWaitingTasks(following_task_id,
                                      &deferred_failed_task_ids));
            for (TaskId dependent_id : dependent_tasks) {
              ABSL_RETURN_IF_ERROR(defer_failed_task(
                  dependent_id, status, TaskState::kDependentTaskFailed));
            }
            ABSL_RETURN_IF_ERROR(defer_failed_task(
                following_task_id, status, TaskState::kFailed));
          }
        }
      }
    } else if (!IsTaskEndState(responses->GetTaskState())) {
      return invoke_callback_and_return(absl::InvalidArgumentError(absl::StrCat(
          "Expected task state for responses to be end state, but got ",
          responses->GetTaskState())));
    }

    // The result stays with the single completion callable. TrySchedule only
    // consumes that callable on success; a refused start cannot race the result
    // replacement below or invoke another copy of the completion.
    auto terminal_result =
        std::make_shared<absl::StatusOr<Responses>>(std::move(responses));
    // The return fence does not turn a known failure/cancellation/token limit
    // into provisional success. Done still permits normal continuations.
    task_lookup_.at(task_id).completion_state_after_callback =
        terminal_result->ok() ? (*terminal_result)->GetTaskState()
                              : TaskState::kFailed;
    if (!failed_dependent_callbacks.empty()) {
      failed_dependencies_drained = std::make_shared<absl::Notification>();
    }
    terminal_callback =
        [callback = std::move(callback), terminal_result,
         failed_dependencies_drained, task_id, this]() mutable {
          // Dependent refusal handlers run first on the existing owner outside
          // its lookup lock; do not race a shared native callback-state owner.
          if (failed_dependencies_drained != nullptr) {
            failed_dependencies_drained->WaitForNotification();
          }
          TaskState next_task_state = terminal_result->ok()
                                          ? (*terminal_result)->GetTaskState()
                                          : TaskState::kFailed;
          callback(std::move(*terminal_result));
          absl::MutexLock lock(session_and_task_lookup_mutex_);
          auto status = UpdateTaskState(task_id, next_task_state);
          if (!status.ok()) {
            ABSL_LOG(ERROR) << "Failed to update task state: " << status
                            << " with task id: " << task_id;
          }
        };
    terminal_schedule_status =
        callback_thread_pool_ != nullptr
            ? callback_thread_pool_->TrySchedule(terminal_callback)
            : absl::InternalError("Callback thread pool is null.");
    terminal_callback_scheduled = terminal_schedule_status.ok();
    if (!terminal_callback_scheduled) {
      task_lookup_.at(task_id).completion_state_after_callback =
          TaskState::kFailed;
      *terminal_result = terminal_schedule_status;
      // A successful decode may already have made waiting tasks ready above.
      // This pool has one execution owner, so they have not started yet. Fence
      // their effects before returning the terminal scheduling failure. Queued
      // closures may still be dequeued, but StartTask skips their end state.
      absl::flat_hash_set<TaskId> visited;
      std::vector<TaskId> pending(
          task_lookup_.at(task_id).following_tasks.begin(),
          task_lookup_.at(task_id).following_tasks.end());
      while (!pending.empty()) {
        TaskId dependent_id = pending.back();
        pending.pop_back();
        if (!visited.insert(dependent_id).second ||
            !task_lookup_.contains(dependent_id)) {
          continue;
        }
        auto& dependent = task_lookup_.at(dependent_id);
        if (IsTaskEndState(dependent.task_state)) continue;
        pending.insert(pending.end(), dependent.following_tasks.begin(),
                       dependent.following_tasks.end());
        dependent.cancelled->store(true);
        if (dependent.task_state != TaskState::kCreated &&
            dependent.task_state != TaskState::kQueued) {
          // An already running callback owns its own completion; do not copy it.
          continue;
        }
        ABSL_RETURN_IF_ERROR(defer_failed_task(
            dependent_id, terminal_schedule_status,
            TaskState::kDependentTaskFailed));
      }
    }
    // Keep the task active until its entire callback returns. Publishing this
    // intermediate state also permits a normal continuation from that callback.
    ABSL_RETURN_IF_ERROR(
        UpdateTaskState(task_id, TaskState::kLastCallbackQueued));
  }

  failure_drain.Run();

  if (!terminal_callback_scheduled) {
    // TrySchedule refusal retains the callable and has not enqueued it. Run
    // this accepted task's terminal scheduling error once, outside the lookup
    // mutex. It marks kFailed after callback return, suppressing tool work.
    // No new worker or task is created for the refused completion.
    std::move(terminal_callback)();
    return terminal_schedule_status;
  }

  if (callback_thread_pool_ != nullptr) {
    // TODO b/476205457 - Consider to use a asynchronous approach to handle the
    // callback, and remove this WaitUntilDone.
    ABSL_RETURN_IF_ERROR(
        callback_thread_pool_->WaitUntilDone(absl::Seconds(10)));
  }

  return absl::OkStatus();
}

void ThreadedExecutionManager::FinishTaskAndLogErrors(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  auto status = FinishTask(task_id, std::move(responses), std::move(callback));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to finish task: " << status
                    << " with task id: " << task_id;
  }
}

absl::StatusOr<absl::flat_hash_set<TaskId>>
ThreadedExecutionManager::FollowingWaitingTasks(
    TaskId task_id,
    const absl::flat_hash_set<TaskId>* completion_owned_tasks) {
  absl::flat_hash_set<TaskId> following_waiting_tasks;
  for (TaskId following_task_id : task_lookup_.at(task_id).following_tasks) {
    if (!task_lookup_.contains(following_task_id)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Following task ", following_task_id, " not found in task list."));
    }
    // A different completion owner may already have cancelled/failed this
    // descendant and cleared its dependency edges. Its authoritative terminal
    // state needs neither traversal nor another callback/state mutation.
    if (IsTaskEndState(task_lookup_.at(following_task_id).task_state)) {
      continue;
    }
    if (completion_owned_tasks != nullptr &&
        completion_owned_tasks->contains(following_task_id)) {
      // A diamond can reach an accepted descendant already transferred by
      // another refused branch in this same FinishTask call. Its dependency
      // edges were cleared when its sole callback owner was retained. Skip
      // only that owner, with its active/return-fence state still verified.
      const auto& owned = task_lookup_.at(following_task_id);
      auto session = session_lookup_.find(owned.session_id);
      if (owned.task_state != TaskState::kLastCallbackQueued ||
          owned.callback != nullptr ||
          (owned.completion_state_after_callback != TaskState::kFailed &&
           owned.completion_state_after_callback !=
               TaskState::kDependentTaskFailed) ||
          session == session_lookup_.end() ||
          !session->second->active_tasks.contains(following_task_id)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Completion-owned task ", following_task_id,
            " no longer has its retained callback return fence."));
      }
      continue;
    }
    if (!task_lookup_.at(following_task_id).dependent_tasks.contains(task_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Following task ", following_task_id,
                       " does not depend on task ", task_id));
    }
    following_waiting_tasks.insert(following_task_id);
    ABSL_ASSIGN_OR_RETURN(
        auto next_following_waiting_tasks,
        FollowingWaitingTasks(following_task_id, completion_owned_tasks));
    following_waiting_tasks.insert(next_following_waiting_tasks.begin(),
                                   next_following_waiting_tasks.end());
  }
  return following_waiting_tasks;
}

absl::Status ThreadedExecutionManager::UpdateTaskState(TaskId task_id,
                                                       TaskState task_state) {
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
  if (!IsTaskEndState(task_lookup_.at(task_id).task_state) &&
      IsTaskEndState(task_state)) {
    SessionId session_id = task_lookup_.at(task_id).session_id;
    if (session_lookup_.contains(session_id) &&
        session_lookup_.at(session_id)->active_tasks.contains(task_id)) {
      session_lookup_.at(task_lookup_.at(task_id).session_id)
          ->active_tasks.erase(task_id);
    } else {
      auto error_status = absl::InternalError(absl::StrCat(
          "Task ", task_id, " is not in active tasks of session ", session_id));
      if (task_lookup_.at(task_id).callback != nullptr) {
        task_lookup_.at(task_id).callback(error_status);
      }
      return error_status;
    }
  }
  task_lookup_.at(task_id).task_state = task_state;
  if (IsTaskEndState(task_state)) {
    task_lookup_.at(task_id).completion_state_after_callback.reset();
  }
  return absl::OkStatus();
}

absl::Status ThreadedExecutionManager::UpdateAllTasksToState(
    const absl::flat_hash_set<TaskId>& task_ids, TaskState task_state) {
  for (TaskId task_id : task_ids) {
    task_lookup_.at(task_id).dependent_tasks.clear();
    if (task_lookup_.at(task_id).callback) {
      task_lookup_.at(task_id).callback(Responses(task_state));
    }
    ABSL_RETURN_IF_ERROR(UpdateTaskState(task_id, task_state));
  }
  return absl::OkStatus();
}

absl::StatusOr<ExecutorInputs>
ThreadedExecutionManager::ProcessAndCombineContents(
    const std::vector<InputData>& preprocessed_contents,
    std::optional<BenchmarkInfo>& benchmark_info) {
  std::vector<int> combined_token_ids;
  std::vector<ExecutorVisionData> all_image_data;
  std::vector<ExecutorAudioData> all_audio_data;
  for (const auto& preprocessed_content : preprocessed_contents) {
    if (const auto* input_text =
            std::get_if<InputText>(&preprocessed_content)) {
      ABSL_ASSIGN_OR_RETURN(const auto* token_ids,
                            input_text->GetPreprocessedTextTensor());
      if (token_ids == nullptr) {
        return absl::InvalidArgumentError(
            "Token IDs is null in preprocessed_contents.");
      }
      LITERT_ASSIGN_OR_RETURN(auto ids_buffer_span,
                              ReferTensorBufferAsSpan<int>(*token_ids));
      combined_token_ids.insert(combined_token_ids.end(),
                                ids_buffer_span.begin(), ids_buffer_span.end());
    } else if (const auto* input_image =
                   std::get_if<InputImage>(&preprocessed_content)) {
      if (benchmark_info.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("vision_executor"));
      }
      ExecutorVisionData single_image_data;
      if (input_image->IsTensorBuffer()) {
        ABSL_ASSIGN_OR_RETURN(auto tensor_buffer,
                              input_image->GetPreprocessedImageTensor());
        ABSL_ASSIGN_OR_RETURN(auto vision_executor,
                              resource_manager_->AcquireVisionExecutor());
        ABSL_ASSIGN_OR_RETURN(single_image_data,
                              vision_executor->Encode(*tensor_buffer));
      } else if (input_image->IsTensorBufferMap()) {
        ABSL_ASSIGN_OR_RETURN(auto tensor_buffer_map,
                              input_image->GetPreprocessedImageTensorMap());
        ABSL_ASSIGN_OR_RETURN(auto vision_executor,
                              resource_manager_->AcquireVisionExecutor());
        ABSL_ASSIGN_OR_RETURN(single_image_data,
                              vision_executor->Encode(*tensor_buffer_map));
      } else {
        return absl::FailedPreconditionError(
            "Image tensor or tensor map is null in preprocessed_contents.");
      }
      if (benchmark_info.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("vision_executor"));
      }
      ABSL_ASSIGN_OR_RETURN(auto embeddings_ptr,
                            single_image_data.GetEmbeddingsPtr());
      ABSL_ASSIGN_OR_RETURN(const auto& dimensions,
                            TensorBufferDims(*embeddings_ptr));
      // The last two dimensions are [..., image_token_num, model_dimension].
      const int image_token_num = dimensions.at(dimensions.size() - 2);
      combined_token_ids.insert(combined_token_ids.end(), image_token_num,
                                ExecutorVisionData::kSpecialToken);
      all_image_data.push_back(std::move(single_image_data));
    } else if (const auto* input_image_end =
                   std::get_if<InputImageEnd>(&preprocessed_content)) {
      combined_token_ids.push_back(ExecutorVisionData::kEndToken);
    } else if (const auto* input_audio =
                   std::get_if<InputAudio>(&preprocessed_content)) {
      ABSL_ASSIGN_OR_RETURN(const auto* spectrogram_tensor,
                            input_audio->GetPreprocessedAudioTensor());
      if (benchmark_info.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("audio_executor"));
      }
      ABSL_ASSIGN_OR_RETURN(auto audio_executor,
                            resource_manager_->AcquireAudioExecutor());
      ABSL_ASSIGN_OR_RETURN(auto single_audio_data,
                            audio_executor->Encode(*spectrogram_tensor));
      if (benchmark_info.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("audio_executor"));
      }
      const int num_audio_tokens = single_audio_data.GetValidTokens();
      if (num_audio_tokens > 0) {
        all_audio_data.push_back(std::move(single_audio_data));
        combined_token_ids.insert(combined_token_ids.end(), num_audio_tokens,
                                  ExecutorAudioData::kSpecialToken);
      }
    } else if (const auto* input_audio_end =
                   std::get_if<InputAudioEnd>(&preprocessed_content)) {
      // We allow audio end token even if the audio executor is not
      // available.
      auto audio_executor = resource_manager_->AcquireAudioExecutor();
      if (audio_executor.ok()) {
        // Flush any remaining buffered spectrogram frames from streaming
        // Encode() calls.
        auto flushed_audio_data = (*audio_executor)->Flush();
        if (flushed_audio_data.ok()) {
          const int flushed_tokens = flushed_audio_data->GetValidTokens();
          if (flushed_tokens > 0) {
            all_audio_data.push_back(std::move(*flushed_audio_data));
            combined_token_ids.insert(combined_token_ids.end(), flushed_tokens,
                                      ExecutorAudioData::kSpecialToken);
          }
        } else if (!absl::IsUnimplemented(flushed_audio_data.status())) {
          return flushed_audio_data.status();
        }
      }
      combined_token_ids.push_back(ExecutorAudioData::kEndToken);
    } else {
      return absl::InvalidArgumentError(
          "Unsupported input type in preprocessed_contents.");
    }
  }

  if (combined_token_ids.empty()) {
    return absl::InvalidArgumentError(
        "No token IDs found in preprocessed_contents.");
  }

  std::optional<ExecutorVisionData> combined_image_data = std::nullopt;
  if (!all_image_data.empty()) {
    ABSL_ASSIGN_OR_RETURN(combined_image_data,
                          CombineExecutorVisionData(all_image_data));
  }
  std::optional<ExecutorAudioData> combined_audio_data = std::nullopt;
  if (!all_audio_data.empty()) {
    ABSL_ASSIGN_OR_RETURN(combined_audio_data,
                          CombineExecutorAudioData(all_audio_data));
  }

  last_prefill_token_id_ = combined_token_ids.back();

  ABSL_ASSIGN_OR_RETURN(auto token_ids_buffer,
                        tokenizer_->TokenIdsToTensorBuffer(combined_token_ids));

  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::move(combined_image_data),
                        std::move(combined_audio_data));
  return inputs;
}

absl::StatusOr<std::unique_ptr<ThreadedExecutionManager>>
ThreadedExecutionManager::Create(
    Tokenizer* absl_nonnull tokenizer,
    ModelResources* absl_nullable model_resources,
    std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
    std::unique_ptr<VisionExecutorSettings> absl_nullable
    vision_executor_settings,
    std::unique_ptr<AudioExecutorSettings> absl_nullable
    audio_executor_settings,
    ::litert::Environment* absl_nullable litert_env,
    std::unique_ptr<AudioExecutor> absl_nullable audio_executor,
    std::shared_ptr<RuntimeDebugger> absl_nullable runtime_debugger,
    DecodeInputBufferFactory buffer_factory) {
  ABSL_ASSIGN_OR_RETURN(
      auto resource_manager,
      ResourceManager::Create(model_resources, std::move(llm_executor),
                              std::move(vision_executor_settings),
                              std::move(audio_executor_settings), litert_env,
                              std::move(audio_executor)));
  return absl::WrapUnique(new ThreadedExecutionManager(
      tokenizer, std::move(resource_manager), litert_env,
      std::move(runtime_debugger), std::move(buffer_factory)));
}

absl::Status ThreadedExecutionManager::WaitUntilDone(TaskId task_id,
                                                     absl::Duration timeout) {
  auto task_done = [this, task_id]() {
    session_and_task_lookup_mutex_.AssertReaderHeld();
    return task_lookup_.contains(task_id) &&
           IsTaskEndState(task_lookup_.at(task_id).task_state);
  };
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return session_and_task_lookup_mutex_.AwaitWithTimeout(
             absl::Condition(&task_done), timeout)
             ? absl::OkStatus()
             : absl::DeadlineExceededError(absl::StrCat(
                   "Task ", task_id, " did not complete within the timeout of ",
                   absl::FormatDuration(timeout), "."));
}

absl::Status ThreadedExecutionManager::WaitUntilSessionDone(
    SessionId session_id, absl::Duration timeout) {
  auto session_done = [this, session_id]() {
    session_and_task_lookup_mutex_.AssertReaderHeld();
    return session_lookup_.contains(session_id) &&
           session_lookup_.at(session_id)->active_tasks.empty();
  };
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return session_and_task_lookup_mutex_.AwaitWithTimeout(
             absl::Condition(&session_done), timeout)
             ? absl::OkStatus()
             : absl::DeadlineExceededError(
                   absl::StrCat("Session ", session_id,
                                " did not complete within the timeout of ",
                                absl::FormatDuration(timeout), "."));
}

absl::Status ThreadedExecutionManager::WaitUntilAllDone(
    absl::Duration timeout) {
  const absl::Time deadline = absl::Now() + timeout;
  ABSL_RETURN_IF_ERROR(
      execution_thread_pool_->WaitUntilDone(deadline - absl::Now()));
  ABSL_RETURN_IF_ERROR(
      callback_thread_pool_->WaitUntilDone(deadline - absl::Now()));
  // Accepted immediate-terminal handlers run on their submitting caller rather
  // than either pool. Their existing native active-task return fences must
  // complete too; pool idleness alone cannot authorize owner destruction.
  auto all_tasks_done = [this]() {
    session_and_task_lookup_mutex_.AssertReaderHeld();
    for (const auto& entry : session_lookup_) {
      if (!entry.second->active_tasks.empty()) return false;
    }
    return true;
  };
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return session_and_task_lookup_mutex_.AwaitWithDeadline(
             absl::Condition(&all_tasks_done), deadline)
             ? absl::OkStatus()
             : absl::DeadlineExceededError(
                   "Accepted task callbacks did not return before the "
                   "all-tasks deadline.");
}

absl::Status ThreadedExecutionManager::AddPrefillTask(
    SessionId session_id, TaskId task_id, std::vector<InputData> inputs,
    absl::flat_hash_set<TaskId> dep_tasks,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, session_id,
               inputs = std::move(inputs)]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    // Note AcquireExecutorWithContextHandler include context switching logic,
    // so it should be called before any executor running.
    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      FinishTaskAndLogErrors(task_id, llm_executor.status(),
                             std::move(callback));
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      llm_executor.value().reset();
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    auto executor_inputs =
        ProcessAndCombineContents(inputs, session_info->benchmark_info);
    if (executor_inputs.ok() &&
        session_info->session_config.GetAudioEmbeddingsCallback() != nullptr) {
      auto audio_data_status = executor_inputs->GetAudioDataPtr();
      if (audio_data_status.ok() && *audio_data_status != nullptr) {
        (*session_info->session_config.GetAudioEmbeddingsCallback())(
            **audio_data_status);
      }
    }
    if (!executor_inputs.ok()) {
      llm_executor.value().reset();
      if (executor_inputs.status().message() ==
              "No token IDs found in preprocessed_contents." &&
          session_info->session_config.AudioModalityEnabled()) {
        {
          auto audio_executor = resource_manager_->AcquireAudioExecutor();
          if (!audio_executor.ok()) {
            FinishTaskAndLogErrors(task_id, audio_executor.status(),
                                   std::move(callback));
            return;
          }
          auto audio_executor_properties =
              (*audio_executor)->GetAudioExecutorProperties();
          if (!audio_executor_properties.ok()) {
            audio_executor.value().reset();
            FinishTaskAndLogErrors(task_id, audio_executor_properties.status(),
                                   std::move(callback));
            return;
          }
          if (!audio_executor_properties->is_streaming_model) {
            audio_executor.value().reset();
            FinishTaskAndLogErrors(task_id, executor_inputs.status(),
                                   std::move(callback));
            return;
          }
        }
        ABSL_VLOG(1)
            << "Input audio chunk is smaller than the audio encoder input "
               "size. The input audio chunk is buffered and will be processed "
               "together with the next input audio chunk. Skipping prefill.";
        // We allow empty input for streaming audio use case, so we mark the
        // task as done.
        FinishTaskAndLogErrors(task_id, Responses(TaskState::kDone),
                               std::move(callback));
        return;
      }
      FinishTaskAndLogErrors(task_id, executor_inputs.status(),
                             std::move(callback));
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      llm_executor.value().reset();
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

#if defined(LITERT_LM_DEBUGGER_ENABLED)
    if (runtime_debugger_ != nullptr) {
      runtime_debugger_->SetActiveDebugSession(session_id);
    }
#else
    // Suppress -Wunused-variable for non-debugger builds.
    (void)session_id;
#endif

    auto responses =
        Tasks::Prefill(*llm_executor.value(), *executor_inputs,
                       /*wait_for_completion=*/true,
                       /*benchmark_info=*/session_info->benchmark_info);
    if (!responses.ok()) {
      llm_executor.value().reset();
      FinishTaskAndLogErrors(task_id, responses.status(), std::move(callback));
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
    } else {
      // Keep track of the last_prefill_token_id after prefill is done.
      auto processed_tokens = llm_executor.value()->GetProcessedTokens();
      if (!processed_tokens.ok()) {
        FinishTaskAndLogErrors(task_id, processed_tokens.status(),
                               std::move(callback));
        return;
      }
      auto current_step = llm_executor.value()->GetCurrentStep();
      if (!current_step.ok()) {
        FinishTaskAndLogErrors(task_id, current_step.status(),
                               std::move(callback));
        return;
      }
      session_info->last_prefill_token_id =
          processed_tokens.value()
              ->GetTokenAtStep(current_step.value() - 1)
              .at(0);
    }

    llm_executor.value().reset();
    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

absl::Status ThreadedExecutionManager::AddDecodeTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    RepetitionPenaltyConfig repetition_penalty_config,
    NoRepeatNgramConfig no_repeat_ngram_config,
    SuppressTokensConfig suppress_tokens_config,
    Constraint* absl_nullable constraint,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    int max_output_tokens, std::optional<int> thinking_token_budget,
    std::vector<int> thinking_start_token_ids,
    std::vector<int> thinking_end_token_ids) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

#if defined(LITERT_LM_DEBUGGER_ENABLED)
  if (runtime_debugger_ != nullptr) {
    callback = [this, session_id, cb = std::move(callback)](
                   absl::StatusOr<Responses> responses) mutable {
      if (responses.ok() && runtime_debugger_ != nullptr) {
        runtime_debugger_->ObserveTokens(session_id, *responses);
      }
      cb(std::move(responses));
    };
  }
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)

  auto task = [this, task_id, session_id,
               repetition_penalty_config = std::move(repetition_penalty_config),
               no_repeat_ngram_config = std::move(no_repeat_ngram_config),
               suppress_tokens_config = std::move(suppress_tokens_config),
               constraint, cancelled, max_output_tokens, thinking_token_budget,
               thinking_start_token_ids = std::move(thinking_start_token_ids),
               thinking_end_token_ids =
                   std::move(thinking_end_token_ids)]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      FinishTaskAndLogErrors(task_id, llm_executor.status(),
                             std::move(callback));
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      llm_executor.value().reset();
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    auto num_output_candidates =
        session_info->session_config.GetNumOutputCandidates();
    session_info->stop_token_detector->ResetBatch(num_output_candidates);
    std::optional<Sampler*> optional_sampler = std::nullopt;
    std::optional<litert::TensorBuffer> decoded_ids_buffer = std::nullopt;
      if (session_info->sampler != nullptr) {
        optional_sampler = session_info->sampler.get();
        std::vector<int> decoded_ids(num_output_candidates,
                                     session_info->last_prefill_token_id);
        auto decoded_ids_buffer_or =
            CreateDecodeInputBuffer(decoded_ids);
        if (!decoded_ids_buffer_or.ok()) {
          llm_executor.value().reset();
          FinishTaskAndLogErrors(task_id, decoded_ids_buffer_or.status(),
                                 std::move(callback));
          return;
        }
        decoded_ids_buffer = std::move(decoded_ids_buffer_or.value());
      }

#if defined(LITERT_LM_DEBUGGER_ENABLED)
    if (runtime_debugger_ != nullptr) {
      runtime_debugger_->SetActiveDebugSession(session_id);
    }
#else
    // Suppress -Wunused-variable for non-debugger builds.
    (void)session_id;
#endif

    auto responses = Tasks::Decode(
        *llm_executor.value(), *tokenizer_, *session_info->stop_token_detector,
        num_output_candidates, session_info->benchmark_info, optional_sampler,
        std::move(repetition_penalty_config), std::move(no_repeat_ngram_config),
        std::move(suppress_tokens_config), constraint,
        std::move(decoded_ids_buffer), callback, cancelled.get(),
        max_output_tokens, thinking_token_budget, thinking_end_token_ids,
        thinking_start_token_ids);
    if (!responses.ok() && absl::IsCancelled(responses.status())) {
      responses = Responses(TaskState::kCancelled);
    }

    if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
    }

    llm_executor.value().reset();
    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

absl::Status ThreadedExecutionManager::AddCloneSessionTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    SessionId cloned_session_id,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, session_id, cloned_session_id]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    absl::StatusOr<Responses> result = Responses(TaskState::kDone);
    [&] {
      std::shared_ptr<const SessionInfo> original_session_info;
      {
        absl::MutexLock lock(session_and_task_lookup_mutex_);
        if (!session_lookup_.contains(session_id)) {
          result = absl::InvalidArgumentError(absl::StrCat(
              "Session ", session_id, " not found in session list."));
          return;
        }
        original_session_info = session_lookup_.at(session_id);
      }

      auto cloned_context_handler_or = resource_manager_->CloneContextHandler(
          original_session_info->context_handler);
      if (!cloned_context_handler_or.ok()) {
        result = cloned_context_handler_or.status();
        return;
      }

      std::unique_ptr<Sampler> cloned_sampler;
      if (original_session_info->sampler != nullptr) {
        auto sampler = CreateSampler(
            original_session_info->session_config.GetSamplerBackend(),
            original_session_info->session_config.GetNumOutputCandidates(),
            original_session_info->session_config.GetSamplerParams());
        if (!sampler.ok()) {
          result = sampler.status();
          return;
        }
        cloned_sampler = std::move(*sampler);
      }

      auto cloned_stop_token_detector = std::make_unique<StopTokenDetector>(1);
      for (const auto& stop_token_sequence :
           original_session_info->session_config.GetStopTokenIds()) {
        auto status = cloned_stop_token_detector->AddStopTokenSequence(
            stop_token_sequence);
        if (!status.ok()) {
          result = status;
          return;
        }
      }

      {
        absl::MutexLock lock(session_and_task_lookup_mutex_);
        if (!session_lookup_.contains(cloned_session_id)) {
          result = absl::InvalidArgumentError(
              absl::StrCat("Cloned session ", cloned_session_id,
                           " not found in session list."));
          return;
        }
        session_lookup_.at(cloned_session_id)->session_config =
            original_session_info->session_config;
        session_lookup_.at(cloned_session_id)->context_handler =
            std::move(cloned_context_handler_or.value());
        session_lookup_.at(cloned_session_id)->sampler =
            std::move(cloned_sampler);
        session_lookup_.at(cloned_session_id)->last_prefill_token_id =
            original_session_info->last_prefill_token_id;
        session_lookup_.at(cloned_session_id)->stop_token_detector =
            std::move(cloned_stop_token_detector);
        session_lookup_.at(cloned_session_id)->benchmark_info =
            original_session_info->benchmark_info;
      }
    }();

    if (cancelled != nullptr && cancelled->load()) {
      result = Responses(TaskState::kCancelled);
    }

    FinishTaskAndLogErrors(task_id, result, std::move(callback));
    return;
  };

  return CreateTask(cloned_session_id, task_id, std::move(task),
                    std::move(dep_tasks), cancelled, std::move(callback));
}

absl::Status ThreadedExecutionManager::AddTextScoringTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    const std::vector<absl::string_view>& target_text, bool store_token_lengths,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, target_text,
               store_token_lengths]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      FinishTaskAndLogErrors(task_id, llm_executor.status(),
                             std::move(callback));
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      llm_executor.value().reset();
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    const int num_output_candidates =
        session_info->session_config.GetNumOutputCandidates();
    std::vector<int> decoded_ids(num_output_candidates,
                                 session_info->last_prefill_token_id);
    auto decoded_ids_buffer =
        CreateDecodeInputBuffer(decoded_ids);
    if (!decoded_ids_buffer.ok()) {
      llm_executor.value().reset();
      FinishTaskAndLogErrors(
          task_id, decoded_ids_buffer.status(),
          std::move(callback));
      return;
    }

    // TODO(b/435040163): Handle the temperature. Should it be calculated from
    // the sampler or the sampler parameters? For now, hardcode it to 1.0f for
    // testing.
    auto temperature = 1.0f;
    auto responses = Tasks::Score(
        *llm_executor.value(), *tokenizer_, target_text, temperature,
        std::move(decoded_ids_buffer.value()), store_token_lengths);

    if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
    }

    llm_executor.value().reset();
    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    cancelled, std::move(callback));
}

absl::StatusOr<int> ThreadedExecutionManager::GetCurrentStep(
    const SessionInfo& session_info) {
  ABSL_ASSIGN_OR_RETURN(auto llm_executor,
                        resource_manager_->AcquireExecutorWithContextHandler(
                            session_info.context_handler));
  return llm_executor->GetCurrentStep();
}

absl::Status ThreadedExecutionManager::SetCurrentStep(
    const SessionInfo& session_info, int target_step) {
  ABSL_ASSIGN_OR_RETURN(auto llm_executor,
                        resource_manager_->AcquireExecutorWithContextHandler(
                            session_info.context_handler));
  ABSL_ASSIGN_OR_RETURN(int current_step, llm_executor->GetCurrentStep());
  if (target_step > current_step) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Target step is greater than the current step: ", current_step));
  }
  return llm_executor->SetCurrentStep(target_step);
}

absl::StatusOr<AudioExecutorProperties>
ThreadedExecutionManager::GetAudioExecutorProperties() const {
  return resource_manager_->GetAudioExecutorProperties();
}

absl::StatusOr<VisionExecutorProperties>
ThreadedExecutionManager::GetVisionExecutorProperties() const {
  return resource_manager_->GetVisionExecutorProperties();
}

}  // namespace litert::lm
