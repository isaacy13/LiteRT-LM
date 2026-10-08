// Copyright 2026 The ODML Authors.
// Licensed under the Apache License, Version 2.0.

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "c/conversation.h"
#include "c/conversation_internal.h"
#include "c/engine_internal.h"
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/core/session_advanced.h"
#include "runtime/core/session_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "runtime/proto/engine.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "support/tokenizer/tokenizer.h"
#include "support/tokenizer/sentencepiece_tokenizer.h"

namespace litert::lm {
namespace {

constexpr int kVocabSize = 16;
constexpr int kBos = 7;
constexpr int kStop = 6;
constexpr const char* kTemplate = R"jinja(
{{ tools | tojson }}
{% for m in messages %}{{ m.role }}:{{ m.content | tojson }};{% endfor %}
{% if add_generation_prompt %}assistant:{% endif %}
{{ enable_thinking | default(false) }}
{{ strftime_now('%Y-%m-%d %H:%M:%S') }}
)jinja";

bool Check(bool value, const char* detail) {
  if (!value) std::fprintf(stderr, "%s\n", detail);
  return value;
}

struct Effects {
  std::atomic<int> prefills = 0;
  std::atomic<int> decodes = 0;
  std::atomic<int> context_restores = 0;
  std::atomic<int> step_reads = 0;
  std::vector<int> prefilled_ids;
};

// Deliberately non-additive native tokenizer pre/post processing makes a
// concatenated-string or fixed BOS+body shortcut disagree with real prefill.
class TestTokenizer : public support::Tokenizer {
 public:
  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override {
    encoded_texts.emplace_back(text);
    if (on_encode) on_encode();
    if (fail_encoding) return absl::InvalidArgumentError("Tokenizer rejected input.");
    std::vector<int> ids = {8};
    for (unsigned char byte : text) ids.push_back(byte % 5 + 1);
    ids.push_back(9);
    return ids;
  }
  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (token == "<B>") return kBos;
    return absl::NotFoundError("Unknown token.");
  }
  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> ids, bool skip_special_tokens) override {
    std::string text;
    for (int id : ids) {
      if (id == kBos) {
        if (!skip_special_tokens) text += "<B>";
      } else if (id != kStop) {
        text += "x";
      }
    }
    return text;
  }
  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kUnspecified;
  }
  std::vector<std::string> GetTokens() const override {
    return std::vector<std::string>(kVocabSize, "x");
  }
  int GetVocabSize() const override { return kVocabSize; }
  std::vector<std::string> encoded_texts;
  bool fail_encoding = false;
  std::function<void()> on_encode;
};

// No model is loaded. Capture the integer IDs actually delivered through the
// real ThreadedExecutionManager, and maintain its ordinary context bookkeeping.
class CaptureExecutor : public FakeLlmExecutor {
 public:
  explicit CaptureExecutor(std::shared_ptr<Effects> effects, int vocab_size = kVocabSize)
      : FakeLlmExecutor(vocab_size, {}, {{kStop}}), effects_(effects) {}
  absl::Status Prefill(const ExecutorInputs& input,
                       const ExecutorPrefillParams&) override {
    ABSL_ASSIGN_OR_RETURN(auto text, input.GetTextDataPtr());
    LITERT_ASSIGN_OR_RETURN(auto ids,
                            ReferTensorBufferAsSpan<int>(text->GetTokenIds()));
    effects_->prefilled_ids.insert(effects_->prefilled_ids.end(), ids.begin(), ids.end());
    processed_.AddProcessedTokens(std::vector<int>(ids.begin(), ids.end()));
    step_ += ids.size();
    ++effects_->prefills;
    return absl::OkStatus();
  }
  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams&) override {
    ++effects_->decodes;
    processed_.AddProcessedTokens({kStop});
    ++step_;
    return std::vector<std::vector<int>>{{kStop}};
  }
  absl::StatusOr<int> GetCurrentStep() const override {
    ++effects_->step_reads;
    return step_;
  }
  absl::Status SetCurrentStep(int step) override {
    step_ = step;
    return processed_.RollBackToStep(step);
  }
  absl::StatusOr<const ProcessedTokens*> GetProcessedTokens() const override {
    return &processed_;
  }
  absl::StatusOr<std::unique_ptr<LlmContext>> CloneContext() const override {
    auto state = std::make_unique<RuntimeState>();
    state->current_step = step_;
    return std::make_unique<LlmContext>(
        nullptr, std::make_unique<RuntimeConfig>(runtime_config_), std::move(state));
  }
  absl::Status RestoreContext(std::unique_ptr<LlmContext> context) override {
    ++effects_->context_restores;
    runtime_config_ = context->runtime_config();
    step_ = context->runtime_state().current_step;
    return processed_.RollBackToStep(step_);
  }
  absl::StatusOr<RuntimeConfig> GetRuntimeConfig() const override {
    return runtime_config_;
  }
  absl::Status UpdateRuntimeConfig(const RuntimeConfig& config) override {
    runtime_config_ = config;
    return absl::OkStatus();
  }
 private:
  std::shared_ptr<Effects> effects_;
  ProcessedTokens processed_;
  RuntimeConfig runtime_config_;
  int step_ = 0;
};

class TestEngine : public Engine {
 public:
  TestEngine(EngineSettings settings, support::Tokenizer& tokenizer,
             std::shared_ptr<ThreadedExecutionManager> owner)
      : settings_(std::move(settings)), tokenizer_(tokenizer), owner_(owner) {}
  const EngineSettings& GetEngineSettings() const override { return settings_; }
  const support::Tokenizer& GetTokenizer() const override { return tokenizer_; }
  absl::StatusOr<std::unique_ptr<SessionInterface>> CreateSession(
      const SessionConfig& input_config) override {
    auto config = input_config;
    ABSL_RETURN_IF_ERROR(config.MaybeUpdateAndValidate(settings_));
    ABSL_ASSIGN_OR_RETURN(auto session,
                          SessionAdvanced::Create(owner_, &tokenizer_, config,
                                                  std::nullopt));
    latest_session = session.get();
    ++created_sessions;
    return std::unique_ptr<SessionInterface>(std::move(session));
  }
  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties() const override {
    return absl::UnimplementedError("No audio fixture.");
  }
  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties() const override {
    return absl::UnimplementedError("No vision fixture.");
  }
  SessionAdvanced* latest_session = nullptr;
  int created_sessions = 0;
 private:
  EngineSettings settings_;
  support::Tokenizer& tokenizer_;
  std::shared_ptr<ThreadedExecutionManager> owner_;
};

class Harness {
 public:
  ~Harness() {
    if (owner && !owner->WaitUntilAllDone(absl::Seconds(5)).ok()) _exit(2);
    conversations.clear();
    raw_sessions.clear();
    c_engine.engine.reset();
    owner.reset();
  }
  bool Initialize(bool bos = true, bool gemma4 = false, int forced_prefill = 0,
                  support::Tokenizer* actual = nullptr) {
    auto& native_tokenizer = actual == nullptr ? static_cast<support::Tokenizer&>(tokenizer) : *actual;
    effects = std::make_shared<Effects>();
    auto created = ThreadedExecutionManager::Create(
        &native_tokenizer, nullptr, std::make_unique<CaptureExecutor>(effects, native_tokenizer.GetVocabSize()),
        nullptr, nullptr, nullptr, nullptr, nullptr);
    if (!created.ok()) return false;
    owner = std::move(*created);
    auto assets = ModelAssets::Create("fixture_model");
    if (!assets.ok()) return false;
    auto settings = EngineSettings::CreateDefault(*assets);
    if (!settings.ok()) return false;
    proto::LlmMetadata metadata;
    metadata.mutable_stop_tokens()->Add()->mutable_token_ids()->mutable_ids()->Add(kStop);
    if (bos) metadata.mutable_start_token()->mutable_token_ids()->mutable_ids()->Add(kBos);
    if (gemma4) metadata.mutable_llm_model_type()->mutable_gemma4();
    else metadata.mutable_llm_model_type()->mutable_generic_model();
    if (forced_prefill > 0) settings->GetMutableBenchmarkParams().set_num_prefill_tokens(forced_prefill);
    if (!settings->MaybeUpdateAndValidate(&native_tokenizer, &metadata).ok()) return false;
    c_engine.engine = std::make_unique<TestEngine>(*settings, native_tokenizer, owner);
    return true;
  }
  SessionConfig Config() {
    auto config = SessionConfig::CreateDefault();
    config.SetUseExternalSampler(false);
    config.SetApplyPromptTemplateInSession(false);
    config.SetMaxOutputTokens(1);
    config.MaybeUpdateAndValidate(c_engine.engine->GetEngineSettings()).IgnoreError();
    return config;
  }
  using ConfigOwner = std::unique_ptr<LiteRtLmConversationConfig,
      decltype(&litert_lm_conversation_config_delete)>;
  ConfigOwner Configuration(const std::string& preface = "[]",
                            const char* prompt_template = kTemplate) {
    ConfigOwner config(litert_lm_conversation_config_create(),
                       litert_lm_conversation_config_delete);
    LiteRtLmSessionConfig session_config;
    session_config.config = std::make_unique<SessionConfig>(Config());
    litert_lm_conversation_config_set_session_config(config.get(), &session_config);
    litert_lm_conversation_config_set_messages(config.get(), preface.c_str());
    litert_lm_conversation_config_set_prompt_template(config.get(), prompt_template);
    litert_lm_conversation_config_set_enable_constrained_decoding(config.get(), false);
    litert_lm_conversation_config_set_tools(config.get(),
        R"json([{"name":"lookup","description":"Look up \"nearby\" café 北","parameters":{"type":"object"}}])json");
    return config;
  }
  SessionAdvanced* Session(std::optional<BenchmarkInfo> benchmark = std::nullopt) {
    auto session = SessionAdvanced::Create(owner, &tokenizer, Config(), benchmark);
    if (!session.ok()) return nullptr;
    auto* result = session->get();
    raw_sessions.push_back(std::move(*session));
    return result;
  }
  TestEngine& EngineOwner() { return *static_cast<TestEngine*>(c_engine.engine.get()); }
  TestTokenizer tokenizer;
  std::shared_ptr<Effects> effects;
  std::shared_ptr<ThreadedExecutionManager> owner;
  LiteRtLmEngine c_engine;
 private:
  std::vector<std::unique_ptr<LiteRtLmConversation>> conversations;
  std::vector<std::unique_ptr<SessionAdvanced>> raw_sessions;
};

bool HasNoEffects(const Harness& h, int reads, int restores) {
  return h.effects->prefills == 0 && h.effects->decodes == 0 &&
         h.effects->step_reads == reads && h.effects->context_restores == restores;
}


constexpr const char* kExtra = R"json({"enable_thinking":true,"now":1992124800})json";
constexpr const char* kInput = R"json({"role":"user","content":[{"type":"text","text":"Use the \"earlier\" plan\nCafé 北"}]})json";
constexpr const char* kPacketTemplate = R"jinja(
{{ tools | tojson }}
{% for m in messages %}{{ m | tojson }};{% endfor %}
{% if add_generation_prompt %}assistant:{% endif %}
{{ enable_thinking | default(false) }}
{{ strftime_now('%Y-%m-%d %H:%M:%S') }}
)jinja";

bool Differential(bool bos, bool gemma4 = false, const char* tmpl = kTemplate,
                  const char* preface = R"json([{"role":"system","content":"Keep dates literal"},{"role":"user","content":"Café 北"},{"role":"assistant","content":"A draft"}])json",
                  const char* current = kInput) {
  Harness h;
  if (!h.Initialize(bos, gemma4)) return false;
  auto config = h.Configuration(preface, tmpl);
  const int reads = h.effects->step_reads;
  const int restores = h.effects->context_restores;
  size_t measured = 0;
  for (int candidate = 0; candidate < 12; ++candidate) {
    h.tokenizer.encoded_texts.clear();
    size_t count = 99;
    const int status = litert_lm_engine_count_first_input_text_tokens(
        &h.c_engine, config.get(), current, kExtra, nullptr, &count);
    if (status != 0 || count == 0 || (candidate > 0 && count != measured)) return false;
    measured = count;
  }
  if (!HasNoEffects(h, reads, restores) || h.EngineOwner().created_sessions != 0)
    return Check(false, "candidate counts allocated/touched a native session/executor");
  const auto counted_chunk_bytes = h.tokenizer.encoded_texts;
  auto conversation = std::unique_ptr<LiteRtLmConversation,
      decltype(&litert_lm_conversation_delete)>(
      litert_lm_conversation_create(&h.c_engine, config.get()),
      litert_lm_conversation_delete);
  if (!conversation || !conversation->conversation->GetHistory().empty()) return false;
  h.tokenizer.encoded_texts.clear();
  auto response = std::unique_ptr<LiteRtLmJsonResponse,
      decltype(&litert_lm_json_response_delete)>(
      litert_lm_conversation_send_message(conversation.get(), current, kExtra, nullptr),
      litert_lm_json_response_delete);
  return Check(counted_chunk_bytes == h.tokenizer.encoded_texts &&
               response != nullptr && h.EngineOwner().created_sessions == 1 &&
               h.effects->prefills > 0 && h.effects->prefilled_ids.size() == measured,
               "canonical count differs from actual processed prefill IDs");
}

TEST(CanonicalFactoryCount, ActualProcessedPrefillWithBOS) {
  EXPECT_TRUE(Differential(true));
}
TEST(CanonicalFactoryCount, ActualProcessedPrefillWithoutBOS) {
  EXPECT_TRUE(Differential(false));
}
TEST(CanonicalFactoryCount, ActualGemma4BareToolHistoryAndResponse) {
  // Actual app replay leaves tool-only assistant content absent. Preserve all
  // packet fields and arbitrary nested result data; do not normalize them.
  EXPECT_TRUE(Differential(true, true, kPacketTemplate,
      R"json([{"role":"system","content":"Keep dates literal"},{"role":"user","content":"Plan Café 北"},{"role":"assistant","tool_calls":[{"id":"native-call-1","type":"function","function":{"name":"lookup","arguments":{"date":"2033-02-17"}}}]},{"role":"tool","name":"lookup","tool_call_id":"native-call-1","content":{"event_date":"2033-02-17","type":"image","path":"arbitrary tool data"}},{"role":"assistant","content":"Draft"}])json"));
}
TEST(CanonicalFactoryCount, CurrentObjectOrderingMatchesExistingCSendConversion) {
  EXPECT_TRUE(Differential(true, true, kPacketTemplate, "[]",
      R"json({"role":"user","z_field":{"z":3,"a":1},"content":"Café 北","a_field":2})json"));
}
TEST(CanonicalFactoryCount, ActualSingleTurnTemplate) {
  std::ifstream file(std::string(::testing::SrcDir()) +
      "/litert_lm/runtime/components/testdata/generic-model-multi-prefill.jinja");
  std::stringstream text; text << file.rdbuf();
  const auto source = text.str();
  ASSERT_FALSE(source.empty());
  ASSERT_TRUE(PromptTemplate(source).GetCapabilities().supports_single_turn);
  EXPECT_TRUE(Differential(true, false, source.c_str()));
}
TEST(CanonicalFactoryCount, MediaRefusalBeforeNativeSessionOrConverterIO) {
  for (const char* type : {"image", "audio"}) {
    for (bool in_preface : {false, true}) {
      Harness h; ASSERT_TRUE(h.Initialize(true, true));
      const auto media = Message{{"role", "user"}, {"content", Message::array({
          {{"type", type}, {"path", "/must-not-read/absent-input"}}})}};
      auto config = h.Configuration(in_preface ? Message::array({media}).dump() : "[]",
                                    kPacketTemplate);
      const auto input = media.dump(); size_t count = 99;
      EXPECT_EQ(litert_lm_engine_count_first_input_text_tokens(
          &h.c_engine, config.get(), in_preface ? kInput : input.c_str(),
          kExtra, nullptr, &count), static_cast<int>(absl::StatusCode::kUnimplemented));
      EXPECT_EQ(count, 0); EXPECT_EQ(h.EngineOwner().created_sessions, 0);
      EXPECT_EQ(h.effects->prefills, 0); EXPECT_EQ(h.effects->decodes, 0);
    }
  }
}
TEST(CanonicalFactoryCount, InvalidShapeJSONAndFrozenTimeRefusals) {
  Harness h; ASSERT_TRUE(h.Initialize()); auto config = h.Configuration();
  for (const char* input : {"{", "null", "[]", R"({"role":3,"content":"x"})",
                           R"({"role":"user","content":["x"]})",
                           R"({"role":"user","content":[{"type":"text","text":3}]})",
                           R"({"role":"assistant","tool_calls":[{"function":{"name":"lookup"}}]})"}) {
    size_t count = 99;
    const int result = litert_lm_engine_count_first_input_text_tokens(
        &h.c_engine, config.get(), input, kExtra, nullptr, &count);
    // [] may render a valid preface-only first send; its behavior is canonical.
    if (std::string(input) != "[]") { EXPECT_NE(result, 0); EXPECT_EQ(count, 0); }
  }
  for (const char* extra : std::vector<const char*>{nullptr, "{", "null", "[]", "{}", R"({"now":"tomorrow"})"}) {
    size_t count = 99;
    EXPECT_EQ(litert_lm_engine_count_first_input_text_tokens(
        &h.c_engine, config.get(), kInput, extra, nullptr, &count),
        static_cast<int>(absl::StatusCode::kInvalidArgument));
    EXPECT_EQ(count, 0);
  }
  EXPECT_EQ(h.EngineOwner().created_sessions, 0);
  EXPECT_EQ(h.effects->prefills, 0); EXPECT_EQ(h.effects->decodes, 0);
}
TEST(CanonicalFactoryCount, MalformedConfigBytesAreRefusedInsteadOfDropped) {
  Harness h; ASSERT_TRUE(h.Initialize());
  for (int field : {0, 1, 2}) {
    auto config = h.Configuration();
    if (field == 0) config->messages_json = "{";
    if (field == 1) config->tools_json = "{}";
    if (field == 2) config->extra_context_json = "[]";
    size_t count = 99;
    EXPECT_EQ(litert_lm_engine_count_first_input_text_tokens(
        &h.c_engine, config.get(), kInput, kExtra, nullptr, &count),
        static_cast<int>(absl::StatusCode::kInvalidArgument));
    EXPECT_EQ(count, 0);
  }
  EXPECT_EQ(h.EngineOwner().created_sessions, 0);
}
TEST(CanonicalFactoryCount, ForcedPrefillBenchmarkIsRefused) {
  Harness h; ASSERT_TRUE(h.Initialize(true, false, 7));
  auto config = h.Configuration(); size_t count = 99;
  EXPECT_EQ(litert_lm_engine_count_first_input_text_tokens(
      &h.c_engine, config.get(), kInput, kExtra, nullptr, &count),
      static_cast<int>(absl::StatusCode::kFailedPrecondition));
  EXPECT_EQ(count, 0); EXPECT_EQ(h.EngineOwner().created_sessions, 0);
}
TEST(CanonicalFactoryCount, PrefilledPrefaceAndPendingAppendAreRefused) {
  Harness h; ASSERT_TRUE(h.Initialize());
  JsonPreface preface; preface.messages = Message::array({{{"role","system"},{"content","seed"}}});
  for (bool prefilled : {false, true}) {
    auto config = ConversationConfig::Builder().SetSessionConfig(h.Config())
        .SetPreface(preface).SetOverwritePromptTemplate(PromptTemplate(kTemplate))
        .SetPrefillPrefaceOnInit(prefilled).Build(h.EngineOwner());
    ASSERT_TRUE(config.ok()); OptionalArgs args;
    args.extra_context = Message{{"now",1992124800}};
    args.has_pending_message = !prefilled;
    auto count = Conversation::CountFirstInputTextTokens(h.EngineOwner(), *config,
        Message{{"role","user"},{"content","x"}}, args);
    EXPECT_FALSE(count.ok()); EXPECT_TRUE(absl::IsFailedPrecondition(count.status()));
  }
  EXPECT_EQ(h.EngineOwner().created_sessions, 0);
}
TEST(CanonicalFactoryCount, BorrowedMoveOnlyOptionsRemainUnconsumed) {
  Harness h;
  ASSERT_TRUE(h.Initialize(true));
  auto config = ConversationConfig::Builder().SetSessionConfig(h.Config())
      .SetOverwritePromptTemplate(PromptTemplate(kTemplate))
      .SetPrefillPrefaceOnInit(false).Build(h.EngineOwner());
  ASSERT_TRUE(config.ok());
  OptionalArgs args;
  args.extra_context = nlohmann::ordered_json::parse(kExtra);
  args.max_output_tokens = 17;
  LlGuidanceConstraintArg constraint;
  constraint.constraint_type = LlgConstraintType::kJsonSchema;
  constraint.constraint_string = R"({"type":"object","properties":{"x":{"type":"string"}}})";
  args.decoding_constraint = std::move(constraint);
  const auto original_context = args.extra_context;
  const auto original_schema = std::get<LlGuidanceConstraintArg>(*args.decoding_constraint).constraint_string;
  const int reads = h.effects->step_reads;
  const int restores = h.effects->context_restores;
  for (int i = 0; i < 3; ++i) {
    auto count = Conversation::CountFirstInputTextTokens(h.EngineOwner(), *config,
        Message{{"role", "user"}, {"content", nlohmann::ordered_json::array({
          {{"type", "text"}, {"text", "Exact options"}}})}}, args);
    ASSERT_TRUE(count.ok());
    EXPECT_GT(*count, 0u);
    ASSERT_TRUE(args.decoding_constraint.has_value());
    ASSERT_TRUE(std::holds_alternative<LlGuidanceConstraintArg>(*args.decoding_constraint));
    EXPECT_EQ(std::get<LlGuidanceConstraintArg>(*args.decoding_constraint).constraint_string, original_schema);
    EXPECT_EQ(args.extra_context, original_context);
    EXPECT_EQ(args.max_output_tokens, 17);
  }
  EXPECT_TRUE(HasNoEffects(h, reads, restores));
  EXPECT_EQ(h.EngineOwner().created_sessions, 0);
}

TEST(CanonicalFactoryCount, TokenizerFailureAndReentryLeaveNoNativeState) {
  Harness h; ASSERT_TRUE(h.Initialize()); auto config = h.Configuration();
  size_t count = 99; h.tokenizer.fail_encoding = true;
  EXPECT_NE(litert_lm_engine_count_first_input_text_tokens(
      &h.c_engine, config.get(), kInput, kExtra, nullptr, &count), 0);
  EXPECT_EQ(count, 0); h.tokenizer.fail_encoding = false;
  bool attempted = false; int nested_status = 0; size_t nested_count = 99;
  h.tokenizer.on_encode = [&] {
    if (attempted) return; attempted = true;
    nested_status = litert_lm_engine_count_first_input_text_tokens(
        &h.c_engine, config.get(), kInput, kExtra, nullptr, &nested_count);
  };
  EXPECT_EQ(litert_lm_engine_count_first_input_text_tokens(
      &h.c_engine, config.get(), kInput, kExtra, nullptr, &count), 0);
  EXPECT_TRUE(attempted); EXPECT_EQ(nested_status,
      static_cast<int>(absl::StatusCode::kFailedPrecondition));
  EXPECT_EQ(nested_count, 0); EXPECT_GT(count, 0);
  EXPECT_EQ(h.EngineOwner().created_sessions, 0);
  EXPECT_EQ(h.effects->prefills, 0); EXPECT_EQ(h.effects->decodes, 0);
}
TEST(CanonicalFactoryCount, ProcessedTensorBranchAndChunkSensitiveBOS) {
  Harness h; ASSERT_TRUE(h.Initialize()); auto* session = h.Session(); ASSERT_NE(session,nullptr);
  std::vector<InputData> chunks; chunks.emplace_back(InputText(""));
  chunks.emplace_back(InputText("<B>ab")); chunks.emplace_back(InputText("c"));
  auto processed = PreparePrefillContents(chunks, h.Config(), h.tokenizer,
                                         std::nullopt, true, false);
  ASSERT_TRUE(processed.ok()); auto count = CalculateProcessedTextTokens(*processed);
  ASSERT_TRUE(count.ok()); EXPECT_TRUE(session->RunPrefill(chunks).ok());
  EXPECT_EQ(h.effects->prefilled_ids.size(), *count);
  std::vector<InputData> raw; raw.emplace_back(InputText("must be processed first"));
  auto refused = CalculateProcessedTextTokens(raw);
  EXPECT_FALSE(refused.ok()); EXPECT_TRUE(absl::IsUnimplemented(refused.status()));
}
TEST(CanonicalFactoryCount, CountDoesNotTouchUsedRewoundCancelledSession) {
  // No live Session is an input to this factory census. Existing315 controls
  // remain the regression gate for all submission/cancellation semantics.
  Harness h; ASSERT_TRUE(h.Initialize()); auto* session = h.Session(); ASSERT_NE(session,nullptr);
  bool rejected = false; h.tokenizer.on_encode = [&] {
    if (rejected) return; rejected = true;
    std::vector<InputData> nested; nested.emplace_back(InputText("nested"));
    auto result = session->RunPrefillAsync(nested, [](auto){});
    rejected = !result.ok() && absl::IsFailedPrecondition(result.status());
  };
  std::vector<InputData> input; input.emplace_back(InputText("normal"));
  ASSERT_TRUE(session->RunPrefill(input).ok()); EXPECT_TRUE(rejected);
  h.tokenizer.on_encode = nullptr;
  ASSERT_TRUE(session->RewindToStep(0).ok()); session->CancelProcess();
  const int prefills = h.effects->prefills, decodes = h.effects->decodes;
  const int reads = h.effects->step_reads, restores = h.effects->context_restores;
  const int sessions = h.EngineOwner().created_sessions;
  auto config = h.Configuration(); size_t count = 99;
  EXPECT_EQ(litert_lm_engine_count_first_input_text_tokens(
      &h.c_engine, config.get(), kInput, kExtra, nullptr, &count), 0);
  EXPECT_GT(count, 0); EXPECT_EQ(h.EngineOwner().created_sessions, sessions);
  EXPECT_EQ(h.effects->prefills, prefills); EXPECT_EQ(h.effects->decodes, decodes);
  EXPECT_EQ(h.effects->step_reads, reads); EXPECT_EQ(h.effects->context_restores, restores);
}

TEST(CanonicalFactoryCount, RealSentencePieceActualProcessedPrefill) {
  auto tokenizer = support::SentencePieceTokenizer::CreateFromFile(
      std::string(::testing::SrcDir()) +
      "/litert_lm/runtime/components/testdata/gemma3_sentencepiece.model");
  ASSERT_TRUE(tokenizer.ok());
  Harness h; ASSERT_TRUE(h.Initialize(true, false, 0, tokenizer->get()));
  auto config = h.Configuration("[]", kPacketTemplate);
  size_t measured = 99;
  ASSERT_EQ(litert_lm_engine_count_first_input_text_tokens(
      &h.c_engine, config.get(), kInput, kExtra, nullptr, &measured), 0);
  EXPECT_EQ(h.EngineOwner().created_sessions, 0);
  auto conversation = std::unique_ptr<LiteRtLmConversation,
      decltype(&litert_lm_conversation_delete)>(
      litert_lm_conversation_create(&h.c_engine, config.get()),
      litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);
  auto response = std::unique_ptr<LiteRtLmJsonResponse,
      decltype(&litert_lm_json_response_delete)>(
      litert_lm_conversation_send_message(conversation.get(), kInput, kExtra, nullptr),
      litert_lm_json_response_delete);
  ASSERT_NE(response,nullptr);
  EXPECT_EQ(h.effects->prefilled_ids.size(), measured);
}

// Configure only a fresh fixture: no Conversation or Session can retain the
// original Engine. Production validation still reads the actual EngineSettings.
void ConfigureFreshVisualBudgetFixture(Harness& h, int maximum_budget) {
  ASSERT_EQ(h.EngineOwner().created_sessions, 0);
  auto settings = h.EngineOwner().GetEngineSettings();
  settings.SetMaxVisionTokensPerImage(maximum_budget);
  h.c_engine.engine = std::make_unique<TestEngine>(
      std::move(settings), h.tokenizer, h.owner);
}

TEST(CanonicalFactoryCount, VisualTokenBudgetRefusalsMatchActualSend) {
  for (int budget : {-1, 0, 201}) {
    SCOPED_TRACE(budget);
    Harness h;
    ASSERT_TRUE(h.Initialize(true, true));
    ConfigureFreshVisualBudgetFixture(h, 200);
    auto config = h.Configuration("[]", kPacketTemplate);
    auto options = std::unique_ptr<LiteRtLmConversationOptionalArgs,
        decltype(&litert_lm_conversation_optional_args_delete)>(
        litert_lm_conversation_optional_args_create(),
        litert_lm_conversation_optional_args_delete);
    ASSERT_NE(options, nullptr);
    litert_lm_conversation_optional_args_set_visual_token_budget(
        options.get(), budget);
    const int reads = h.effects->step_reads;
    const int restores = h.effects->context_restores;
    h.tokenizer.encoded_texts.clear();
    size_t count = 99;
    EXPECT_EQ(litert_lm_engine_count_first_input_text_tokens(
        &h.c_engine, config.get(), kInput, kExtra, options.get(), &count),
        static_cast<int>(absl::StatusCode::kInvalidArgument));
    EXPECT_EQ(count, 0);
    EXPECT_EQ(h.EngineOwner().created_sessions, 0);
    EXPECT_TRUE(HasNoEffects(h, reads, restores));
    EXPECT_TRUE(h.tokenizer.encoded_texts.empty());
    EXPECT_EQ(options->visual_token_budget, budget);

    auto conversation = std::unique_ptr<LiteRtLmConversation,
        decltype(&litert_lm_conversation_delete)>(
        litert_lm_conversation_create(&h.c_engine, config.get()),
        litert_lm_conversation_delete);
    ASSERT_NE(conversation, nullptr);
    ASSERT_TRUE(conversation->conversation->GetHistory().empty());
    const int send_reads = h.effects->step_reads;
    const int send_restores = h.effects->context_restores;
    h.tokenizer.encoded_texts.clear();
    OptionalArgs args;
    args.extra_context = nlohmann::ordered_json::parse(kExtra);
    args.args = Gemma4DataProcessorArguments{.visual_token_budget = budget};
    auto result = conversation->conversation->SendMessage(
        nlohmann::json::parse(kInput), std::move(args));
    ASSERT_FALSE(result.ok());
    EXPECT_TRUE(absl::IsInvalidArgument(result.status()));
    if (budget <= 0) {
      EXPECT_EQ(result.status().message(), "Visual token budget must be positive.");
    } else {
      EXPECT_EQ(result.status().message(),
          "Visual token budget (201) cannot be larger than the engine's max vision "
          "tokens per image (200).");
    }
    auto response = std::unique_ptr<LiteRtLmJsonResponse,
        decltype(&litert_lm_json_response_delete)>(
        litert_lm_conversation_send_message(
            conversation.get(), kInput, kExtra, options.get()),
        litert_lm_json_response_delete);
    EXPECT_EQ(response, nullptr);
    EXPECT_EQ(h.EngineOwner().created_sessions, 1);
    EXPECT_TRUE(HasNoEffects(h, send_reads, send_restores));
    EXPECT_TRUE(h.tokenizer.encoded_texts.empty());
    EXPECT_TRUE(conversation->conversation->GetHistory().empty());
  }
}

TEST(CanonicalFactoryCount, VisualTokenBudgetAtEngineLimitMatchesProcessedPrefill) {
  Harness h;
  ASSERT_TRUE(h.Initialize(true, true));
  ConfigureFreshVisualBudgetFixture(h, 200);
  auto config = h.Configuration("[]", kPacketTemplate);
  auto options = std::unique_ptr<LiteRtLmConversationOptionalArgs,
      decltype(&litert_lm_conversation_optional_args_delete)>(
      litert_lm_conversation_optional_args_create(),
      litert_lm_conversation_optional_args_delete);
  ASSERT_NE(options, nullptr);
  litert_lm_conversation_optional_args_set_visual_token_budget(options.get(), 200);
  const int reads = h.effects->step_reads;
  const int restores = h.effects->context_restores;
  h.tokenizer.encoded_texts.clear();
  size_t count = 99;
  ASSERT_EQ(litert_lm_engine_count_first_input_text_tokens(
      &h.c_engine, config.get(), kInput, kExtra, options.get(), &count), 0);
  EXPECT_GT(count, 0u);
  EXPECT_EQ(h.EngineOwner().created_sessions, 0);
  EXPECT_TRUE(HasNoEffects(h, reads, restores));
  EXPECT_EQ(options->visual_token_budget, 200);
  const auto counted_chunk_bytes = h.tokenizer.encoded_texts;
  auto conversation = std::unique_ptr<LiteRtLmConversation,
      decltype(&litert_lm_conversation_delete)>(
      litert_lm_conversation_create(&h.c_engine, config.get()),
      litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);
  ASSERT_TRUE(conversation->conversation->GetHistory().empty());
  h.tokenizer.encoded_texts.clear();
  auto response = std::unique_ptr<LiteRtLmJsonResponse,
      decltype(&litert_lm_json_response_delete)>(
      litert_lm_conversation_send_message(
          conversation.get(), kInput, kExtra, options.get()),
      litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(h.EngineOwner().created_sessions, 1);
  EXPECT_GT(h.effects->prefills, 0);
  EXPECT_EQ(h.effects->prefilled_ids.size(), count);
  EXPECT_EQ(h.tokenizer.encoded_texts, counted_chunk_bytes);
}

}  // namespace
}  // namespace litert::lm
