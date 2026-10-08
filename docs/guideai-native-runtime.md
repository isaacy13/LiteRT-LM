# GuideAI native runtime extensions

The fork retains the upstream renderers, tokenizers and model processors. Its
Apple frameworks are consumed through Swift Package Manager; native source and
build products stay outside the GuideAI app repository.

## First-input text count

`litert_lm_engine_count_first_input_text_tokens` measures canonical processed
text tokens for a complete first send. It shares conversation rendering,
deferred preface assembly, session configuration resolution and prefill
preprocessing with inference. Its private conversation has no native session,
KV cache, executor context, sampler or task. The function is synchronous and
borrows the initialized engine: the caller must serialize it against inference,
destruction, session operations and tokenizer use until it returns.

The caller supplies the exact conversation configuration, current message,
optional arguments and an explicit integer `extra_context.now`. Count and later
create/send must use those same captured values. Media, forced benchmark
prefill, pending messages, already-prefilled prefaces and scoped LoRA are refused.
Unsupported converters or canonical processing errors remain errors. Success
does not establish continuation capacity, model quality, output constraint
availability or future send identity.

Independent processors retain the configured constrained-decoding setting;
native tool mode can repeat vocabulary/parser work per count. Initial structured
response mode does not enable that tool processor. Count allocates JSON,
processor and token data, so latency and memory need device measurement.

The additive `//c:conversation_factory_count_test` controls compare canonical
tokenizer chunk bytes and processed prefill IDs with actual first sends. They
also cover no-session allocation, deferred preface, native BOS handling, frozen
time, C JSON conversion, malformed input, media refusal, reentry and tokenizer
failure. The existing native regression controls remain required. Framework and
Swift adoption require matching headers, exports and device qualification.
