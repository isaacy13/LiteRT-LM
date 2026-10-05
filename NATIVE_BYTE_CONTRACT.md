# Constraint vocabulary patch

Based on LiteRT-LM v0.17.1 (`5e58e9a`). GuideAI's app repository retains its
small Swift package and checksum-pinned binary dependency. This fork does not
change app prompts, repair generated strings, change model weights or add a
second decoding library.

LLGuidance requires decoded continuation bytes and control-token metadata.
SentencePiece's raw `GetTokens()` exposes vocabulary spellings, including its
space symbol and byte-piece names, which differ from the decoder's bytes.
`GetConstraintVocabulary()` preserves that existing API and supplies the native
byte mapping separately. Control, unknown and unused pieces receive special
metadata. Byte pieces remain individual bytes so a valid Unicode sequence can
span several tokens. Isolated single-token decoding would strip leading spaces
and replace incomplete Unicode pieces, so it is not used here.

HuggingFace tokenizers retain their original tokenizer JSON. LLGuidance's existing
JSON decoder interprets byte-level/fallback vocabularies and special added tokens;
unsupported decoders fail at native tokenizer creation. Native encoding and the
ordinary chat detokenizer are unchanged.

Every configured single-token stop, including an explicitly provided EOS, is
marked as control. Components of multi-token stop sequences remain ordinary
tokens because globally banning them could prevent valid text. Sequence-aware
stopping is unchanged and requires separate coverage. Input tokenization is
declared string-based, allowing LLGuidance's existing invalid UTF8 fallback.

The SentencePiece constraint callback verifies that ordinary model IDs reproduce
the requested continuation bytes without using provider-configured control IDs.
Each provider owns immutable callback metadata for its lifetime; the shared model
tokenizer is not mutated. Faithful ordinary encodings retain their IDs. A source
character changed by the normal encoder, a literal control spelling, or partial
Unicode instead uses the model's native byte pieces. Models lacking the needed
byte pieces refuse that input. This is tokenizer encoding at a grammar boundary;
generated replies are neither transformed nor repaired. Direct C ABI probes retain
the original failures and validate native mask-driven generation of a literal
space symbol and Unicode against the normal SentencePiece decoder.

`constraint_vocabulary_test` checks exact string enums with spaces and Unicode,
literal angle brackets, Unicode byte fallback (including a literal SentencePiece
space symbol), premature controls and stop IDs, explicit special-token tool
grammar, HF byte-level metadata and invalid IDs. Existing tokenizer and constraint
tests remain enabled. The pinned upstream FC test had an orphaned assertion
argument and numeric cases using fences different from its explicit fixture
options. The assertion call and those fences are corrected; all integer/number
acceptance and rejection cases remain enabled. No model quality result is inferred from those tests.

The dedicated workflow runs those controls and only then builds the iOS device
and simulator framework, retaining source identity, binary checksum and logs.
GuideAI must validate actual native output, tool execution, cancellation and
physical performance before changing its binary dependency.

Provider references:

- [LLGuidance byte and special-token contract](https://github.com/guidance-ai/llguidance/blob/v1.3.0/docs/special_tokens.md)
- [LLGuidance native tokenizer JSON decoder](https://github.com/guidance-ai/llguidance/blob/v1.3.0/parser/src/tokenizer_json.rs)
- [SentencePiece v0.2.2 decoder](https://github.com/google/sentencepiece/blob/v0.2.2/src/sentencepiece_processor.cc)

The first compiled gate at `07f9fa1` failed before executing tests on that
upstream FC fixture syntax error; its logs are retained and no framework was
built. Further C ABI probes also found that the current callback treated a
configured stop's literal spelling as the stop ID. Passing the provider's stop
metadata to the constraint encoder preserved literal `<end_of_turn>` and
`<unused44>` text through native masks and normal decoding. These probes use the
Gemma3 test tokenizer with zero LLM queries; compiled and phone acceptance
remain required.
