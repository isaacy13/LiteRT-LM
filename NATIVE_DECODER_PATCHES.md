# GuideAI native decoder fork

This fork starts at LiteRT-LM v0.17.1 (`5e58e9a`). GuideAI keeps the
native source here and consumes a verified binary through its small Swift package.

## Constraint ownership and token bytes

The constraint vocabulary preserves decoded bytes and tokenizer control metadata.
SentencePiece grammar-input encoding uses ordinary token IDs only when their
constraint bytes exactly match the input; otherwise it uses byte fallback tokens.
Configured single-token stops cannot encode ordinary grammar text. Callback
metadata remains owned by constraints and cloned states. Each committed token
advances an independent parser clone so speculative rejection preserves earlier
grammar positions and a failed commit cannot poison its input state.

## Apple runtime linkage

The pinned LiteRT dependency routes its iOS dynamic runtime to a `macos_dylib`
target. The arm64 simulator framework consequently tried to link an x86_64 macOS
runtime. `patches/litert_ios_runtime_import.patch` selects LiteRT-LM's pinned
Apple runtime binary for each iOS platform. macOS and other platform routing is
preserved. The release workflow rejects unfetched LFS pointers, wrong
architectures, wrong Apple platforms, and a missing runtime ABI before compilation.

The framework build and runtime dependency packaging still require acceptance.
A successful link does not establish Metal plugin loading, native tool grammar,
model accuracy, cancellation, phone memory use, or battery performance. GuideAI
must pass those device checks before adopting the fork's binary.

## Retained validation

`native-byte-contract.yml` runs eight tokenizer and constraint targets, then
builds device and simulator framework slices. Host XML and logs are copied before
switching Bazel configurations: `bazel-testlogs` otherwise points at the later
iOS configuration. Evidence is uploaded even when a later step fails.

## Complete Apple runtime artifacts

`package_apple_runtime.py` produces five explicit binary artifacts: CLiteRTLM,
LiteRtRuntime, GemmaModelConstraintProvider, LiteRtMetalAccelerator and
LiteRtTopKMetalSampler. Every dependency has its own iOS framework and simulator
framework, declared install name, platform metadata and checksum. Packaging
updates only Mach-O install/load names and code signatures in copies of the pinned
libraries. Original source/LFS objects remain unchanged. It verifies the complete
dependency graph, architectures, platforms, C headers, required native symbols and
signatures before retaining output. There are no nested frameworks or standalone
third-party dylibs in the app-facing artifacts.

On iOS, the sampler binds the pinned C APIs through a strong link dependency.
The native environment receives the linked Metal accelerator definition through
the pinned runtime's custom GPU accelerator option for GPU executors. This
internal SDK binding preserves the actual definition's lifetime and avoids
searching for standalone dylib filenames. Other platforms retain their existing
runtime loading. App code gets no native handle or internal configuration API.

Package verification is not model/device acceptance. SwiftPM embedding and
signing, actual Metal execution and sampling, constrained text/tool generation,
cancellation, full-app memory/thermal behavior and battery cost remain required
before GuideAI adopts this SDK. The package still requires its new build gate;
these source changes alone make no runtime acceptance claim.
