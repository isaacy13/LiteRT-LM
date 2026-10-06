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

## Engine ownership of the constraint vocabulary

Conversation creation and cloning now obtain their LLGuidance provider from the
engine. One mutex-protected entry retains the immutable vocabulary, keyed by
tokenizer identity, explicit EOS, configured special IDs and the exact ordered
stop sequences. A changed key replaces that entry; failed initialization preserves
the prior valid provider. Existing conversations retain their provider, and each
grammar and parser state is still created independently. External providers retain
their existing per-conversation ownership and do not evict the vocabulary.

The engine clears its cache before tearing down its tokenizer. Native constraints
still require their engine's tokenizer to remain alive. There is no global cache,
unbounded configuration map, cached conversation or shared parser state.

Ten controls exercise the actual engine API: repeated vocabulary construction,
stop/EOS/special-ID and tokenizer changes, separate engines, failure preservation,
external providers, retained parsers after eviction/clear and concurrent creation.
Two conversation controls (four existing configurations each) count real
SentencePiece vocabulary construction through actual Create and Clone paths.
Both Bazel and CMake describe the cache dependency. Native build/package gates,
actual model semantics and physical initialization/RAM/energy measurements remain
required; source changes alone establish no device performance improvement.

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

`native-byte-contract.yml` declares eleven complete tokenizer, constraint and
execution-manager targets, then eight selected conversation-cache cases before
building device and simulator framework slices. Host XML and logs are copied before
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

## Package failure diagnostics

The e730273 native gate passed 105 test cases and built both framework slices,
but `install_name_tool -id` failed during packaging. Its captured diagnostic was
missing from the traceback, and the temporary packaging files were deleted. The
root cause is unverified. Command failures now report the captured output and exit
code. Failed work is retained separately with source/archive identity and an
explicit rejection marker; the untouched Bazel framework archive is uploaded even
on failure. Existing evidence is never replaced. This enables diagnosis of the
actual binary without weakening any package or runtime acceptance check.

## Preserve signatures until final packaging

The exact 0e5e7aa gate built both slices and passed 105 native cases, but runner
Xcode 16.4 rejected the simulator C API ID rewrite after signature removal.
Retained headers show eight padding bytes between the string table and the old
signature location. Removing that signature leaves those bytes at the end of
`__LINKEDIT`. This is the observed failing tool sequence; complete binary
validation and the runner-specific cause still require verification.

Linkage changes now operate on signed copies before the existing final framework
signing replaces their invalidated signatures. Original binaries remain intact;
all source, platform, signature and dependency checks remain required. One real
retained simulator copy accepted ID/dependency changes and final signing on the
local newer Xcode. Its signature-removal output differs from the runner's, and
both command orders succeed locally, so this is not a runner reproduction or full
package acceptance. The first local probe's import failure and second probe's
incorrect byte-equality assumption are retained separately. A producer control
checks that linkage rewriting never strips signatures and still rewrites every
required dependency. The next complete SDK gate remains required before adoption.

## Allocation errors finish their native task

The external-sampler decoded-ID allocation error previously called the threaded
callback directly and returned without the native task finalizer. That could
leave an active task after the worker exited, delaying session completion.
The error now releases the executor and uses the existing finalizer, which owns
terminal callback delivery, dependent-task failure and active-task removal.

Serial and threaded managers share one decoded-ID allocation policy. Production
uses the same managed tensor-buffer allocation and shape; a per-manager injected
allocator permits deterministic allocation-failure coverage without global state
or real memory exhaustion. The new native regression runs for both managers,
requires exactly one error, bounded task/session/pool completion, explicit release,
and a fresh successful session on the same manager with the real allocator.
The full existing execution-manager suite is added to the host gate. The 292dec9
gate compiled and executed nine targets and 163 cases without skips; two fresh
session recovery variants failed. This does not establish phone OOM, cancellation,
memory, performance or SDK/package acceptance.

The full execution-manager target previously skipped three serial variants. Its
serial waits drive queued work and cannot interrupt an already running decode;
the timeout controls now require the delayed task to finish, the caller to wait
for the actual configured delay, and zero-time task/session/pool checks to succeed.
Threaded variants retain their original exact timeout assertions. The destructor
control runs its original pending-work/completed-callback check for both managers.
No skipped cases are accepted. The first owned 6c8706d gate was canceled after this
coverage issue was identified; its partial logs/results are retained, not treated
as a native test or SDK acceptance result.

That gate's injected error, callback and bounded completion checks succeeded;
the next session failed with `Decode called without prior prefill or decode`.
`FakeLlmExecutor::Reset` cleared its counters and decode state while retaining
its processed-token ledger. The resource manager could then skip a cached prefix
the fake could no longer decode. Reset now clears that stale ledger. A direct
native regression checks repeated reset, empty processed tokens, decode refusal
before prefill, and identical fresh prefill/decode output. The manager recovery
test and its original assertions are unchanged. The complete fake-executor target
joins the gate: ten targets, 175 cases, no failures, errors or skips required.
This corrects test-executor state; it is not evidence of a phone backend failure
or proof of fresh-session recovery until the corrected native gate executes.

CMake now compiles the shared `execution_manager.cc` implementation in one static
target linked by both derived manager targets and the resource-manager facade.
Its prior explicit target list omitted the new implementation; the generated
source glob only copied it. Apple Bazel compilation does not validate a complete
CMake build, which remains a separate verification requirement.
