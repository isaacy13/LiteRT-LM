# GuideAI patches on LiteRT-LM v0.18.0

Branch `guideai/ios-gpu-vision-v0.18.0` = tag `v0.18.0` plus:

1. **Vision encoder GPU hint** — cherry-pick of upstream `903455d1` ("Hint that embedding
   text and vision encoders are fully GPU-delegated"). Without
   `SetHintFullyDelegatedToSingleDelegate(true)`, `Subgraph::ModifyGraphWithDelegateImpl`
   prepares every CPU kernel before delegation. Each `STABLEHLO_COMPOSITE` (`odml.rms_norm`)
   `Prepare` allocates its decomposition subgraph's arena for the full patch count and never
   frees it. Gemma 4's encoder has 112 such composites per signature
   (`vision_70/140/280`, up to `[1, 2520, 768]` fp32), roughly 3 GB in all. On iOS that
   allocation fails part-way through the third signature
   (`Node number N (STABLEHLO_COMPOSITE) failed to prepare` →
   `Failed to modify graph with delegate`); on macOS it succeeds and wastes the memory.
   Upstream main has the fix; v0.18.1 does not.

2. **iOS runtime packaging** — the release `CLiteRTLM.xcframework` links the Metal
   accelerator, Metal top-k sampler and Gemma constraint provider statically (Google-internal
   build). The open-source build links them dynamically from `prebuilt/ios_*`, so
   `tools/package_apple_runtime.py` wraps each pinned prebuilt as its own framework and
   rewrites install names (`patches/litert_*.patch`, `runtime/util/litert_util.cc`,
   `runtime/components/sampler_factory.cc`). CI: `.github/workflows/guideai-ios-runtime.yml`.
