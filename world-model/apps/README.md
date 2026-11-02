# Interactive world-model inspection

The active application is [r3b-playground](r3b-playground/README.md). Its folder name is retained for historical compatibility, while its default model is now **HMF500**, selected by the owner for the app's H1/AR4 use. HMF3000 is an explicit alternative for the saved 16-step comparison. Old R3B is not the default.

The app reads weights and pinned upstream source from configurable external paths; it never copies them into Git. A local `runtime.local.json` is ignored. Missing assets and startup failures are reported directly by the launcher. GPU inference uses one numerical CPU thread; physics, action conventions and the original sampler are unchanged.

Historical app bytes, including the original R3B loader and its cross-physics limitation, remain under `archive/app-before-hmf-20261103/`. Source/evidence mappings continue to point at those preserved historical versions. The current HMF model and current simulator both use the recorded original-physics protocol.
