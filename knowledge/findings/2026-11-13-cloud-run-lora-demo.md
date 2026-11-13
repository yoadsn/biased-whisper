---
type: Finding
title: Cloud Run L4 adapter demo provenance and bias semantics
description: Selected Phase 4 digit and cardinal checkpoints, base model identity, slider behavior, and private GPU deployment constraints.
tags: [finding, phase4, lora, cloud-run, deployment]
status: draft
generated: { by: opencode/gpt-6-sol, at: 2026-11-13T13:00:00Z }
---

# Checkpoints selected

The newest **completed** run for each direction on `technion-compute-113`
under `/home/yoadsnapir/whisper/biased-whisper/data/runs/` is:

- Cardinals: `phase4-20261107-173859-1395/epoch-01-step-001681` (1,681 updates).
- Digits: `phase4-digits-20261107-204452-834d/epoch-01-step-001269` (1,269 updates).

Both contain `bwhisper-lora-v1` adapter matrices at 32 identical decoder
sites, rank 8 and alpha 16. Both were trained on the same server-local
`ivrit-ai/whisper-large-v3-turbo` checkpoint. The deploy image downloads
HF revision `f33172a8c3c6efbc040a7200e00835257cac0447`; the checkpoint
metadata names the server path, not the HF revision. The downloaded HF
`model.safetensors` and `config.json` have the same SHA-256 hashes as the
server-local files (`600f71638a2c8e674a9bead39ddc16553efc969541e0b74e74e25572b58e1a58`
and `22df89eb892c95148f6beaf3eb6ed2a70cf496953923e8e1667730d06efb3a69`).

The 50-clip training monitor reported digit-clip rates of 74% for the
unadapted base, 2% for the cardinal adapter and 88% for the digit adapter.
These are prevalence measurements, **not transcription accuracy** or
independent confirmation. The default midpoint is unadapted; position
-100..-1 scales *only* the digit adapter, +1..+100 scales *only* the cardinal
adapter. Never sum both contributions.

See `deploy/README.md` for the local amd64 build and personal-account Cloud
Run commands. The service is intentionally IAM-private and limits itself to
30-second clips; longer files would otherwise be silently truncated by the
Whisper feature extractor. No model weights or adapters are version-controlled.

# Deployment verification

On 2026-11-13, the local amd64 image was built and pushed to Artifact Registry
as `hebrew-tts/biased-whisper-l4:20261113`, digest
`sha256:ddc34bd9084e745b3ab6fc7724bec3c26f314dc5ef9e0d560af095a717fe10eb`.
Cloud Run revision `biased-whisper-l4-00001-j9h` is ready in `europe-west1`
with one L4, 4 CPU, 16 GiB, concurrency 1, min instances 0 and initially
IAM-only invocation. `/ready` reports digits step 1269 and cardinals step 1681.
Authenticated requests for a 2.2-second existing Explore speech clip returned
`אנחנו ב-1850.` at -100 and midpoint, versus
`אנחנו באלף שמונה מאות חמישים.` at +100. This is a functional smoke test,
not an accuracy evaluation. The HTML UI loaded through the authenticated
Cloud Run proxy, with its slider, upload and submit controls visible.

# Browser access correction

The owner's ordinary browser received Forbidden: Cloud Run IAM-only invoker
authentication does not initiate Google login. Direct Cloud Run IAP was
enabled on the existing service on 2026-11-13. The IAP service agent received
service-scoped `roles/run.invoker`; `yoadsn@gmail.com` received service-scoped
`roles/iap.httpsResourceAccessor`. An unauthenticated HTTP request now returns
302 to Google's OAuth sign-in page instead of 403, and the IAP policy lists
only that user initially. The service-scoped IAP allowlist now contains
`yoadsn@gmail.com`, `yair@lifshitz.io` and `yanirmr@gmail.com`. A completed
Google-account login has **not yet** been verified; external identities might
require one-time External OAuth consent/client configuration in the Cloud
Console. No public invocation was enabled.
