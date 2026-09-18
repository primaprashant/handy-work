# Cohere long-form transcription

Ported from the final four commits of the sibling `handy-cohere-exp` repository: pause-aware chunking, background processing, boundary-quality tuning, and retained results. Measurements below are retained from the August 2026 experiments, not a fresh benchmark.

## Current implementation

- Applies to `transcribe-cpp` sessions with architecture `cohere_asr`. Model calls are sequential through one loaded session; background inference overlaps recording.
- Live recording uses a **50-second lookahead**, choosing one prefix without treating the lookahead edge as the end of the recording. Background chunks are **15–35 seconds**, preferably around 30 seconds. The first background call starts after 50 seconds of audio.
- At stop, the remaining suffix uses global dynamic programming, also used for complete recordings in CLI/history transcription. Once the worker catches up, fewer than 50 seconds remain; the suffix needs at most two calls. Backlog can add to the stop wait. Inputs of at most 35 seconds use one call.
- Silero VAD ranks pauses of at least **200 ms**; **100 ms** energy candidates provide fallback boundaries. Planning preserves every supplied sample exactly once, without overlap. Disable capture VAD to preserve the original waveform; this code does not override that setting and cannot restore audio already discarded during capture.
- Chunk text is trimmed and space-joined, then normal output cleanup runs once. Cancellation is checked between calls; a failed chunk fails the operation rather than returning a partial transcript.

Implementation: [chunk planner](../src-tauri/src/managers/transcription/cohere/chunking.rs), [rolling coordinator](../src-tauri/src/managers/transcription/cohere/chunking/rolling.rs), [Cohere worker](../src-tauri/src/managers/transcription/cohere.rs).

## Final results

Four English recordings: **1,366.37 seconds, 2,909 reference words**, 16 kHz mono WAV. Model: `handy-computer/cohere-transcribe-03-2026-gguf/cohere-transcribe-03-2026-Q5_K_M.gguf`; Apple M5, Metal `MTL0`. One accuracy pass per recording; timing excludes model loading.

| Strategy                      |   Micro WER |   Macro WER | Sub / Del / Ins  | Surface CER | Total processing |
| ----------------------------- | ----------: | ----------: | ---------------- | ----------: | ---------------: |
| Full-recording global planner |     6.2564% |     5.8978% | 76 / 83 / 23     |     6.5865% |         31.013 s |
| Original rolling planner      |     6.8752% |     6.5889% | 74 / 99 / 27     |     7.0725% |         34.468 s |
| **Current rolling planner**   | **6.4627%** | **6.0502%** | **76 / 88 / 24** | **6.6313%** |     **33.951 s** |

| Recording stem                         |  Duration | Reference words | Full-global WER | Current rolling WER |
| -------------------------------------- | --------: | --------------: | --------------: | ------------------: |
| `08F5AF68-3968-43B9-A8D6-F360F9FE4B1B` | 399.080 s |             978 |         5.9305% |             6.1350% |
| `handy-1787958773`                     | 411.600 s |             842 |         7.7197% |             8.0760% |
| `handy-1788049411`                     | 330.450 s |             681 |         6.7548% |             7.0485% |
| `handy-1788061728`                     | 225.240 s |             408 |         3.1863% |             2.9412% |

The retained changes remove **12 of 200 rolling word errors (6%)**, closing two thirds of the original gap to full-global planning; six extra errors remain. Replay timing includes all inference and **does not measure microphone stop-to-result latency**. That latency still needs a manual measurement; a sub-second result was not established.

WER uses NFKC, lowercase, punctuation/symbols replaced with spaces, whitespace collapse, and literal numbers. Micro pools all words; macro averages per-recording WER. Surface CER retains punctuation and case. Filler cleanup was enabled: 34 of the global baseline's 83 deletions were reference `uh` tokens intentionally removed by cleanup. These scores measure cleaned output, not raw ASR alone. Older edited references and punctuation-removal scoring produced incompatible absolute scores; use the current references and scorer.

## Alternatives already evaluated

Avoid repeating these unchanged experiments on this corpus; new models or evidence may justify revisiting them.

| Alternative                                        | Finding and reason not retained                                                                                                                     |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| One long model call                                | Lost most of the original 399-second recording: 90.24% WER under the old scoring/reference, versus 5.04% for global chunking.                       |
| Fixed 30 / 35-second cuts                          | Old single-recording WER was 9.23% / 5.88%; arbitrary boundaries were worse than pause-aware planning (5.04%).                                      |
| Energy-only global planning                        | Old single-recording WER was 5.25%, with three more deletions than VAD + energy. Retain as fallback, not default.                                   |
| Increase old rolling horizon from 50 to 65 seconds | Still 200 errors; macro WER worsened and runtime rose 4.5%. Moving the artificial endpoint did not fix its planning bias.                           |
| Open-ended planning without a hard commit minimum  | Improved to 189 errors, but selected a 7.9-second chunk. The retained 15-second floor removed one more error without per-file regression.           |
| Raise rolling commit floor to 20 seconds           | 190 errors (6.5315% micro WER), with regressions on two recordings versus the retained 15-second floor.                                             |
| Require 300 ms VAD pauses                          | Full-global errors rose from 182 to 186 (6.3940% WER); keep 200 ms.                                                                                 |
| Blind suffix/prefix text deduplication             | Rejected before inference: the corrected reference confirms that the apparent repeated phrase was spoken twice. Removing it would create deletions. |

Punctuation/capitalization at joins remains imperfect. Conditional 1–2-second acoustic overlap with alignment, greedy pause-aware planning, and frequent VAD segmentation were **not benchmarked**; they are not demonstrated failures. Overlap needs evidence of better deletion and surface-error scores before accepting its reconciliation complexity.

## Maintenance and validation

The planner and rolling coordinator retain the experiment's policy unchanged. Cohere worker execution, batch planning, offline VAD, and their tests are isolated under `src-tauri/src/managers/transcription/cohere.rs` and `cohere/`. The only existing Rust files changed are `actions.rs` (start/finalize hooks and cancellation) and `managers/transcription.rs` (shared worker lifecycle and batch dispatch).

Reuse the existing audio router and engine lease when merging upstream. Preserve ordered delivery before finalize, return the engine before replying, distinguish an unavailable worker from a failed chunk, and treat empty Cohere output as a complete result. Apply output cleanup once after joining every successful chunk. The loaded architecture must be `cohere_asr`; Qwen and other batch models retain their original path. The engine pin, lockfile, app identity, signing configuration, and personal install scripts are unaffected by this port.

Run regression tests with:

```bash
cargo test --manifest-path src-tauri/Cargo.toml --locked --lib managers::transcription
```

The four WAV/reference pairs and Python scorer remain in the sibling checkout's `handy-long-form-corpus/`; large recordings and experiment tooling are not copied into this fork. To re-evaluate, run this fork's debug `handy --transcribe-file` against those WAVs with the same model and cleanup settings, saving same-stem `--json` outputs outside the corpus. Score them using that checkout's `score_long_form.py`. For rolling replay set `HANDY_ROLLING_REPLAY_BENCHMARK=1` (debug builds only); omit it for full-global planning.

Replay measures quality and total processing, not microphone stop latency. For that measurement, record over 50 seconds and inspect the background chunk logs and `Recording worker stop-to-raw-transcript latency` entry. Also exercise cancel, silence, and a subsequent recording. The existing 30-second finalize timeout still applies; substantial inference backlog can exceed it. Failed operations retain the existing WAV/history recovery path rather than pasting partial text.
