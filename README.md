# Handy-work: Fixes for Qwen3-ASR

Handy supports Qwen3-ASR models, but the transcription engine [has a bug](https://github.com/handy-computer/transcribe.cpp/issues/95) that makes them unusable for long recordings. This fork makes Qwen3-ASR models work inside Handy for long-form dictation on macOS.

---

[Handy](https://github.com/cjpais/Handy) is the largest open-source voice typing tool. I use it every day. Big thanks to [CJ Pais](https://github.com/cjpais) for making it.

My main use for voice typing is long-form dictation. I often talk for five to ten minutes while explaining an idea, especially when I am giving an LLM or a coding agent enough context to do useful work.

Handy [doesn't work well](https://github.com/cjpais/Handy/issues/1332) for long-form dictation unless you're using a streaming model such as Parakeet Unified EN 0.6B or Nemotron Streaming 3.5. These models are fast, but their transcription quality is not good enough for my needs.

Handy doesn't implement any chunking. For non-streaming models, the entire recording is sent to the model for transcription at once. Models like **Cohere Transcribe** only handle about 30 to 35 seconds of audio at a time. So, when used for long-form dictation, these models start producing low-quality transcriptions and leave out a lot of what was said.

The Qwen3-ASR runtime accepts up to 1,200 seconds of audio. The 1,200-second input limit is enough for a 10-to-15-minute recording, so the full recording can be transcribed in one pass without splitting it into chunks. I also find that Qwen3-ASR 1.7B gives me good enough results in terms of WER and latency. All of this makes Qwen3-ASR 1.7B the best model for me.

So, I created this fork so that I can use the Qwen3-ASR 1.7B model. This is not an official Handy release.

## How Handy uses Qwen3-ASR through transcribe.cpp

The official Qwen3-ASR runtime accepts up to 1,200 seconds of audio. You can see this in Qwen's [`MAX_ASR_INPUT_SECONDS`](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py) setting. Qwen's own example uses `max_new_tokens=256` and says to increase it for long audio in the [Qwen3-ASR documentation](https://github.com/QwenLM/Qwen3-ASR/blob/main/README.md#python-package-usage).

The `transcribe.cpp` implementation used by Handy had a different limit. Its Qwen3-ASR decoder used a fixed `k_max_new = 256` for every recording. This limited the transcript to 256 generated tokens even when the model could accept much more audio. Once the decoder reached that limit, it returned an error and no transcript. This [hard limit of 256 tokens is a bug](https://github.com/handy-computer/transcribe.cpp/issues/95). In my use, recordings longer than roughly 30 seconds to one minute could hit the limit.

## What changed

I have made three changes:

1. The [`transcribe.cpp` fork](https://github.com/primaprashant/transcribe.cpp) scales the Qwen3-ASR output budget with audio length for single and batch transcription.
2. The engine fork's regression tests check both an 11-second recording and a 197-second recording. The long recording was truncated by the old 256-token limit and completes with the fix.
3. Handy pins `transcribe-cpp` to a tested revision of the engine fork in [`src-tauri/Cargo.toml`](src-tauri/Cargo.toml). The macOS build script signs and installs the local app in a repeatable way.

The personal build also disables Handy's official updater. An official release
therefore cannot replace this build. Update it from this repository instead.

## Keeping the forks up to date

Both this repository and the `transcribe.cpp` fork continue to incorporate upstream changes while preserving the long-form transcription fix. Changes stay small and localized to minimize merge conflicts and maintenance.

The engine revision is pinned in [`src-tauri/Cargo.toml`](src-tauri/Cargo.toml) and resolved in [`src-tauri/Cargo.lock`](src-tauri/Cargo.lock) for reproducible builds. Advance these together as the engine fork evolves, verifying that long-form transcription still works. These files are the source of truth for the dependency version.

## Install on macOS

The commands below are for macOS. I have tested this on Apple Silicon.

Install these tools first:

- [Rust](https://rustup.rs/)
- [Bun](https://bun.sh/)
- [CMake](https://cmake.org/)
- Xcode Command Line Tools: `xcode-select --install`

Then clone and install the app:

```bash
git clone https://github.com/primaprashant/handy-work.git
cd handy-work
make personal-setup
make personal-install
```

`make personal-setup` is a one-time step. It installs the project dependencies and opens Keychain Access so you can create a local code-signing certificate. Follow the instructions printed in the terminal. If Keychain asks whether `codesign` may use the private key, choose **Always Allow**.

`make personal-install` builds Handy, signs it with that certificate, replaces `/Applications/Handy.app`, and opens it. Grant Microphone and Accessibility access when macOS asks.

The stable certificate matters because macOS uses an app's code-signing requirement when it tracks permissions. Reusing the same certificate lets new local builds keep their existing permissions. The install script resets Handy's permissions only when that requirement changes. Apple's [code-signing requirements note](https://developer.apple.com/documentation/technotes/tn3127-inside-code-signing-requirements) explains this identity model.

After Handy opens, go to the model list and download either Qwen3-ASR 0.6B or Qwen3-ASR 1.7B. Select it as the active model and use Handy normally.

See [`BUILD.md`](BUILD.md) for the full build notes.

## More information

- [`BUILD.md`](BUILD.md) has detailed build and troubleshooting notes.
- [The original Handy repository](https://github.com/cjpais/Handy) documents the main application.
- [The modified `transcribe.cpp` repository](https://github.com/primaprashant/transcribe.cpp) contains the inference fix and its tests.

This project keeps Handy's [MIT license](LICENSE).
