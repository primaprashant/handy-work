//! Fork-local Cohere long-form inference. Keep policy, worker execution, and
//! regression tests here so upstream lifecycle changes stay easy to merge.
use super::*;

mod chunking;

impl TranscriptionManager {
    pub(super) fn run_cohere_rolling_worker(
        &self,
        rx: mpsc::Receiver<StreamCmd>,
        session: &mut Session,
        run_options: &RunOptions,
        output_language: OutputLanguageEvidence,
        supported_languages: Vec<String>,
    ) -> Option<(mpsc::Sender<RecordingWorkerReply>, RecordingWorkerReply)> {
        if get_settings(&self.app_handle).vad_enabled {
            warn!(
                "Capture VAD is enabled in the current settings; Cohere rolling planning remains sample-preserving but cannot recover pauses already discarded during recording"
            );
        }

        let mut vad = None;
        run_rolling_batch_commands(
            rx,
            chunking::ChunkPolicy::default(),
            output_language,
            supported_languages,
            || self.rolling_cancel_requested.load(Ordering::Acquire),
            |audio| {
                if vad.is_none() {
                    vad = Some(self.create_long_form_vad()?);
                }
                chunking::detect_pause_intervals(
                    audio,
                    vad.as_mut().expect("VAD was initialized above"),
                )
            },
            |audio, absolute_range, call_number| {
                session
                    .run(audio, run_options)
                    .map(|transcript| (transcript.text, transcript.language))
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "Cohere rolling transcription failed on call {} for samples {}..{}: {}",
                            call_number,
                            absolute_range.start,
                            absolute_range.end,
                            error
                        )
                    })
            },
        )
    }

    pub(super) fn transcribe_cohere_chunks<C: Fn() -> bool>(
        &self,
        session: &mut Session,
        audio: &[f32],
        run_options: &RunOptions,
        settings: &AppSettings,
        is_cancelled: &C,
        model_detected_language: &mut Option<String>,
    ) -> Result<String> {
        if settings.vad_enabled {
            warn!(
                "Capture VAD is enabled in the current settings; Cohere long-form planning remains sample-preserving but cannot recover pauses already discarded during recording"
            );
        }

        let policy = chunking::ChunkPolicy::default();
        // The replay switch exists only to compare the live
        // coordinator against the authoritative full-global
        // batch path in local debug benchmarks. Release
        // builds must never inherit this process-wide knob.
        #[cfg(debug_assertions)]
        let rolling_replay = std::env::var("HANDY_ROLLING_REPLAY_BENCHMARK").as_deref() == Ok("1");
        #[cfg(not(debug_assertions))]
        let rolling_replay = false;
        let (planner, spans) = if rolling_replay {
            let spans = chunking::plan_rolling_replay_spans(audio, policy, |window| {
                self.detect_long_form_pauses(window).inspect_err(|error| {
                        warn!(
                            "Offline Silero analysis failed; falling back to energy-only Cohere rolling replay planning: {error}"
                        );
                    })
            });
            ("rolling-replay", spans)
        } else {
            let pauses = match self.detect_long_form_pauses(audio) {
                Ok(pauses) => pauses,
                Err(error) => {
                    warn!(
                        "Offline Silero analysis failed; falling back to energy-only Cohere chunk planning: {error}"
                    );
                    Vec::new()
                }
            };
            ("full-global", chunking::plan_spans(audio, &pauses, policy))
        };
        info!(
            "Selected Cohere long-form planner={planner}; planned {} chunks for {:.2}s of audio; exact sample ranges={spans:?}",
            spans.len(),
            audio.len() as f64 / 16_000.0
        );

        let transcripts = run_chunk_sequence(&spans, is_cancelled, |span, index, total| {
            session
                .run(&audio[span.clone()], run_options)
                .map(|transcript| (transcript.text, transcript.language))
                .map_err(|e| {
                    anyhow::anyhow!(
                        "transcribe-cpp transcription failed on chunk {}/{}: {}",
                        index + 1,
                        total,
                        e
                    )
                })
        })?;

        let mut texts = Vec::with_capacity(transcripts.len());
        for (text, language) in transcripts {
            // Keep the first model-provided language as the
            // evidence for the combined English transcript.
            if model_detected_language.is_none() {
                *model_detected_language = language;
            }
            texts.push(text);
        }
        Ok(chunking::merge_english_chunks(texts))
    }

    fn detect_long_form_pauses(&self, audio: &[f32]) -> Result<Vec<std::ops::Range<usize>>> {
        let mut vad = self.create_long_form_vad()?;
        chunking::detect_pause_intervals(audio, &mut vad)
    }

    fn create_long_form_vad(&self) -> Result<crate::audio_toolkit::SileroVad> {
        const SILERO_VAD_THRESHOLD: f32 = 0.3;
        let vad_path = self
            .app_handle
            .path()
            .resolve(
                "resources/models/silero_vad_v4.onnx",
                tauri::path::BaseDirectory::Resource,
            )
            .map_err(|error| anyhow::anyhow!("Failed to resolve VAD path: {error}"))?;
        crate::audio_toolkit::SileroVad::new(vad_path, SILERO_VAD_THRESHOLD)
            .map_err(|error| anyhow::anyhow!("Failed to create offline Silero VAD: {error}"))
    }
}

pub(super) fn should_use_long_form_chunking(model_arch: &str, audio_len: usize) -> bool {
    model_arch == "cohere_asr" && audio_len > chunking::ChunkPolicy::default().hard_max_samples
}

pub(super) fn recording_worker_path_available(
    intent: RecordingWorkerIntent,
    model_arch: Option<&str>,
    supports_streaming: bool,
) -> bool {
    match intent {
        RecordingWorkerIntent::NativeStream => supports_streaming,
        RecordingWorkerIntent::CohereRollingBatch => model_arch == Some("cohere_asr"),
    }
}

fn run_rolling_batch_commands<P, R, C>(
    rx: mpsc::Receiver<StreamCmd>,
    policy: chunking::ChunkPolicy,
    output_language: OutputLanguageEvidence,
    supported_languages: Vec<String>,
    is_cancelled: C,
    mut analyze_pauses: P,
    mut run_chunk: R,
) -> Option<(mpsc::Sender<RecordingWorkerReply>, RecordingWorkerReply)>
where
    P: FnMut(&[f32]) -> Result<Vec<std::ops::Range<usize>>>,
    R: FnMut(&[f32], std::ops::Range<usize>, usize) -> Result<(String, Option<String>)>,
    C: Fn() -> bool,
{
    let mut coordinator = chunking::RollingChunkCoordinator::new(policy);
    let mut raw_texts = Vec::new();
    let mut detected_language = None;
    let mut sticky_error = None;
    let mut vad_failed = false;
    let mut cancelled = false;
    let mut received_samples = 0usize;
    let mut background_calls = 0usize;

    while let Ok(cmd) = rx.recv() {
        match cmd {
            StreamCmd::Feed(pcm) => {
                received_samples = received_samples.saturating_add(pcm.len());
                if sticky_error.is_some() || cancelled {
                    continue;
                }
                coordinator.push(&pcm);

                while coordinator.planning_window().is_some() {
                    if is_cancelled() {
                        cancelled = true;
                        break;
                    }

                    let pauses = if vad_failed {
                        Vec::new()
                    } else {
                        match analyze_pauses(coordinator.planning_window().unwrap()) {
                            Ok(pauses) => pauses,
                            Err(error) => {
                                warn!(
                                    "Offline Silero analysis failed; using energy-only Cohere rolling planning for this recording: {error}"
                                );
                                vad_failed = true;
                                Vec::new()
                            }
                        }
                    };
                    let local_span = coordinator
                        .next_ready_span(&pauses)
                        .expect("a full rolling horizon always has a planned span");
                    let absolute_span = coordinator.absolute_range(&local_span);
                    let call_number = raw_texts.len() + 1;
                    let call_started = Instant::now();
                    let result = run_chunk(
                        &coordinator.uncommitted_audio()[local_span.clone()],
                        absolute_span.clone(),
                        call_number,
                    );
                    let call_elapsed = call_started.elapsed();

                    match result {
                        Ok((text, language)) => {
                            if detected_language.is_none() {
                                detected_language = language;
                            }
                            raw_texts.push(text);
                            let committed = coordinator.commit(local_span);
                            background_calls += 1;
                            info!(
                                "Cohere rolling chunk {} committed samples {}..{} ({:.2}s, {:?}, planner={})",
                                background_calls,
                                committed.start,
                                committed.end,
                                committed.len() as f64 / 16_000.0,
                                call_elapsed,
                                if vad_failed { "energy" } else { "vad+energy" }
                            );
                        }
                        Err(error) => {
                            error!("Cohere rolling chunk failed: {error}");
                            sticky_error = Some(error.to_string());
                            break;
                        }
                    }
                }
            }
            StreamCmd::Finalize(reply) => {
                let finalize_started = Instant::now();
                info!(
                    "Finalizing Cohere rolling transcription: received {:.2}s, background_calls={}, uncommitted {:.2}s, high_water {:.2}s",
                    received_samples as f64 / 16_000.0,
                    background_calls,
                    coordinator.uncommitted_samples() as f64 / 16_000.0,
                    coordinator.high_water_samples() as f64 / 16_000.0,
                );

                if cancelled || is_cancelled() {
                    return Some((
                        reply,
                        RecordingWorkerReply::Failed(
                            "Cohere rolling transcription was cancelled".to_string(),
                        ),
                    ));
                }
                if let Some(message) = sticky_error {
                    return Some((reply, RecordingWorkerReply::Failed(message)));
                }

                let suffix_samples = coordinator.uncommitted_samples();
                let pauses = if suffix_samples <= policy.hard_max_samples || vad_failed {
                    Vec::new()
                } else {
                    match analyze_pauses(coordinator.uncommitted_audio()) {
                        Ok(pauses) => pauses,
                        Err(error) => {
                            warn!(
                                "Offline Silero analysis failed; using energy-only Cohere final planning for this recording: {error}"
                            );
                            Vec::new()
                        }
                    }
                };
                let final_spans = coordinator.final_spans(&pauses);
                if suffix_samples < coordinator.rolling_horizon_samples() {
                    debug_assert!(final_spans.len() <= 2);
                }

                for (index, local_span) in final_spans.iter().enumerate() {
                    if is_cancelled() {
                        return Some((
                            reply,
                            RecordingWorkerReply::Failed(
                                "Cohere rolling transcription was cancelled".to_string(),
                            ),
                        ));
                    }

                    let absolute_span = coordinator.absolute_range(local_span);
                    let call_number = raw_texts.len() + 1;
                    let call_started = Instant::now();
                    match run_chunk(
                        &coordinator.uncommitted_audio()[local_span.clone()],
                        absolute_span.clone(),
                        call_number,
                    ) {
                        Ok((text, language)) => {
                            if detected_language.is_none() {
                                detected_language = language;
                            }
                            raw_texts.push(text);
                            info!(
                                "Cohere final chunk {}/{} transcribed samples {}..{} ({:.2}s, {:?})",
                                index + 1,
                                final_spans.len(),
                                absolute_span.start,
                                absolute_span.end,
                                absolute_span.len() as f64 / 16_000.0,
                                call_started.elapsed(),
                            );
                        }
                        Err(error) => {
                            error!("Cohere final chunk failed: {error}");
                            return Some((reply, RecordingWorkerReply::Failed(error.to_string())));
                        }
                    }
                }

                let output_language =
                    with_model_detected_language(output_language, detected_language);
                let text = chunking::merge_english_chunks(raw_texts);
                info!(
                    "Cohere rolling finalize produced {} raw characters in {:?}; final_suffix={:.2}s final_calls={}",
                    text.len(),
                    finalize_started.elapsed(),
                    suffix_samples as f64 / 16_000.0,
                    final_spans.len(),
                );
                return Some((
                    reply,
                    RecordingWorkerReply::Completed(FinalizedStreamText {
                        text,
                        output_language,
                        supported_languages,
                        empty_is_complete: true,
                    }),
                ));
            }
            StreamCmd::Cancel => return None,
        }
    }

    None
}

fn run_chunk_sequence<T, C, F>(
    spans: &[std::ops::Range<usize>],
    is_cancelled: &C,
    mut run: F,
) -> Result<Vec<T>>
where
    C: Fn() -> bool,
    F: FnMut(&std::ops::Range<usize>, usize, usize) -> Result<T>,
{
    let mut results = Vec::with_capacity(spans.len());
    for (index, span) in spans.iter().enumerate() {
        if index > 0 && is_cancelled() {
            anyhow::bail!("Transcription cancelled before chunk {}", index + 1);
        }
        results.push(run(span, index, spans.len())?);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn languages(codes: &[&str]) -> Vec<String> {
        codes.iter().map(|code| (*code).to_string()).collect()
    }

    fn rolling_test_policy() -> chunking::ChunkPolicy {
        chunking::ChunkPolicy {
            hard_max_samples: 35,
            preferred_samples: 30,
            soft_min_samples: 15,
            min_pause_samples: 2,
            energy_window_samples: 1,
        }
    }

    fn run_fake_rolling<P, R, C>(
        feeds: &[Vec<f32>],
        is_cancelled: C,
        analyze_pauses: P,
        run_chunk: R,
    ) -> RecordingWorkerReply
    where
        P: FnMut(&[f32]) -> Result<Vec<std::ops::Range<usize>>>,
        R: FnMut(&[f32], std::ops::Range<usize>, usize) -> Result<(String, Option<String>)>,
        C: Fn() -> bool,
    {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        for feed in feeds {
            cmd_tx.send(StreamCmd::Feed(feed.clone())).unwrap();
        }
        let (reply_tx, reply_rx) = mpsc::channel();
        cmd_tx.send(StreamCmd::Finalize(reply_tx)).unwrap();
        drop(cmd_tx);

        let (reply, result) = run_rolling_batch_commands(
            cmd_rx,
            rolling_test_policy(),
            OutputLanguageEvidence::Unknown,
            languages(&["en"]),
            is_cancelled,
            analyze_pauses,
            run_chunk,
        )
        .expect("fake worker should finalize");
        reply.send(result).unwrap();
        reply_rx.recv().unwrap()
    }

    #[test]
    fn long_form_chunking_is_only_for_long_cohere_asr_audio() {
        let policy = chunking::ChunkPolicy::default();

        assert!(!should_use_long_form_chunking(
            "cohere_asr",
            policy.hard_max_samples,
        ));
        assert!(should_use_long_form_chunking(
            "cohere_asr",
            policy.hard_max_samples + 1,
        ));
        for architecture in ["whisper", "parakeet", "moonshine", "cohere", "qwen3_asr"] {
            assert!(!should_use_long_form_chunking(
                architecture,
                policy.hard_max_samples + 1,
            ));
        }
    }

    #[test]
    fn recording_worker_intent_runtime_gates_cohere_by_architecture() {
        assert!(recording_worker_path_available(
            RecordingWorkerIntent::CohereRollingBatch,
            Some("cohere_asr"),
            false,
        ));
        for architecture in [
            None,
            Some("whisper"),
            Some("cohere"),
            Some("parakeet"),
            Some("qwen3_asr"),
        ] {
            assert!(!recording_worker_path_available(
                RecordingWorkerIntent::CohereRollingBatch,
                architecture,
                false,
            ));
        }
        assert!(recording_worker_path_available(
            RecordingWorkerIntent::NativeStream,
            Some("parakeet"),
            true,
        ));
        assert!(!recording_worker_path_available(
            RecordingWorkerIntent::NativeStream,
            Some("cohere_asr"),
            false,
        ));
    }

    #[test]
    fn rolling_inference_runs_before_finalize_is_sent() {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (call_tx, call_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let result = run_rolling_batch_commands(
                cmd_rx,
                rolling_test_policy(),
                OutputLanguageEvidence::Unknown,
                languages(&["en"]),
                || false,
                |_| Ok(Vec::new()),
                |_, range, _| {
                    call_tx.send(range).unwrap();
                    Ok(("speech".to_string(), None))
                },
            );
            if let Some((reply, result)) = result {
                reply.send(result).unwrap();
            }
        });
        cmd_tx.send(StreamCmd::Feed(vec![0.5; 50])).unwrap();
        let first = call_rx.recv_timeout(Duration::from_secs(5));
        let (reply_tx, reply_rx) = mpsc::channel();
        cmd_tx.send(StreamCmd::Finalize(reply_tx)).unwrap();
        let result = reply_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
        assert_eq!(first.unwrap().start, 0);
        assert!(matches!(result, RecordingWorkerReply::Completed(_)));
    }

    #[test]
    fn rolling_vad_failure_uses_energy_for_remaining_chunks() {
        let analyses = Cell::new(0);
        let calls = Cell::new(0);
        let reply = run_fake_rolling(
            &[vec![0.5; 180]],
            || false,
            |_| {
                analyses.set(analyses.get() + 1);
                anyhow::bail!("VAD unavailable");
            },
            |_, _, _| {
                calls.set(calls.get() + 1);
                Ok(("speech".to_string(), None))
            },
        );
        assert!(matches!(reply, RecordingWorkerReply::Completed(_)));
        assert_eq!(analyses.get(), 1);
        assert!(calls.get() > 2);
    }

    #[test]
    fn short_rolling_audio_runs_once_only_at_finalize() {
        let analyzed = Cell::new(0);
        let calls = Cell::new(0);
        let reply = run_fake_rolling(
            &[vec![0.5; 35]],
            || false,
            |_| {
                analyzed.set(analyzed.get() + 1);
                Ok(Vec::new())
            },
            |audio, range, call_number| {
                calls.set(calls.get() + 1);
                assert_eq!(call_number, 1);
                assert_eq!(range, 0..35);
                assert_eq!(audio.len(), 35);
                Ok(("short result".to_string(), Some("en".to_string())))
            },
        );

        let RecordingWorkerReply::Completed(finalized) = reply else {
            panic!("short rolling audio should complete");
        };
        assert_eq!(calls.get(), 1);
        assert_eq!(analyzed.get(), 0, "short audio should not initialize VAD");
        assert_eq!(finalized.text, "short result");
        assert!(finalized.empty_is_complete);
    }

    #[test]
    fn long_rolling_audio_has_exact_ordered_coverage_and_two_or_fewer_final_calls() {
        let audio = vec![0.5; 180];
        let visited = std::cell::RefCell::new(Vec::new());
        let reply = run_fake_rolling(
            std::slice::from_ref(&audio),
            || false,
            |_| Ok(Vec::new()),
            |chunk, range, call_number| {
                assert_eq!(chunk, &audio[range.clone()]);
                visited.borrow_mut().push(range);
                Ok((format!("chunk-{call_number}"), None))
            },
        );

        let RecordingWorkerReply::Completed(finalized) = reply else {
            panic!("long rolling audio should complete");
        };
        let visited = visited.into_inner();
        assert!(visited.len() > 2);
        assert_eq!(visited.first().unwrap().start, 0);
        assert_eq!(visited.last().unwrap().end, audio.len());
        for (index, range) in visited.iter().enumerate() {
            assert!(!range.is_empty());
            assert!(range.len() <= rolling_test_policy().hard_max_samples);
            if index > 0 {
                assert_eq!(visited[index - 1].end, range.start);
            }
        }

        let mut simulation = chunking::RollingChunkCoordinator::new(rolling_test_policy());
        simulation.push(&audio);
        let mut background_calls = 0;
        while simulation.planning_window().is_some() {
            let span = simulation.next_ready_span(&[]).unwrap();
            simulation.commit(span);
            background_calls += 1;
        }
        let final_calls = visited.len() - background_calls;
        assert!(background_calls > 0);
        assert!(final_calls <= 2);
        assert_eq!(
            finalized.text,
            (1..=visited.len())
                .map(|index| format!("chunk-{index}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    #[test]
    fn rolling_merge_omits_empty_chunks_after_all_calls_complete() {
        let reply = run_fake_rolling(
            &[vec![0.5; 80]],
            || false,
            |_| Ok(Vec::new()),
            |_, _, call_number| {
                let text = if call_number == 1 {
                    "  first  ".to_string()
                } else {
                    String::new()
                };
                Ok((text, None))
            },
        );

        let RecordingWorkerReply::Completed(finalized) = reply else {
            panic!("empty chunks are successful model results");
        };
        assert_eq!(finalized.text, "first");
    }

    #[test]
    fn rolling_middle_error_is_sticky_and_stops_later_calls() {
        let calls = Cell::new(0);
        let reply = run_fake_rolling(
            &[vec![0.5; 180], vec![0.5; 40]],
            || false,
            |_| Ok(Vec::new()),
            |_, _, call_number| {
                calls.set(calls.get() + 1);
                if call_number == 2 {
                    anyhow::bail!("simulated middle failure");
                }
                Ok((format!("chunk-{call_number}"), None))
            },
        );

        let RecordingWorkerReply::Failed(message) = reply else {
            panic!("a real chunk error must fail the whole transcription");
        };
        assert!(message.contains("simulated middle failure"));
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn rolling_cancellation_stops_before_the_next_model_call() {
        let cancelled = AtomicBool::new(false);
        let calls = Cell::new(0);
        let reply = run_fake_rolling(
            &[vec![0.5; 180]],
            || cancelled.load(Ordering::Acquire),
            |_| Ok(Vec::new()),
            |_, _, call_number| {
                calls.set(calls.get() + 1);
                cancelled.store(true, Ordering::Release);
                Ok((format!("chunk-{call_number}"), None))
            },
        );

        assert!(matches!(reply, RecordingWorkerReply::Failed(_)));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn chunk_sequence_runs_in_order_and_stops_on_error() {
        let spans = vec![0..10, 10..20, 20..30];
        let visited = Cell::new(0);

        let error = run_chunk_sequence(&spans, &|| false, |span, index, _| -> Result<usize> {
            visited.set(visited.get() + 1);
            if index == 1 {
                anyhow::bail!("middle chunk failed");
            }
            Ok(span.len())
        })
        .unwrap_err();

        assert_eq!(error.to_string(), "middle chunk failed");
        assert_eq!(visited.get(), 2);
    }

    #[test]
    fn chunk_sequence_returns_every_successful_result_in_order() {
        let spans = vec![0..10, 10..25, 25..30];

        let results = run_chunk_sequence(&spans, &|| false, |span, index, total| {
            assert_eq!(total, 3);
            Ok((index, span.clone()))
        })
        .unwrap();

        assert_eq!(results, vec![(0, 0..10), (1, 10..25), (2, 25..30)]);
    }

    #[test]
    fn chunk_sequence_observes_cancellation_between_calls() {
        let spans = vec![0..10, 10..20, 20..30];
        let completed = Cell::new(0);

        let error = run_chunk_sequence(&spans, &|| completed.get() == 1, |_, _, _| {
            completed.set(completed.get() + 1);
            Ok(())
        })
        .unwrap_err();

        assert_eq!(completed.get(), 1);
        assert!(error.to_string().contains("before chunk 2"));
    }
}
