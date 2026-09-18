use crate::audio_toolkit::VoiceActivityDetector;
use anyhow::Result;
use std::cmp::Ordering;
use std::ops::Range;

mod rolling;

pub(super) use rolling::{plan_rolling_replay_spans, RollingChunkCoordinator};

const SAMPLE_RATE: usize = 16_000;

// These weights are deliberately private policy. The hard maximum and energy
// window come from Cohere's preprocessing; the remaining values are the
// quality-first starting point and can be tuned against the evaluation corpus
// without changing the integration API.
const CHUNK_CALL_COST: f64 = 2.0;
const LENGTH_COST_WEIGHT: f64 = 2.0;
const SHORT_SPAN_COST_WEIGHT: f64 = 20.0;
const FINAL_SHORT_SPAN_EXTRA_WEIGHT: f64 = 30.0;
const ENERGY_BOUNDARY_BASE_COST: f64 = 6.0;
const ENERGY_RMS_COST_WEIGHT: f64 = 2.0;
const VAD_RMS_COST_WEIGHT: f64 = 2.0;
const VAD_DURATION_COST_WEIGHT: f64 = 0.5;

#[derive(Clone, Copy, Debug)]
pub(super) struct ChunkPolicy {
    pub hard_max_samples: usize,
    pub preferred_samples: usize,
    pub soft_min_samples: usize,
    pub min_pause_samples: usize,
    pub energy_window_samples: usize,
}

impl Default for ChunkPolicy {
    fn default() -> Self {
        Self {
            hard_max_samples: 35 * SAMPLE_RATE,
            preferred_samples: 30 * SAMPLE_RATE,
            soft_min_samples: 15 * SAMPLE_RATE,
            min_pause_samples: SAMPLE_RATE / 5,      // 200 ms
            energy_window_samples: SAMPLE_RATE / 10, // 100 ms
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundaryKind {
    Start,
    VadPause,
    EnergyFallback,
    End,
}

#[derive(Clone, Copy, Debug)]
struct BoundaryCandidate {
    sample: usize,
    kind: BoundaryKind,
    pause_samples: usize,
    rms: f32,
}

#[derive(Clone, Copy, Debug)]
struct PlanState {
    cost: f64,
    chunks: usize,
    predecessor: Option<usize>,
}

impl PlanState {
    const UNREACHABLE: Self = Self {
        cost: f64::INFINITY,
        chunks: usize::MAX,
        predecessor: None,
    };
}

/// Run VAD over the complete waveform and return coalesced non-speech ranges.
/// The waveform is only inspected; no samples are copied into a speech-only
/// buffer or otherwise removed.
pub(super) fn detect_pause_intervals(
    audio: &[f32],
    vad: &mut dyn VoiceActivityDetector,
) -> Result<Vec<Range<usize>>> {
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let frame_samples = vad.frame_samples();
    if frame_samples == 0 {
        anyhow::bail!("VAD frame size must be non-zero");
    }

    vad.reset();
    let mut pauses = Vec::new();
    let mut pause_start = None;
    let mut padded_frame = vec![0.0; frame_samples];

    for offset in (0..audio.len()).step_by(frame_samples) {
        let end = (offset + frame_samples).min(audio.len());
        let is_speech = if end - offset == frame_samples {
            vad.is_voice(&audio[offset..end])?
        } else {
            padded_frame.fill(0.0);
            padded_frame[..end - offset].copy_from_slice(&audio[offset..end]);
            vad.is_voice(&padded_frame)?
        };

        if is_speech {
            if let Some(start) = pause_start.take() {
                pauses.push(start..offset);
            }
        } else if pause_start.is_none() {
            pause_start = Some(offset);
        }
    }

    if let Some(start) = pause_start {
        pauses.push(start..audio.len());
    }

    Ok(pauses)
}

/// Find a globally scored sequence of contiguous, non-overlapping spans.
/// Every returned span is non-empty and at most `hard_max_samples` long.
pub(super) fn plan_spans(
    audio: &[f32],
    pauses: &[Range<usize>],
    policy: ChunkPolicy,
) -> Vec<Range<usize>> {
    if audio.is_empty() {
        return Vec::new();
    }

    assert!(policy.hard_max_samples > 0);
    assert!(policy.preferred_samples > 0);
    assert!(policy.soft_min_samples > 0);
    assert!(policy.soft_min_samples <= policy.hard_max_samples);
    assert!(policy.energy_window_samples > 0);

    if audio.len() <= policy.hard_max_samples {
        return std::iter::once(0..audio.len()).collect();
    }

    let unique_candidates = boundary_candidates(audio, pauses, policy);

    let mut states = vec![PlanState::UNREACHABLE; unique_candidates.len()];
    states[0] = PlanState {
        cost: 0.0,
        chunks: 0,
        predecessor: None,
    };

    for end_index in 1..unique_candidates.len() {
        let end_candidate = unique_candidates[end_index];
        for start_index in (0..end_index).rev() {
            let start_sample = unique_candidates[start_index].sample;
            let span_samples = end_candidate.sample - start_sample;
            if span_samples > policy.hard_max_samples {
                break;
            }
            if !states[start_index].cost.is_finite() {
                continue;
            }

            let cost = states[start_index].cost
                + edge_cost(
                    span_samples,
                    &end_candidate,
                    end_index == unique_candidates.len() - 1,
                    policy,
                );
            let chunks = states[start_index].chunks + 1;
            let current = states[end_index];
            let cost_order = cost.total_cmp(&current.cost);
            let is_better = cost_order == Ordering::Less
                || (cost_order == Ordering::Equal
                    && (chunks < current.chunks
                        || (chunks == current.chunks
                            && current
                                .predecessor
                                .is_none_or(|previous| start_index < previous))));

            if is_better {
                states[end_index] = PlanState {
                    cost,
                    chunks,
                    predecessor: Some(start_index),
                };
            }
        }
    }

    let last_index = unique_candidates.len() - 1;
    if !states[last_index].cost.is_finite() {
        return fixed_maximum_spans(audio.len(), policy.hard_max_samples);
    }

    let mut boundaries = Vec::with_capacity(states[last_index].chunks + 1);
    let mut cursor = last_index;
    boundaries.push(unique_candidates[cursor].sample);
    while let Some(predecessor) = states[cursor].predecessor {
        cursor = predecessor;
        boundaries.push(unique_candidates[cursor].sample);
    }
    boundaries.reverse();

    let spans: Vec<_> = boundaries
        .windows(2)
        .map(|boundary| boundary[0]..boundary[1])
        .collect();
    debug_assert!(spans_have_full_coverage(
        &spans,
        audio.len(),
        policy.hard_max_samples
    ));
    spans
}

/// Choose the next committed prefix without treating the end of a rolling
/// lookahead window as the real end of the recording. Unlike `plan_spans`, this
/// scores only the first edge. The remaining audio is deliberately left for a
/// later window or the authoritative global suffix plan at stop time.
pub(super) fn plan_first_span(
    audio: &[f32],
    pauses: &[Range<usize>],
    policy: ChunkPolicy,
) -> Option<Range<usize>> {
    if audio.is_empty() {
        return None;
    }
    assert!(policy.hard_max_samples > 0);
    assert!(policy.preferred_samples > 0);
    assert!(policy.soft_min_samples > 0);
    assert!(policy.soft_min_samples <= policy.hard_max_samples);
    assert!(policy.energy_window_samples > 0);
    if audio.len() <= policy.hard_max_samples {
        return Some(0..audio.len());
    }

    boundary_candidates(audio, pauses, policy)
        .into_iter()
        .filter(|candidate| {
            candidate.sample >= policy.soft_min_samples
                && candidate.sample <= policy.hard_max_samples
                && candidate.kind != BoundaryKind::End
        })
        .min_by(|left, right| {
            edge_cost(left.sample, left, false, policy)
                .total_cmp(&edge_cost(right.sample, right, false, policy))
                .then_with(|| boundary_rank(left.kind).cmp(&boundary_rank(right.kind)))
                .then_with(|| left.sample.cmp(&right.sample))
        })
        .map(|candidate| 0..candidate.sample)
        .or_else(|| Some(0..policy.hard_max_samples.min(audio.len())))
}

fn boundary_candidates(
    audio: &[f32],
    pauses: &[Range<usize>],
    policy: ChunkPolicy,
) -> Vec<BoundaryCandidate> {
    let mut candidates =
        Vec::with_capacity(audio.len() / policy.energy_window_samples + pauses.len() + 2);
    candidates.push(BoundaryCandidate {
        sample: 0,
        kind: BoundaryKind::Start,
        pause_samples: 0,
        rms: 0.0,
    });

    for pause in pauses {
        let start = pause.start.min(audio.len());
        let end = pause.end.min(audio.len());
        let pause_samples = end.saturating_sub(start);
        if pause_samples < policy.min_pause_samples || start >= end {
            continue;
        }

        let (sample, rms) = quietest_cut(audio, start..end, policy.energy_window_samples);
        if sample > 0 && sample < audio.len() {
            candidates.push(BoundaryCandidate {
                sample,
                kind: BoundaryKind::VadPause,
                pause_samples,
                rms,
            });
        }
    }

    // A 100 ms grid makes the graph connected even during continuous speech.
    // Each point is scored with the RMS of the surrounding 100 ms window. The
    // grid includes exact 35-second multiples, avoiding accidental tiny spans
    // caused solely by a half-window offset.
    let mut sample = policy.energy_window_samples;
    while sample < audio.len() {
        let half_window = policy.energy_window_samples / 2;
        let window_start = sample.saturating_sub(half_window);
        let window_end = (window_start + policy.energy_window_samples).min(audio.len());
        candidates.push(BoundaryCandidate {
            sample,
            kind: BoundaryKind::EnergyFallback,
            pause_samples: 0,
            rms: rms(audio, window_start..window_end),
        });
        sample = match sample.checked_add(policy.energy_window_samples) {
            Some(next) => next,
            None => break,
        };
    }

    candidates.push(BoundaryCandidate {
        sample: audio.len(),
        kind: BoundaryKind::End,
        pause_samples: 0,
        rms: 0.0,
    });
    candidates.sort_by(|left, right| {
        left.sample
            .cmp(&right.sample)
            .then_with(|| boundary_rank(left.kind).cmp(&boundary_rank(right.kind)))
    });

    // Prefer a VAD-backed candidate when it lands on the same sample as an
    // energy-grid candidate. Start/end are unique because internal candidates
    // exclude those sample positions.
    let mut unique_candidates: Vec<BoundaryCandidate> = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        match unique_candidates.last_mut() {
            Some(previous) if previous.sample == candidate.sample => {
                if boundary_rank(candidate.kind) < boundary_rank(previous.kind) {
                    *previous = candidate;
                }
            }
            _ => unique_candidates.push(candidate),
        }
    }

    unique_candidates
}

pub(super) fn merge_english_chunks(texts: impl IntoIterator<Item = String>) -> String {
    texts
        .into_iter()
        .filter_map(|text| {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn edge_cost(
    span_samples: usize,
    ending_boundary: &BoundaryCandidate,
    is_final: bool,
    policy: ChunkPolicy,
) -> f64 {
    let preferred = policy.preferred_samples as f64;
    let length_deviation = (span_samples as f64 - preferred) / preferred;
    let mut cost = CHUNK_CALL_COST + LENGTH_COST_WEIGHT * length_deviation * length_deviation;

    if span_samples < policy.soft_min_samples {
        let shortfall =
            (policy.soft_min_samples - span_samples) as f64 / policy.soft_min_samples as f64;
        cost += SHORT_SPAN_COST_WEIGHT * shortfall * shortfall;
        if is_final {
            cost += FINAL_SHORT_SPAN_EXTRA_WEIGHT * shortfall * shortfall;
        }
    }

    cost + match ending_boundary.kind {
        BoundaryKind::VadPause => {
            let duration_ratio =
                policy.min_pause_samples as f64 / ending_boundary.pause_samples.max(1) as f64;
            VAD_RMS_COST_WEIGHT * ending_boundary.rms as f64
                + VAD_DURATION_COST_WEIGHT * duration_ratio
        }
        BoundaryKind::EnergyFallback => {
            ENERGY_BOUNDARY_BASE_COST + ENERGY_RMS_COST_WEIGHT * ending_boundary.rms as f64
        }
        BoundaryKind::Start | BoundaryKind::End => 0.0,
    }
}

fn quietest_cut(audio: &[f32], search: Range<usize>, energy_window_samples: usize) -> (usize, f32) {
    let search_len = search.end - search.start;
    let window_samples = energy_window_samples.min(search_len).max(1);
    let last_start = search.end - window_samples;
    let mut window_start = search.start;
    let mut energy_sum = squared_energy_sum(&audio[window_start..window_start + window_samples]);
    let mut best_start = window_start;
    let mut best_energy = energy_sum;

    while window_start < last_start {
        energy_sum -= sample_energy(audio[window_start]);
        energy_sum += sample_energy(audio[window_start + window_samples]);
        window_start += 1;
        if energy_sum.total_cmp(&best_energy) == Ordering::Less {
            best_energy = energy_sum;
            best_start = window_start;
        }
    }

    (
        best_start + window_samples / 2,
        (best_energy.max(0.0) / window_samples as f64).sqrt() as f32,
    )
}

fn rms(audio: &[f32], range: Range<usize>) -> f32 {
    if range.is_empty() {
        return 0.0;
    }
    (squared_energy_sum(&audio[range.clone()]) / range.len() as f64).sqrt() as f32
}

fn squared_energy_sum(audio: &[f32]) -> f64 {
    audio.iter().map(|&sample| sample_energy(sample)).sum()
}

fn sample_energy(sample: f32) -> f64 {
    // PCM should be finite and in [-1, 1]. Treat invalid/out-of-range values as
    // high energy so they cannot create an attractive but invalid boundary.
    let normalized = if sample.is_finite() {
        sample.clamp(-1.0, 1.0) as f64
    } else {
        1.0
    };
    normalized * normalized
}

fn boundary_rank(kind: BoundaryKind) -> u8 {
    match kind {
        BoundaryKind::Start => 0,
        BoundaryKind::VadPause => 1,
        BoundaryKind::EnergyFallback => 2,
        BoundaryKind::End => 3,
    }
}

fn fixed_maximum_spans(audio_len: usize, hard_max_samples: usize) -> Vec<Range<usize>> {
    let mut spans = Vec::new();
    let mut start = 0;
    while start < audio_len {
        let end = (start + hard_max_samples).min(audio_len);
        spans.push(start..end);
        start = end;
    }
    spans
}

fn spans_have_full_coverage(
    spans: &[Range<usize>],
    audio_len: usize,
    hard_max_samples: usize,
) -> bool {
    if audio_len == 0 {
        return spans.is_empty();
    }
    if spans.is_empty() || spans[0].start != 0 || spans.last().unwrap().end != audio_len {
        return false;
    }

    spans.iter().enumerate().all(|(index, span)| {
        span.start < span.end
            && span.end - span.start <= hard_max_samples
            && (index == 0 || spans[index - 1].end == span.start)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_toolkit::vad::VadFrame;

    fn samples(seconds: usize) -> Vec<f32> {
        vec![0.5; seconds * SAMPLE_RATE]
    }

    fn default_spans(audio: &[f32], pauses: &[Range<usize>]) -> Vec<Range<usize>> {
        plan_spans(audio, pauses, ChunkPolicy::default())
    }

    fn assert_full_coverage(spans: &[Range<usize>], audio_len: usize) {
        assert!(spans_have_full_coverage(
            spans,
            audio_len,
            ChunkPolicy::default().hard_max_samples
        ));
    }

    #[test]
    fn empty_input_has_no_spans() {
        assert!(default_spans(&[], &[]).is_empty());
    }

    #[test]
    fn short_and_exact_maximum_inputs_remain_single_spans() {
        let short = samples(20);
        assert_eq!(default_spans(&short, &[]), vec![0..short.len()]);

        let exact = samples(35);
        assert_eq!(default_spans(&exact, &[]), vec![0..exact.len()]);
    }

    #[test]
    fn uses_a_real_pause_near_the_preferred_duration() {
        let mut audio = samples(60);
        let pause = 29 * SAMPLE_RATE + SAMPLE_RATE / 2..30 * SAMPLE_RATE + SAMPLE_RATE / 2;
        audio[pause.clone()].fill(0.0);

        let spans = default_spans(&audio, std::slice::from_ref(&pause));

        assert_eq!(spans.len(), 2);
        assert!(pause.contains(&spans[0].end));
        assert_full_coverage(&spans, audio.len());
    }

    #[test]
    fn prefers_a_longer_pause_over_a_shorter_later_pause() {
        let mut audio = samples(55);
        let long_pause = 26 * SAMPLE_RATE + SAMPLE_RATE / 2..27 * SAMPLE_RATE + SAMPLE_RATE / 2;
        let short_pause = 30 * SAMPLE_RATE..30 * SAMPLE_RATE + SAMPLE_RATE / 5;
        audio[long_pause.clone()].fill(0.0);
        audio[short_pause.clone()].fill(0.0);

        let spans = default_spans(&audio, &[short_pause, long_pause.clone()]);

        assert_eq!(spans.len(), 2);
        assert!(long_pause.contains(&spans[0].end));
    }

    #[test]
    fn global_plan_avoids_a_greedy_short_final_tail() {
        let mut audio = samples(36);
        let late_pause = 34 * SAMPLE_RATE..35 * SAMPLE_RATE;
        audio[late_pause.clone()].fill(0.0);

        let spans = default_spans(&audio, &[late_pause]);

        assert_eq!(spans.len(), 2);
        assert!(spans
            .iter()
            .all(|span| span.end - span.start >= 15 * SAMPLE_RATE));
        assert_full_coverage(&spans, audio.len());
    }

    #[test]
    fn rolling_first_span_is_independent_of_the_artificial_window_end() {
        let mut short_window = samples(50);
        let pause = 29 * SAMPLE_RATE..31 * SAMPLE_RATE;
        short_window[pause.clone()].fill(0.0);
        let mut long_window = short_window.clone();
        long_window.extend(samples(15));

        let short = plan_first_span(
            &short_window,
            std::slice::from_ref(&pause),
            ChunkPolicy::default(),
        );
        let long = plan_first_span(&long_window, &[pause], ChunkPolicy::default());

        assert_eq!(short, long);
        assert!((29 * SAMPLE_RATE..31 * SAMPLE_RATE).contains(&short.unwrap().end));
    }

    #[test]
    fn rolling_first_span_uses_a_preferred_energy_fallback_in_continuous_speech() {
        let audio = samples(50);

        let span = plan_first_span(&audio, &[], ChunkPolicy::default()).unwrap();

        assert!((29 * SAMPLE_RATE..=31 * SAMPLE_RATE).contains(&span.end));
        assert!(span.len() <= ChunkPolicy::default().hard_max_samples);
    }

    #[test]
    fn rolling_first_span_never_commits_before_the_soft_minimum() {
        let mut audio = samples(50);
        let early_pause = 7 * SAMPLE_RATE..9 * SAMPLE_RATE;
        audio[early_pause.clone()].fill(0.0);

        let span = plan_first_span(
            &audio,
            std::slice::from_ref(&early_pause),
            ChunkPolicy::default(),
        )
        .unwrap();

        assert!(span.len() >= ChunkPolicy::default().soft_min_samples);
        assert!(!early_pause.contains(&span.end));
    }

    #[test]
    fn continuous_speech_uses_energy_fallback_with_full_coverage() {
        let audio = samples(75);
        let spans = default_spans(&audio, &[]);

        assert!(spans.len() >= 3);
        assert_full_coverage(&spans, audio.len());
    }

    #[test]
    fn quiet_background_noise_can_be_an_energy_only_boundary() {
        let mut audio = samples(60);
        let quiet = 29 * SAMPLE_RATE + SAMPLE_RATE / 2..30 * SAMPLE_RATE + SAMPLE_RATE / 2;
        audio[quiet].fill(0.01);

        let spans = default_spans(&audio, &[]);

        assert_eq!(spans.len(), 2);
        assert!((29 * SAMPLE_RATE..=31 * SAMPLE_RATE).contains(&spans[0].end));
    }

    #[test]
    fn ignores_short_pauses_and_uses_unusually_long_pauses() {
        let mut audio = samples(60);
        let short = 20 * SAMPLE_RATE..20 * SAMPLE_RATE + SAMPLE_RATE / 10;
        let long = 28 * SAMPLE_RATE..32 * SAMPLE_RATE;
        audio[short.clone()].fill(0.0);
        audio[long.clone()].fill(0.0);

        let spans = default_spans(&audio, &[short.clone(), long.clone()]);

        assert!(long.contains(&spans[0].end));
        assert!(!short.contains(&spans[0].end));
    }

    #[test]
    fn constant_audio_has_deterministic_tie_breaking() {
        let audio = samples(80);
        let first = default_spans(&audio, &[]);
        let second = default_spans(&audio, &[]);

        assert_eq!(first, second);
    }

    #[test]
    fn non_finite_samples_do_not_break_planning() {
        let mut audio = samples(72);
        audio[30 * SAMPLE_RATE] = f32::NAN;
        audio[60 * SAMPLE_RATE] = f32::INFINITY;

        let spans = default_spans(&audio, &[]);

        assert_full_coverage(&spans, audio.len());
    }

    #[test]
    fn merging_trims_and_omits_empty_chunks() {
        assert_eq!(
            merge_english_chunks([
                "  first chunk  ".to_string(),
                String::new(),
                "   ".to_string(),
                "second chunk\n".to_string(),
            ]),
            "first chunk second chunk"
        );
    }

    struct ScriptedVad {
        script: Vec<bool>,
        cursor: usize,
        frame_samples: usize,
    }

    impl VoiceActivityDetector for ScriptedVad {
        fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> Result<VadFrame<'a>> {
            let is_speech = self.script[self.cursor];
            self.cursor += 1;
            if is_speech {
                Ok(VadFrame::Speech(frame))
            } else {
                Ok(VadFrame::Noise)
            }
        }

        fn frame_samples(&self) -> usize {
            self.frame_samples
        }

        fn reset(&mut self) {
            self.cursor = 0;
        }
    }

    #[test]
    fn vad_analysis_coalesces_noise_without_changing_audio() {
        let frame_samples = 4;
        let audio: Vec<_> = (0..24).map(|sample| sample as f32).collect();
        let original = audio.clone();
        let mut vad = ScriptedVad {
            script: vec![true, false, false, true, false, false],
            cursor: 0,
            frame_samples,
        };

        let pauses = detect_pause_intervals(&audio, &mut vad).unwrap();

        assert_eq!(pauses, vec![4..12, 16..24]);
        assert_eq!(audio, original);
    }
}
