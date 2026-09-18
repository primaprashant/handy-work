use super::{plan_first_span, plan_spans, ChunkPolicy};
use std::ops::Range;

/// Sample-preserving coordinator for Cohere's rolling 50-second planning
/// horizon. It owns only the uncommitted suffix; VAD analysis and model calls
/// remain in the transcription worker.
pub(in crate::managers::transcription) struct RollingChunkCoordinator {
    uncommitted: Vec<f32>,
    absolute_start: usize,
    high_water_samples: usize,
    policy: ChunkPolicy,
}

impl RollingChunkCoordinator {
    pub(in crate::managers::transcription) fn new(policy: ChunkPolicy) -> Self {
        assert!(policy.soft_min_samples <= policy.hard_max_samples);
        Self {
            uncommitted: Vec::new(),
            absolute_start: 0,
            high_water_samples: 0,
            policy,
        }
    }

    pub(in crate::managers::transcription) fn push(&mut self, samples: &[f32]) {
        self.uncommitted.extend_from_slice(samples);
        self.high_water_samples = self.high_water_samples.max(self.uncommitted.len());
    }

    pub(in crate::managers::transcription) fn planning_window(&self) -> Option<&[f32]> {
        let horizon = self.rolling_horizon_samples();
        (self.uncommitted.len() >= horizon).then(|| &self.uncommitted[..horizon])
    }

    /// Choose a prefix from the exact rolling window without treating its
    /// artificial end as the end of the recording. Everything after the
    /// committed prefix remains available for the next decision.
    pub(in crate::managers::transcription) fn next_ready_span(
        &self,
        pauses: &[Range<usize>],
    ) -> Option<Range<usize>> {
        self.planning_window()
            .and_then(|window| plan_first_span(window, pauses, self.policy))
    }

    pub(in crate::managers::transcription) fn commit(
        &mut self,
        span: Range<usize>,
    ) -> Range<usize> {
        assert_eq!(span.start, 0, "rolling commits must remove an exact prefix");
        assert!(span.end > 0 && span.end <= self.uncommitted.len());

        let absolute = self.absolute_start..self.absolute_start + span.end;
        self.uncommitted.drain(..span.end);
        self.absolute_start = absolute.end;
        absolute
    }

    pub(in crate::managers::transcription) fn final_spans(
        &self,
        pauses: &[Range<usize>],
    ) -> Vec<Range<usize>> {
        plan_spans(&self.uncommitted, pauses, self.policy)
    }

    pub(in crate::managers::transcription) fn absolute_range(
        &self,
        local: &Range<usize>,
    ) -> Range<usize> {
        self.absolute_start + local.start..self.absolute_start + local.end
    }

    pub(in crate::managers::transcription) fn uncommitted_audio(&self) -> &[f32] {
        &self.uncommitted
    }

    pub(in crate::managers::transcription) fn uncommitted_samples(&self) -> usize {
        self.uncommitted.len()
    }

    pub(in crate::managers::transcription) fn high_water_samples(&self) -> usize {
        self.high_water_samples
    }

    pub(in crate::managers::transcription) fn rolling_horizon_samples(&self) -> usize {
        self.policy
            .hard_max_samples
            .saturating_add(self.policy.soft_min_samples)
    }
}

/// Replay a complete waveform through the same retained-window policy used by
/// the live rolling worker. Analysis failures are sticky: after the first
/// failure, every remaining boundary is planned with energy only.
pub(in crate::managers::transcription) fn plan_rolling_replay_spans<F, E>(
    audio: &[f32],
    policy: ChunkPolicy,
    mut analyze_pauses: F,
) -> Vec<Range<usize>>
where
    F: FnMut(&[f32]) -> Result<Vec<Range<usize>>, E>,
{
    let mut coordinator = RollingChunkCoordinator::new(policy);
    let mut spans = Vec::new();
    let mut analysis_failed = false;
    coordinator.push(audio);

    while let Some(window) = coordinator.planning_window() {
        let pauses = if analysis_failed {
            Vec::new()
        } else {
            match analyze_pauses(window) {
                Ok(pauses) => pauses,
                Err(_) => {
                    analysis_failed = true;
                    Vec::new()
                }
            }
        };
        let local_span = coordinator
            .next_ready_span(&pauses)
            .expect("a full rolling horizon always has a planned span");
        spans.push(coordinator.commit(local_span));
    }

    let final_pauses =
        if coordinator.uncommitted_samples() <= policy.hard_max_samples || analysis_failed {
            Vec::new()
        } else {
            analyze_pauses(coordinator.uncommitted_audio()).unwrap_or_default()
        };
    spans.extend(
        coordinator
            .final_spans(&final_pauses)
            .iter()
            .map(|span| coordinator.absolute_range(span)),
    );

    debug_assert!(super::spans_have_full_coverage(
        &spans,
        audio.len(),
        policy.hard_max_samples
    ));
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ChunkPolicy {
        // One sample represents one second in these pure range-policy tests.
        ChunkPolicy {
            hard_max_samples: 35,
            preferred_samples: 30,
            soft_min_samples: 15,
            min_pause_samples: 2,
            energy_window_samples: 1,
        }
    }

    fn audio(seconds: usize) -> Vec<f32> {
        vec![0.5; seconds]
    }

    fn zero_runs(samples: &[f32]) -> Vec<Range<usize>> {
        let mut pauses = Vec::new();
        let mut start = None;
        for (index, sample) in samples.iter().enumerate() {
            if *sample == 0.0 {
                start.get_or_insert(index);
            } else if let Some(pause_start) = start.take() {
                pauses.push(pause_start..index);
            }
        }
        if let Some(pause_start) = start {
            pauses.push(pause_start..samples.len());
        }
        pauses
    }

    fn simulate(samples: &[f32], feed_samples: usize) -> (Vec<Range<usize>>, usize) {
        simulate_with(samples, feed_samples, zero_runs)
    }

    fn simulate_with<F>(
        samples: &[f32],
        feed_samples: usize,
        mut detect_pauses: F,
    ) -> (Vec<Range<usize>>, usize)
    where
        F: FnMut(&[f32]) -> Vec<Range<usize>>,
    {
        let mut coordinator = RollingChunkCoordinator::new(policy());
        let mut spans = Vec::new();

        for feed in samples.chunks(feed_samples.max(1)) {
            coordinator.push(feed);
            while let Some(window) = coordinator.planning_window() {
                let pauses = detect_pauses(window);
                let span = coordinator.next_ready_span(&pauses).unwrap();
                spans.push(coordinator.commit(span));
            }
        }

        let suffix_samples = coordinator.uncommitted_samples();
        let final_pauses = if suffix_samples > policy().hard_max_samples {
            detect_pauses(coordinator.uncommitted_audio())
        } else {
            Vec::new()
        };
        spans.extend(
            coordinator
                .final_spans(&final_pauses)
                .iter()
                .map(|span| coordinator.absolute_range(span)),
        );

        (spans, suffix_samples)
    }

    fn assert_invariants(spans: &[Range<usize>], audio_len: usize) {
        if audio_len == 0 {
            assert!(spans.is_empty());
            return;
        }

        assert_eq!(spans.first().unwrap().start, 0);
        assert_eq!(spans.last().unwrap().end, audio_len);
        for (index, span) in spans.iter().enumerate() {
            assert!(span.start < span.end);
            assert!(span.len() <= policy().hard_max_samples);
            if index > 0 {
                assert_eq!(spans[index - 1].end, span.start);
            }
        }
    }

    #[test]
    fn empty_audio_has_no_spans() {
        let (spans, suffix) = simulate(&[], 7);
        assert!(spans.is_empty());
        assert_eq!(suffix, 0);
    }

    #[test]
    fn short_and_exact_hard_maximum_audio_finalize_once() {
        for seconds in [20, 35] {
            let (spans, suffix) = simulate(&audio(seconds), 7);
            assert_eq!(spans, vec![0..seconds]);
            assert_eq!(suffix, seconds);
        }
    }

    #[test]
    fn suffix_between_hard_maximum_and_horizon_uses_two_calls() {
        let (spans, suffix) = simulate(&audio(49), 8);
        assert_eq!(spans.len(), 2);
        assert_eq!(suffix, 49);
        assert_invariants(&spans, 49);
    }

    #[test]
    fn exact_horizon_commits_in_background() {
        let horizon = policy().hard_max_samples + policy().soft_min_samples;
        let (spans, suffix) = simulate(&audio(horizon), horizon);
        assert_eq!(spans.len(), 2);
        assert!(suffix < horizon);
        assert_invariants(&spans, horizon);
    }

    #[test]
    fn long_recording_leaves_less_than_one_horizon() {
        let samples = audio(10 * 60);
        let (spans, suffix) = simulate(&samples, 3);

        assert!(spans.len() > 10);
        assert!(suffix < policy().hard_max_samples + policy().soft_min_samples);
        assert_invariants(&spans, samples.len());
    }

    #[test]
    fn rolling_plan_uses_a_good_pause_near_the_preferred_boundary() {
        let mut samples = audio(80);
        samples[29..32].fill(0.0);

        let (spans, _) = simulate(&samples, 5);

        assert!((29..32).contains(&spans[0].end));
        assert_invariants(&spans, samples.len());
    }

    #[test]
    fn continuous_speech_uses_energy_fallback() {
        let samples = audio(140);
        let (spans, suffix) = simulate(&samples, 4);

        assert!(spans.len() >= 4);
        assert!(suffix < policy().hard_max_samples + policy().soft_min_samples);
        assert_invariants(&spans, samples.len());
    }

    #[test]
    fn rolling_plan_uses_one_of_the_available_pause_competitors() {
        let mut samples = audio(80);
        samples[26..30].fill(0.0);
        samples[32..34].fill(0.0);

        let (spans, _) = simulate(&samples, 6);

        assert!((26..30).contains(&spans[0].end) || (32..34).contains(&spans[0].end));
        assert_invariants(&spans, samples.len());
    }

    #[test]
    fn constant_audio_and_ties_are_deterministic() {
        let samples = audio(180);
        let first = simulate(&samples, 11);
        let second = simulate(&samples, 11);
        assert_eq!(first, second);
    }

    #[test]
    fn non_finite_samples_preserve_range_invariants() {
        let mut samples = audio(130);
        samples[30] = f32::NAN;
        samples[90] = f32::INFINITY;

        let (spans, suffix) = simulate(&samples, 9);

        assert!(suffix < policy().hard_max_samples + policy().soft_min_samples);
        assert_invariants(&spans, samples.len());
    }

    #[test]
    fn final_suffix_can_require_one_or_two_calls() {
        let (one_call, _) = simulate(&audio(34), 10);
        assert_eq!(one_call.len(), 1);

        let (two_calls, _) = simulate(&audio(49), 10);
        assert_eq!(two_calls.len(), 2);
    }

    #[test]
    fn feed_granularity_does_not_change_ranges() {
        let mut samples = audio(240);
        samples[28..31].fill(0.0);
        samples[87..90].fill(0.0);
        samples[145..148].fill(0.0);

        let sample_at_a_time = simulate(&samples, 1);
        let queued_frames = simulate(&samples, 73);

        assert_eq!(sample_at_a_time, queued_frames);
        assert_invariants(&sample_at_a_time.0, samples.len());
    }

    #[test]
    fn complete_replay_matches_incremental_rolling_planning() {
        let mut samples = audio(240);
        samples[28..31].fill(0.0);
        samples[87..90].fill(0.0);
        samples[145..148].fill(0.0);

        let (incremental, _) = simulate(&samples, 13);
        let replay =
            plan_rolling_replay_spans(&samples, policy(), |window| Ok::<_, ()>(zero_runs(window)));

        assert_eq!(replay, incremental);
        assert_invariants(&replay, samples.len());
    }

    #[test]
    fn complete_replay_analyzes_exact_retained_windows_after_feeding_all_audio() {
        let samples: Vec<_> = (0..180).map(|sample| sample as f32).collect();
        let mut analyzed = Vec::new();

        let spans = plan_rolling_replay_spans(&samples, policy(), |window| {
            analyzed.push((window.len(), window[0]));
            Ok::<_, ()>(Vec::new())
        });

        let horizon = policy().hard_max_samples + policy().soft_min_samples;
        let rolling_windows: Vec<_> = analyzed.iter().filter(|(len, _)| *len == horizon).collect();
        assert!(!rolling_windows.is_empty());
        for (index, (_, first_sample)) in rolling_windows.iter().enumerate() {
            assert_eq!(*first_sample, spans[index].start as f32);
        }
        assert_invariants(&spans, samples.len());
    }

    #[test]
    fn complete_replay_uses_sticky_energy_fallback_after_analysis_error() {
        let samples = audio(180);
        let mut analysis_calls = 0;

        let fallback = plan_rolling_replay_spans(&samples, policy(), |_| {
            analysis_calls += 1;
            Err::<Vec<Range<usize>>, _>("offline analysis failed")
        });
        let energy_only =
            plan_rolling_replay_spans(&samples, policy(), |_| Ok::<_, ()>(Vec::new()));

        assert_eq!(analysis_calls, 1);
        assert_eq!(fallback, energy_only);
        assert_invariants(&fallback, samples.len());
    }
}
