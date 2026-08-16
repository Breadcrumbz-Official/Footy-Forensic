import type { AnalysisResult, Metric, Phase, PhaseKey } from './api';
import { PHASE_KEYS } from './api';


export type MetricStatus = 'good' | 'warn' | 'bad' | 'unknown';

export interface DisplayMetric {
  id: string;
  name: string;
  
  value: string | null;
  ideal: string;
  status: MetricStatus;
  score: number | null;
  
  reason: string | null;
  caveat: string | null;
  what: string | null;
  why: string | null;
  tip: string | null;
}

export interface DisplayPhase {
  key: PhaseKey;
  label: string;
  color: string;
  score: number | null;
  
  insufficient: boolean;
  partial: boolean;
  counted: number;
  skipped: number;
  metrics: DisplayMetric[];
  
  image: string | null;
  time: number | null;
  shiftedMs: number;
  ballFound: boolean;
  
  reposed: boolean;
  
  aiNote: string | null;
}


export interface PhaseFeedback {
  key: PhaseKey;
  label: string;
  color: string;
  strengths: string[];
  improvements: string[];
}

export interface DisplayResults {
  overall: number | null;
  phases: DisplayPhase[];
  feedback: PhaseFeedback[];
  warnings: string[];
  context: AnalysisResult['context'];
  timing: AnalysisResult['timing'];
}

export const PHASE_COLOR: Record<PhaseKey, string> = {
  plant: '#ffb800',
  contact: '#00df54',
  followThrough: '#38bdf8',
};


export const PHASE_SHORT: Record<PhaseKey, string> = {
  plant: 'PLANT',
  contact: 'CONTACT',
  followThrough: 'FOLLOW',
};

export const PHASE_HINT: Record<PhaseKey, string> = {
  plant: 'Moment your non-kicking foot plants beside the ball.',
  contact: 'Moment your kicking foot strikes the ball.',
  followThrough: 'Body position immediately after the ball leaves your foot.',
};

export function scoreColor(score: number | null): string {
  if (score === null) return '#6b7280';
  if (score >= 80) return '#00df54';
  if (score >= 60) return '#ffb800';
  return '#e03c3c';
}

function metricStatus(m: Metric): MetricStatus {
  
  
  if (m.uncertain || m.score === undefined || m.score === null) return 'unknown';
  if (m.score >= 80) return 'good';
  if (m.score >= 60) return 'warn';
  return 'bad';
}

function toDisplayMetric(m: Metric): DisplayMetric {
  return {
    id: m.id,
    name: m.label,
    value: m.valueText,
    ideal: m.idealText,
    status: metricStatus(m),
    score: m.score ?? null,
    reason: m.reason ?? null,
    caveat: m.caveat ?? null,
    what: m.feedback?.what ?? null,
    why: m.feedback?.why ?? null,
    tip: m.feedback?.tip ?? null,
  };
}


function phaseCoaching(phase: Phase): { strengths: string[]; improvements: string[] } {
  const strengths: { text: string; rank: number }[] = [];
  const improvements: { text: string; rank: number }[] = [];

  for (const m of phase.metrics) {
    const status = metricStatus(m);
    const what = m.feedback?.what;
    if (!what || status === 'unknown') continue;

    if (status === 'good') {
      strengths.push({ text: what, rank: (m.score ?? 0) * m.weight });
    } else {
      const tip = m.feedback?.tip;
      improvements.push({
        text: tip ? `${what} ${tip}` : what,
        rank: (100 - (m.score ?? 0)) * m.weight,
      });
    }
  }

  const top = (xs: { text: string; rank: number }[]) =>
    xs.sort((a, b) => b.rank - a.rank).map(x => x.text);

  return { strengths: top(strengths), improvements: top(improvements) };
}

export function toDisplay(result: AnalysisResult): DisplayResults {
  const phases: DisplayPhase[] = PHASE_KEYS.map(key => {
    const p = result.phases[key];
    const frame = result.frames?.[key];
    return {
      key,
      label: p.label,
      color: PHASE_COLOR[key],
      score: p.score,
      insufficient: p.insufficient,
      partial: p.partial,
      counted: p.counted,
      skipped: p.skipped,
      metrics: p.metrics.map(toDisplayMetric),
      image: frame?.image ?? null,
      time: frame?.time ?? null,
      shiftedMs: frame?.shiftedMs ?? 0,
      ballFound: Boolean(frame?.ball),
      reposed: Boolean(frame?.reposed),
      aiNote: result.aiFeedback?.[key] ?? null,
    };
  });

  const feedback: PhaseFeedback[] = PHASE_KEYS.map(key => {
    const p = result.phases[key];
    return { key, label: p.label, color: PHASE_COLOR[key], ...phaseCoaching(p) };
  });

  return {
    overall: result.overall,
    phases,
    feedback,
    warnings: result.warnings ?? [],
    context: result.context,
    timing: result.timing,
  };
}
