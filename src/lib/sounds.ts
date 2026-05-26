/**
 * VisCurator — Web Audio Sound Synthesizer
 * -----------------------------------------
 * Generates distinct, pleasant UI notification sounds using the Web Audio API.
 * No external files required — all sounds are synthesized in real-time.
 *
 * Sound design:
 *   success  — rising two-tone chime (C5 → E5), soft and affirming
 *   error    — descending minor third buzz (A4 → F4), urgent but not harsh
 *   warning  — single mid-tone pulse (D5) with a slight wobble
 *   info     — short bright click (G5), neutral and quiet
 */

const STORAGE_KEY = 'vc_sounds_muted';

export function isMuted(): boolean {
  return localStorage.getItem(STORAGE_KEY) === 'true';
}

export function setMuted(muted: boolean): void {
  localStorage.setItem(STORAGE_KEY, muted ? 'true' : 'false');
  window.dispatchEvent(new CustomEvent('vc-mute-change', { detail: { muted } }));
}

export function toggleMuted(): boolean {
  const next = !isMuted();
  setMuted(next);
  return next;
}

/** Lazily created shared AudioContext — avoids hitting the browser limit. */
let _ctx: AudioContext | null = null;
function getCtx(): AudioContext {
  if (!_ctx || _ctx.state === 'closed') {
    _ctx = new AudioContext();
  }
  return _ctx;
}

/** Low-level helper: schedule an oscillator burst. */
function _tone(
  ctx: AudioContext,
  freq: number,
  startAt: number,
  duration: number,
  gain: number,
  type: OscillatorType = 'sine',
  detune = 0,
): void {
  const osc = ctx.createOscillator();
  const env = ctx.createGain();

  osc.type = type;
  osc.frequency.setValueAtTime(freq, startAt);
  osc.detune.setValueAtTime(detune, startAt);

  env.gain.setValueAtTime(0, startAt);
  env.gain.linearRampToValueAtTime(gain, startAt + 0.01);
  env.gain.exponentialRampToValueAtTime(0.0001, startAt + duration);

  osc.connect(env);
  env.connect(ctx.destination);
  osc.start(startAt);
  osc.stop(startAt + duration + 0.05);
}

// ── Public sound functions ────────────────────────────────────────────────────

export function playSuccess(): void {
  if (isMuted()) return;
  try {
    const ctx = getCtx();
    const t = ctx.currentTime;
    // Rising two-tone chime: C5 (523 Hz) then E5 (659 Hz)
    _tone(ctx, 523, t,        0.18, 0.22, 'sine');
    _tone(ctx, 659, t + 0.14, 0.24, 0.18, 'sine');
    // Soft harmonic shimmer
    _tone(ctx, 1046, t + 0.14, 0.20, 0.06, 'sine');
  } catch { /* AudioContext blocked — silently ignore */ }
}

export function playError(): void {
  if (isMuted()) return;
  try {
    const ctx = getCtx();
    const t = ctx.currentTime;
    // Descending minor third: A4 (440 Hz) → F4 (349 Hz), slight triangle buzz
    _tone(ctx, 440, t,        0.14, 0.20, 'triangle');
    _tone(ctx, 349, t + 0.10, 0.22, 0.18, 'triangle');
    // Low sub thump for urgency
    _tone(ctx, 110, t,        0.10, 0.12, 'sine');
  } catch { /* ignore */ }
}

export function playWarning(): void {
  if (isMuted()) return;
  try {
    const ctx = getCtx();
    const t = ctx.currentTime;
    // Single D5 pulse (587 Hz) with subtle frequency wobble (LFO via rapid param change)
    const osc = ctx.createOscillator();
    const env = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(587, t);
    osc.frequency.linearRampToValueAtTime(570, t + 0.08);
    osc.frequency.linearRampToValueAtTime(587, t + 0.16);
    env.gain.setValueAtTime(0, t);
    env.gain.linearRampToValueAtTime(0.20, t + 0.015);
    env.gain.exponentialRampToValueAtTime(0.0001, t + 0.28);
    osc.connect(env);
    env.connect(ctx.destination);
    osc.start(t);
    osc.stop(t + 0.32);
  } catch { /* ignore */ }
}

export function playInfo(): void {
  if (isMuted()) return;
  try {
    const ctx = getCtx();
    const t = ctx.currentTime;
    // Short bright G5 click (784 Hz), very quiet and quick
    _tone(ctx, 784, t, 0.10, 0.10, 'sine');
  } catch { /* ignore */ }
}

/** Map a toast type to its sound function. */
export const SOUND_MAP: Record<string, () => void> = {
  success: playSuccess,
  error:   playError,
  warning: playWarning,
  info:    playInfo,
};
