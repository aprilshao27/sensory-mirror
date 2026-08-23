// Minimal procedural sound effects. No audio assets exist in the repo yet
// (see content_config.json's gestures.*.sound and objects.marker_1.sound),
// so gesture payoffs synthesize short tones with the Web Audio API instead
// of depending on files that don't exist.

const SensoryAudio = (() => {
  let ctx = null;

  function getContext() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  // Browsers block audio before a user gesture, but the mirror's own
  // triggers are camera-driven and never count as one — unlock on the
  // first click/keypress this page happens to see, so playback works once
  // that's occurred without requiring it up front.
  function unlock() {
    getContext();
    window.removeEventListener("pointerdown", unlock);
    window.removeEventListener("keydown", unlock);
  }
  window.addEventListener("pointerdown", unlock);
  window.addEventListener("keydown", unlock);

  function playTone({ freq, duration, type = "sine", startGain = 0.25, delay = 0 }) {
    const audioCtx = getContext();
    const t0 = audioCtx.currentTime + delay;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t0);
    gain.gain.setValueAtTime(0, t0);
    gain.gain.linearRampToValueAtTime(startGain, t0 + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, t0 + duration);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(t0);
    osc.stop(t0 + duration + 0.05);
  }

  // Held-smile payoff: bright ascending triad reads as an affirming "ding".
  function playChime() {
    playTone({ freq: 880, duration: 0.35, type: "sine", startGain: 0.2 });
    playTone({ freq: 1108.73, duration: 0.35, type: "sine", startGain: 0.15, delay: 0.06 });
    playTone({ freq: 1318.51, duration: 0.5, type: "sine", startGain: 0.15, delay: 0.12 });
  }

  // Mouth-open payoff: sawtooth with a fast downward pitch sweep, kazoo-honk-ish.
  function playBoing() {
    const audioCtx = getContext();
    const t0 = audioCtx.currentTime;
    const duration = 0.4;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(520, t0);
    osc.frequency.exponentialRampToValueAtTime(140, t0 + duration);
    gain.gain.setValueAtTime(0.001, t0);
    gain.gain.linearRampToValueAtTime(0.22, t0 + 0.03);
    gain.gain.exponentialRampToValueAtTime(0.001, t0 + duration);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(t0);
    osc.stop(t0 + duration + 0.05);
  }

  return { playChime, playBoing };
})();
