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

  // Mouth-open payoff: upbeat "ta-da" — a short anticipatory note followed by
  // a bright sustained major chord — reads as a cheer/fanfare instead of the
  // old downward-sweep honk, matching the reframe from startle to celebration.
  function playCheer() {
    playTone({ freq: 523.25, duration: 0.15, type: "triangle", startGain: 0.22 });
    playTone({ freq: 659.25, duration: 0.5, type: "triangle", startGain: 0.22, delay: 0.15 });
    playTone({ freq: 783.99, duration: 0.5, type: "triangle", startGain: 0.18, delay: 0.15 });
    playTone({ freq: 1046.5, duration: 0.6, type: "triangle", startGain: 0.15, delay: 0.17 });
  }

  // Cat-pet affirmation payoff: a real recorded meow rather than a synth
  // tone. "Cat Meow" by Mafon2, CC0 —
  // https://freesound.org/people/Mafon2/sounds/436541/ (Freesound HQ
  // preview mp3). Decoded once on first use through the same context the
  // tones use, so it's subject to the same unlock-on-first-gesture rule.
  let meowBuffer = null;
  let meowLoading = null;
  function loadMeow() {
    if (!meowLoading) {
      meowLoading = fetch("assets/sounds/meow.mp3")
        .then((res) => res.arrayBuffer())
        .then((data) => getContext().decodeAudioData(data))
        .then((buf) => (meowBuffer = buf))
        .catch((err) => console.warn("meow.mp3 failed to load", err));
    }
    return meowLoading;
  }

  function playMeow(volume = 0.6) {
    const play = () => {
      if (!meowBuffer) return;
      const audioCtx = getContext();
      const src = audioCtx.createBufferSource();
      const gain = audioCtx.createGain();
      src.buffer = meowBuffer;
      gain.gain.value = volume;
      src.connect(gain).connect(audioCtx.destination);
      src.start();
    };
    if (meowBuffer) play();
    else loadMeow().then(play);
  }

  // Preload so the first squeeze doesn't wait on the fetch.
  loadMeow();

  return { playChime, playCheer, playMeow };
})();
