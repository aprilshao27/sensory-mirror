// Phase 2 — Object recognition rendering.
// Renders per-marker visuals (see config.objects[marker_id] in
// content_config.json) in response to {"modality": "object", "markers": [
// {marker_id, x, y}, ...]} messages from backend/modules/object_recognition.py
// — a YOLO11s model fine-tuned on the 5 real items, marker_id being its
// predicted class id (0-4, matching content_config.json's marker_0..marker_4
// 1:1) — where x/y are normalised [0,1] mirroring gesture.js's
// face_landmarks convention. No angle: a bounding-box detector doesn't
// produce rotation the way the ArUco detector this replaced did; every
// visual below already treats angle as optional and degrades to
// translation-only reactivity without it.
//
// The DEV_SIMULATOR block at the bottom drives this from the mouse instead
// of real detections, for tuning visuals independently of camera/lighting/
// model conditions. Delete it once that's no longer needed.
//
// Which marker plays which visual is entirely config-driven (objCfg.visual
// dispatches to a renderer below), not hardcoded per marker_id here, so
// re-assigning markers to different physical items is a content_config.json
// edit, not a code change.
//
// drawAmbientWaveField is the one exception to "per-marker" — a shared,
// canvas-wide backdrop (config.ambient_wave) that colour-matches whichever
// adaptive_colour_ripple marker (the pop-it balls) was most recently active
// and gets pushed apart around that marker's live position, layered under
// everything else.

const ObjectModule = (() => {
  const markers = new Map(); // marker_id -> per-marker tracking + animation state

  // Shared, canvas-wide state for the ambient wave field (see drawAmbientWaveField
  // below) — not per-marker, since it's one layer shared across whichever pop-it
  // ball is currently the "active" one. Colors start null (not color(...)) since
  // this file loads before p5's global-mode functions exist; they're created
  // lazily in updateAmbientWaveField, the first time a candidate marker is found.
  const waveField = { presence: 0, pushX: 0, pushY: 0, colorA: null, colorB: null, colorC: null };

  const PRESENCE_EASE_IN = 0.25;
  const PRESENCE_EASE_OUT = 0.06;
  const MARKER_TIMEOUT_MS = 250; // no sighting within this long reads as "gone", not a blip
  const HISTORY_RETAIN_MS = 1200; // comfortably longer than any visual's max chain lag

  function getOrCreateState(id) {
    let state = markers.get(id);
    if (!state) {
      state = {
        history: [], // [{x, y, t}], oldest first, canvas pixel space
        lastSeen: -Infinity,
        presence: 0,
        smoothX: null,
        smoothY: null,
        ripples: [], // adaptive_colour_ripple
        nextRippleAt: 0,
        rippleEnergy: 0,
        squish: 0, // virtual_pet_companion
        wasSqueezed: false,
        lastAffirmationAt: -Infinity,
        affirmationShownAt: -Infinity,
        lastAffirmationText: null,
        nextBlinkAt: undefined, // simple_cat_animation
        blinkStart: -Infinity,
        pulsePhase: 0, // segmented_light_wave
      };
      markers.set(id, state);
    }
    return state;
  }

  function handleMessage(msg) {
    const now = millis();
    for (const m of msg.markers ?? []) {
      const state = getOrCreateState(m.marker_id);
      state.history.push({ x: m.x * width, y: m.y * height, angle: m.angle ?? null, t: now });
      state.lastSeen = now;
    }
  }

  function update(cfg) {
    updateSimulator();
    const now = millis();
    for (const [id, state] of markers) {
      const visible = now - state.lastSeen < MARKER_TIMEOUT_MS;
      state.presence += ((visible ? 1 : 0) - state.presence) * (visible ? PRESENCE_EASE_IN : PRESENCE_EASE_OUT);

      while (state.history.length && now - state.history[0].t > HISTORY_RETAIN_MS) {
        state.history.shift();
      }

      if (state.history.length) {
        const latest = state.history[state.history.length - 1];
        if (state.smoothX === null) {
          state.smoothX = latest.x;
          state.smoothY = latest.y;
        } else {
          state.smoothX += (latest.x - state.smoothX) * 0.2;
          state.smoothY += (latest.y - state.smoothY) * 0.2;
        }
      }

      if (state.presence < 0.005 && state.history.length === 0) markers.delete(id);
    }
    updateAmbientWaveField(cfg);
  }

  // Picks whichever adaptive_colour_ripple marker was most recently seen and
  // eases the shared wave field's colour/push-position/presence toward it —
  // deliberately "most recent wins", not a blend, so multiple balls held up
  // at once don't muddy the field's colour into something none of them
  // actually are.
  function updateAmbientWaveField(cfg) {
    let candidateId = null;
    let candidateLastSeen = -Infinity;
    if (cfg) {
      for (const [id, state] of markers) {
        if (cfg[`marker_${id}`]?.visual !== "adaptive_colour_ripple") continue;
        if (state.presence < 0.05) continue;
        if (state.lastSeen > candidateLastSeen) {
          candidateLastSeen = state.lastSeen;
          candidateId = id;
        }
      }
    }

    const easeRate = 0.1;
    if (candidateId !== null) {
      const state = markers.get(candidateId);
      const objCfg = cfg[`marker_${candidateId}`];
      const targetA = color(objCfg.ball_color_primary ?? "#8FE3FF");
      const targetB = color(objCfg.ball_color_secondary ?? objCfg.ball_color_primary ?? "#8FE3FF");
      const targetC = color(
        objCfg.ball_color_tertiary ?? objCfg.ball_color_secondary ?? objCfg.ball_color_primary ?? "#8FE3FF"
      );
      waveField.colorA = waveField.colorA ? lerpColor(waveField.colorA, targetA, easeRate) : targetA;
      waveField.colorB = waveField.colorB ? lerpColor(waveField.colorB, targetB, easeRate) : targetB;
      waveField.colorC = waveField.colorC ? lerpColor(waveField.colorC, targetC, easeRate) : targetC;
      waveField.pushX += (state.smoothX - waveField.pushX) * easeRate;
      waveField.pushY += (state.smoothY - waveField.pushY) * easeRate;
      waveField.presence += (state.presence - waveField.presence) * easeRate;
    } else {
      waveField.presence += (0 - waveField.presence) * 0.05; // slower fade-out than fade-in, less abrupt
    }
  }

  // ---- shared helpers ------------------------------------------------------

  // Interpolates the tracked path to find where the marker was at targetT,
  // clamping to the oldest sample once targetT runs off the front of the
  // buffer (e.g. right after a marker first appears, before enough history
  // has built up for a segment's full lag) rather than returning nothing.
  function sampleHistoryAt(history, targetT) {
    if (history.length === 0) return null;
    if (targetT <= history[0].t) return history[0];
    for (let i = 1; i < history.length; i++) {
      if (history[i].t >= targetT) {
        const a = history[i - 1];
        const b = history[i];
        const span = b.t - a.t;
        const tt = span > 0 ? (targetT - a.t) / span : 0;
        return { x: lerp(a.x, b.x, tt), y: lerp(a.y, b.y, tt) };
      }
    }
    return history[history.length - 1];
  }

  // Recent px/ms speed over a short trailing window — the shared "how
  // vigorously is this thing being handled right now" signal every visual
  // below scales its reactivity off, so a still object stays calm and an
  // actively-squeezed/waved one reads as energised.
  function recentSpeed(history, windowMs = 150) {
    if (history.length < 2) return 0;
    const now = history[history.length - 1].t;
    const cutoff = now - windowMs;
    let i = history.length - 1;
    while (i > 0 && history[i - 1].t > cutoff) i--;
    const a = history[i];
    const b = history[history.length - 1];
    const dt = b.t - a.t;
    return dt > 0 ? dist(a.x, a.y, b.x, b.y) / dt : 0;
  }

  // Recent rad/ms angular speed, same trailing-window shape as recentSpeed
  // above — a marker being twisted/flexed in place (imagine bending the
  // caterpillar fidget without moving it much through space) barely
  // registers on recentSpeed's translational distance, so this is the
  // second half of "how vigorously is this being handled": rotation, not
  // just position change. Returns 0 for markers with no angle data (the
  // dev simulator's synthetic messages don't set one), degrading gracefully
  // to translation-only reactivity rather than throwing.
  function recentAngularSpeed(history, windowMs = 150) {
    if (history.length < 2) return 0;
    const now = history[history.length - 1].t;
    const cutoff = now - windowMs;
    let i = history.length - 1;
    while (i > 0 && history[i - 1].t > cutoff) i--;
    const a = history[i];
    const b = history[history.length - 1];
    const dt = b.t - a.t;
    if (dt <= 0 || a.angle == null || b.angle == null) return 0;
    // Shortest signed angular distance, so a wrap from +PI to -PI reads as
    // a small turn rather than a spurious near-2*PI jump.
    const dAngle = Math.atan2(Math.sin(b.angle - a.angle), Math.cos(b.angle - a.angle));
    return Math.abs(dAngle) / dt;
  }

  // Like proxemic.js's cyclePalette, but takes an explicit phase instead of
  // deriving one from millis() internally, so callers (e.g. the caterpillar
  // chain below) can offset phase per-segment to make the colour travel
  // along a chain rather than pulse everywhere in sync.
  function paletteColorAt(colors, phase) {
    const n = colors.length;
    if (n === 0) return color(255);
    if (n === 1) return color(colors[0]);
    const t = ((phase % n) + n) % n;
    const i0 = Math.floor(t);
    const i1 = (i0 + 1) % n;
    return lerpColor(color(colors[i0]), color(colors[i1]), t - i0);
  }

  // Full-canvas ambient backdrop, sensory-room-style — concentric rings
  // centred on the screen, colour-matched to whichever pop-it ball is
  // currently active (see updateAmbientWaveField), reading as a steady
  // mandala/ripple-pond rather than something tied to any one ball's
  // position. Each ring's sample points are pushed radially away from the
  // ball's live position, so the rings visibly part around it as it moves
  // — like the ball disturbing a still pond — rather than it just sitting
  // on top of a static pattern. Deliberately not per-marker — this draws
  // once, under everything else, from the shared waveField state.
  //
  // The crest shape is a plain sine (broad, symmetric) plus a "dome" term —
  // sin() remapped to [0,1] and exponentiated, which rounds its peaks and
  // flattens its valleys into a repeating ring of soft convex humps, echoing
  // the balls' domed bump texture — kept low-amplitude and drifting at the
  // same slow pace as the broad sine so it reads as texture, not jitter.
  // The broad flow terms are counted "per revolution" (an integer-ish
  // number of humps around the ring) rather than a spatial frequency, since
  // there's no fixed "x axis" to measure a frequency against on a circle —
  // but the bump term specifically is scaled per-ring from an arc-length
  // spacing instead of a fixed count, so each bump occupies roughly the
  // same physical size on every ring, inner or outer, rather than the
  // outer rings' bumps stretching wider just because their circumference
  // is bigger — matching the real balls, whose nubs are a consistent size
  // across the whole surface regardless of where they sit on the sphere.
  function drawAmbientWaveField(cfg) {
    if (waveField.presence < 0.01 || !waveField.colorA) return;
    const now = millis();
    const ringCount = cfg?.ring_count ?? 16;
    const minSamplesPerRing = cfg?.samples_per_ring ?? 72;
    const influenceRadius = cfg?.influence_radius ?? 220;
    const pushStrength = cfg?.push_strength ?? 70;
    const ampA = cfg?.wave_amp_1 ?? 18;
    const ampB = cfg?.wave_amp_2 ?? 10;
    const countA = cfg?.wave_count_1 ?? 3;
    const countB = cfg?.wave_count_2 ?? 5;
    const speedA = cfg?.wave_speed_1 ?? 0.0006;
    const speedB = cfg?.wave_speed_2 ?? 0.0004;
    const domeAmp = cfg?.wave_bump_amp ?? 9;
    const domeArcSpacing = cfg?.wave_bump_arc_spacing ?? 42; // px of ring circumference per bump
    const domeSharpness = cfg?.wave_bump_sharpness ?? 2.2;

    const palette = [waveField.colorA, waveField.colorB, waveField.colorC];
    const centerX = width / 2;
    const centerY = height / 2;
    // Sized off the diagonal (centre-to-corner distance), not the smaller
    // of width/height — on a wide canvas the latter leaves the left/right
    // edges bare since the rings never reach past a centred circle inscribed
    // in the shorter dimension. The diagonal is the actual farthest point
    // from centre, so the outermost ring reaching it covers every corner.
    const maxRadius = Math.sqrt(centerX * centerX + centerY * centerY) * (cfg?.max_radius_fraction ?? 1.05);
    const spacing = maxRadius / (ringCount + 1);

    noFill();
    blendMode(ADD);
    drawingContext.filter = `blur(${cfg?.glow_blur ?? 3}px)`;
    for (let li = 0; li < ringCount; li++) {
      const baseRadius = spacing * (li + 1);
      const domeCount = Math.max(4, Math.round((TWO_PI * baseRadius) / domeArcSpacing));
      // At least ~6 samples per bump, and never fewer than the configured
      // floor, so bigger rings (more, tighter bumps) still render smoothly.
      const samplesPerRing = Math.max(minSamplesPerRing, domeCount * 6);
      const angleStep = TWO_PI / samplesPerRing;

      const c = paletteColorAt(palette, (li / ringCount) * palette.length + now * 0.00008);
      stroke(red(c), green(c), blue(c), (cfg?.stroke_alpha ?? 90) * waveField.presence);
      strokeWeight(cfg?.stroke_weight ?? 2);
      beginShape();
      for (let a = 0; a <= TWO_PI + angleStep; a += angleStep) {
        const domePhase = Math.sin(a * domeCount + li * 1.1 + now * speedA * 0.6);
        const dome = Math.pow((domePhase + 1) / 2, domeSharpness) * domeAmp;
        const radius =
          baseRadius +
          Math.sin(a * countA + li * 0.7 + now * speedA) * ampA +
          Math.sin(a * countB - li * 1.3 + now * speedB) * ampB +
          dome;
        const x = centerX + Math.cos(a) * radius;
        const y = centerY + Math.sin(a) * radius;

        const dx = x - waveField.pushX;
        const dy = y - waveField.pushY;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d > 0.001 && d < influenceRadius) {
          const falloff = 1 - d / influenceRadius;
          const push = falloff * falloff * pushStrength;
          vertex(x + (dx / d) * push, y + (dy / d) * push);
        } else {
          vertex(x, y);
        }
      }
      endShape();
    }
    drawingContext.filter = "none";
    blendMode(BLEND);
  }

  const VISUAL_RENDERERS = {
    adaptive_colour_ripple: drawColourRipple,
    virtual_pet_companion: drawVirtualPet,
    segmented_light_wave: drawSegmentedLightWave,
  };

  function draw(cfg, ambientCfg) {
    if (!cfg) return;
    drawAmbientWaveField(ambientCfg);
    for (const [id, state] of markers) {
      if (state.presence < 0.005 && state.history.length === 0) continue;
      const objCfg = cfg[`marker_${id}`];
      const renderer = objCfg && VISUAL_RENDERERS[objCfg.visual];
      if (renderer) renderer(state, objCfg);
    }
    drawDevOverlay();
  }

  // ---- marker_0-2: adaptive_colour_ripple (Pop It squishy balls) ----------
  //
  // Deliberately abstract, not a literal render of the ball — soft rings
  // emanate from the marker's live position and travel outward with it, and
  // a small marbled "aura" sits at the core. What makes each ball read
  // differently is purely its own ball_color_primary/secondary/tertiary in
  // content_config.json, cycled through as rings age rather than tied to
  // any literal geometry, so it reads as "this ball's colours in motion"
  // rather than a picture of the ball itself.
  //
  // Both the ring spawn rate and the aura's energy track how vigorously the
  // marker has been moving (recentSpeed), so a still ball settles into a
  // slow, quiet pulse and active handling ripples faster and wider — same
  // handling-energy-in -> visual-energy-out echo the other visuals use.

  function drawColourRipple(state, cfg) {
    if (state.presence < 0.01 || state.smoothX === null) return;
    const now = millis();

    const palette = [
      color(cfg.ball_color_primary ?? "#FF8FD6"),
      color(cfg.ball_color_secondary ?? cfg.ball_color_primary ?? "#FF8FD6"),
      color(cfg.ball_color_tertiary ?? cfg.ball_color_secondary ?? cfg.ball_color_primary ?? "#FF8FD6"),
    ];

    const speed = recentSpeed(state.history);
    const energyTarget = constrain(speed / (cfg.ripple_speed_ref ?? 0.5), 0, 1);
    state.rippleEnergy += (energyTarget - state.rippleEnergy) * 0.15;

    const spawnInterval = lerp(cfg.ripple_interval_ms ?? 650, cfg.ripple_min_interval_ms ?? 140, state.rippleEnergy);
    if (state.presence > 0.4 && now > state.nextRippleAt) {
      state.ripples.push({ x: state.smoothX, y: state.smoothY, spawnedAt: now, phaseSeed: random() });
      state.nextRippleAt = now + spawnInterval;
    }

    const maxRadius = lerp(cfg.ripple_max_radius_min ?? 60, cfg.ripple_max_radius_max ?? 150, state.rippleEnergy);
    const lifeMs = cfg.ripple_life_ms ?? 1300;

    noFill();
    blendMode(ADD);
    drawingContext.filter = `blur(${cfg.ripple_blur ?? 8}px)`;
    for (let i = state.ripples.length - 1; i >= 0; i--) {
      const r = state.ripples[i];
      const life = (now - r.spawnedAt) / lifeMs;
      if (life >= 1) {
        state.ripples.splice(i, 1);
        continue;
      }
      const radius = lerp(0, maxRadius, life);
      const alpha = (1 - life) * 210 * state.presence;
      const c = paletteColorAt(palette, (life + r.phaseSeed) * palette.length);
      stroke(red(c), green(c), blue(c), alpha);
      strokeWeight(cfg.ripple_stroke_weight ?? 4);
      circle(r.x, r.y, radius * 2);
    }
    drawingContext.filter = "none";
    blendMode(BLEND);

    // Marbled core aura: a handful of soft, drifting colour blobs orbiting
    // the current position — an abstract stand-in for the ball's tie-dye
    // marbling, not a literal render of it.
    const coreSize = cfg.core_size ?? 46;
    const auraLayers = cfg.aura_layers ?? 5;
    noStroke();
    blendMode(ADD);
    drawingContext.filter = `blur(${coreSize * 0.35}px)`;
    for (let i = 0; i < auraLayers; i++) {
      const angle = (i / auraLayers) * TWO_PI + now * 0.0004;
      const orbit = coreSize * 0.28 * (0.4 + 0.6 * noise(i * 3, now * 0.0006));
      const px = state.smoothX + Math.cos(angle) * orbit;
      const py = state.smoothY + Math.sin(angle) * orbit;
      const c = paletteColorAt(palette, noise(i * 1.7, now * 0.0003) * palette.length);
      fill(red(c), green(c), blue(c), state.presence * 130);
      circle(px, py, coreSize * lerp(0.6, 1, state.rippleEnergy));
    }
    drawingContext.filter = "none";
    blendMode(BLEND);
  }

  // ---- marker_3: virtual_pet_companion (purple cartoon cat) ---------------
  //
  // A stylised chibi cartoon cat — big round head, small body, doesn't try
  // to be a literal render of the real Squeezibo plush — just purple and
  // cute. Squash-stretch reactivity is driven by recentSpeed as a proxy for
  // squeezing — marker tracking only gives position, not actual pressure —
  // and that same signal doubles as the affirmation trigger: crossing
  // squeeze_trigger_threshold on a rising edge (not just "above it", so a
  // sustained squeeze doesn't refire every frame) counts as a deliberate
  // squeeze, subject to affirmation_cooldown_ms so it can't spam.

  function drawVirtualPet(state, cfg) {
    if (state.presence < 0.01 || state.smoothX === null) return;
    const now = millis();
    const alpha = state.presence;
    const bodyColor = color(cfg.body_color ?? "#8C5FCE");
    const bellyColor = color(cfg.belly_color ?? "#D9C6F0");
    const accentColor = color(cfg.accent_color ?? "#F2A6D8");
    const darkColor = color(cfg.dark_color ?? "#3A2E4A");
    const size = cfg.size ?? 340;

    const speed = recentSpeed(state.history);
    const squishTarget = constrain(speed / (cfg.squish_speed_ref ?? 0.5), 0, 1);
    state.squish += (squishTarget - state.squish) * 0.25;

    const squeezeThreshold = cfg.squeeze_trigger_threshold ?? 0.55;
    const isSqueezed = state.squish > squeezeThreshold;
    if (isSqueezed && !state.wasSqueezed && now - state.lastAffirmationAt > (cfg.affirmation_cooldown_ms ?? 4000)) {
      const pool = cfg.affirmations ?? ["you're doing great"];
      let pick = random(pool);
      if (pool.length > 1) while (pick === state.lastAffirmationText) pick = random(pool);
      state.lastAffirmationText = pick;
      state.affirmationShownAt = now;
      state.lastAffirmationAt = now;
    }
    state.wasSqueezed = isSqueezed;

    const breathe = 1 + 0.04 * Math.sin(now * 0.0015);
    const scaleX = (1 + state.squish * 0.3) * breathe;
    const scaleY = (1 - state.squish * 0.22) * breathe;

    // Blink timer — its own state fields, independent of every other
    // marker's state instance.
    if (state.nextBlinkAt === undefined) state.nextBlinkAt = now + random(2200, 4200);
    if (now > state.nextBlinkAt) {
      state.blinkStart = now;
      state.nextBlinkAt = now + random(3200, 5200);
    }
    const blinking = now - state.blinkStart < 130;

    // Body stays put at screen centre — only the affirmation bubble below
    // tracks the marker's live position — so it can sit large and steady
    // rather than jittering around with tracking noise.
    push();
    translate(width / 2, height / 2);
    scale(scaleX, scaleY);

    const headR = size * 0.34;
    const headY = -size * 0.12;

    // Curled tail from the body's side — reads as "cat" on its own even
    // before the ears/face register.
    const tailSway = Math.sin(now * 0.0018) * 0.05;
    noFill();
    stroke(red(bodyColor), green(bodyColor), blue(bodyColor), alpha * 255);
    strokeWeight(size * 0.09);
    strokeCap(ROUND);
    beginShape();
    vertex(size * 0.22, size * 0.4);
    bezierVertex(
      size * 0.42, size * 0.42,
      size * (0.52 + tailSway), size * 0.28,
      size * (0.46 + tailSway), size * 0.12
    );
    endShape();

    // Paws peeking out from under the body.
    noStroke();
    fill(red(bodyColor), green(bodyColor), blue(bodyColor), alpha * 255);
    circle(-size * 0.16, size * 0.48, size * 0.16);
    circle(size * 0.16, size * 0.48, size * 0.16);
    fill(red(bellyColor), green(bellyColor), blue(bellyColor), alpha * 255);
    circle(-size * 0.16, size * 0.5, size * 0.08);
    circle(size * 0.16, size * 0.5, size * 0.08);

    // Small body beneath the (dominant, chibi-proportioned) head, with a
    // lighter belly patch for a classic two-tone cartoon-cat look.
    fill(red(bodyColor), green(bodyColor), blue(bodyColor), alpha * 255);
    ellipse(0, size * 0.28, size * 0.56, size * 0.46);
    fill(red(bellyColor), green(bellyColor), blue(bellyColor), alpha * 255);
    ellipse(0, size * 0.34, size * 0.3, size * 0.26);

    // Ears, rooted at the top of the head circle — a small circle at each
    // tip rounds the triangle's point into a softer, kitten-like curve.
    for (const side of [-1, 1]) {
      push();
      translate(side * headR * 0.65, headY - headR * 0.72);
      rotate(side * 0.3);
      fill(red(bodyColor), green(bodyColor), blue(bodyColor), alpha * 255);
      triangle(-headR * 0.32, headR * 0.25, headR * 0.32, headR * 0.25, 0, -headR * 0.55);
      circle(0, -headR * 0.5, headR * 0.18);
      fill(red(accentColor), green(accentColor), blue(accentColor), alpha * 255);
      triangle(-headR * 0.18, headR * 0.12, headR * 0.18, headR * 0.12, 0, -headR * 0.32);
      circle(0, -headR * 0.28, headR * 0.1);
      pop();
    }

    // Head.
    noStroke();
    fill(red(bodyColor), green(bodyColor), blue(bodyColor), alpha * 255);
    circle(0, headY, headR * 2);

    // Blush cheeks — soft, low-opacity, sits behind the eyes so it reads as
    // warmth rather than a hard-edged pink circle.
    fill(255, 150, 190, alpha * 90);
    circle(-headR * 0.55, headY + headR * 0.25, headR * 0.4);
    circle(headR * 0.55, headY + headR * 0.25, headR * 0.4);

    // Eyes: big and chibi-round, with a double highlight for sparkle;
    // collapse to a soft blink line on the timer above.
    const eyeOffsetX = headR * 0.4;
    const eyeY = headY - headR * 0.02;
    const eyeSize = headR * 0.42;
    for (const ex of [-eyeOffsetX, eyeOffsetX]) {
      if (blinking) {
        stroke(red(darkColor), green(darkColor), blue(darkColor), alpha * 255);
        strokeWeight(size * 0.012);
        noFill();
        line(ex - eyeSize * 0.5, eyeY, ex + eyeSize * 0.5, eyeY);
        continue;
      }
      noStroke();
      fill(red(darkColor), green(darkColor), blue(darkColor), alpha * 255);
      circle(ex, eyeY, eyeSize);
      fill(255, 255, 255, alpha * 235);
      circle(ex - eyeSize * 0.22, eyeY - eyeSize * 0.22, eyeSize * 0.34);
      fill(255, 255, 255, alpha * 160);
      circle(ex + eyeSize * 0.18, eyeY + eyeSize * 0.15, eyeSize * 0.14);
    }

    // Nose.
    fill(red(accentColor), green(accentColor), blue(accentColor), alpha * 255);
    const noseY = eyeY + headR * 0.34;
    triangle(-headR * 0.06, noseY, headR * 0.06, noseY, 0, noseY + headR * 0.12);

    // Mouth: classic cartoon-cat "w" smile, opening a touch wider the more
    // it's squeezed instead of the flat whisker-swirl this used to have.
    noFill();
    stroke(red(darkColor), green(darkColor), blue(darkColor), alpha * 255);
    strokeWeight(size * 0.012);
    const mouthY = noseY + headR * 0.16;
    const mouthOpen = lerp(0.8, 1.3, state.squish);
    for (const side of [-1, 1]) {
      beginShape();
      vertex(0, mouthY);
      quadraticVertex(side * headR * 0.14, mouthY + headR * 0.1 * mouthOpen, side * headR * 0.2, mouthY);
      endShape();
    }

    // Whiskers.
    strokeWeight(size * 0.006);
    for (const [ex, dir] of [
      [-headR * 0.75, -1],
      [headR * 0.75, 1],
    ]) {
      for (let i = -1; i <= 1; i++) {
        const spread = i * headR * 0.12;
        line(ex, eyeY + headR * 0.35 + spread * 0.3, ex + dir * headR * 0.5, eyeY + headR * 0.38 + spread);
      }
    }

    pop();

    drawAffirmationBubble(state, cfg);
  }

  // Speech-bubble payoff for the squeeze trigger above — tracks the
  // marker's own live position (not the fixed-centre body above), since
  // that's the thing actually being held and moved around. Fades in,
  // holds, fades out, timed off affirmationShownAt rather than any
  // ongoing state, so it plays out fully even if the pet is set back down
  // mid-display.
  function drawAffirmationBubble(state, cfg) {
    if (!state.lastAffirmationText) return;
    const now = millis();
    const holdMs = cfg.affirmation_hold_ms ?? 4200;
    const fadeMs = cfg.affirmation_fade_ms ?? 400;
    const elapsed = now - state.affirmationShownAt;
    if (elapsed > holdMs + fadeMs) return;

    const alpha =
      elapsed < fadeMs
        ? elapsed / fadeMs
        : constrain(1 - (elapsed - holdMs) / fadeMs, 0, 1);
    if (alpha <= 0.01) return;

    const text_ = `Meow, ${state.lastAffirmationText}`;
    const bubbleColor = color(cfg.bubble_color ?? "#FFFFFF");
    const textColor = color(cfg.bubble_text_color ?? "#2A2030");

    push();
    translate(state.smoothX, state.smoothY - (cfg.bubble_offset_y ?? 60));
    textSize(cfg.bubble_text_size ?? 19);
    textFont("sans-serif");
    textAlign(CENTER, CENTER);
    const padX = 20;
    const w = Math.min(textWidth(text_) + padX * 2, 400);

    noStroke();
    fill(red(bubbleColor), green(bubbleColor), blue(bubbleColor), alpha * state.presence * 235);
    rectMode(CENTER);
    rect(0, 0, w, 44, 18);
    triangle(-10, 20, 10, 20, 0, 35);

    fill(red(textColor), green(textColor), blue(textColor), alpha * state.presence * 255);
    text(text_, 0, 0, w - padX);
    pop();
  }

  // ---- marker_4: segmented_light_wave (Hand Caterpillar Fidget) -----------
  //
  // The trailing-chain technique: each segment samples the SAME recorded
  // path at an increasing time delay, so the chain's shape is literally the
  // marker's own recent trajectory, link by link, rather than anything
  // simulated independently per segment. Colour travels along the chain as
  // a phase offset per segment index, at a speed tied to how fast the
  // marker is actually moving.
  //
  // The real item is a row of glossy silver beads threaded on black cord —
  // segments are close to uniform size (not the sharp head->tail taper a
  // generic "trail" would have) and each gets an unblurred highlight dot
  // for a bead-like glint, with a thin dark cord drawn underneath threading
  // through them. The glow/colour-cycle itself stays as originally
  // designed — vivid colour is the point of the visual reaction, not a
  // literal render of the (mostly black/silver) real object.
  //
  // Energy blends translational speed (recentSpeed) with angular speed
  // (recentAngularSpeed) rather than translation alone — a single marker
  // can't see the caterpillar's own segments articulating relative to each
  // other, but a marker being twisted/flexed in place at least shows up as
  // rotation even when it barely moves through the frame, which pure
  // position-based speed would miss entirely.
  //
  // The trailing chain and per-segment colour travel below are a real
  // record of the marker's own path, not fabricated — but they're still
  // only one rigid point's motion, not the toy's actual segment-to-segment
  // articulation, so treating "the chain's shape" as the harmony-with-the-
  // -action signal overclaims what's actually being sensed. What IS
  // reliably sensed is how vigorously the thing is being handled — so
  // pulseHz below drives a synchronised "breathing" pulse across every
  // segment together (size + brightness, all in phase, unlike the
  // per-segment colour travel's staggered phase), directly tempo-matched to
  // that energy: slow and calm when still, quick and energised when
  // handled vigorously. That tempo match — reaction pace mirroring action
  // pace — is the actual harmony-of-modalities signal here, not the
  // chain's spatial shape.

  // Full-screen ambient backdrop, same calming-wave style as the pop-it
  // balls' ring field (broad sine + rounded "dome" bump texture) but laid
  // out as horizontal trails rather than rings — no particular reason this
  // item's backdrop needs to match theirs geometrically, horizontal just
  // suits a single long marker better than a centred mandala would.
  //
  // The interaction is different too: instead of physically pushing the
  // trails apart, the marker darkens the trail colour near its own
  // position. p5 can't vary one continuous shape's stroke colour per
  // vertex, so each trail is drawn as many short segments rather than one
  // beginShape()/endShape() polyline — the only way to get per-point colour
  // out of the immediate-mode API.
  //
  // The whole pattern also continuously scrolls sideways (scrollX below) —
  // vertices stay on the fixed screen sample grid, but the phase fed into
  // the wave functions drifts with time, so the shape sampled at each grid
  // point keeps changing and the bumps visibly travel across the screen and
  // off the edge, link by link, rather than oscillating in place.
  function drawCaterpillarTrails(state, cfg) {
    if (state.presence < 0.01) return;
    const now = millis();
    const lineCount = cfg.trail_line_count ?? 12;
    const colors = (cfg.color_cycle ?? ["#4A90D9", "#7FD1E0", "#B39DDB", "#FF8FD6"]).map((h) => color(h));
    const ampA = cfg.trail_amp_1 ?? 16;
    const ampB = cfg.trail_amp_2 ?? 9;
    const freqA = cfg.trail_freq_1 ?? 0.007;
    const freqB = cfg.trail_freq_2 ?? 0.013;
    const domeAmp = cfg.trail_bump_amp ?? 8;
    const domeFreq = cfg.trail_bump_freq ?? 0.045;
    const domeSharpness = cfg.trail_bump_sharpness ?? 2.2;
    const scrollSpeed = cfg.trail_scroll_speed ?? 0.03; // px/ms the whole pattern drifts sideways
    const sampleStep = cfg.trail_sample_step ?? 24;
    const darkenRadius = cfg.trail_darken_radius ?? 220;
    const maxDarken = cfg.trail_max_darken ?? 0.8;
    const hasMarkerPos = state.smoothX !== null;

    const spacing = height / (lineCount + 1);
    const coreWeight = cfg.trail_stroke_weight ?? 2;
    const glowWeight = coreWeight * (cfg.trail_glow_weight_mult ?? 3.5);

    blendMode(ADD);
    for (let li = 0; li < lineCount; li++) {
      const baseY = spacing * (li + 1);
      const baseColor = paletteColorAt(colors, (li / lineCount) * colors.length + now * 0.00008);

      let prevX = null;
      let prevY = null;
      for (let x = -sampleStep; x <= width + sampleStep; x += sampleStep) {
        // The pattern travels by shifting the PHASE fed into the wave
        // functions over time (scrollX), not by moving each vertex's drawn
        // x position — the vertex stays on the fixed screen grid, but the
        // shape sampled at that grid point keeps changing, so what's
        // rendered there flows steadily across the screen and off the
        // edge, link by link, rather than wiggling in place.
        const scrollX = x - now * scrollSpeed;
        const domePhase = Math.sin(scrollX * domeFreq + li * 1.1);
        const dome = Math.pow((domePhase + 1) / 2, domeSharpness) * domeAmp;
        const y =
          baseY +
          Math.sin(scrollX * freqA + li * 0.7) * ampA +
          Math.sin(scrollX * freqB - li * 1.3) * ampB +
          dome;

        if (prevX !== null) {
          const midX = (prevX + x) / 2;
          const midY = (prevY + y) / 2;
          let c = baseColor;
          if (hasMarkerPos) {
            const d = dist(midX, midY, state.smoothX, state.smoothY);
            if (d < darkenRadius) {
              c = lerpColor(baseColor, color(0, 0, 0), (1 - d / darkenRadius) * maxDarken);
            }
          }
          const alpha = (cfg.trail_stroke_alpha ?? 90) * state.presence;
          // Manual two-pass glow (wide+faint, then thin+bright) instead of
          // drawingContext.filter blur — CSS canvas filters are applied per
          // draw call, not once per frame, so blurring ~650 tiny segments
          // individually (one per sample step per line) was the actual
          // cause of the slowdown, not the segment-based colouring itself.
          stroke(red(c), green(c), blue(c), alpha * 0.3);
          strokeWeight(glowWeight);
          line(prevX, prevY, x, y);
          stroke(red(c), green(c), blue(c), alpha);
          strokeWeight(coreWeight);
          line(prevX, prevY, x, y);
        }
        prevX = x;
        prevY = y;
      }
    }
    blendMode(BLEND);
  }

  function drawSegmentedLightWave(state, cfg) {
    if (state.history.length === 0) return;
    drawCaterpillarTrails(state, cfg);
    const now = millis();
    const segmentCount = cfg.segment_count ?? 11;
    const lagMs = cfg.segment_lag_ms ?? 45;
    const colors = cfg.color_cycle ?? ["#4A90D9", "#7FD1E0", "#B39DDB", "#FF8FD6"];
    const headSize = cfg.head_size ?? 24;
    const tailSize = cfg.tail_size ?? 17;
    const cordColor = color(cfg.cord_color ?? "#1a1a1a");

    const translateEnergy = constrain(recentSpeed(state.history) / (cfg.wave_speed_ref ?? 0.5), 0, 1);
    const angularEnergy = constrain(recentAngularSpeed(state.history) / (cfg.angular_speed_ref ?? 0.004), 0, 1);
    const energy = constrain(translateEnergy + angularEnergy, 0, 1);
    const speedMult = lerp(1, cfg.max_wave_speed_mult ?? 3, energy);
    const wavePos = now * 0.001 * (cfg.wave_speed ?? 1.4) * speedMult;

    const pulseHz = lerp(cfg.pulse_min_hz ?? 0.5, cfg.pulse_max_hz ?? 3, energy);
    state.pulsePhase += (deltaTime / 1000) * pulseHz * TWO_PI;
    const pulse = 0.5 + 0.5 * Math.sin(state.pulsePhase); // 0..1, shared across every segment

    const points = [];
    for (let i = 0; i < segmentCount; i++) {
      points.push(sampleHistoryAt(state.history, now - i * lagMs));
    }

    noFill();
    stroke(red(cordColor), green(cordColor), blue(cordColor), state.presence * 200);
    strokeWeight(cfg.cord_weight ?? 3);
    beginShape();
    for (const p of points) if (p) vertex(p.x, p.y);
    endShape();

    noStroke();
    blendMode(ADD);
    for (let i = 0; i < segmentCount; i++) {
      const p = points[i];
      if (!p) continue;
      const t = i / Math.max(segmentCount - 1, 1); // 0 at head, 1 at tail
      const size = lerp(headSize, tailSize, t) * lerp(cfg.pulse_size_min ?? 0.85, cfg.pulse_size_max ?? 1.2, pulse);
      const c = paletteColorAt(colors, wavePos - i * (cfg.color_step ?? 0.35));
      const alpha = lerp(1, 0.55, t) * lerp(cfg.pulse_alpha_min ?? 0.7, 1, pulse) * state.presence * 230;

      drawingContext.filter = `blur(${cfg.glow_blur ?? 6}px)`;
      fill(red(c), green(c), blue(c), alpha);
      circle(p.x, p.y, size);

      drawingContext.filter = "none";
      fill(255, 255, 255, alpha * 0.7);
      circle(p.x - size * 0.22, p.y - size * 0.22, size * 0.3);
    }
    drawingContext.filter = "none";
    blendMode(BLEND);
  }

  // ---- dev-only simulator ---------------------------------------------------
  //
  // Drives handleMessage() the same way a real backend message would, from
  // the mouse, so every visual above can be tuned in-browser independently
  // of camera/lighting/model conditions. Press 0-4 to toggle a mouse-following
  // marker of that ID; move the mouse to "hold" it. Delete this block and
  // the keyPressed() hookup in core.js once real detection is reliable
  // enough that it's no longer needed for tuning.
  const DEV_SIMULATOR = true;
  let simulatedMarkerId = null;

  function handleDevKey(key) {
    if (!DEV_SIMULATOR) return;
    const id = "01234".indexOf(key);
    if (id === -1) return;
    simulatedMarkerId = simulatedMarkerId === id ? null : id;
  }

  function updateSimulator() {
    if (!DEV_SIMULATOR || simulatedMarkerId === null) return;
    if (mouseX < 0 || mouseX > width || mouseY < 0 || mouseY > height) return;
    handleMessage({ markers: [{ marker_id: simulatedMarkerId, x: mouseX / width, y: mouseY / height }] });
  }

  function drawDevOverlay() {
    if (!DEV_SIMULATOR) return;
    push();
    noStroke();
    fill(0, 0, 0, 160);
    rect(10, 64, 460, 32, 6);
    fill(255);
    textSize(13);
    textFont("monospace");
    const active = simulatedMarkerId === null ? "none" : simulatedMarkerId;
    text(`object sim: keys 0-4 toggle a mouse-driven marker (active: ${active})`, 20, 85);
    pop();
  }

  return { handleMessage, update, draw, handleDevKey };
})();
