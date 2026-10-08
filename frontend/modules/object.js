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
        grains: null, // segmented_light_wave — fixed cluster of grain offsets, generated once
        rollPhase: 0, // segmented_light_wave — eased brightness-wave phase travelling through the cluster
        activity: 0, // segmented_light_wave — eased 0 (still) .. 1 (being handled)
        grainTrackX: null, // segmented_light_wave — own, faster-tracking position (see drawSegmentedLightWave)
        grainTrackY: null,
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
  }

  // ---- marker_0-2: adaptive_colour_ripple (Pop It squishy balls) ----------
  //
  // Deliberately abstract, not a literal render of the ball — soft rings
  // emanate from the marker's live position and travel outward with it, and
  // a small marbled "aura" sits at the core. What makes each ball read
  // differently is purely its own ball_color_primary/secondary/tertiary in
  // content_config.json — sampled straight from the real toy's photos (see
  // backend/dataset/raw/*_pop_it/*.jpg) — cycled through as rings age
  // rather than tied to any literal geometry, so it reads as "this ball's
  // colours in motion" rather than a picture of the ball itself.
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

    // Core aura: a handful of soft, drifting blobs orbiting the current
    // position, all in the ball's own primary colour — so the one visual
    // element that's always right on the marker (as opposed to the ripples
    // above, which travel outward and fade) reads as a stable, recognisable
    // "this is that ball" colour rather than cycling through its palette.
    const coreSize = cfg.core_size ?? 46;
    const auraLayers = cfg.aura_layers ?? 5;
    const primary = palette[0];
    noStroke();
    blendMode(ADD);
    drawingContext.filter = `blur(${coreSize * 0.35}px)`;
    for (let i = 0; i < auraLayers; i++) {
      const angle = (i / auraLayers) * TWO_PI + now * 0.0004;
      const orbit = coreSize * 0.28 * (0.4 + 0.6 * noise(i * 3, now * 0.0006));
      const px = state.smoothX + Math.cos(angle) * orbit;
      const py = state.smoothY + Math.sin(angle) * orbit;
      fill(red(primary), green(primary), blue(primary), state.presence * 130);
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
      SensoryAudio.playMeow(cfg.meow_volume ?? 0.6);
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
    textSize(cfg.bubble_text_size ?? 24);
    textFont("sans-serif");
    textAlign(CENTER, CENTER);
    const padX = 26;
    const w = Math.min(textWidth(text_) + padX * 2, 520);

    noStroke();
    fill(red(bubbleColor), green(bubbleColor), blue(bubbleColor), alpha * state.presence * 235);
    rectMode(CENTER);
    rect(0, 0, w, 58, 24);
    triangle(-13, 27, 13, 27, 0, 46);

    fill(red(textColor), green(textColor), blue(textColor), alpha * state.presence * 255);
    text(text_, 0, 0, w - padX);
    pop();
  }

  // ---- marker_4: segmented_light_wave (Hand Caterpillar Fidget) -----------
  //
  // "Granular roll": a fixed field of soft grains covers the ENTIRE canvas
  // (state.grains, see generateGrainField) — a calm, near-invisible ambient
  // texture at rest, sensory-room-style, rather than a small indicator that
  // tracks the object like a cursor. Grains within influenceRadius of the
  // caterpillar's own tracked position brighten in a rolling sequence,
  // falling off with distance — so the field reads as responding TO the
  // object at its actual location (contingency stays legible: this
  // brightened because of what's happening right here), not as a generic
  // whole-screen shimmer unrelated to it. Same falloff shape as
  // drawAmbientWaveField's push-radius above, adapted to brightening grains
  // instead of displacing rings.
  //
  // Grounded in crossmodal-correspondence research (Spence, 2011,
  // "Crossmodal correspondences: A tutorial review"): texture and rhythm are
  // amodal qualities that transfer between touch and sight without either
  // channel needing to resemble the other, so nothing here tries to redraw
  // the toy's silhouette or pose the way an earlier line/spiral-morphing
  // version did.
  //
  // Only one real signal drives the roll: recentSpeed (translation), eased
  // into state.activity with a faster attack than decay so it reads as
  // settling rather than reacting. It controls both how fast the roll
  // travels and how sharply its brightness is focused within the influence
  // radius — calm holds a narrow, slow-moving highlight; active handling
  // broadens it and speeds it up. Intensity and tempo are the only things
  // that ever change — never hue-cycling, never a shape swap.
  function drawSegmentedLightWave(state, cfg) {
    if (state.presence < 0.01 || state.history.length === 0) return;
    if (!state.grains) state.grains = generateGrainField(cfg.grain_count ?? 220, cfg.grain_min_spacing_px ?? 28);

    // Tracks the latest raw detection directly, on its own (faster) ease
    // rate, instead of the shared state.smoothX/Y every other marker visual
    // uses (see update() above) — the highlight needs to sit right where
    // the caterpillar physically is without lagging behind quick moves.
    const latest = state.history[state.history.length - 1];
    const trackEase = cfg.position_ease ?? 0.45;
    if (state.grainTrackX === null) {
      state.grainTrackX = latest.x;
      state.grainTrackY = latest.y;
    } else {
      state.grainTrackX += (latest.x - state.grainTrackX) * trackEase;
      state.grainTrackY += (latest.y - state.grainTrackY) * trackEase;
    }

    const activityTarget = constrain(recentSpeed(state.history) / (cfg.speed_ref ?? 0.5), 0, 1);
    const activityEase =
      activityTarget > state.activity ? cfg.activity_ease_up ?? 0.05 : cfg.activity_ease_down ?? 0.02;
    state.activity += (activityTarget - state.activity) * activityEase;

    const rollSpeed = (cfg.roll_speed_min ?? 0.00025) + state.activity * (cfg.roll_speed_range ?? 0.0022);
    state.rollPhase += deltaTime * rollSpeed;

    const influenceRadius = cfg.influence_radius ?? 260;
    const falloffSharpness = cfg.falloff_sharpness ?? 1.6;
    const waveFreq = cfg.wave_freq ?? 6;
    const sharpness = (cfg.pulse_sharpness_base ?? 3) + (1 - state.activity) * (cfg.pulse_sharpness_range ?? 4);
    const ambientBrightness = cfg.ambient_brightness ?? 0.05;
    const brightRange = (cfg.brightness_range ?? 0.35) + state.activity * (cfg.brightness_active_range ?? 0.65);
    const sizeMin = cfg.grain_size_min ?? 6;
    const sizeMax = cfg.grain_size_max ?? 11;
    const minAlpha = cfg.min_alpha ?? 18;
    const colorDim = color(cfg.body_color_dim ?? "#3E6B5C");
    const colorActive = color(cfg.body_color_active ?? "#7FD9B0");

    const mx = state.grainTrackX;
    const my = state.grainTrackY;
    const cosP = Math.cos(state.rollPhase);
    const sinP = Math.sin(state.rollPhase);

    noStroke();
    blendMode(ADD);
    for (const g of state.grains) {
      const x = g.fx * width;
      const y = g.fy * height;
      const dx = x - mx;
      const dy = y - my;
      const d = Math.sqrt(dx * dx + dy * dy);

      // Grains outside the influence radius sit at the flat ambient
      // baseline — skipping the wave math for them is also a real
      // shortcut, since most of a full-canvas field falls outside it.
      let t = ambientBrightness;
      if (d < influenceRadius) {
        const falloff = Math.pow(1 - d / influenceRadius, falloffSharpness);
        // Projecting onto a rotating axis anchored at the marker's OWN
        // current position (not a fixed field-wide origin) gives a
        // brightness band that sweeps outward from wherever the object
        // actually is right now, following it as it moves.
        const proj = dx * cosP + dy * sinP;
        const wave = 0.5 + 0.5 * Math.sin((proj / influenceRadius) * waveFreq);
        const pulse = Math.pow(wave, sharpness);
        t = constrain(ambientBrightness + falloff * pulse * brightRange, 0, 1);
      }

      const size = lerp(sizeMin, sizeMax, g.sizeT) * (0.85 + t * 0.3);
      const c = lerpColor(colorDim, colorActive, t);
      const alpha = state.presence * lerp(minAlpha, 235, t);

      // Two-pass soft glow (wide+faint, then core+bright) instead of a
      // per-grain blur filter — cheap for one shape, costly repeated once
      // per grain per frame (same tradeoff as the ambient-field renderers
      // elsewhere in this file, now doubly true at full-canvas grain counts).
      fill(red(c), green(c), blue(c), alpha * 0.35);
      circle(x, y, size * 2.2);
      fill(red(c), green(c), blue(c), alpha);
      circle(x, y, size);
    }
    blendMode(BLEND);
  }

  // A scatter of grain positions covering the entire canvas, stored as
  // fractions of width/height (not absolute pixels) so the field stays
  // correctly laid out if the canvas is ever resized. Simple rejection
  // sampling keeps grains from overlapping while staying organic rather
  // than a grid. Generated once per marker and held fixed — the field
  // itself never moves; only which grains brighten does.
  function generateGrainField(count, minSpacingPx) {
    const grains = [];
    let tries = 0;
    const maxTries = count * 40;
    while (grains.length < count && tries < maxTries) {
      tries++;
      const x = Math.random() * width;
      const y = Math.random() * height;
      let ok = true;
      for (const g of grains) {
        if (dist(g.fx * width, g.fy * height, x, y) < minSpacingPx) {
          ok = false;
          break;
        }
      }
      if (ok) grains.push({ fx: x / width, fy: y / height, sizeT: Math.random() });
    }
    return grains;
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

  return { handleMessage, update, draw, handleDevKey };
})();
