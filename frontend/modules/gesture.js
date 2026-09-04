// Phase 3 — Intentional facial gesture rendering.
// Renders the two gestures in config.gestures (see content_config.json) in
// response to {"modality": "gesture", "gesture_type", "hold_progress",
// "triggered"} messages from backend/modules/gesture.py.
//
// Reward the holding, not just detection: both visuals build continuously
// across hold_progress rather than snapping in at the end, and the payoff
// (burst / character popup + sound) fires exactly once per hold, on the
// frame the backend reports triggered === true.

const Gesture = (() => {
  let activeType = null; // raw current gesture from the backend, or null
  let displayType = null; // sticky last-active type, held through the fade-down after release
  let rawProgress = 0;
  let smoothProgress = 0;
  const PROGRESS_SMOOTHING = 0.25;

  let pendingTrigger = null; // gesture_type of a just-triggered payoff, consumed on the next draw()

  let faceLandmarks = null; // {left_eye, right_eye, nose, mouth}: normalised [0,1] points, or null when no face tracked
  let facePresence = 0; // eases toward 1 while a face is tracked, 0 while not — smooths the underlay's appear/disappear
  const FACE_PRESENCE_SMOOTHING = 0.12;

  let smileFlash = 0; // one-shot bright pulse on the held_smile payoff, decays each frame

  let character = null; // jack-in-the-box popup state, or null when off screen

  // Temporary calibration aid: shows the live width_ratio/open_ratio the
  // backend is comparing against its thresholds (see gesture.py), since
  // those were set from rough estimates, not a live camera. Only drawn while
  // gesture is the active modality (see draw() below) — this module's draw()
  // still runs every frame regardless of active_modality, same as
  // Proxemic/ObjectModule, so without that gate this overlay would sit on
  // top of the other modalities too. Safe to delete this whole block (and
  // its draw call below) once the thresholds are confirmed to trigger
  // reliably.
  const DEBUG_OVERLAY = true;
  let lastDebug = { width_ratio: null, open_ratio: null };

  function handleMessage(msg) {
    activeType = msg.gesture_type ?? null;
    if (activeType) displayType = activeType;
    rawProgress = msg.hold_progress ?? 0;
    lastDebug = { width_ratio: msg.width_ratio ?? null, open_ratio: msg.open_ratio ?? null };
    if (msg.triggered) pendingTrigger = activeType;
    faceLandmarks = msg.face_landmarks ?? null;
  }

  function drawDebugOverlay(activeModality) {
    if (!DEBUG_OVERLAY || activeModality !== "gesture") return;
    const wr = lastDebug.width_ratio;
    const or_ = lastDebug.open_ratio;
    const line1 = `gesture: ${activeType ?? "none"}  hold: ${(smoothProgress * 100).toFixed(0)}%`;
    const line2 = wr === null ? "no face detected" : `width_ratio: ${wr.toFixed(3)} (smile > 0.62)   open_ratio: ${or_.toFixed(3)} (open > 0.30)`;

    push();
    noStroke();
    fill(0, 0, 0, 160);
    rect(10, 10, 480, 46, 6);
    fill(255);
    textSize(13);
    textFont("monospace");
    text(line1, 20, 28);
    text(line2, 20, 46);
    pop();
  }

  function update() {
    const target = activeType ? rawProgress : 0;
    smoothProgress += (target - smoothProgress) * PROGRESS_SMOOTHING;
    if (!activeType && smoothProgress < 0.001) {
      smoothProgress = 0;
      displayType = null;
    }
    facePresence += ((faceLandmarks ? 1 : 0) - facePresence) * FACE_PRESENCE_SMOOTHING;
  }

  function draw(cfg, activeModality) {
    if (!cfg) return;

    if (pendingTrigger === "held_smile") {
      triggerSmileFlash(cfg.held_smile);
      pendingTrigger = null;
    } else if (pendingTrigger === "mouth_open") {
      triggerCharacterPopup();
      pendingTrigger = null;
    }

    drawSkyWash(cfg.held_smile);
    drawFaceUnderlay(cfg.face_underlay);
    drawHeldSmile(cfg.held_smile);
    drawMouthOpen(cfg.mouth_open);
    drawDebugOverlay(activeModality);
  }

  // ---- face_underlay: soft eye/nose/mouth glow (styled after proxemic's
  // body-silhouette underlay) ------------------------------------------------

  // Landmarks arrive unordered per feature (see gesture.py) — a convex hull
  // is enough to get a soft, recognisable eye/nose/mouth blob without needing
  // the mesh's exact edge topology, and reuses the same hull/expand helpers
  // proxemic already uses for its glow-rope body outline.
  function computeFeatureHull(points) {
    if (!points || points.length < 3) return [];
    const pts = points.map((p) => ({ x: p.x * width, y: p.y * height }));
    const hull = convexHull(pts);
    if (hull.length < 3) return [];
    // A looser expansion than proxemic's body hull (1.3) reads as a diffuse
    // aura around the feature rather than a tight, sticker-like trace of it
    // — softer and less literal.
    return expandHull(hull, 1.4);
  }

  // Slow, gently asynchronous "breathing" so the four blobs feel alive
  // rather than a static overlay — each feature gets its own phase offset
  // so they drift in and out of sync instead of pulsing together.
  function breathe(phase) {
    return 1 + 0.12 * Math.sin(millis() * 0.0009 + phase);
  }

  // Soft layered-blur-pass fill, same technique as proxemic's silhouette
  // underlay but applied to a small per-feature hull instead of a full-body
  // segmentation mask. No hard contour stroke — at this size a crisp edge
  // reads as a sticker outline rather than a soft glow, so definition comes
  // entirely from the blur falloff.
  function drawFeatureBlob(hull, fillColor, opacity, cfg) {
    if (hull.length < 3 || opacity <= 0.001) return;
    const softness = cfg.softness ?? 16;

    push();
    noStroke();
    blendMode(ADD);
    const layers = [
      { blur: softness * 0.5, alpha: opacity * 0.55 },
      { blur: softness * 1.4, alpha: opacity * 0.35 },
      { blur: softness * 3.2, alpha: opacity * 0.2 },
    ];
    for (const layer of layers) {
      drawingContext.filter = `blur(${layer.blur}px)`;
      fill(red(fillColor), green(fillColor), blue(fillColor), layer.alpha * 255);
      beginShape();
      for (const p of hull) vertex(p.x, p.y);
      endShape(CLOSE);
    }
    drawingContext.filter = "none";
    blendMode(BLEND);
    pop();
  }

  const FACE_UNDERLAY_PHASES = { left_eye: 0, right_eye: 1.7, nose: 3.4, mouth: 5.1 };

  function drawFaceUnderlay(cfg) {
    if (!cfg || facePresence < 0.01 || !faceLandmarks) return;

    const baseColor = color(cfg.color ?? "#8FD8E8");
    const eyeAccent = color(cfg.eye_accent ?? cfg.color ?? "#8FD8E8");
    const mouthAccent = color(cfg.mouth_accent ?? cfg.color ?? "#8FD8E8");
    const baseOpacity = (cfg.opacity ?? 0.13) * facePresence;

    // held_smile/mouth_open progress gently warms the relevant feature
    // toward its accent colour rather than fully replacing it, so the
    // "engaged" cue stays a tint, not a colour swap.
    const smileProgress = displayType === "held_smile" ? smoothProgress : 0;
    const mouthOpenProgress = displayType === "mouth_open" ? smoothProgress : 0;

    const eyeColor = lerpColor(baseColor, eyeAccent, smileProgress * 0.6);
    const eyeOpacity = baseOpacity * lerp(1, 1.3, smileProgress);
    const mouthColor = lerpColor(baseColor, mouthAccent, mouthOpenProgress * 0.6);
    const mouthOpacity = baseOpacity * lerp(1, 1.3, mouthOpenProgress);

    drawFeatureBlob(
      computeFeatureHull(faceLandmarks.left_eye),
      eyeColor,
      eyeOpacity * breathe(FACE_UNDERLAY_PHASES.left_eye),
      cfg
    );
    drawFeatureBlob(
      computeFeatureHull(faceLandmarks.right_eye),
      eyeColor,
      eyeOpacity * breathe(FACE_UNDERLAY_PHASES.right_eye),
      cfg
    );
    drawFeatureBlob(
      computeFeatureHull(faceLandmarks.nose),
      baseColor,
      baseOpacity * breathe(FACE_UNDERLAY_PHASES.nose),
      cfg
    );
    drawFeatureBlob(
      computeFeatureHull(faceLandmarks.mouth),
      mouthColor,
      mouthOpacity * breathe(FACE_UNDERLAY_PHASES.mouth),
      cfg
    );
  }

  // ---- held_smile: calming sky-blue wash -> bright flash + chime ---------

  // Fills the whole canvas with a soft vertical gradient (deeper blue up top,
  // pale near the "horizon") that eases in with hold progress and back out
  // if the smile is released early — a full-frame mood shift rather than a
  // localized effect. Drawn before the face underlay/mouth_open layers so
  // it sits behind them as a true background rather than a tint on top.
  function drawSkyWash(cfg) {
    if (!cfg) return;
    const progress = displayType === "held_smile" ? smoothProgress : 0;
    if (progress <= 0.001) return;

    const topColor = color(cfg.sky_color_top ?? "#4A90D9");
    const horizonColor = color(cfg.sky_color_horizon ?? "#BFEFFF");
    const opacity = (cfg.sky_max_opacity ?? 0.85) * progress;

    push();
    noStroke();
    const grad = drawingContext.createLinearGradient(0, 0, 0, height);
    grad.addColorStop(0, `rgba(${red(topColor)}, ${green(topColor)}, ${blue(topColor)}, ${opacity})`);
    grad.addColorStop(1, `rgba(${red(horizonColor)}, ${green(horizonColor)}, ${blue(horizonColor)}, ${opacity})`);
    drawingContext.fillStyle = grad;
    rect(0, 0, width, height);
    pop();
  }

  // The payoff on trigger: a brief bright overlay that decays back into the
  // steady sky wash, giving the hold a distinct "moment" without sparkles.
  function drawHeldSmile(cfg) {
    if (!cfg || smileFlash <= 0.001) return;

    const flashColor = color(cfg.flash_color ?? "#FFFFFF");
    const maxOpacity = cfg.flash_max_opacity ?? 0.45;

    push();
    noStroke();
    fill(red(flashColor), green(flashColor), blue(flashColor), smileFlash * maxOpacity * 255);
    rect(0, 0, width, height);
    pop();

    smileFlash *= cfg.flash_decay ?? 0.9;
    if (smileFlash < 0.01) smileFlash = 0;
  }

  function triggerSmileFlash(cfg) {
    smileFlash = 1;
    SensoryAudio.playChime();
  }

  // ---- mouth_open: edge intensify -> jack-in-the-box popup + boing -------

  function drawMouthOpen(cfg) {
    if (!cfg) return;
    const progress = displayType === "mouth_open" ? smoothProgress : 0;
    drawEdgeIntensify(cfg, progress);
    drawCharacter(cfg);
  }

  function drawEdgeIntensify(cfg, progress) {
    if (progress <= 0.01) return;
    const c = color(cfg.edge_color ?? "#FF6B6B");
    const maxOpacity = cfg.edge_max_opacity ?? 0.5;
    const thickness = cfg.edge_thickness ?? 90;
    const blurPx = cfg.edge_blur ?? 40;
    // Tension: a small noise-driven jitter on the border inset that grows
    // with hold progress, rather than a static glow that just fades in.
    const shakeAmount = (cfg.tension_shake_amount ?? 4) * progress;
    const shake = shakeAmount * (noise(frameCount * 0.3) * 2 - 1);
    const opacity = maxOpacity * progress;

    push();
    blendMode(ADD);
    drawingContext.filter = `blur(${blurPx}px)`;
    noFill();
    stroke(red(c), green(c), blue(c), opacity * 255);
    strokeWeight(thickness);
    rect(shake, shake, width - shake * 2, height - shake * 2);
    drawingContext.filter = "none";
    blendMode(BLEND);
    pop();
  }

  function triggerCharacterPopup() {
    character = { spawnTime: millis() };
    SensoryAudio.playBoing();
  }

  // Animation timeline, in ms from trigger: bounce up with overshoot ->
  // wiggle side to side -> one full spin -> settle into an idle bob -> fade out.
  const BOUNCE_MS = 350;
  const WIGGLE_MS = 500;
  const SPIN_MS = 400;
  const SETTLE_MS = 1800;
  const FADE_MS = 500;
  const BOUNCE_END = BOUNCE_MS;
  const WIGGLE_END = BOUNCE_END + WIGGLE_MS;
  const SPIN_END = WIGGLE_END + SPIN_MS;
  const SETTLE_END = SPIN_END + SETTLE_MS;
  const FADE_END = SETTLE_END + FADE_MS;

  function easeOutBack(t) {
    const c1 = 1.7;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  }

  function drawCharacter(cfg) {
    if (!character) return;
    const elapsed = millis() - character.spawnTime;
    if (elapsed >= FADE_END) {
      character = null;
      return;
    }

    const restX = width / 2;
    const restY = height * 0.68;
    const startY = height + 150;

    let y = restY;
    let scale = 1;
    let rotation = 0;
    let alpha = 255;

    if (elapsed < BOUNCE_END) {
      const t = easeOutBack(elapsed / BOUNCE_MS);
      y = lerp(startY, restY, t);
      scale = lerp(0.4, 1, constrain(elapsed / BOUNCE_MS, 0, 1));
    } else if (elapsed < WIGGLE_END) {
      const t = (elapsed - BOUNCE_END) / WIGGLE_MS;
      const decay = 1 - t;
      rotation = sin(t * TWO_PI * 3) * 0.35 * decay;
    } else if (elapsed < SPIN_END) {
      const t = (elapsed - WIGGLE_END) / SPIN_MS;
      rotation = t * TWO_PI;
    } else if (elapsed < SETTLE_END) {
      y = restY + sin(millis() * 0.003) * 6;
    } else {
      y = restY + sin(millis() * 0.003) * 6;
      alpha = lerp(255, 0, (elapsed - SETTLE_END) / FADE_MS);
    }

    drawSillyCharacter(restX, y, rotation, scale, alpha, cfg);
  }

  function drawSillyCharacter(x, y, rotation, scale_, alpha, cfg) {
    const size = cfg.character_size ?? 130;
    const bodyColor = color(cfg.character_color ?? "#FFB84D");
    const accentColor = color(cfg.character_accent_color ?? "#3B2A1A");

    push();
    translate(x, y);
    rotate(rotation);
    scale(scale_);

    // Spring: a simple zig-zag, visible while the character is still low
    // enough to plausibly be connected to a box below the frame.
    if (y > height * 0.55) {
      stroke(red(accentColor), green(accentColor), blue(accentColor), alpha * 0.8);
      strokeWeight(4);
      noFill();
      beginShape();
      const coils = 5;
      for (let i = 0; i <= coils; i++) {
        const sx = (i % 2 === 0 ? -1 : 1) * size * 0.18;
        const sy = size * 0.5 + i * (size * 0.12);
        vertex(sx, sy);
      }
      endShape();
    }

    noStroke();
    fill(red(bodyColor), green(bodyColor), blue(bodyColor), alpha);
    ellipse(0, 0, size, size);

    // Eyes
    fill(red(accentColor), green(accentColor), blue(accentColor), alpha);
    const eyeOffsetX = size * 0.2;
    const eyeY = -size * 0.08;
    circle(-eyeOffsetX, eyeY, size * 0.12);
    circle(eyeOffsetX, eyeY, size * 0.12);

    // Big friendly open-mouth smile
    noFill();
    stroke(red(accentColor), green(accentColor), blue(accentColor), alpha);
    strokeWeight(size * 0.05);
    arc(0, size * 0.05, size * 0.5, size * 0.4, 0.15 * PI, 0.85 * PI);

    // Cheeks
    noStroke();
    fill(255, 150, 150, alpha * 0.6);
    circle(-size * 0.32, size * 0.08, size * 0.14);
    circle(size * 0.32, size * 0.08, size * 0.14);

    pop();
  }

  return { handleMessage, update, draw };
})();
