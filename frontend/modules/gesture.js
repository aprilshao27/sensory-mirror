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

  let ambientSparkles = []; // builds during a held_smile hold
  let burstSparkles = []; // one-shot payoff burst

  let character = null; // jack-in-the-box popup state, or null when off screen

  // Temporary calibration aid: shows the live width_ratio/open_ratio the
  // backend is comparing against its thresholds (see gesture.py), since
  // those were set from rough estimates, not a live camera. Safe to delete
  // this block (and its draw call below) once the thresholds are confirmed
  // to trigger reliably.
  const DEBUG_OVERLAY = true;
  let lastDebug = { width_ratio: null, open_ratio: null };

  function handleMessage(msg) {
    activeType = msg.gesture_type ?? null;
    if (activeType) displayType = activeType;
    rawProgress = msg.hold_progress ?? 0;
    lastDebug = { width_ratio: msg.width_ratio ?? null, open_ratio: msg.open_ratio ?? null };
    if (msg.triggered) pendingTrigger = activeType;
  }

  function drawDebugOverlay() {
    if (!DEBUG_OVERLAY) return;
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
  }

  function draw(cfg) {
    if (!cfg) return;

    if (pendingTrigger === "held_smile") {
      triggerSmileBurst(cfg.held_smile);
      pendingTrigger = null;
    } else if (pendingTrigger === "mouth_open") {
      triggerCharacterPopup();
      pendingTrigger = null;
    }

    drawHeldSmile(cfg.held_smile);
    drawMouthOpen(cfg.mouth_open);
    drawDebugOverlay();
  }

  // ---- held_smile: ambient sparkle build -> burst + chime ----------------

  function spawnAmbientSparkle(cfg) {
    const colors = cfg.sparkle_colors ?? ["#FFD966"];
    const cx = width / 2;
    const cy = height / 2;
    // Averaging three random samples biases spawns toward the center
    // (roughly where a face fills the mirror) without needing real face
    // position — this module only gets gesture_type/hold_progress, not
    // landmarks.
    const bx = (random() + random() + random()) / 3;
    const by = (random() + random() + random()) / 3;
    return {
      x: cx + (bx - 0.5) * width * 0.8,
      y: cy + (by - 0.5) * height * 0.8,
      size: random(cfg.sparkle_min_size ?? 3, cfg.sparkle_max_size ?? 9),
      col: color(random(colors)),
      twinklePhase: random(TWO_PI),
      twinkleSpeed: random(0.06, 0.12),
    };
  }

  function drawHeldSmile(cfg) {
    if (!cfg) return;

    // Ambient sparkles only build/persist while held_smile is the
    // (possibly fading-out) displayed gesture; the count target tracks
    // smoothProgress directly so it grows during the hold and shrinks back
    // out smoothly if released early, rather than an abrupt clear.
    const targetCount = displayType === "held_smile" ? Math.floor((cfg.max_ambient_sparkles ?? 40) * smoothProgress) : 0;
    while (ambientSparkles.length < targetCount) {
      ambientSparkles.push(spawnAmbientSparkle(cfg));
    }
    while (ambientSparkles.length > targetCount) {
      ambientSparkles.pop();
    }

    noStroke();
    blendMode(ADD);
    for (const s of ambientSparkles) {
      s.twinklePhase += s.twinkleSpeed;
      const twinkle = 0.4 + 0.6 * (0.5 + 0.5 * sin(s.twinklePhase));
      const alpha = twinkle * smoothProgress * 220;
      drawSparkle(s.x, s.y, s.size, s.col, alpha);
    }

    for (let i = burstSparkles.length - 1; i >= 0; i--) {
      const s = burstSparkles[i];
      s.x += s.vx;
      s.y += s.vy;
      s.vx *= 0.97;
      s.vy *= 0.97;
      s.life -= 0.018;
      if (s.life <= 0) {
        burstSparkles.splice(i, 1);
        continue;
      }
      drawSparkle(s.x, s.y, s.size * s.life, s.col, s.life * 255);
    }
    blendMode(BLEND);
  }

  // Four-point star glyph reads as "sparkle" more clearly than a plain dot.
  function drawSparkle(x, y, size, col, alpha) {
    if (alpha <= 1) return;
    push();
    translate(x, y);
    fill(red(col), green(col), blue(col), alpha);
    beginShape();
    for (let i = 0; i < 8; i++) {
      const r = i % 2 === 0 ? size : size * 0.35;
      const a = (PI / 4) * i;
      vertex(cos(a) * r, sin(a) * r);
    }
    endShape(CLOSE);
    pop();
  }

  function triggerSmileBurst(cfg) {
    const cfgS = cfg ?? {};
    const colors = cfgS.sparkle_colors ?? ["#FFD966"];
    const count = cfgS.burst_sparkle_count ?? 70;
    const speedMin = cfgS.burst_speed_min ?? 3;
    const speedMax = cfgS.burst_speed_max ?? 9;
    const cx = width / 2;
    const cy = height / 2;
    for (let i = 0; i < count; i++) {
      const angle = random(TWO_PI);
      const speed = random(speedMin, speedMax);
      burstSparkles.push({
        x: cx,
        y: cy,
        vx: cos(angle) * speed,
        vy: sin(angle) * speed,
        size: random(cfgS.sparkle_min_size ?? 3, cfgS.sparkle_max_size ?? 9) * 1.4,
        col: color(random(colors)),
        life: 1,
      });
    }
    ambientSparkles = []; // the burst replaces the ambient build-up
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
