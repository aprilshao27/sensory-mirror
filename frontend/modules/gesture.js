// Phase 3 — Intentional facial gesture rendering.
// Renders the two gestures in config.gestures (see content_config.json) in
// response to {"modality": "gesture", "gesture_type", "hold_progress",
// "triggered"} messages from backend/modules/gesture.py.
//
// Reward the holding, not just detection: both visuals build continuously
// across hold_progress rather than snapping in at the end, and the payoff
// (flash / mascot cheer + sound) fires exactly once per hold, on the
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

  let character = null; // mascot cheer state, or null when not showing

  let voiceRings = []; // {x, y, born, color}: expanding warm-coloured rings visualising the held "voice" (shout/laugh/sung note)
  let lastRingSpawn = 0;
  let ringColorIndex = 0;
  const RING_COLORS_DEFAULT = ["#FFB84D", "#FFD966", "#FF8C69", "#FFE29A"];

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
      triggerMascotCheer();
      pendingTrigger = null;
    }

    drawSkyWash(cfg.held_smile);
    drawFaceUnderlay(cfg.face_underlay);
    drawSmileBloom(cfg.held_smile);
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
  // Default expansion (1.4) is looser than proxemic's body hull (1.3) so it
  // reads as a diffuse aura around the feature rather than a tight,
  // sticker-like trace of it — softer and less literal. held_smile's bloom
  // ring (see drawSmileBloom below) reuses this same hull-from-landmarks
  // technique against face_oval, with its own growing expansion factor.
  function computeFeatureHull(points, expansion = 1.4) {
    if (!points || points.length < 3) return [];
    const pts = points.map((p) => ({ x: p.x * width, y: p.y * height }));
    const hull = convexHull(pts);
    if (hull.length < 3) return [];
    return expandHull(hull, expansion);
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

  // ---- held_smile: bloom ring/flowers growing from the participant's own
  // facial outline -----------------------------------------------------

  // On top of the full-screen sky wash, grows a ring of soft light and
  // flowers directly off the participant's own face_oval landmarks (see
  // gesture.py) rather than a shape centered arbitrarily on screen — the
  // positive payoff reads as being about their own reflection, not an
  // abstract screen effect.
  function drawSmileBloom(cfg) {
    if (!cfg || !faceLandmarks || !faceLandmarks.face_oval) return;
    const progress = displayType === "held_smile" ? smoothProgress : 0;
    if (progress <= 0.001) return;

    // Starts just outside the actual face outline (1.15) and grows toward
    // its max expansion across the hold, so the ring visibly emerges from
    // the silhouette rather than appearing already-expanded.
    const expansion = lerp(1.15, cfg.bloom_ring_max_expansion ?? 1.9, progress);
    const hull = computeFeatureHull(faceLandmarks.face_oval, expansion);
    if (hull.length < 3) return;

    drawBloomRing(cfg, hull, progress);
    drawBloomFlowers(cfg, hull, progress);
  }

  // Soft glowing outline traced along the expanding hull, same layered-blur
  // stroke technique as proxemic's silhouette contour.
  function drawBloomRing(cfg, hull, progress) {
    const c = color(cfg.bloom_ring_color ?? "#FFF3D6");
    const softness = cfg.bloom_ring_softness ?? 10;
    const weight = cfg.bloom_ring_width ?? 3;
    const baseOpacity = (cfg.bloom_ring_opacity ?? 0.8) * progress;

    push();
    noFill();
    blendMode(ADD);
    const passes = [
      { blur: softness * 0.3, alpha: baseOpacity * 0.9, w: weight },
      { blur: softness * 1.0, alpha: baseOpacity * 0.5, w: weight * 1.6 },
      { blur: softness * 2.4, alpha: baseOpacity * 0.25, w: weight * 2.6 },
    ];
    for (const p of passes) {
      drawingContext.filter = `blur(${p.blur}px)`;
      stroke(red(c), green(c), blue(c), p.alpha * 255);
      strokeWeight(p.w);
      beginShape();
      for (const pt of hull) vertex(pt.x, pt.y);
      endShape(CLOSE);
    }
    drawingContext.filter = "none";
    blendMode(BLEND);
    pop();
  }

  // A small radial-petal bloom at one point on the ring. Petals are soft
  // additive ellipses around a bright center rather than a literal flower
  // sprite, so it reads as "bloom of light" first and "flower" second —
  // consistent with the rest of the module's abstract, glow-based visuals.
  function drawFlower(x, y, size, rotation, fillColor, alpha, petalCount) {
    if (size <= 0.5 || alpha <= 0.001) return;
    const petalLen = size * 0.6;
    const petalWidth = size * 0.34;

    push();
    translate(x, y);
    rotate(rotation);
    noStroke();
    blendMode(ADD);
    drawingContext.filter = `blur(${size * 0.06}px)`;
    for (let i = 0; i < petalCount; i++) {
      push();
      rotate((i / petalCount) * TWO_PI);
      fill(red(fillColor), green(fillColor), blue(fillColor), alpha * 255);
      ellipse(0, -petalLen * 0.5, petalWidth, petalLen);
      pop();
    }
    fill(255, 255, 255, alpha * 230);
    circle(0, 0, size * 0.3);
    drawingContext.filter = "none";
    blendMode(BLEND);
    pop();
  }

  // Flowers are placed at evenly spaced points around the ring and blossom
  // in sequence as hold_progress advances (bloom_stagger controls how much
  // of the hold is spent staggering around the ring vs. each flower's own
  // bloom-in), so the ring visibly grows/blooms rather than popping in at
  // once. Each flower keeps a slow individual rotation drift so the ring
  // feels alive rather than static once fully bloomed.
  function drawBloomFlowers(cfg, hull, progress) {
    const count = cfg.flower_count ?? 10;
    if (count <= 0) return;
    const points = resamplePolyline(hull, count);
    const colors = (cfg.flower_colors ?? ["#FFFFFF"]).map((hex) => color(hex));
    const maxSize = cfg.flower_max_size ?? 34;
    const petalCount = cfg.flower_petal_count ?? 6;
    const maxOpacity = cfg.flower_max_opacity ?? 0.85;
    const stagger = constrain(cfg.flower_bloom_stagger ?? 0.55, 0, 0.95);

    for (let i = 0; i < points.length; i++) {
      const start = (i / count) * stagger;
      const local = constrain((progress - start) / (1 - stagger), 0, 1);
      if (local <= 0.001) continue;
      const eased = easeOutBack(local);
      const size = maxSize * constrain(eased, 0, 1.15);
      const alpha = constrain(local * 1.3, 0, 1) * maxOpacity;
      const rotation = i * 0.7 + millis() * 0.00015;
      drawFlower(points[i].x, points[i].y, size, rotation, colors[i % colors.length], alpha, petalCount);
    }
  }

  // ---- mouth_open: warm expanding voice rings -> mascot cheer + tada ----
  //
  // Reframed from "tension -> startle" to "big voice -> celebration": same
  // hold mechanic (see gesture.py), but the build is warm concentric rings
  // reading as a shout/laugh/sung note radiating from the mouth, and the
  // payoff is a mascot that appears in place and cheers/dances rather than
  // launching up from off-screen — no jump-scare for a vulnerable population.

  function drawMouthOpen(cfg) {
    if (!cfg) return;
    const progress = displayType === "mouth_open" ? smoothProgress : 0;
    drawVoiceRings(cfg, progress);
    drawCharacter(cfg);
  }

  // Finds the mouth's on-screen centroid from its landmarks (see gesture.py's
  // FACE_FEATURE_INDICES) so the rings visibly radiate from the participant's
  // own mouth rather than an arbitrary screen point; falls back to screen
  // center if no face is tracked yet.
  function mouthCenter() {
    const pts = faceLandmarks?.mouth;
    if (!pts || pts.length === 0) return { x: width / 2, y: height / 2 };
    const cx = pts.reduce((sum, p) => sum + p.x, 0) / pts.length;
    const cy = pts.reduce((sum, p) => sum + p.y, 0) / pts.length;
    return { x: cx * width, y: cy * height };
  }

  // Spawns and ages a set of expanding rings. Spawn rate and reach both climb
  // with hold_progress, so the "voice" audibly/visibly builds rather than
  // appearing at full intensity immediately — same reward-the-holding intent
  // as the old edge glow, just recast as sound-wave ripples instead of a
  // tightening red border.
  function drawVoiceRings(cfg, progress) {
    const center = mouthCenter();
    const colors = (cfg.ring_colors ?? RING_COLORS_DEFAULT).map((hex) => color(hex));

    if (progress > 0.01) {
      const spawnInterval = lerp(cfg.ring_spawn_interval_min_ms ?? 550, cfg.ring_spawn_interval_max_ms ?? 140, progress);
      if (millis() - lastRingSpawn > spawnInterval) {
        voiceRings.push({ x: center.x, y: center.y, born: millis(), color: colors[ringColorIndex % colors.length] });
        ringColorIndex++;
        lastRingSpawn = millis();
      }
    }

    const lifeMs = cfg.ring_life_ms ?? 900;
    const maxRadius = cfg.ring_max_radius ?? 260;
    const baseWeight = cfg.ring_stroke_weight ?? 5;
    const blurPx = cfg.ring_glow_blur ?? 6;

    push();
    noFill();
    blendMode(ADD);
    drawingContext.filter = `blur(${blurPx}px)`;
    for (let i = voiceRings.length - 1; i >= 0; i--) {
      const r = voiceRings[i];
      const age = millis() - r.born;
      if (age > lifeMs) {
        voiceRings.splice(i, 1);
        continue;
      }
      const t = age / lifeMs;
      const radius = lerp(16, maxRadius * lerp(0.55, 1, progress), t);
      const alpha = (1 - t) * lerp(0.35, 1, progress);
      stroke(red(r.color), green(r.color), blue(r.color), alpha * 255);
      strokeWeight(baseWeight * (1 - t * 0.5));
      circle(r.x, r.y, radius * 2);
    }
    drawingContext.filter = "none";
    blendMode(BLEND);
    pop();
  }

  function triggerMascotCheer() {
    character = { spawnTime: millis() };
    SensoryAudio.playCheer();
  }

  // Animation timeline, in ms from trigger: appear in place (grow + fade in,
  // with a one-shot "tada" burst) -> a few cheerful hops with arms raised ->
  // settle into a gentle side-to-side dance -> fade out. Unlike the old
  // timeline, the character never travels from off-screen — it simply
  // materializes at rest position, so there's no "launch/pop out" moment.
  const APPEAR_MS = 350;
  const CHEER_MS = 900;
  const DANCE_MS = 1500;
  const FADE_MS = 500;
  const APPEAR_END = APPEAR_MS;
  const CHEER_END = APPEAR_END + CHEER_MS;
  const DANCE_END = CHEER_END + DANCE_MS;
  const FADE_END = DANCE_END + FADE_MS;

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
    const restY = height * 0.62;

    let y = restY;
    let scale = 1;
    let rotation = 0;
    let armLift = 0;
    let alpha = 255;
    let burst = 0; // 0..1 one-shot radiating "tada" burst, only present right at appearance

    if (elapsed < APPEAR_END) {
      const t = constrain(elapsed / APPEAR_MS, 0, 1);
      scale = easeOutBack(t);
      alpha = lerp(0, 255, Math.min(t * 1.6, 1));
      armLift = t;
      burst = 1 - t;
    } else if (elapsed < CHEER_END) {
      const t = (elapsed - APPEAR_END) / CHEER_MS;
      const hopPhase = t * TWO_PI * 3; // a few cheerful hops
      y = restY - Math.abs(sin(hopPhase)) * (cfg.hop_height ?? 22);
      armLift = 0.6 + 0.4 * Math.abs(sin(hopPhase));
      rotation = sin(hopPhase * 0.5) * 0.06;
    } else if (elapsed < DANCE_END) {
      const swayPhase = millis() * 0.004;
      y = restY + sin(swayPhase) * 8;
      rotation = sin(swayPhase) * 0.18;
      armLift = 0.5 + 0.25 * sin(swayPhase * 1.3);
    } else {
      const swayPhase = millis() * 0.004;
      y = restY + sin(swayPhase) * 8;
      rotation = sin(swayPhase) * 0.18;
      armLift = 0.5 + 0.25 * sin(swayPhase * 1.3);
      alpha = lerp(255, 0, (elapsed - DANCE_END) / FADE_MS);
    }

    if (burst > 0.01) drawCheerBurst(restX, y, burst, cfg);
    drawMascotCharacter(restX, y, rotation, scale, armLift, alpha, cfg);
  }

  // One-shot radiating rays at the moment the mascot appears — the visual
  // "tada!" beat, using the same warm ring palette so the build and the
  // payoff read as one continuous idea.
  function drawCheerBurst(x, y, intensity, cfg) {
    const colors = (cfg.ring_colors ?? RING_COLORS_DEFAULT).map((hex) => color(hex));
    const rayCount = 10;
    const length = lerp(10, cfg.burst_ray_length ?? 90, 1 - intensity);

    push();
    translate(x, y);
    blendMode(ADD);
    strokeWeight(4);
    for (let i = 0; i < rayCount; i++) {
      const angle = (i / rayCount) * TWO_PI;
      const c = colors[i % colors.length];
      stroke(red(c), green(c), blue(c), intensity * 220);
      line(cos(angle) * length * 0.4, sin(angle) * length * 0.4, cos(angle) * length, sin(angle) * length);
    }
    blendMode(BLEND);
    pop();
  }

  function drawMascotCharacter(x, y, rotation, scale_, armLift, alpha, cfg) {
    const size = cfg.character_size ?? 130;
    const bodyColor = color(cfg.character_color ?? "#FFB84D");
    const accentColor = color(cfg.character_accent_color ?? "#3B2A1A");

    push();
    translate(x, y);
    rotate(rotation);
    scale(scale_);

    // Arms swept up into a "hooray" pose as armLift climbs — replaces the
    // old spring/zig-zag, which read as a jack-in-the-box being launched
    // rather than a character celebrating.
    const shoulderY = -size * 0.05;
    const shoulderX = size * 0.42;
    const downAngle = PI * 0.35;
    const upAngle = -PI * 0.15;
    const armAngle = lerp(downAngle, upAngle, armLift);
    const armLen = size * 0.42;

    stroke(red(accentColor), green(accentColor), blue(accentColor), alpha * 0.9);
    strokeWeight(size * 0.06);
    strokeCap(ROUND);
    line(shoulderX, shoulderY, shoulderX + cos(armAngle) * armLen, shoulderY + sin(armAngle) * armLen);
    line(-shoulderX, shoulderY, -shoulderX - cos(armAngle) * armLen, shoulderY + sin(armAngle) * armLen);

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
