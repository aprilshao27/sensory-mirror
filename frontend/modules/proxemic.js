// Phase 1 — Proxemic interaction: placeholder visuals.
// Renders a smooth interpolation between three continuous states published
// by the backend over WebSocket. Every state's visual
// carries a "weight" that eases toward 1 (active) or 0 (inactive) each frame,
// so transitions read as continuous.
//
// Content (colours, particle counts, speeds...) is not hardcoded here -
// it all comes from config.proxemic, loaded by core.js and passed into
// draw() each frame.

const Proxemic = (() => {
  let targetState = "retreat";
  let stateWeights = { close_still: 0, active_movement: 0, retreat: 1 };
  const STATE_SMOOTHING = 0.05;

  let landmarks = [];

  let smoothedMovement = 0;
  let targetMovement = 0;
  const MOVEMENT_SMOOTHING = 0.15;

  // closeness: 1 = right at the mirror, 0 = at/past the retreat threshold.
  // Drives the underlay's continuous shrink/fade, independent of the
  // discrete state, so retreat reads as "distance".
  let smoothedCloseness = 1;
  let targetCloseness = 1;
  const CLOSENESS_SMOOTHING = 0.08;

  let maskImg = null; // decoded segmentation mask, low-res + degradation-blurred server-side

  let bubbles = [];
  let stars = [];

  function handleMessage(msg) {
    targetState = msg.state;
    landmarks = msg.landmarks ?? [];
    targetMovement = msg.movement ?? 0;
    if (msg.mask) {
      targetCloseness = msg.mask.closeness ?? 1;
      maskImg = decodeMask(msg.mask);
    } else {
      // No mask this frame (no pose detected) — let the underlay dissolve
      // instead of freezing at its last shape/size.
      targetCloseness = 0;
    }
  }

  function decodeMask(maskMsg) {
    const binary = atob(maskMsg.data);
    const img = createImage(maskMsg.width, maskMsg.height);
    img.loadPixels();
    for (let i = 0; i < binary.length; i++) {
      const v = binary.charCodeAt(i);
      img.pixels[i * 4] = 255;
      img.pixels[i * 4 + 1] = 255;
      img.pixels[i * 4 + 2] = 255;
      img.pixels[i * 4 + 3] = v; // mask strength lives in alpha
    }
    img.updatePixels();
    return img;
  }

  function update() {
    for (const state of Object.keys(stateWeights)) {
      const target = state === targetState ? 1 : 0;
      stateWeights[state] += (target - stateWeights[state]) * STATE_SMOOTHING;
    }
    smoothedMovement += (targetMovement - smoothedMovement) * MOVEMENT_SMOOTHING;
    smoothedCloseness += (targetCloseness - smoothedCloseness) * CLOSENESS_SMOOTHING;
  }

  function draw(cfg) {
    if (maskImg) maskImg.loadPixels();

    const hull = computeHull();
    drawSilhouetteUnderlay(cfg.underlay);
    drawBubbleTube(cfg.close_still, hull);
    drawGlowRope(cfg.active_movement, hull);
    drawStarryProjection(cfg.retreat, hull);
  }

  // ---- Silhouette (segmentation mask underlay) --------------------------

  function drawSilhouetteUnderlay(cfg) {
    if (!maskImg) return;
    const c = color(cfg.color ?? "#e6e6e6");

    // Present across all three states, but continuously shrinks and fades
    // toward 0 as the person retreats, letting it visually dissolve into the star field.
    const maskScale = lerp(0.35, 1.0, smoothedCloseness);
    const opacity = (cfg.base_opacity ?? 0.35) * lerp(0.1, 1.0, smoothedCloseness);

    push();
    translate(width / 2, height / 2);
    scale(maskScale);
    translate(-width / 2, -height / 2);

    drawingContext.filter = `blur(${cfg.glow_softness ?? 18}px)`;
    tint(red(c), green(c), blue(c), opacity * 255);
    image(maskImg, 0, 0, width, height);
    noTint();
    drawingContext.filter = "none";
    pop();
  }

  function isInsideSilhouette(x, y) {
    if (!maskImg) return false;

    // Match the same centered scale transform used by the silhouette underlay.
    const maskScale = lerp(0.35, 1.0, smoothedCloseness);
    if (maskScale <= 0) return false;

    const scaledX = (x - width / 2) / maskScale + width / 2;
    const scaledY = (y - height / 2) / maskScale + height / 2;

    if (scaledX < 0 || scaledX >= width || scaledY < 0 || scaledY >= height) return false;

    const mx = floor((scaledX / width) * maskImg.width);
    const my = floor((scaledY / height) * maskImg.height);
    const clampedX = constrain(mx, 0, maskImg.width - 1);
    const clampedY = constrain(my, 0, maskImg.height - 1);

    const idx = (clampedY * maskImg.width + clampedX) * 4 + 3;
    const alpha = maskImg.pixels[idx] ?? 0;
    return alpha > 40;
  }

  // ---- Body hull (from pose landmarks) -----------------------------------

  // MediaPipe BlazePose landmark indices: 0 = nose, 11/12 = shoulders,
  // 13-32 = elbows/wrists/hips/legs. Fine finger points (17-22) are left out
  // of the body hull — they make the outline spiky without reading as "person".
  const BODY_HULL_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32];

  function computeHull() {
    if (landmarks.length < 33) return [];
    const pts = landmarks.map((lm) => ({ x: lm.x * width, y: lm.y * height }));
    const bodyPoints = BODY_HULL_INDICES.map((i) => pts[i]);
    const hull = convexHull(bodyPoints);
    if (hull.length < 3) return [];
    return expandHull(hull, 1.3);
  }

  // ---- close_still: sensory bubble tube ----------------------------------

  function drawBubbleTube(cfg, hull) {
    const w = stateWeights.close_still;
    const c = color(cfg.bubble_color ?? "#50C878");
    const density = cfg.base_density ?? 0.5;
    const riseMin = cfg.base_rise_speed ?? 1.2;
    const riseMax = cfg.max_rise_speed ?? 5.5;
    const sizeMin = cfg.min_bubble_size ?? 6;
    const sizeMax = cfg.max_bubble_size ?? 22;

    // 0.05 is comfortably above the backend's active_movement threshold
    // (0.015), giving headroom for "how fast" to keep scaling once triggered.
    const movementFactor = constrain(smoothedMovement / 0.05, 0, 1);

    if (w > 0.02) {
      // Spawn across the full canvas width, not just the hull's bounds, so
      // the tube fills the screen at any distance instead of scaling with
      // how big the person reads in frame.
      const spawnRate = density * (0.3 + movementFactor * 1.5) * w;
      if (random() < spawnRate) {
        const candidateX = random(0, width);
        const candidateY = height + random(0, 20);
        if (!isInsideSilhouette(candidateX, candidateY)) {
          bubbles.push({
            x: candidateX,
            y: candidateY,
            r: random(sizeMin, sizeMax),
            speed: lerp(riseMin, riseMax, movementFactor) * random(0.7, 1.3),
          });
        }
      }
    }

    noStroke();
    for (let i = bubbles.length - 1; i >= 0; i--) {
      const b = bubbles[i];
      b.y -= b.speed;
      if (isInsideSilhouette(b.x, b.y)) {
        continue;
      }
      const alpha = (140 + movementFactor * 110) * w;
      fill(red(c), green(c), blue(c), alpha);
      circle(b.x, b.y, b.r);
      if (b.y < -40 || w < 0.01) bubbles.splice(i, 1);
    }
  }

  // ---- active_movement: glow-rope / fibre-optic strands ------------------

  function drawGlowRope(cfg, hull) {
    const w = stateWeights.active_movement;
    if (w < 0.01 || hull.length < 3) return;
    const path = resamplePolyline(hull, 140);
    const colors = (cfg.strand_colors ?? ["#ffffff"]).map((hex) => color(hex));
    const strandCount = cfg.strand_count ?? 6;
    const speed = cfg.strand_speed ?? 0.4;
    const glowSize = cfg.strand_glow_size ?? 10;
    const trailLength = 8;

    noStroke();
    for (let i = 0; i < strandCount; i++) {
      const phase = (millis() * 0.0002 * speed + i / strandCount) % 1;
      const headIdx = floor(phase * path.length);
      const c = colors[i % colors.length];
      for (let t = 0; t < trailLength; t++) {
        const p = path[(headIdx - t + path.length) % path.length];
        const alpha = (1 - t / trailLength) * 220 * w;
        fill(red(c), green(c), blue(c), alpha);
        circle(p.x, p.y, glowSize * (1 - t / (trailLength * 1.5)));
      }
    }
  }

  // ---- retreat: starry projection ------------------------------------------

  function randomEdgePoint() {
    const edge = floor(random(4));
    if (edge === 0) return { x: random(width), y: 0 };
    if (edge === 1) return { x: random(width), y: height };
    if (edge === 2) return { x: 0, y: random(height) };
    return { x: width, y: random(height) };
  }

  function drawStarryProjection(cfg, hull) {
    const w = stateWeights.retreat;
    const c = color(cfg.star_color ?? "#ffffff");
    const starCount = cfg.star_count ?? 90;
    const drift = cfg.drift_speed ?? 0.3;

    if (w > 0.02 && stars.length < starCount && random() < w * 0.6) {
      // Spawn from the screen edges rather than the hull outline, so stars
      // fill the whole canvas at any distance instead of clustering around
      // how big the person reads in frame. They still drift toward wherever
      // the person is (or screen center, if nobody's tracked at all).
      const origin = randomEdgePoint();
      const centroid = hull.length >= 3 ? hullCentroid(hull) : { x: width / 2, y: height / 2 };
      const dx = origin.x - centroid.x;
      const dy = origin.y - centroid.y;
      const mag = max(Math.hypot(dx, dy), 0.001);
      stars.push({
        x: origin.x,
        y: origin.y,
        vx: (dx / mag) * random(0.2, 0.6) * drift,
        vy: (dy / mag) * random(0.2, 0.6) * drift,
        alpha: 0,
        twinklePhase: random(TWO_PI),
        size: random(1.5, 3.5),
      });
    }

    noStroke();
    for (let i = stars.length - 1; i >= 0; i--) {
      const s = stars[i];
      s.x += s.vx;
      s.y += s.vy;
      s.alpha = min(s.alpha + 4, 200);
      const twinkle = 0.6 + 0.4 * sin(frameCount * 0.05 + s.twinklePhase);
      fill(red(c), green(c), blue(c), s.alpha * twinkle * w);
      circle(s.x, s.y, s.size);
      if (w < 0.02 || s.x < -50 || s.x > width + 50 || s.y < -50 || s.y > height + 50) {
        stars.splice(i, 1);
      }
    }
  }

  return { handleMessage, update, draw };
})();
