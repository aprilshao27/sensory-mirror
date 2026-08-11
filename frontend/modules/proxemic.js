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
  // No target until the first real message arrives, so nothing (including
  // retreat's starfield) renders during the gap right after page load —
  // previously defaulting to "retreat" made the starfield flash in at full
  // weight before the backend had said anything at all.
  let targetState = null;
  let stateWeights = { close_still: 0, active_movement: 0, retreat: 0 };
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
  let silhouetteContour = []; // smoothed edge traced from maskImg, normalised [0,1] mask-space points

  let bubbles = [];
  let stars = [];

  function handleMessage(msg) {
    targetState = msg.state;
    landmarks = msg.landmarks ?? [];
    targetMovement = msg.movement ?? 0;
    if (msg.mask) {
      targetCloseness = msg.mask.closeness ?? 1;
      maskImg = decodeMask(msg.mask);
      // Only recomputed when a new mask arrives (not every draw frame) —
      // the trace itself is cheap, but there's no reason to redo it faster
      // than the underlying data changes.
      silhouetteContour = computeContour(maskImg);
    } else {
      // No mask this frame (no pose detected) — let the underlay dissolve
      // instead of freezing at its last shape/size.
      targetCloseness = 0;
    }
  }

  // ---- Silhouette contour (smooth vector edge traced from the mask) -----
  //
  // Full marching-squares would need per-cell edge segments stitched into
  // ordered loops, which is a lot of fiddly linking code for a mask this
  // small. Radial ray-casting gets the same result — an ordered ring of
  // points tracing the mask's outer edge — for far less code: cast rays out
  // from the centroid, keep the farthest point on each ray that's still
  // inside the mask (robust to the blur/noise fringe left by the backend's
  // distance-based degradation), then smooth the resulting polygon.
  const CONTOUR_ALPHA_THRESHOLD = 50;
  const CONTOUR_RAY_COUNT = 72;
  const CONTOUR_RAY_STEP = 0.75;
  const CONTOUR_MIN_PIXELS = 8; // below this the mask is just noise — skip tracing
  const CONTOUR_SMOOTH_ITERATIONS = 2;

  function computeContour(mask) {
    const w = mask.width;
    const h = mask.height;
    const pixels = mask.pixels;
    const alphaAt = (x, y) => pixels[(y * w + x) * 4 + 3];

    let sumX = 0;
    let sumY = 0;
    let n = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (alphaAt(x, y) > CONTOUR_ALPHA_THRESHOLD) {
          sumX += x;
          sumY += y;
          n++;
        }
      }
    }
    if (n < CONTOUR_MIN_PIXELS) return [];

    const cx = sumX / n;
    const cy = sumY / n;
    const maxRadius = Math.hypot(w, h) / 2;

    const raw = [];
    for (let i = 0; i < CONTOUR_RAY_COUNT; i++) {
      const angle = (i / CONTOUR_RAY_COUNT) * TWO_PI;
      const dx = Math.cos(angle);
      const dy = Math.sin(angle);
      let farX = cx;
      let farY = cy;
      for (let r = CONTOUR_RAY_STEP; r <= maxRadius; r += CONTOUR_RAY_STEP) {
        const px = cx + dx * r;
        const py = cy + dy * r;
        const ix = Math.round(px);
        const iy = Math.round(py);
        if (ix < 0 || ix >= w || iy < 0 || iy >= h) break;
        if (alphaAt(ix, iy) > CONTOUR_ALPHA_THRESHOLD) {
          farX = px;
          farY = py;
        }
      }
      raw.push({ x: farX / w, y: farY / h });
    }
    return chaikinSmooth(raw, CONTOUR_SMOOTH_ITERATIONS);
  }

  // Chaikin corner-cutting: replaces each edge with two points 1/4 and 3/4
  // along it. A few iterations turns the ray-cast polygon's straight,
  // slightly jittery segments into a smooth closed curve without needing a
  // full spline implementation.
  function chaikinSmooth(points, iterations) {
    let pts = points;
    for (let iter = 0; iter < iterations; iter++) {
      const next = [];
      const n = pts.length;
      for (let i = 0; i < n; i++) {
        const p0 = pts[i];
        const p1 = pts[(i + 1) % n];
        next.push({ x: p0.x * 0.75 + p1.x * 0.25, y: p0.y * 0.75 + p1.y * 0.25 });
        next.push({ x: p0.x * 0.25 + p1.x * 0.75, y: p0.y * 0.25 + p1.y * 0.75 });
      }
      pts = next;
    }
    return pts;
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
    drawSilhouetteUnderlay(cfg.underlay, cfg.active_movement);
    drawBubbleTube(cfg.close_still, hull);
    drawGlowRope(cfg.active_movement, hull);
    drawStarryProjection(cfg.retreat, hull);
  }

  // ---- Silhouette (segmentation mask underlay) --------------------------

  function drawSilhouetteUnderlay(cfg, activeCfg) {
    if (!maskImg) return;

    const activeWeight = stateWeights.active_movement;
    const movementFactor = constrain(smoothedMovement / (activeCfg?.movement_intensity_ref ?? 0.05), 0, 1);

    // As active_movement engages, the underlay shifts from its resting
    // colour toward a distinct accent and brightens further with how fast
    // the movement is — the silhouette itself reads as "energised" instead
    // of staying a flat backdrop while only the glow rope reacts.
    const baseColor = color(cfg.color ?? "#e6e6e6");
    const accentColor = color(activeCfg?.underlay_accent ?? cfg.color ?? "#e6e6e6");
    const c = lerpColor(baseColor, accentColor, activeWeight * lerp(0.6, 1, movementFactor));
    const intensity = 1 + activeWeight * lerp(0.3, 1.1, movementFactor);

    // close_still keeps the underlay much more transparent — a quiet
    // backdrop for the bubble tube rather than competing with it — while
    // active_movement's intensity boost above is untouched.
    const closeStillFade = lerp(1, cfg.close_still_opacity_mult ?? 0.35, stateWeights.close_still);

    // Present across all three states, but continuously shrinks and fades
    // toward 0 as the person retreats, letting it visually dissolve into the star field.
    const maskScale = lerp(0.35, 1.0, smoothedCloseness);
    const opacity = (cfg.base_opacity ?? 0.35) * lerp(0.1, 1.0, smoothedCloseness) * intensity * closeStillFade;
    const softness = cfg.glow_softness ?? 18;

    push();
    translate(width / 2, height / 2);
    scale(maskScale);
    translate(-width / 2, -height / 2);

    // Additive stack of increasingly blurred, increasingly faint passes reads
    // as a soft glow/bloom radiating off the silhouette, rather than one flat
    // blurred shape — that's the "ethereal" quality a single blur pass can't give.
    blendMode(ADD);
    const layers = [
      { blur: softness * 0.4, alpha: opacity * 0.7 },
      { blur: softness * 1.2, alpha: opacity * 0.45 },
      { blur: softness * 2.8, alpha: opacity * 0.25 },
    ];
    for (const layer of layers) {
      drawingContext.filter = `blur(${layer.blur}px)`;
      tint(red(c), green(c), blue(c), layer.alpha * 255);
      image(maskImg, 0, 0, width, height);
    }
    noTint();
    drawingContext.filter = "none";
    blendMode(BLEND);
    pop();

    drawSilhouetteContour(cfg, maskScale, opacity);
  }

  // Strokes the traced, smoothed mask edge (see computeContour above) on top
  // of the soft fill wash — a crisp, intentional line instead of relying on
  // the low-res mask's blurred/blocky pixel edge for definition.
  function drawSilhouetteContour(cfg, maskScale, baseOpacity) {
    if (silhouetteContour.length < 3) return;

    const c = color(cfg.contour_color ?? cfg.color ?? "#ffffff");
    const strokeWeightPx = cfg.contour_width ?? 2;
    const softness = cfg.contour_softness ?? 6;
    const opacity = (cfg.contour_opacity ?? 1.0) * baseOpacity;

    push();
    translate(width / 2, height / 2);
    scale(maskScale);
    translate(-width / 2, -height / 2);

    noFill();
    blendMode(ADD);
    // Same layered-blur-pass trick as the fill wash, but stroking the traced
    // curve instead of the pixel mask, so it reads as a glowing line rather
    // than a flat blurred blob.
    const passes = [
      { blur: softness * 0.3, alpha: opacity * 0.9, weight: strokeWeightPx },
      { blur: softness * 1.0, alpha: opacity * 0.5, weight: strokeWeightPx * 1.5 },
      { blur: softness * 2.2, alpha: opacity * 0.25, weight: strokeWeightPx * 2.5 },
    ];
    for (const p of passes) {
      drawingContext.filter = `blur(${p.blur}px)`;
      stroke(red(c), green(c), blue(c), p.alpha * 255);
      strokeWeight(p.weight);
      beginShape();
      for (const pt of silhouetteContour) {
        vertex(pt.x * width, pt.y * height);
      }
      endShape(CLOSE);
    }
    drawingContext.filter = "none";
    blendMode(BLEND);
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

  // Steps through a colour list in a continuous loop (wrapping back to the
  // start), blending smoothly between consecutive entries rather than
  // jumping — used to make the bubble tube's colour drift gradually instead
  // of holding one flat colour.
  function cyclePalette(colors, speed) {
    const n = colors.length;
    if (n === 0) return color(255);
    if (n === 1) return color(colors[0]);
    const t = (millis() * speed) % n;
    const i0 = Math.floor(t);
    const i1 = (i0 + 1) % n;
    return lerpColor(color(colors[i0]), color(colors[i1]), t - i0);
  }

  function drawBubbleTube(cfg, hull) {
    // Multiplying by (1 - active_movement weight) suppresses the tube as
    // soon as movement engages, rather than only fading with close_still's
    // own (shared, slower) state-transition easing — otherwise bubbles keep
    // rising through the glow rope for a while after movement is triggered.
    const w = stateWeights.close_still * (1 - stateWeights.active_movement);
    const palette = cfg.color_cycle ?? ["#4A90D9", "#9B59B6", "#E85D9E", "#FF8C42"];
    const cycleSpeed = cfg.color_cycle_speed ?? 0.00025;
    const c = cyclePalette(palette, cycleSpeed);
    const density = cfg.base_density ?? 0.5;
    const riseMin = cfg.base_rise_speed ?? 1.2;
    const riseMax = cfg.max_rise_speed ?? 5.5;
    const sizeMin = cfg.min_bubble_size ?? 6;
    const sizeMax = cfg.max_bubble_size ?? 22;
    const wobbleAmount = cfg.wobble_amount ?? 6;
    const wobbleSpeed = cfg.wobble_speed ?? 0.05;
    const softness = cfg.bubble_softness ?? 2;

    // 0.05 is comfortably above the backend's active_movement threshold
    // (0.015), giving headroom for "how fast" to keep scaling once triggered.
    const movementFactor = constrain(smoothedMovement / 0.05, 0, 1);

    if (w > 0.02) {
      // Spawn across the full canvas width, not just the hull's bounds, so
      // the tube fills the whole screen at any distance instead of scaling
      // with how big the person reads in frame.
      const spawnRate = density * (0.3 + movementFactor * 1.5) * w;
      if (random() < spawnRate) {
        const candidateX = random(0, width);
        const candidateY = height + random(0, 20);
        if (!isInsideSilhouette(candidateX, candidateY)) {
          bubbles.push({
            x: candidateX,
            baseX: candidateX,
            y: candidateY,
            r: random(sizeMin, sizeMax),
            speed: lerp(riseMin, riseMax, movementFactor) * random(0.7, 1.3),
            wobblePhase: random(TWO_PI),
            wobbleFreq: random(0.7, 1.3),
          });
        }
      }
    }

    noStroke();
    // Soft edges read as liquid/translucent rather than crisp, solid discs.
    drawingContext.filter = `blur(${softness}px)`;
    for (let i = bubbles.length - 1; i >= 0; i--) {
      const b = bubbles[i];
      b.y -= b.speed;
      // Buoyant side-to-side wobble as it rises, rather than a straight
      // line, so it reads as jostling through liquid.
      b.x = b.baseX + sin(b.y * wobbleSpeed * b.wobbleFreq + b.wobblePhase) * wobbleAmount;
      if (isInsideSilhouette(b.x, b.y)) {
        continue;
      }
      const alpha = (140 + movementFactor * 110) * w;
      // Faster-rising bubbles stretch slightly along their motion, like
      // surface tension dragging, instead of staying a rigid circle.
      const stretch = 1 + constrain(b.speed / riseMax, 0, 1) * 0.35;

      push();
      translate(b.x, b.y);
      fill(red(c), green(c), blue(c), alpha);
      ellipse(0, 0, b.r, b.r * stretch);
      // Glossy highlight suggests light refracting through the bubble's
      // surface, the detail that most reads as "liquid" over "flat circle".
      fill(255, 255, 255, alpha * 0.5);
      ellipse(-b.r * 0.22, -b.r * 0.25 * stretch, b.r * 0.35, b.r * 0.35);
      pop();

      if (b.y < -40 || w < 0.01) bubbles.splice(i, 1);
    }
    drawingContext.filter = "none";
  }

  // ---- active_movement: glow-rope / fibre-optic strands ------------------

  // Maps the traced mask contour (normalised [0,1], mask-space — see
  // computeContour above) into canvas pixel space, applying the same
  // centered scale the silhouette underlay uses. Following the actual mask
  // edge instead of the landmark convex hull lets the rope hug concavities
  // (waist, armpits, neck) rather than riding a blocky, convex outline.
  function contourToCanvasSpace() {
    if (silhouetteContour.length < 3) return [];
    const maskScale = lerp(0.35, 1.0, smoothedCloseness);
    const cx = width / 2;
    const cy = height / 2;
    return silhouetteContour.map((p) => ({
      x: cx + (p.x * width - cx) * maskScale,
      y: cy + (p.y * height - cy) * maskScale,
    }));
  }

  function drawGlowRope(cfg, hull) {
    const w = stateWeights.active_movement;
    if (w < 0.01) return;

    // Prefer the organic mask contour; fall back to the landmark hull if no
    // mask has arrived yet (e.g. right at state entry).
    const contourPath = contourToCanvasSpace();
    const source = contourPath.length >= 3 ? contourPath : hull;
    if (source.length < 3) return;
    const path = resamplePolyline(source, 160);

    const colors = (cfg.strand_colors ?? ["#ffffff"]).map((hex) => color(hex));
    const strandCount = cfg.strand_count ?? 6;
    const speed = cfg.strand_speed ?? 0.4;
    const glowSize = cfg.strand_glow_size ?? 10;
    const trailLength = 8;

    // Mirrors the bubble tube's movement ramp (0.05 threshold) so speed,
    // glow size and brightness all climb together as movement intensity
    // increases, instead of the rope pulsing at one fixed rate regardless
    // of how much the person is actually moving.
    const movementFactor = constrain(smoothedMovement / (cfg.movement_intensity_ref ?? 0.05), 0, 1);
    const speedMult = lerp(1, cfg.max_speed_mult ?? 2.2, movementFactor);
    const glowMult = lerp(1, cfg.max_glow_mult ?? 1.8, movementFactor);
    const alphaMult = lerp(0.85, 1, movementFactor);
    const blurPx = (cfg.strand_glow_blur ?? 4) * lerp(1, 1.6, movementFactor);

    noStroke();
    // Additive blend + blur makes the strands read as a soft glowing rope
    // wrapping the body rather than a string of flat, hard-edged discs.
    blendMode(ADD);
    drawingContext.filter = `blur(${blurPx}px)`;
    for (let i = 0; i < strandCount; i++) {
      const phase = (millis() * 0.0002 * speed * speedMult + i / strandCount) % 1;
      const headIdx = floor(phase * path.length);
      const c = colors[i % colors.length];
      for (let t = 0; t < trailLength; t++) {
        const p = path[(headIdx - t + path.length) % path.length];
        const alpha = (1 - t / trailLength) * 220 * w * alphaMult;
        fill(red(c), green(c), blue(c), alpha);
        circle(p.x, p.y, glowSize * glowMult * (1 - t / (trailLength * 1.5)));
      }
    }
    drawingContext.filter = "none";
    blendMode(BLEND);
  }

  // ---- retreat: starry projection ------------------------------------------

  function drawStarryProjection(cfg, hull) {
    const w = stateWeights.retreat;
    const c = color(cfg.star_color ?? "#ffffff");
    const starCount = cfg.star_count ?? 90;
    const drift = cfg.drift_speed ?? 0.3;
    // Sized/lit to stay legible at the distance retreat is actually viewed
    // from — a person far enough back to trigger retreat is too far to
    // resolve the old 1.5-3.5px, alpha-200 dots.
    const sizeMin = cfg.star_min_size ?? 3;
    const sizeMax = cfg.star_max_size ?? 7;
    const glowSize = cfg.star_glow_size ?? 16;
    const maxAlpha = cfg.star_max_alpha ?? 255;
    // Slow bob layered on top of the linear drift below — the drift alone is
    // too faint (a few px/sec) to read as "moving" over a short glance; the
    // wobble gives each star its own small, steady, ongoing motion without
    // relying on it actually crossing the screen.
    const wobbleAmount = cfg.star_wobble_amount ?? 10;
    const wobbleSpeed = cfg.star_wobble_speed ?? 0.15;

    // Spawn straight up to the weight-scaled target each frame, rather than
    // a per-frame coin flip for at most one star — that coin flip was the
    // main source of "takes forever to appear". The state-weight ramp (`w`,
    // shared easing used by all three states) still paces how fast the
    // field fills in overall; this just stops adding a second, slower delay
    // on top of it. Each star's own drift/twinkle speed is untouched, so the
    // motion stays calm even though the field now populates quickly.
    const targetVisible = Math.floor(starCount * w);
    // Reaches the far corners, so the field ends up covering the whole
    // screen rather than just a center-ish circle.
    const maxRadius = Math.hypot(width, height) / 2;
    while (w > 0.02 && stars.length < targetVisible) {
      // Each star spawns directly at its "final" spot rather than having to
      // drift there — the spawn radius grows from 0 (center) outward as the
      // field fills in, so the expansion itself reads as fast and radiating
      // from the middle, while every star's own motion afterward stays a
      // slow, calm wobble instead of a long trip across the screen.
      const progress = stars.length / starCount;
      const radius = maxRadius * progress;
      const angle = random(TWO_PI);
      const angle2 = random(TWO_PI);
      stars.push({
        // bx/by is the slow-drifting anchor point; x/y (computed each frame
        // below) adds a bounded wobble on top, so the wobble stays a gentle
        // bob in place rather than compounding into a random walk.
        bx: width / 2 + cos(angle) * radius,
        by: height / 2 + sin(angle) * radius,
        x: 0,
        y: 0,
        vx: cos(angle2) * random(0.1, 0.3) * drift,
        vy: sin(angle2) * random(0.1, 0.3) * drift,
        alpha: 0,
        twinklePhase: random(TWO_PI),
        wobblePhase: random(TWO_PI),
        wobbleFreq: random(0.7, 1.3),
        size: random(sizeMin, sizeMax),
      });
    }

    // Physics/lifecycle pass, kept separate from rendering below so the glow
    // halo and the crisp core can each do their own full pass over `stars`
    // without double-updating position/alpha.
    for (let i = stars.length - 1; i >= 0; i--) {
      const s = stars[i];
      s.bx += s.vx;
      s.by += s.vy;
      const t = millis() * 0.0006 * wobbleSpeed * s.wobbleFreq + s.wobblePhase;
      s.x = s.bx + cos(t) * wobbleAmount;
      s.y = s.by + sin(t * 1.3) * wobbleAmount;
      // Fast fade-in per star (independent of the slow drift/twinkle) so a
      // newly spawned star pops in quickly instead of adding its own delay.
      s.alpha = min(s.alpha + 25, maxAlpha);
      if (w < 0.02 || s.bx < -50 || s.bx > width + 50 || s.by < -50 || s.by > height + 50) {
        stars.splice(i, 1);
      }
    }

    noStroke();
    // Soft additive bloom around each star — the thing that actually reads
    // from a distance, since a blurred, oversized halo stays visible well
    // past the point where a sharp few-pixel dot disappears into the screen.
    blendMode(ADD);
    drawingContext.filter = `blur(${glowSize}px)`;
    for (const s of stars) {
      const twinkle = 0.6 + 0.4 * sin(frameCount * 0.05 + s.twinklePhase);
      fill(red(c), green(c), blue(c), s.alpha * twinkle * w * 0.8);
      circle(s.x, s.y, s.size * 4);
    }
    drawingContext.filter = "none";
    blendMode(BLEND);

    // Crisp core on top of the glow, so up close the field still reads as
    // distinct points of light rather than a uniform haze.
    for (const s of stars) {
      const twinkle = 0.6 + 0.4 * sin(frameCount * 0.05 + s.twinklePhase);
      fill(red(c), green(c), blue(c), s.alpha * twinkle * w);
      circle(s.x, s.y, s.size);
    }
  }

  return { handleMessage, update, draw };
})();
