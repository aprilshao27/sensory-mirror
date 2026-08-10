// Shared geometry helpers — generic hull/polyline math. Not tied to any one
// modality; any module that renders along or within a tracked outline can
// use these (currently proxemic.js, via its pose-landmark hull).

function expandHull(hull, factor) {
  const cx = hull.reduce((sum, p) => sum + p.x, 0) / hull.length;
  const cy = hull.reduce((sum, p) => sum + p.y, 0) / hull.length;
  return hull.map((p) => ({ x: cx + (p.x - cx) * factor, y: cy + (p.y - cy) * factor }));
}

function hullCentroid(hull) {
  const cx = hull.reduce((sum, p) => sum + p.x, 0) / hull.length;
  const cy = hull.reduce((sum, p) => sum + p.y, 0) / hull.length;
  return { x: cx, y: cy };
}

function convexHull(points) {
  const sorted = points.slice().sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

// Even-spacing resample of a closed polyline loop, so effects that travel
// along the outline (glow-rope strands, star spawn points) move at a
// constant visual speed instead of bunching up near short hull edges.
function resamplePolyline(hull, n) {
  const closed = hull.concat([hull[0]]);
  const segLens = [];
  let total = 0;
  for (let i = 0; i < closed.length - 1; i++) {
    const d = dist(closed[i].x, closed[i].y, closed[i + 1].x, closed[i + 1].y);
    segLens.push(d);
    total += d;
  }
  if (total === 0) return hull;

  const step = total / n;
  const out = [];
  for (let i = 0; i < n; i++) {
    const targetDist = i * step;
    let acc = 0;
    let j = 0;
    for (; j < segLens.length; j++) {
      if (acc + segLens[j] >= targetDist) break;
      acc += segLens[j];
    }
    const segLen = segLens[j] || 1;
    const tt = (targetDist - acc) / segLen;
    const a = closed[j];
    const b = closed[j + 1] || closed[0];
    out.push({ x: lerp(a.x, b.x, tt), y: lerp(a.y, b.y, tt) });
  }
  return out;
}
