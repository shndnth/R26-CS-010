// Geometry for the line drawings: point lists, curves, even resampling,
// and a seeded random so the dot clouds look the same on every visit.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});

  const N = 300; // points per shape
  const lerp = (a, b, t) => a + (b - a) * t;
  const rad = (d) => (d * Math.PI) / 180;

  const seg = (a, b, n = 24) =>
    Array.from({ length: n + 1 }, (_, i) => [lerp(a[0], b[0], i / n), lerp(a[1], b[1], i / n)]);

  const cubic = (p0, p1, p2, p3, n = 48) =>
    Array.from({ length: n + 1 }, (_, i) => {
      const t = i / n, u = 1 - t;
      const a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
      return [a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]];
    });

  const arc = (cx, cy, r, a0, a1) => {
    const n = Math.max(8, Math.ceil(Math.abs(a1 - a0) / 3));
    return Array.from({ length: n + 1 }, (_, i) => {
      const a = rad(lerp(a0, a1, i / n));
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    });
  };

  const join = (...parts) => parts.reduce((out, p) => {
    p.forEach((pt, i) => {
      const l = out[out.length - 1];
      if (i === 0 && l && Math.hypot(l[0] - pt[0], l[1] - pt[1]) < 0.5) return;
      out.push(pt);
    });
    return out;
  }, []);

  // Smooth curve through points (Catmull-Rom as cubic Beziers).
  const through = (pts) => join(...pts.slice(0, -1).map((p1, i) => {
    const p0 = pts[i - 1] || p1, p2 = pts[i + 1], p3 = pts[i + 2] || p2;
    return cubic(
      p1,
      [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6],
      [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6],
      p2
    );
  }));

  // Evenly spaced points along a polyline, as a flat [x0, y0, x1, y1, ...] array.
  const resample = (pts) => {
    const cum = [0];
    for (let i = 1; i < pts.length; i++) {
      cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    }
    const total = cum[cum.length - 1];
    const out = new Float32Array(N * 2);
    let j = 0;
    for (let k = 0; k < N; k++) {
      const target = (total * k) / (N - 1);
      while (j < pts.length - 2 && cum[j + 1] < target) j++;
      const t = Math.min(1, Math.max(0, (target - cum[j]) / (cum[j + 1] - cum[j] || 1)));
      out[2 * k] = lerp(pts[j][0], pts[j + 1][0], t);
      out[2 * k + 1] = lerp(pts[j][1], pts[j + 1][1], t);
    }
    return out;
  };

  // Seeded random, so the dot clouds look the same on every visit.
  let seed = 11;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const scatter = (cx, cy, r, count, minD) => {
    const pts = [];
    for (let tries = 0; pts.length < count && tries < 5000; tries++) {
      const a = rand() * Math.PI * 2, d = r * Math.sqrt(rand());
      const p = [cx + d * Math.cos(a), cy + d * Math.sin(a)];
      if (pts.every((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) >= minD)) pts.push(p);
    }
    return pts;
  };

  // Round to one decimal, for compact SVG attributes.
  const f1 = (v) => Math.round(v * 10) / 10;

  Object.assign(M, { N, lerp, rad, seg, cubic, arc, join, through, resample, rand, scatter, f1 });
})();
