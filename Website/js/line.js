// The one line: L.m picks the shape (fractions blend between shapes),
// L.d0 and L.d1 are the visible span along it (0 to 1).
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $, N, f1, SHAPES, lastShape } = M;

  function createLine(stage) {
    const linePath = $('.line', stage);
    const L = { m: 0, d0: 0, d1: 0 }; // shape position, visible span start and end (0 to 1)
    const buf = new Float32Array(N * 2);
    let lastM = -1, lastD0 = -1, lastD1 = -1;

    const toD = (a) => {
      let d = `M${f1(a[0])} ${f1(a[1])}`;
      for (let k = 2; k < a.length - 2; k += 2) {
        d += `Q${f1(a[k])} ${f1(a[k + 1])} ${f1((a[k] + a[k + 2]) / 2)} ${f1((a[k + 1] + a[k + 3]) / 2)}`;
      }
      return `${d}L${f1(a[a.length - 2])} ${f1(a[a.length - 1])}`;
    };

    const renderLine = () => {
      const m = Math.min(Math.max(L.m, 0), lastShape);
      if (m !== lastM) {
        const i = Math.min(Math.floor(m), lastShape - 1);
        const t = m - i, A = SHAPES[i], B = SHAPES[i + 1];
        for (let k = 0; k < buf.length; k++) buf[k] = A[k] + (B[k] - A[k]) * t;
        linePath.setAttribute('d', toD(buf));
        lastM = m;
      }
      if (L.d0 !== lastD0 || L.d1 !== lastD1) {
        const span = L.d1 - L.d0;
        if (span <= 0.0005) {
          linePath.style.visibility = 'hidden';
        } else {
          linePath.style.visibility = '';
          linePath.style.strokeDasharray = `${span.toFixed(4)} 2`;
          linePath.style.strokeDashoffset = (-L.d0).toFixed(4);
        }
        lastD0 = L.d0;
        lastD1 = L.d1;
      }
    };

    return { L, linePath, renderLine };
  }

  Object.assign(M, { createLine });
})();
