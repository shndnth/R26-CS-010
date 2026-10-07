// Keep every line drawing's strokes near 1.75px on screen, at any size.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $$ } = M;

  function sizeLines() {
      $$('svg.lineart').forEach((svg) => {
        const vb = svg.viewBox.baseVal;
        const r = svg.getBoundingClientRect();
        if (!r.width || !r.height || !vb) return;
        const scale = Math.min(r.width / vb.width, r.height / vb.height);
        svg.style.setProperty('--sw', (1.75 / scale).toFixed(3));
      });
  }

  Object.assign(M, { sizeLines });
})();
