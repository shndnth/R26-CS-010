// Reduced motion: no pinning or scrubbing. Each scene section gets a still
// drawing, taken from the same timeline at that scene's finished stop.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $, buildTimeline, sizeLines } = M;

  function renderStills(ctx) {
    const { stage } = ctx;
    const { renderLine } = ctx.line;
    const tl = buildTimeline(ctx, false);
    const stills = { 'glance-1': 'g1', 'glance-2': 'g2', 'glance-3': 'g3', 'glance-4': 'g4', 'glance-5': 'g5b' };
    Object.keys(stills).forEach((id) => {
      tl.seek(stills[id], false);
      renderLine();
      const svg = stage.cloneNode(true);
      svg.setAttribute('class', 'lineart');
      $(`#${id} .figure`).appendChild(svg);
    });
    tl.kill();
    sizeLines();
  }

  Object.assign(M, { renderStills });
})();
