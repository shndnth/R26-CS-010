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
    const stills = { problem: 'problem-3', product: 'product-2', privacy: 'privacy-2', quality: 'quality-1', legal: 'legal-1', how: 'how-end' };
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
