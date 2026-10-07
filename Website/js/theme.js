// Theme colors, read from the stylesheet so the drawing always matches it.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { root } = M;

  function readTheme() {
    const css = getComputedStyle(root);
    const ACCENT = css.getPropertyValue('--accent').trim() || '#f2a900';
    const INK = css.getPropertyValue('--ink').trim() || '#eceae4';
    return { ACCENT, INK };
  }

  Object.assign(M, { readTheme });
})();
