// Small DOM helpers shared by every module.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});

  const root = document.documentElement;
  const $ = (s, c = document) => c.querySelector(s);
  const $$ = (s, c = document) => Array.from(c.querySelectorAll(s));

  // Setup runs in short chunks with a pause between them, so the page stays
  // responsive while the drawings and the timeline are built.
  const breathe = () => new Promise((r) => {
    if (window.scheduler && typeof window.scheduler.yield === 'function') window.scheduler.yield().then(r);
    else setTimeout(r, 0);
  });

  Object.assign(M, { root, $, $$, breathe });
})();
