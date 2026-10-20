// In-page links and the address bar. The "Research at a glance" steps live
// inside the pinned story, so links to them jump to their scroll stops.
// A shared link's #section is visited once, then cleared, so a refresh
// always starts at the top.
(() => {
  'use strict';
  const M = (window.R26SITE = window.R26SITE || {});
  const { sizeLines } = M;

  function initLinks(ctx) {
    const { deck, scrollToY } = ctx;
    const targets = { home: () => 0, domain: () => deck.pin.start };
    [['glance-1', 'g1'], ['glance-2', 'g2'], ['glance-3', 'g3'], ['glance-4', 'g4'], ['glance-5', 'g5a']]
      .forEach(([id, stop]) => { targets[id] = () => deck.labelToScroll(stop); });

    // Where to scroll so a section's first line sits just below the floating nav.
    const landing = (el) => {
      const navBottom = document.querySelector('.nav').getBoundingClientRect().bottom;
      const pad = parseFloat(getComputedStyle(el).paddingTop) || 0;
      return Math.max(0, el.getBoundingClientRect().top + window.scrollY + pad - navBottom - 28);
    };

    document.addEventListener('click', (e) => {
      const a = e.target.closest('a[href^="#"]');
      if (!a) return;
      const id = a.getAttribute('href').slice(1);
      if (!id || id === 'main') return;
      const el = document.getElementById(id);
      if (!el) return;
      e.preventDefault();
      const y = targets[id] ? targets[id]() : landing(el);
      scrollToY(y);
      if (!targets[id]) {
        el.setAttribute('tabindex', '-1');
        el.focus({ preventScroll: true });
      }
    });

    // Go to the #section in the address bar, measured now that the pinned
    // story has its full height.
    const goToHash = (immediate) => {
      const id = location.hash.slice(1);
      if (!id) { scrollToY(0, true); return; }
      if (id === 'main') return;
      const el = document.getElementById(id);
      if (targets[id]) scrollToY(targets[id](), immediate);
      else if (el) scrollToY(landing(el), immediate);
    };
    const clearHash = () => {
      if (location.hash) history.replaceState(null, '', location.pathname + location.search);
    };
    const onLoaded = () => {
      sizeLines();
      // Run after the browser's own jump to the anchor, which would otherwise win.
      requestAnimationFrame(() => setTimeout(() => { goToHash(true); clearHash(); }, 60));
    };
    window.addEventListener('hashchange', () => { goToHash(false); clearHash(); });
    if (document.readyState === 'complete') onLoaded();
    else window.addEventListener('load', onLoaded);
  }

  Object.assign(M, { initLinks });
})();
