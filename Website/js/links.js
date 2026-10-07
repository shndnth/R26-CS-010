// In-page links and the address bar. Scenes and slides live inside the
// pinned deck, so links jump to their scroll stops. A shared link's #section
// is visited once, then cleared, so a refresh always starts at the top.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { sizeLines } = M;

  function initLinks(ctx) {
    const { deck, END, scrollToY } = ctx;
    const targets = { top: () => 0 };
    [['problem', 'problem-1'], ['product', 'product-1'], ['privacy', 'privacy-1'], ['quality', 'quality-1'], ['legal', 'legal-1'], ['how', 'how-end']]
      .forEach(([id, stop]) => { targets[id] = () => deck.labelToScroll(stop); });
    if (END) {
      [['packages', 'packages'], ['faq', 'faq'], ['contact', 'contact']]
        .forEach(([id, stop]) => { targets[id] = () => deck.labelToScroll(stop); });
    }
    document.addEventListener('click', (e) => {
      const a = e.target.closest('a[href^="#"]');
      if (!a) return;
      const id = a.getAttribute('href').slice(1);
      if (!id || id === 'main') return;
      const el = document.getElementById(id);
      if (!el) return;
      e.preventDefault();
      const y = targets[id] ? targets[id]() : el.getBoundingClientRect().top + window.scrollY;
      scrollToY(y);
      if (!targets[id]) {
        el.setAttribute('tabindex', '-1');
        el.focus({ preventScroll: true });
      }
    });

    // Go to the #section in the address bar. Scenes and slides live inside the
    // pinned deck, so they use their scroll stop; other sections use their spot
    // on the page, measured now that the pinned story has its full height.
    const goToHash = (immediate) => {
      const id = location.hash.slice(1);
      if (!id) { scrollToY(0, true); return; }
      if (id === 'main') return;
      const el = document.getElementById(id);
      if (targets[id]) scrollToY(targets[id](), immediate);
      else if (el) scrollToY(el.getBoundingClientRect().top + window.scrollY, immediate);
    };
    // A shared link's #section is visited once, then cleared from the address
    // bar, so refreshing the page always starts again from the top.
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
