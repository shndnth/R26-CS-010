// The small-screen menu: open, close on link click, outside click or Escape.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $ } = M;

  function initNavMenu() {
    const toggle = $('.nav__toggle');
    const links = $('#nav-links');
    const setMenu = (open) => {
      toggle.setAttribute('aria-expanded', String(open));
      links.classList.toggle('is-open', open);
    };
    toggle.addEventListener('click', () => setMenu(toggle.getAttribute('aria-expanded') !== 'true'));
    links.addEventListener('click', (e) => { if (e.target.closest('a')) setMenu(false); });
    document.addEventListener('click', (e) => { if (!e.target.closest('.nav__menu')) setMenu(false); });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') {
        setMenu(false);
        toggle.focus();
      }
    });
  }

  Object.assign(M, { initNavMenu });
})();
