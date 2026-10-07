// The hero image reveal (top to bottom, on load).
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $ } = M;

  function createLogo() {
    const logo = $('.hero-photo');
    const logoIntro = { v: 0 };
    const renderLogo = () => logo.style.setProperty('--r', logoIntro.v.toFixed(4));
    // The whole logo draws in on load.
    const intro = () => window.gsap.to(logoIntro, { v: 1, duration: 1.1, ease: 'power2.out', onUpdate: renderLogo });
    return { render: renderLogo, intro };
  }

  Object.assign(M, { createLogo });
})();
