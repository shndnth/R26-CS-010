// Smooth scrolling (Lenis): turns stepped mouse-wheel jumps into one
// continuous glide, so the scrubbed drawing never lurches. Touch keeps
// native scrolling. `scrollToY` is the one way the page jumps anywhere.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});

  function createSmoothScroll() {
    const gsap = window.gsap;
    const ST = window.ScrollTrigger;
    let lenis = null;
    if (window.Lenis) {
      lenis = new window.Lenis({ lerp: 0.1, smoothWheel: true });
      lenis.on('scroll', ST.update);
      gsap.ticker.add((time) => lenis.raf(time * 1000));
      gsap.ticker.lagSmoothing(0);
    }
    const scrollToY = (y, immediate = false) => {
      // Re-measure first: the pinned story grows the page after smooth scrolling
      // starts, and a stale page height would cut jumps short.
      if (lenis) lenis.resize();
      if (lenis) lenis.scrollTo(y, immediate ? { immediate: true, force: true } : { duration: 1.4, force: true });
      else window.scrollTo({ top: y, behavior: immediate ? 'auto' : 'smooth' });
    };
    return { lenis, scrollToY };
  }

  Object.assign(M, { createSmoothScroll });
})();
