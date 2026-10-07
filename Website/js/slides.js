// Desktop: Packages, FAQ, the closing call and the footer become slides in
// the pinned deck. Reaching one plays its whole entrance on its own, with the
// same lift-and-rise as the opening, and keys move one whole slide per press.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { root, $, $$, setupClosing } = M;

  const SLIDE_QUERY = '(min-width: 900px) and (min-height: 600px)';

  // Before the timeline is built: move the sections into the deck.
  function prepareSlides(ctx) {
    const gsap = window.gsap;
    if (window.matchMedia(SLIDE_QUERY).matches) {
      root.classList.add('slides');
      const deckEl = $('.deck');
      ctx.SLIDES = ['#packages', '#faq', '#contact', '.footer'].map((sel) => deckEl.appendChild($(sel)));
      gsap.set($$('.qa__rule'), { scaleX: 0 });
      gsap.set($$('.radar__ring'), { strokeDasharray: '1 1.02', strokeDashoffset: 1.01 });
      ctx.END = setupClosing(true);
    }
    // Crossing between the phone and desktop layouts needs a fresh setup.
    window.matchMedia(SLIDE_QUERY).addEventListener('change', () => location.reload());
  }

  // After the timeline is built: returns `showSlideAt(time)` for the deck.
  function createSlideSwitcher(ctx) {
    const gsap = window.gsap;
    const { END, SLIDES, ZONE } = ctx;
    if (!END) return () => {};

    const [sPk, sFaq] = SLIDES;
    const rise = { opacity: 1, y: 0, duration: 0.5, ease: 'power3.out' };
    const intros = [
      // 07: title and cards rise in turn.
      gsap.timeline({ paused: true })
        .fromTo([$('.eyebrow', sPk), $('.section__title', sPk), ...$$('.card', sPk)], { opacity: 0, y: 28 }, { ...rise, stagger: 0.06 }),
      // 08: intro and questions rise, dividers draw in, the scanner's rings draw.
      gsap.timeline({ paused: true })
        .fromTo([...$$('.faq__intro > *:not(.radar)', sFaq), ...$$('.qa-row', sFaq)], { opacity: 0, y: 28 }, { ...rise, stagger: 0.05 })
        .fromTo($$('.qa__rule', sFaq), { scaleX: 0 }, { scaleX: 1, duration: 0.45, ease: 'power2.out', stagger: 0.07 }, 0.15)
        .fromTo($$('.radar__ring', sFaq), { strokeDashoffset: 1.01 }, { strokeDashoffset: 0, duration: 0.9, ease: 'power2.out', stagger: 0.15 }, 0.1),
      // 09: the hood draws, the headline rises, "protects" underlines.
      gsap.timeline({ paused: true })
        .fromTo(END.hoodState, { p: 0 }, { p: 1, duration: 1, ease: 'power2.inOut', onUpdate: END.renderHood })
        .fromTo(END.ctaWords, { yPercent: 110 }, { yPercent: 0, duration: 0.6, ease: 'power3.out', stagger: 0.05 }, 0.2)
        .fromTo(END.ctaMark, { '--u': 0 }, { '--u': 1, duration: 0.45, ease: 'power2.out' }, 0.75)
        .fromTo(END.ctaButtons, { opacity: 0, y: 20 }, { ...rise, stagger: 0.08 }, 0.6),
      // Footer: the outline draws, the columns rise, the name lifts.
      gsap.timeline({ paused: true })
        .fromTo(END.outState, { p: 0 }, { p: 1, duration: 1, ease: 'power2.inOut', onUpdate: END.renderOutline })
        .fromTo(END.footerParts, { opacity: 0, y: 24 }, { ...rise, stagger: 0.06 }, 0.15)
        .fromTo(END.letters, { yPercent: 105 }, { yPercent: 0, duration: 0.6, ease: 'power3.out', stagger: 0.05 }, 0.35),
    ];

    let active = -1;
    const showSlide = (next) => {
      if (next === active) return;
      const prev = active;
      active = next;
      if (prev >= 0) {
        // The old slide lifts away...
        intros[prev].pause();
        gsap.to(SLIDES[prev], {
          opacity: 0,
          y: -24,
          duration: 0.3,
          ease: 'power2.in',
          overwrite: true,
          onComplete: () => gsap.set(SLIDES[prev], { pointerEvents: 'none', y: 0 }),
        });
      }
      if (next >= 0) {
        // ...then the new one appears and plays its entrance.
        const wait = prev >= 0 ? 0.28 : 0.05;
        gsap.set(SLIDES[next], { pointerEvents: 'auto' });
        gsap.to(SLIDES[next], { opacity: 1, y: 0, duration: 0.01, delay: wait, overwrite: true });
        intros[next].delay(wait).restart(true);
      }
    };

    return (time) => {
      let k = -1;
      for (let i = 0; i < SLIDES.length; i++) if (time >= ctx.storyEnd + i * ZONE - 0.1) k = i;
      showSlide(k);
    };
  }

  // Tabbing into a slide that isn't showing brings it on screen.
  function bindSlideFocus(ctx) {
    const { END, SLIDES, deck, scrollToY } = ctx;
    if (!END) return;
    const slideStop = new Map([[SLIDES[0], 'packages'], [SLIDES[1], 'faq'], [SLIDES[2], 'contact'], [SLIDES[3], 'footer']]);
    document.addEventListener('focusin', (e) => {
      const slide = SLIDES.find((sl) => sl.contains(e.target));
      if (slide && slide.style.opacity !== '1') scrollToY(deck.labelToScroll(slideStop.get(slide)), true);
    });
  }

  // Keyboard: from the end of 06 to the footer, each key press moves one whole
  // part (an arrow press alone only scrolls a few lines).
  function bindSlideKeys(ctx) {
    const { END, deck, scrollToY } = ctx;
    if (!END) return;
    const STOPS = ['how-end', 'packages', 'faq', 'contact', 'footer'];
    document.addEventListener('keydown', (e) => {
      const dirs = { ArrowDown: 1, PageDown: 1, ' ': 1, ArrowUp: -1, PageUp: -1 };
      let dir = dirs[e.key];
      if (!dir || e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.target.closest('input, textarea, select, [contenteditable]')) return;
      if (e.key === ' ') {
        if (e.target.closest('a, button, summary')) return;
        if (e.shiftKey) dir = -1;
      }
      const ys = STOPS.map((l) => Math.round(deck.labelToScroll(l)));
      const y = window.scrollY;
      if (y < ys[0] - 4 || y > ys[ys.length - 1] + 4) return; // outside the slides: normal keys
      const target = dir > 0 ? ys.find((v) => v > y + 4) : [...ys].reverse().find((v) => v < y - 4);
      if (target === undefined) return;
      e.preventDefault();
      scrollToY(target);
    });
  }

  Object.assign(M, { SLIDE_QUERY, prepareSlides, createSlideSwitcher, bindSlideFocus, bindSlideKeys });
})();
