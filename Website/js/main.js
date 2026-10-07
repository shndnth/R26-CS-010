/*
 * MURAGALA: one protective line, drawn by scroll.
 *
 * The page is a single pinned "deck". One GSAP timeline, scrubbed by scroll,
 * crossfades the words and drives one SVG path through every scene. On
 * desktop, Packages, FAQ, the closing call and the footer follow as slides.
 * With prefers-reduced-motion (or if GSAP fails to load) the scenes are plain
 * sections, each with a still drawing taken from the same timeline.
 *
 * Each file in js/ adds its parts to one shared namespace (window.MURAGALA);
 * index.html loads them in order and this file, last, wires them together.
 * Plain scripts (not modules), so the page also works when opened as a file.
 * Where to edit:
 *   shapes.js        the line's shapes (road, car, camera, ... route)
 *   scenes.js        the details drawn around the line in each scene
 *   timeline.js      what happens at each scroll step, and how far apart
 *   life.js          the small loops that keep each scene moving
 *   slides.js        07, 08, 09 and the footer on desktop
 *   closing.js       the hood, headline and footer animations
 *   links.js         in-page links and the address bar
 */
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { root, $, breathe, initNavMenu, sizeLines, createLine, buildScenes, centerScenes, finishScenes, createLogo, readTheme, createPager, buildTimeline, pinDeck, renderStills, createSmoothScroll, prepareSlides, createSlideSwitcher, bindSlideFocus, bindSlideKeys, setupClosing, startLife, initLinks } = M;

  async function run() {
    const gsap = window.gsap;
    const ST = window.ScrollTrigger;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const motion = !reduce && !window.MURAGALA_FORCE_STATIC && !!gsap && !!ST;
    if (!motion) {
      root.classList.remove('motion');
      root.classList.add('static');
    }

    initNavMenu();
    sizeLines();
    window.addEventListener('resize', sizeLines);

    // Without GSAP there is nothing more to draw; the page reads as plain sections.
    if (!gsap) return;

    // Everything the modules share. ZONE is the scroll given to each desktop slide.
    const stage = $('.stage');
    const ctx = { motion, stage, ZONE: 1.1, END: null, SLIDES: null, storyEnd: 0 };

    ctx.line = createLine(stage);
    await breathe();
    ctx.art = buildScenes(stage);
    await breathe();
    centerScenes(stage, ctx.art);
    finishScenes(stage, ctx.art);
    await breathe();

    ctx.logo = createLogo();
    ctx.theme = readTheme();
    ctx.pager = createPager(motion);

    if (!motion) {
      renderStills(ctx);
      return;
    }

    gsap.registerPlugin(ST);
    ST.config({ ignoreMobileResize: true });
    Object.assign(ctx, createSmoothScroll());
    prepareSlides(ctx);

    ctx.logo.render();
    ctx.line.renderLine();

    ctx.tl = buildTimeline(ctx, true);
    ctx.pager.bind(ctx.storyEnd || ctx.tl.duration());
    const showSlideAt = createSlideSwitcher(ctx);
    await breathe();
    ctx.deck = pinDeck(ctx, showSlideAt);

    startLife(ctx);
    ctx.logo.intro();

    // Phones: the closing section and the footer set up once the browser is idle.
    if (!ctx.END) {
      if ('requestIdleCallback' in window) requestIdleCallback(() => setupClosing(false), { timeout: 1500 });
      else setTimeout(() => setupClosing(false), 300);
    }

    initLinks(ctx);
    bindSlideFocus(ctx);
    bindSlideKeys(ctx);
    ST.addEventListener('refresh', sizeLines);
  }

  M.started = true; // tells the safety net in index.html that the scripts loaded
  run().catch((err) => {
    // Setup crashed part way (for example, a browser serving a stale copy of
    // one file). Rather than leave a half-built page, reload once in the
    // plain readable layout. If even that fails, just show the plain layout.
    console.error('MURAGALA setup failed:', err);
    if (!window.MURAGALA_FORCE_STATIC) {
      try {
        sessionStorage.setItem('muragala-static', '1');
        location.reload();
        return;
      } catch (e) {}
    }
    root.classList.remove('motion', 'slides', 'lenis');
    root.classList.add('static');
  });
})();
