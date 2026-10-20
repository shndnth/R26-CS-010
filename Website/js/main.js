/*
 * R26-CS-010 research website.
 *
 * A single page of readable sections. One part of Domain, "Research at a
 * glance", is a short pinned story: one GSAP timeline, scrubbed by scroll,
 * crossfades five steps and drives one SVG line through the drawings. With
 * prefers-reduced-motion, or without JavaScript, every part is a plain
 * section and the story steps show still drawings.
 *
 * Each file in js/ adds its parts to one shared namespace (window.R26SITE);
 * index.html loads them in order and this file, last, wires them together.
 * Plain scripts (not modules), so the page also works when opened as a file.
 * Where to edit:
 *   shapes.js        the line's shapes
 *   scenes.js        the details drawn around the line in each step
 *   timeline.js      what happens at each scroll step, and how far apart
 *   life.js          the small loops that keep each drawing moving
 *   sections.js      hero counters, nav highlight, pipeline, document buttons, transitions
 *   contact-form.js  the contact form and its confirmation dialog
 *   footer.js        the footer's outline, fact strip and name animations
 *   links.js         in-page links and the address bar
 */
(() => {
  'use strict';
  const M = (window.R26SITE = window.R26SITE || {});
  const {
    root, $, breathe, initNavMenu, sizeLines, createLine, buildScenes, centerScenes, finishScenes,
    createLogo, readTheme, createPager, buildTimeline, pinDeck, renderStills, createSmoothScroll,
    startLife, initLinks, initCounters, initActiveNav, initPipeline, initFiles, initReveal,
    initContactForm, initFooter,
  } = M;

  async function run() {
    const gsap = window.gsap;
    const ST = window.ScrollTrigger;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const motion = !reduce && !window.R26_FORCE_STATIC && !!gsap && !!ST;
    if (!motion) {
      root.classList.remove('motion');
      root.classList.add('static');
    }

    initNavMenu();
    initActiveNav();
    initFiles();
    initContactForm();
    sizeLines();
    window.addEventListener('resize', sizeLines);

    // Without GSAP there is nothing more to draw; the page reads as plain sections.
    if (!gsap) return;

    const stage = $('.stage');
    const ctx = { motion, stage, storyEnd: 0 };

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
      initFooter(false);
      return;
    }

    gsap.registerPlugin(ST);
    ST.config({ ignoreMobileResize: true });
    Object.assign(ctx, createSmoothScroll());

    ctx.logo.render();
    ctx.logo.intro();
    initCounters(motion);
    initPipeline(motion);
    initReveal(motion);
    ctx.line.renderLine();

    ctx.tl = buildTimeline(ctx, true);
    ctx.pager.bind(ctx.storyEnd || ctx.tl.duration());
    await breathe();
    ctx.deck = pinDeck(ctx);

    startLife(ctx);
    initFooter(true);
    initLinks(ctx);
    ST.addEventListener('refresh', sizeLines);
  }

  M.started = true; // tells the safety net in index.html that the scripts loaded
  run().catch((err) => {
    // Setup crashed part way (for example, a browser serving a stale copy of
    // one file). Rather than leave a half-built page, reload once in the
    // plain readable layout. If even that fails, just show the plain layout.
    console.error('R26 site setup failed:', err);
    if (!window.R26_FORCE_STATIC) {
      try {
        sessionStorage.setItem('r26-static', '1');
        location.reload();
        return;
      } catch (e) {}
    }
    root.classList.remove('motion', 'lenis');
    root.classList.add('static');
  });
})();
