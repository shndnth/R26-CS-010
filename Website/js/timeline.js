// "Research at a glance": one GSAP timeline, scrubbed by scroll, that
// crossfades five short steps and drives the one line through the drawings:
//   1 road with people (the problem)      2 simulated road and sky (simulators)
//   3 two separable groups (leakage)      4 groups merge in a protective circle
//   5 results chart, then the seal (utility and verification)
// Labels mark the scroll stops: g1 ... g4, g5a, g5b.
// With `live` false only the drawings are built (for the reduced-motion stills).
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $, $$, f1 } = M;

  function buildTimeline(ctx, live) {
    const gsap = window.gsap;
    const { L, linePath, renderLine } = ctx.line;
    const { pager, chapters, render: renderPager } = ctx.pager;
    const { ACCENT, INK } = ctx.theme;
    const {
      gProblem, horizon, lanes, people, circled,
      gScenario, hills, roadDashes, skyIcons, skyLabels, picker,
      gPrivacy, privacyDots, groupLabels, mixedOrder,
      gQuality, ticks, good, bad, scan, BAD, SCAN_FROM, SCAN_TO,
      gLegal, sealInner, sealCheck, sealDots,
    } = ctx.art;

    const tl = gsap.timeline({
      paused: true,
      defaults: { ease: 'none' },
      onUpdate() { renderLine(); renderPager(this.time()); },
    });
    let t = 0;
    const label = (name) => tl.addLabel(name, t);
    // A step: its changes start at t and finish within `change`; the scroll stop
    // lands there, then the finished state holds briefly before the next step.
    const stop = (name, change = 1.05, hold = 0.45) => { t += change; label(name); t += hold; };
    // Chapters feed the progress bar at the bottom of the story.
    chapters.length = 0;
    const chapter = (id) => chapters.push({ id, start: t });

    // Every step change runs in the same order, so nothing overlaps:
    //   0.00 to 0.30  old words and old details fade out
    //   0.00 to 0.75  the line runs out, then draws its next shape
    //   0.35 to 0.75  new words rise in
    //   0.75 onward   new details draw onto the finished shape
    const OUT = 0, IN = 0.35, DETAIL = 0.75;

    const textIn = (els, at, stagger = 0.08) => {
      if (live) tl.fromTo(els, { opacity: 0, y: 28 }, { opacity: 1, y: 0, duration: 0.4, ease: 'power3.out', stagger, immediateRender: false }, at);
    };
    const textOut = (els, at) => {
      if (live) tl.to(els, { opacity: 0, y: -24, duration: 0.3, ease: 'power2.in' }, at);
    };
    const show = (els, at, d = 0.3) => tl.to(els, { opacity: 1, duration: d }, at);
    const hide = (els, at, d = 0.3) => tl.to(els, { opacity: 0, duration: d }, at);
    const draw = (els, at, d = 0.4, stagger = 0) => tl.to(els, { strokeDashoffset: 0, duration: d, ease: 'power1.inOut', stagger }, at);
    const pop = (els, at, stagger = 0.05) => tl.to(els, { scale: 1, duration: 0.22, stagger, ease: 'back.out(2)' }, at);
    const morph = (to, at, d = 0.6) => tl.to(L, { m: to, duration: d, ease: 'power2.inOut' }, at);
    // Redraw: the current drawing runs out along its own path, then the next
    // shape draws in from its start. Every frame shows a real drawing, never a blend.
    const redraw = (to, at) => {
      tl.to(L, { d0: 1, duration: 0.28, ease: 'power1.in' }, at);
      tl.set(L, { m: to, d0: 0, d1: 0 }, at + 0.29);
      tl.to(L, { d1: 1, duration: 0.45, ease: 'power1.inOut' }, at + 0.3);
    };

    const scene = (id) => document.getElementById(id);
    const parts = (el) => [$('.glance-label', el), $('.step', el)];

    // 1. The problem: a road draws, people appear beside it, one is singled out.
    //    This plays while the story scrolls into view, before it pins.
    const S1 = scene('glance-1');
    chapter('glance-1');
    if (live) tl.to(pager, { autoAlpha: 1, duration: 0.3 }, 0.6);
    show(gProblem, t, 0.01);
    tl.to(L, { d1: 1, duration: 0.7, ease: 'power1.inOut' }, t);
    textIn(parts(S1), t + 0.1);
    draw(horizon, t + 0.45, 0.35);
    tl.to(lanes, { opacity: 1, duration: 0.06, stagger: 0.04 }, t + 0.55);
    pop(people, t + 0.65, 0.04);
    draw(circled, t + 1.15, 0.35);
    stop('g1', 1.6);

    // 2. Simulators: the line becomes a simulated road, with day, overcast
    //    and night options in the sky: tuned from real statistics.
    const S2 = scene('glance-2');
    chapter('glance-2');
    textOut([$('.domain-head', S1), ...parts(S1)], t + OUT); // the Domain intro leaves with step 1
    hide(gProblem, t + OUT);
    redraw(2, t);
    textIn(parts(S2), t + IN);
    show(gScenario, t + DETAIL, 0.01);
    draw(hills, t + DETAIL, 0.3, 0.05);
    show(roadDashes, t + DETAIL + 0.1, 0.2);
    pop(skyIcons, t + 0.95, 0.15);
    show(skyLabels, t + 1.1, 0.2);
    show(picker, t + 1.35, 0.15);
    stop('g2', 1.6);

    // 3. Leakage: records in and out of the training data are told apart.
    const S3 = scene('glance-3');
    chapter('glance-3');
    textOut(parts(S2), t + OUT);
    hide(gScenario, t + OUT);
    redraw(4, t);
    textIn(parts(S3), t + IN);
    show(gPrivacy, t + 0.6, 0.01);
    pop(privacyDots, t + 0.6, 0.01);
    show(groupLabels, t + 0.8, 0.2);
    stop('g3', 1.2);

    // 4. The approach: the groups become indistinguishable, and the line
    //    closes around them in the protection color.
    const S4 = scene('glance-4');
    chapter('glance-4');
    textOut(parts(S3), t + OUT);
    hide(groupLabels, t + OUT);
    textIn(parts(S4), t + IN);
    privacyDots.forEach((c, i) => {
      const [x, y] = mixedOrder[i];
      tl.to(c, { attr: { cx: f1(x), cy: f1(y) }, duration: 0.8, ease: 'power2.inOut' }, t + 0.1 + (i % 9) * 0.02);
    });
    morph(5, t + 0.2, 0.8);
    tl.to(linePath, { stroke: ACCENT, duration: 0.4 }, t + 0.6);
    stop('g4', 1.3);

    // 5. Evidence: utility measured on a results chart (a scan removes bad
    //    labels), then the line becomes the seal of independent verification.
    const S5 = scene('glance-5');
    chapter('glance-5');
    textOut(parts(S4), t + OUT);
    hide(gPrivacy, t + OUT);
    redraw(6, t);
    tl.to(linePath, { stroke: INK, duration: 0.01 }, t + 0.29); // back to ink while the line is hidden
    textIn(parts(S5), t + IN);
    show(gQuality, t + DETAIL, 0.01);
    show(ticks, t + DETAIL, 0.2);
    pop([...good, ...bad], t + DETAIL, 0.01);
    const SCAN = 1.1, SCAN_AT = t + 1.0;
    show(scan, SCAN_AT, 0.1);
    tl.to(scan, { x: SCAN_TO - SCAN_FROM, duration: SCAN }, SCAN_AT);
    BAD.forEach(([x], i) => {
      tl.to(bad[i], { opacity: 0, scale: 0.3, duration: 0.1 }, SCAN_AT + ((x - SCAN_FROM) / (SCAN_TO - SCAN_FROM)) * SCAN);
    });
    hide(scan, SCAN_AT + SCAN - 0.05, 0.12);
    stop('g5a', 1.0 + SCAN + 0.1);

    hide(gQuality, t + OUT);
    redraw(7, t);
    show(gLegal, t + 0.85, 0.01);
    draw(sealInner, t + 0.85, 0.3);
    draw(sealCheck, t + 1.05, 0.3);
    pop(sealDots, t + 1.2, 0.025);
    stop('g5b', 1.5, 0.6);

    ctx.storyEnd = t;
    tl.set({}, {}, t); // marks the end of the timeline (labels alone don't extend it)
    return tl;
  }

  // Pin the story while it plays. The timeline starts a little before the pin
  // (step 1 draws as the story scrolls into view) and ends with the pin.
  function pinDeck(ctx) {
    const ST = window.ScrollTrigger;
    const { tl, lenis } = ctx;
    // Scroll distance per timeline unit: short, so the story moves at a brisk pace.
    const unit = () => Math.max(window.innerHeight * 0.32, 240);
    const lead = () => window.innerHeight * 0.6; // scroll before the pin, while step 1 draws

    const pin = ST.create({
      trigger: '.deck',
      start: 'top top',
      end: () => `+=${Math.max(0, Math.round(tl.duration() * unit() - lead()))}`,
      pin: true,
      anticipatePin: 1,
    });
    const scrub = ST.create({
      trigger: '.deck',
      start: 'top 60%',
      end: () => pin.end,
      scrub: lenis ? 0.3 : 0.8,
      animation: tl,
    });
    scrub.pin = pin;
    return scrub;
  }

  Object.assign(M, { buildTimeline, pinDeck });
})();
