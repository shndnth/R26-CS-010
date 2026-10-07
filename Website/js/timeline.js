// The story timeline: one GSAP timeline, scrubbed by scroll, that crossfades
// the words and drives the line through every scene. Labels mark the scroll
// stops (hero, problem-1 ... how-end, then packages, faq, contact, footer).
// With `live` false only the drawings are built (for the reduced-motion stills).
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $, $$, f1, nodeFrac } = M;

  function buildTimeline(ctx, live) {
    const gsap = window.gsap;
    const { L, linePath, renderLine } = ctx.line;
    const { pager, chapters, render: renderPager } = ctx.pager;
    const { ACCENT, INK } = ctx.theme;
    const { END, SLIDES, ZONE } = ctx;
    const {
      gProblem, horizon, lanes, people, circled, gate, gateState, renderGate, gFrames, smallCars, boxes, boxTags,
      gCar, wheels, hubs, windows, spokes, yoloBox, yoloTag, gScenario, hills, roadDashes, skyIcons, skyLabels, picker,
      gLock, keyhole, files, gPrivacy, privacyDots, groupLabels, mixedOrder,
      gQuality, ticks, good, bad, scan, BAD, SCAN_FROM, SCAN_TO, gLegal, sealInner, sealCheck, sealDots,
      gHow, nodes,
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
    const stop = (name, change = 1.05, hold = 0.4) => { t += change; label(name); t += hold; };
    // Chapters feed the pager at the bottom of the screen.
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
    const eyebrow = (el) => $('.eyebrow', el);

    // 1. Hero: the full logo is already drawn; it steps aside for the line.
    const hero = scene('top');
    label('hero');
    t = 0.5;
    if (live) {
      textOut([$('.display--hero', hero), $('.lede', hero), $('.actions', hero), $('.scroll-hint', hero)], t);
      tl.to($('.hero-art', hero), { opacity: 0, scale: 0.97, duration: 0.3 }, t);
      tl.set([$('.actions', hero)], { visibility: 'hidden' }, t + 0.3);
      tl.set(hero, { pointerEvents: 'none' }, t + 0.3);
      tl.to(pager, { autoAlpha: 1, duration: 0.3 }, t + 0.2);
    }

    // 2. The problem: a road, people beside it, one of them singled out.
    const P = scene('problem');
    const pSteps = $$('.step', P);
    chapter('problem');
    tl.to(L, { d1: 1, duration: 0.7, ease: 'power1.inOut' }, t + 0.3); // the road draws once the logo has gone
    show(gProblem, t + 0.3, 0.01);
    textIn([eyebrow(P), pSteps[0]], t + IN);
    draw(horizon, t + 0.7, 0.35);
    tl.to(lanes, { opacity: 1, duration: 0.06, stagger: 0.04 }, t + 0.8);
    pop(people, t + 0.9, 0.04);
    stop('problem-1', 1.3);

    // Privacy law: a striped gate swings down across the road.
    textOut(pSteps[0], t + OUT);
    textIn(pSteps[1], t + IN);
    show(gate, t + 0.3, 0.15);
    tl.to(gateState, { a: 0, duration: 0.55, ease: 'bounce.out', onUpdate: renderGate }, t + 0.35);
    stop('problem-2', 0.95);

    // The gate lifts away, and one person is singled out.
    textOut(pSteps[1], t + OUT);
    textIn(pSteps[2], t + IN);
    hide(gate, t + OUT, 0.25);
    draw(circled, t + 0.5, 0.35);
    stop('problem-3', 0.85);

    // 3. What you get: labeled cars, a car, a scenario picker, a lock.
    const R = scene('product');
    const rSteps = $$('.step', R);
    chapter('product');
    textOut([pSteps[2], eyebrow(P)], t + OUT);
    hide([...people, circled], t + OUT);
    textIn([eyebrow(R), rSteps[0]], t + IN);
    show(gFrames, t + 0.3, 0.01);
    draw(smallCars, t + 0.4, 0.35, 0.1);
    draw(boxes, t + 0.6, 0.3, 0.1);
    show(boxTags, t + 0.85, 0.12);
    stop('product-1');

    textOut(rSteps[0], t + OUT);
    hide([gFrames, ...horizon, ...lanes], t + OUT);
    redraw(1, t);
    textIn(rSteps[1], t + IN);
    show(gCar, t + DETAIL, 0.01);
    draw([...wheels, ...hubs, windows], t + DETAIL, 0.3);
    show(spokes, t + DETAIL + 0.2, 0.2);
    draw(yoloBox, t + DETAIL + 0.1, 0.3);
    show(yoloTag, t + DETAIL + 0.35, 0.1);
    stop('product-2', 1.2);

    textOut(rSteps[1], t + OUT);
    hide(gCar, t + OUT);
    redraw(2, t);
    textIn(rSteps[2], t + IN);
    // Your roads, your weather, your hours: hills rise, the lane marks appear,
    // then day, overcast and night pop into the sky and the picker ring arrives.
    show(gScenario, t + DETAIL, 0.01);
    draw(hills, t + DETAIL, 0.3, 0.05);
    show(roadDashes, t + DETAIL + 0.1, 0.2);
    pop(skyIcons, t + 0.95, 0.15);
    show(skyLabels, t + 1.1, 0.2);
    show(picker, t + 1.35, 0.15);
    stop('product-3', 1.6);

    textOut(rSteps[2], t + OUT);
    hide(gScenario, t + OUT);
    redraw(3, t);
    textIn(rSteps[3], t + IN);
    show(gLock, t + DETAIL, 0.01);
    draw(keyhole, t + DETAIL, 0.25);
    draw(files, t + DETAIL + 0.1, 0.25, 0.04);
    stop('product-4', 1.25);

    // 4. Privacy: two separable groups merge into one; the line closes around them.
    const V = scene('privacy');
    const statA = $('.stat--a', V), statB = $('.stat--b', V), statement = $('.statement', V);
    chapter('privacy');
    textOut([rSteps[3], eyebrow(R)], t + OUT);
    hide(gLock, t + OUT);
    redraw(4, t);
    textIn([eyebrow(V), statA], t + IN);
    show(gPrivacy, t + 0.6, 0.01);
    pop(privacyDots, t + 0.6, 0.01);
    show(groupLabels, t + 0.8, 0.2);
    stop('privacy-1', 1.2);

    // "Exposed" gives way to "Guesswork" as the groups become indistinguishable,
    // and the conclusion follows on the same stop.
    textOut(statA, t + OUT);
    hide(groupLabels, t + OUT);
    textIn(statB, t + IN);
    privacyDots.forEach((c, i) => {
      const [x, y] = mixedOrder[i];
      tl.to(c, { attr: { cx: f1(x), cy: f1(y) }, duration: 0.8, ease: 'power2.inOut' }, t + 0.1 + (i % 9) * 0.02);
    });
    morph(5, t + 0.2, 0.8);
    // The line that closes around the mixed group turns amber: protection.
    tl.to(linePath, { stroke: ACCENT, duration: 0.4 }, t + 0.6);
    textIn(statement, t + 0.85);
    stop('privacy-2', 1.3);

    // 5. Quality: an axis, results, and a scan that removes bad labels, all on one stop.
    const Q = scene('quality');
    const lead = $('.quality__lead', Q), facts = $$('.facts li', Q);
    chapter('quality');
    textOut([eyebrow(V), statB, statement], t + OUT);
    hide(gPrivacy, t + OUT);
    redraw(6, t);
    tl.to(linePath, { stroke: INK, duration: 0.01 }, t + 0.29); // back to ink while the line is hidden
    textIn([eyebrow(Q), lead], t + IN);
    show(gQuality, t + DETAIL, 0.01);
    show(ticks, t + DETAIL, 0.2);
    pop([...good, ...bad], t + DETAIL, 0.01);
    const SCAN = 1.1, SCAN_AT = t + 1.0;
    show(scan, SCAN_AT, 0.1);
    tl.to(scan, { x: SCAN_TO - SCAN_FROM, duration: SCAN }, SCAN_AT);
    BAD.forEach(([x], i) => {
      tl.to(bad[i], { opacity: 0, scale: 0.3, duration: 0.1 }, SCAN_AT + ((x - SCAN_FROM) / (SCAN_TO - SCAN_FROM)) * SCAN);
    });
    textIn(facts, SCAN_AT + 0.1, 0.25);
    hide(scan, SCAN_AT + SCAN - 0.05, 0.12);
    stop('quality-1', 1.0 + SCAN + 0.1);

    // 6. Legal: the line becomes a seal; all three promises arrive on one stop.
    const G = scene('legal');
    const claims = $$('.claims li', G), note = $('.legal__note', G);
    chapter('legal');
    textOut([eyebrow(Q), lead, ...facts], t + OUT);
    hide(gQuality, t + OUT);
    redraw(7, t);
    textIn([eyebrow(G), ...claims, note], t + IN, 0.12);
    show(gLegal, t + 0.85, 0.01);
    draw(sealInner, t + 0.85, 0.3);
    draw(sealCheck, t + 1.05, 0.3);
    pop(sealDots, t + 1.2, 0.025);
    stop('legal-1', 1.6);

    // 7. How it works: the seal's line runs out, then travels through four nodes.
    const H = scene('how');
    const items = $$('.process li', H);
    chapter('how');
    textOut([eyebrow(G), ...claims, note], t + OUT);
    hide(gLegal, t + OUT);
    tl.to(L, { d0: 1, duration: 0.4, ease: 'power1.in' }, t);
    tl.set(L, { m: 8, d0: 0, d1: 0 }, t + 0.41);
    textIn(eyebrow(H), t + IN);
    if (live) tl.fromTo($('.process', H), { opacity: 0, y: 28 }, { opacity: 1, y: 0, duration: 0.4, ease: 'power3.out', immediateRender: false }, t + IN + 0.05);
    show(gHow, t + 0.45, 0.2);
    t += 0.6;
    items.forEach((li, i) => {
      // The line reaches node i exactly as step i turns black.
      tl.to(L, { d1: nodeFrac[i], duration: 0.45, ease: 'power1.inOut' }, t);
      tl.to(nodes[i], { fill: ACCENT, duration: 0.05 }, t + 0.43);
      if (live) tl.to(li, { color: INK, duration: 0.1 }, t + 0.4);
      t += 0.5;
    });
    tl.to(L, { d1: 1, duration: 0.3 }, t);
    t += 0.35;
    label('how-end');
    t += 0.25;
    ctx.storyEnd = t;

    // 06 steps aside exactly like the opening does (scroll-driven). After that,
    // each slide plays its own entrance as soon as you reach it (see showSlide),
    // so its zone here is just a short stretch of scroll.
    if (live && END) {
      textOut([eyebrow(H), $('.process', H)], t + OUT);
      hide(gHow, t + OUT);
      tl.to(L, { d0: 1, duration: 0.3, ease: 'power1.in' }, t);
      tl.to(pager, { autoAlpha: 0, duration: 0.25 }, t);
      t += 0.35;
      ctx.storyEnd = t;
      ['packages', 'faq', 'contact', 'footer'].forEach((name, k) => tl.addLabel(name, ctx.storyEnd + k * ZONE + 0.3));
      // The footer is the last stop: the page ends exactly where it appears.
      t = ctx.storyEnd + (SLIDES.length - 1) * ZONE + 0.3;
    }
    tl.set({}, {}, t); // marks the end of the timeline (labels alone don't extend it)

    return tl;
  }

  // Pin the deck and let scroll scrub the timeline. `onTime` hears the
  // timeline's position on every update (the desktop slides use it).
  function pinDeck(ctx, onTime) {
    const ST = window.ScrollTrigger;
    const { tl, lenis } = ctx;
    // Scroll distance per timeline unit: short, so the story moves at a brisk pace.
    const unit = () => Math.max(window.innerHeight * 0.32, 240);

    const deck = ST.create({
      trigger: '.deck',
      start: 'top top',
      end: () => `+=${Math.round(tl.duration() * unit())}`,
      pin: true,
      scrub: lenis ? 0.3 : 0.8,
      animation: tl,
      anticipatePin: 1,
      onUpdate: (self) => onTime(self.progress * tl.duration()),
      onRefresh: (self) => onTime(self.progress * tl.duration()),
    });
    return deck;
  }

  Object.assign(M, { buildTimeline, pinDeck });
})();
