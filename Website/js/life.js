// Life: small loops while a scene is on screen. Each scene keeps moving
// gently once it is drawn, and the drawing leans a little toward the cursor.
// Everything here only runs while the deck is in view, and only touches
// attributes the scroll timeline does not animate.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $, N, f1, SHAPES } = M;

  function startLife(ctx) {
    const gsap = window.gsap;
    const { stage, END, SLIDES } = ctx;
    const { L } = ctx.line;
    const { ACCENT } = ctx.theme;
    const {
      gProblem, lanes, people, ripples, gFrames, roadScan,
      gCar, spokes, WHEELS, speed, gScenario, roadDashes, picker, SKY_X, gLock, files,
      gPrivacy, privacyDots, gLegal, stars, gHow, pen,
    } = ctx.art;

    const isOn = (g) => parseFloat(g.style.opacity || g.getAttribute('opacity') || 0) > 0.02;
    const pointer = { x: -1e4, y: -1e4 };
    const finePointer = window.matchMedia('(pointer: fine)').matches;
    if (finePointer) {
      // Only the line drawings lean; the hero image stays still.
      const stageX = gsap.quickTo(stage, 'x', { duration: 0.9, ease: 'power3.out' });
      const stageY = gsap.quickTo(stage, 'y', { duration: 0.9, ease: 'power3.out' });
      window.addEventListener('pointermove', (e) => {
        pointer.x = e.clientX;
        pointer.y = e.clientY;
        stageX((e.clientX / window.innerWidth - 0.5) * -16);
        stageY((e.clientY / window.innerHeight - 0.5) * -12);
      }, { passive: true });
    }

    // Lane marks rush toward the viewer, spaced in perspective.
    const laneD = (u) => {
      const ya = 470 + 490 * Math.pow(u, 1.8);
      const yb = 470 + 490 * Math.pow(Math.min(1, u + 0.05), 1.8);
      return `M500 ${f1(ya)}V${f1(yb)}`;
    };
    // People stroll along the sidewalks.
    const walkers = people.map((p) => gsap.quickSetter(p, 'x'));
    const rippleX = ripples.map((r) => gsap.quickSetter(r, 'x'));
    // Privacy dots drift, and scatter away from the cursor.
    const drift = privacyDots.map((c) => ({ c, sx: gsap.quickSetter(c, 'x'), sy: gsap.quickSetter(c, 'y'), ox: 0, oy: 0 }));
    const toLocal = (g) => {
      const m = g.getScreenCTM();
      return m ? new DOMPoint(pointer.x, pointer.y).matrixTransform(m.inverse()) : null;
    };

    let starsA = 0, wheelA = 0, pickX = SKY_X[0];
    gsap.ticker.add((time, dt) => {
      if (window.scrollY > ctx.deck.end + 50) return;
      const s = Math.min(dt, 50) / 1000;

      if (isOn(gProblem)) {
        const phase = (time * 0.35) % 1;
        lanes.forEach((ln, i) => ln.setAttribute('d', laneD(((i / lanes.length) + phase) % 1)));
        people.forEach((pp, i) => {
          const x = Math.sin(time * 1.3 + i * 1.7) * 7;
          walkers[i](x);
          rippleX[i](x);
          // Each person is tracked: a papaya ripple pulses out once they appear.
          const ph = (time * 0.55 + i * 0.37) % 1;
          const on = gsap.getProperty(pp, 'scale') > 0.9 && gsap.getProperty(pp, 'opacity') > 0.5;
          ripples[i].setAttribute('r', (6 + ph * 30).toFixed(1));
          ripples[i].setAttribute('opacity', on ? (0.9 * (1 - ph)).toFixed(3) : 0);
        });
      }
      if (isOn(gFrames)) {
        const ph = (time * 0.45) % 1;
        roadScan.setAttribute('transform', `translate(0 ${f1(ph * 480)})`);
        roadScan.setAttribute('opacity', (Math.sin(ph * Math.PI) * 0.9).toFixed(3));
      }
      if (isOn(gCar)) {
        wheelA = (wheelA + s * 260) % 360;
        spokes.forEach((g, i) => g.setAttribute('transform', `rotate(${wheelA.toFixed(1)} ${WHEELS[i][0]} ${WHEELS[i][1]})`));
        speed.style.strokeDashoffset = ((time * 140) % 40).toFixed(1);
      }
      if (isOn(gScenario)) {
        // The picker ring glides from day to overcast to night and back, and
        // the lane marks run toward the viewer.
        const k = Math.floor(time / 1.6) % SKY_X.length;
        pickX += (SKY_X[k] - pickX) * 0.12;
        picker.setAttribute('cx', f1(pickX));
        roadDashes.style.strokeDashoffset = ((time * 60) % 40).toFixed(1);
      }
      if (isOn(gLock)) {
        // Each file's checksum lights up in turn.
        const k = Math.floor(time * 2.2) % (files.length + 2);
        files.forEach((f, i) => { f.style.stroke = i === k ? ACCENT : ''; });
      }
      if (isOn(gPrivacy)) {
        const p = finePointer ? toLocal(gPrivacy) : null;
        drift.forEach((d, i) => {
          const bx = +d.c.getAttribute('cx'), by = +d.c.getAttribute('cy');
          let tx = Math.sin(time * 0.9 + i * 2.1) * 5, ty = Math.cos(time * 0.8 + i * 1.3) * 5;
          if (p) {
            const dx = bx - p.x, dy = by - p.y, dist = Math.hypot(dx, dy);
            if (dist < 170 && dist > 0.1) {
              const push = (1 - dist / 170) * 70;
              tx += (dx / dist) * push;
              ty += (dy / dist) * push;
            }
          }
          d.ox += (tx - d.ox) * 0.12;
          d.oy += (ty - d.oy) * 0.12;
          d.sx(d.ox);
          d.sy(d.oy);
        });
      }
      if (isOn(gLegal)) {
        starsA = (starsA + s * 8) % 360;
        stars.setAttribute('transform', `rotate(${starsA.toFixed(2)} 500 430)`);
      }
      // A pen keeps tracing the finished route.
      const routeDone = isOn(gHow) && L.m >= 8 && L.d1 > 0.999 && L.d0 < 0.01;
      pen.setAttribute('opacity', routeDone ? 1 : 0);
      if (routeDone) {
        const u = (Math.sin(time * 0.45 - Math.PI / 2) + 1) / 2;
        const k = Math.round(u * (N - 1)) * 2, w = SHAPES[8];
        pen.setAttribute('cx', f1(w[k]));
        pen.setAttribute('cy', f1(w[k + 1]));
      }
      // The hood's pen traces only while the closing slide is showing and drawn.
      if (END) {
        const ctaOn = SLIDES[2].style.opacity === '1' && END.hoodState.p > 0.999;
        if (ctaOn !== END.penOn) {
          END.penOn = ctaOn;
          END.pen.setAttribute('opacity', ctaOn ? 1 : 0);
          if (ctaOn) END.penLoop.play(); else END.penLoop.pause();
        }
      }
    });
  }

  Object.assign(M, { startLife });
})();
