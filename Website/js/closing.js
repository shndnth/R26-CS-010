// 09 and the footer. The hood draws from its tip, the headline rises word by
// word with "protects" underlined, and a pen keeps tracing the hood. The
// footer's outline draws as one line, its columns rise and the name lifts.
// Desktop (`slideDriven`): returns the parts for the slide entrances.
// Phones: each part plays once as it scrolls into view.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $, $$ } = M;

  function setupClosing(slideDriven) {
    const gsap = window.gsap;
    const ST = window.ScrollTrigger;
    // Final call: the hood draws itself from its center tip out to both curls,
    // the headline rises word by word, then a small pen keeps tracing the hood
    // like a guardian keeping watch.
    const hood = $('.hood path');
    const pen = $('.hood__pen');
    const hoodLen = hood.getTotalLength();
    const hoodState = { p: 0 };
    const renderHood = () => {
      const p = hoodState.p;
      if (p <= 0.001) { hood.style.visibility = 'hidden'; return; }
      hood.style.visibility = '';
      // The path is symmetric, so its midpoint is the center tip: grow both ways from 0.5.
      hood.style.strokeDasharray = `${p.toFixed(4)} 2`;
      hood.style.strokeDashoffset = (-(0.5 - p / 2)).toFixed(4);
    };
    const penState = { u: 0.5 };
    const renderPen = () => {
      const pt = hood.getPointAtLength(penState.u * hoodLen);
      pen.setAttribute('cx', pt.x.toFixed(1));
      pen.setAttribute('cy', pt.y.toFixed(1));
    };
    renderHood();
    renderPen();

    // Split the headline into words, keeping the highlighted word whole.
    const ctaTitle = $('.cta .display');
    const parts = [];
    ctaTitle.childNodes.forEach((n) => {
      if (n.nodeType === 3) n.textContent.trim().split(/\s+/).filter(Boolean).forEach((w) => parts.push(w));
      else parts.push(n.outerHTML);
    });
    ctaTitle.innerHTML = parts.map((w) => `<span class="word"><span>${w}</span></span>`).join(' ');
    const ctaWords = $$('.word > span', ctaTitle);
    const ctaMark = $('.cta__mark', ctaTitle);
    const ctaButtons = $$('.cta .actions .btn');
    gsap.set(ctaWords, { yPercent: 110 });
    gsap.set(ctaButtons, { opacity: 0, y: 20 });
    gsap.set(ctaMark, { '--u': 0 });

    const penLoop = gsap.timeline({ paused: true, repeat: -1, defaults: { ease: 'sine.inOut', onUpdate: renderPen } })
      .to(penState, { u: 1, duration: 4.5 })
      .to(penState, { u: 0, duration: 9 })
      .to(penState, { u: 0.5, duration: 4.5 });

    // Footer: its outline draws itself as one line from the top center, the
    // columns rise in, and the big name lifts letter by letter.
    const card = $('.footer__card');
    const outlineSvg = $('.footer__outline');
    const outline = $('path', outlineSvg);
    const outState = { p: 0 };
    const renderOutline = () => {
      const p = outState.p;
      if (p <= 0.001) { outline.style.visibility = 'hidden'; return; }
      outline.style.visibility = '';
      outline.style.strokeDasharray = `${p.toFixed(4)} 2`;
      outline.style.strokeDashoffset = (-(0.5 - p / 2)).toFixed(4);
    };
    // A rounded rectangle that starts and ends at the bottom center, so its
    // midpoint is the top center and the line can grow both ways from there.
    const buildOutline = () => {
      const w = card.offsetWidth, h = card.offsetHeight, e = 0.5;
      const r = Math.min(parseFloat(getComputedStyle(card).borderTopLeftRadius) || 28, h / 2);
      const q = r - e;
      outlineSvg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      outline.setAttribute('d', `M${w / 2} ${h - e}H${r}A${q} ${q} 0 0 1 ${e} ${h - r}V${r}A${q} ${q} 0 0 1 ${r} ${e}`
        + `H${w - r}A${q} ${q} 0 0 1 ${w - e} ${r}V${h - r}A${q} ${q} 0 0 1 ${w - r} ${h - e}Z`);
    };
    buildOutline();
    renderOutline();
    if ('ResizeObserver' in window) new ResizeObserver(buildOutline).observe(card);

    // The fact strip loops seamlessly by running two copies side by side.
    const track = $('.ticker__track');
    track.appendChild($('.ticker__set', track).cloneNode(true));

    const word = $('.footer__word');
    word.innerHTML = word.textContent.trim().split('')
      .map((c) => `<span class="ch"><span>${c}</span></span>`).join('');
    const letters = $$('.ch > span', word);
    const footerParts = [...$$('.footer__top > *'), $('.ticker'), $('.footer__bottom')];
    gsap.set(letters, { yPercent: 105 });
    gsap.set(footerParts, { opacity: 0, y: 24 });

    if (slideDriven) {
      return { hoodState, renderHood, pen, penLoop, ctaWords, ctaMark, ctaButtons, outState, renderOutline, footerParts, letters };
    }

    // Phones: each part plays once as it scrolls into view.
    const ctaIntro = gsap.timeline({ paused: true })
      .to(hoodState, { p: 1, duration: 1.8, ease: 'power2.inOut', onUpdate: renderHood })
      .to(ctaWords, { yPercent: 0, duration: 0.9, ease: 'power3.out', stagger: 0.06 }, 0.5)
      .to(ctaMark, { '--u': 1, duration: 0.6, ease: 'power2.out' }, 1.1)
      .to(ctaButtons, { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out', stagger: 0.1 }, 1.1)
      .to(pen, { attr: { opacity: 1 }, duration: 0.4 }, 1.7)
      .add(() => penLoop.play(), 1.8);
    ST.create({ trigger: '.cta', start: 'top 72%', once: true, onEnter: () => ctaIntro.play() });
    ST.create({
      trigger: '.cta',
      start: 'top bottom',
      end: 'bottom top',
      onToggle: (self) => { if (ctaIntro.progress() === 1) self.isActive ? penLoop.play() : penLoop.pause(); },
    });
    gsap.timeline({ scrollTrigger: { trigger: card, start: 'top 88%', once: true } })
      .to(outState, { p: 1, duration: 1.6, ease: 'power2.inOut', onUpdate: renderOutline })
      .to(footerParts, { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out', stagger: 0.08 }, 0.35);
    gsap.to(letters, {
      yPercent: 0,
      duration: 1,
      ease: 'power3.out',
      stagger: 0.06,
      scrollTrigger: { trigger: word, start: 'top 95%', once: true },
    });
    return null;
  }

  Object.assign(M, { setupClosing });
})();
