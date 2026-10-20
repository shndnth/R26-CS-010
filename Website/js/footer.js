// The footer card: its outline draws itself as one line from the top center,
// the columns and the fact strip rise in, and the big name lifts letter by
// letter. Motion only; otherwise the footer simply shows, with its border.
(() => {
  'use strict';
  const M = (window.R26SITE = window.R26SITE || {});
  const { $, $$ } = M;

  function initFooter(motion) {
    const card = $('.footer__card');
    if (!card) return;

    // The fact strip loops seamlessly by running two copies side by side.
    const track = $('.ticker__track');
    if (track) track.appendChild($('.ticker__set', track).cloneNode(true));

    // Letters are split so each can lift on hover (and rise in, with motion).
    const word = $('.footer__word');
    word.innerHTML = word.textContent.trim().split('')
      .map((c) => `<span class="ch"><span>${c}</span></span>`).join('');

    if (!motion) return;
    const gsap = window.gsap;
    const ST = window.ScrollTrigger;

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
    card.classList.add('is-drawn');

    const letters = $$('.ch > span', word);
    const parts = [...$$('.footer__top > *'), $('.ticker'), $('.footer__bottom')].filter(Boolean);
    gsap.set(letters, { yPercent: 105 });
    gsap.set(parts, { opacity: 0, y: 24 });

    gsap.timeline({ scrollTrigger: { trigger: card, start: 'top 85%', once: true } })
      .to(outState, { p: 1, duration: 1.6, ease: 'power2.inOut', onUpdate: renderOutline })
      .to(parts, { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out', stagger: 0.08 }, 0.35);
    gsap.to(letters, {
      yPercent: 0,
      duration: 1,
      ease: 'power3.out',
      stagger: 0.06,
      scrollTrigger: { trigger: word, start: 'top 95%', once: true },
    });
    ST.refresh();
  }

  Object.assign(M, { initFooter });
})();
