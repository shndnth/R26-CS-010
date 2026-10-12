// The reading sections: hero counters, the current section in the nav, the
// methodology pipeline drawing in, and document and slide buttons.
// Everything here also works, or degrades to plain content, without motion.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $, $$ } = M;

  // Hero counters count up from zero on load (motion only; the HTML holds the real values).
  function initCounters(motion) {
    if (!motion) return;
    $$('[data-count]').forEach((el) => {
      const to = parseInt(el.textContent, 10);
      const state = { v: 0 };
      el.textContent = '0';
      window.gsap.to(state, {
        v: to,
        duration: 1.2,
        delay: 0.3,
        ease: 'power2.out',
        onUpdate: () => { el.textContent = String(Math.round(state.v)); },
      });
    });
  }

  // Highlight the nav link of the section being read.
  function initActiveNav() {
    const links = $$('.nav__links a[href^="#"]');
    const sections = links.map((a) => document.getElementById(a.getAttribute('href').slice(1))).filter(Boolean);
    if (!('IntersectionObserver' in window) || !sections.length) return;
    const mark = (id) => links.forEach((a) => {
      const on = a.getAttribute('href') === `#${id}`;
      a.classList.toggle('is-current', on);
      if (on) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
    });
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) mark(e.target.id); });
    }, { rootMargin: '-45% 0px -50% 0px' });
    sections.forEach((s) => io.observe(s));
    // At the very bottom (the footer), the last section's link is the current one.
    window.addEventListener('scroll', () => {
      const atEnd = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 4;
      if (atEnd) mark(sections[sections.length - 1].id);
    }, { passive: true });
  }

  // The methodology pipeline draws its line and nodes when it comes into view.
  function initPipeline(motion) {
    const pipe = $('.pipeline');
    if (!pipe || !motion || !('IntersectionObserver' in window)) return;
    pipe.classList.add('is-pending');
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        pipe.classList.remove('is-pending');
        io.disconnect();
      }
    }, { rootMargin: '0px 0px -20% 0px' });
    io.observe(pipe);
  }

  // Does a file exist? Asked differently depending on how the page is opened:
  //  - served (GitHub Pages, localhost): a HEAD request, which downloads nothing;
  //  - opened as a file (double-clicking index.html): browsers block requests
  //    from local pages, so the file is probed with an element instead. Chrome
  //    and Edge use a background prefetch; other browsers load the PDF into a
  //    hidden viewer element for a moment. Both report load or error.
  const SERVED = /^https?:$/.test(location.protocol);
  const viaHead = (href) => fetch(href, { method: 'HEAD', cache: 'no-store' }).then((r) => r.ok).catch(() => false);
  const viaPrefetch = (href) => new Promise((resolve) => {
    const link = document.createElement('link');
    const end = (ok) => { clearTimeout(timer); link.remove(); resolve(ok); };
    const timer = setTimeout(() => end(null), 2500); // no answer: let the next method decide
    link.rel = 'prefetch';
    link.onload = () => end(true);
    link.onerror = () => end(false);
    link.href = href;
    document.head.appendChild(link);
  });
  const viaObject = (href) => new Promise((resolve) => {
    const obj = document.createElement('object');
    const end = (ok) => { clearTimeout(timer); obj.remove(); resolve(ok); };
    const timer = setTimeout(() => end(false), 5000);
    obj.type = 'application/pdf';
    obj.setAttribute('aria-hidden', 'true');
    obj.tabIndex = -1;
    obj.style.cssText = 'position:absolute;left:-9999px;top:0;width:1px;height:1px;opacity:0;pointer-events:none';
    obj.onload = () => end(true);
    obj.onerror = () => end(false);
    obj.data = href;
    document.body.appendChild(obj);
  });
  // Local probes run one at a time, so many PDFs are never loaded at once.
  let localQueue = Promise.resolve();
  const viaLocal = (href) => {
    const run = () => (window.chrome ? viaPrefetch(href).then((ok) => (ok === null ? viaObject(href) : ok)) : viaObject(href));
    const result = localQueue.then(run, run);
    localQueue = result.catch(() => {});
    return result;
  };
  const fileExists = (href) => (SERVED && window.fetch ? viaHead(href) : viaLocal(href));

  const fileLink = (href, text) => {
    const a = document.createElement('a');
    a.className = 'btn btn--sm';
    a.href = href;
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = text;
    return a;
  };

  // Document and slide buttons start disabled ("Coming soon"). Each file is
  // checked, and its button becomes a real link only if the file exists, so
  // adding a file to docs/ is all it takes, and there are no dead links.
  function initFiles() {
    $$('[data-href]').forEach((btn) => {
      const href = btn.getAttribute('data-href');
      fileExists(href).then((ok) => {
        if (ok) btn.replaceWith(fileLink(href, btn.getAttribute('data-label') || 'Open'));
      });
    });

    // Numbered sets (checklist-1.pdf, checklist-2.pdf ...): look for 1, 2, 3 ...
    // and stop at the first number that is missing. One button per file found.
    $$('[data-series]').forEach((btn) => {
      const prefix = btn.getAttribute('data-series');
      const label = btn.getAttribute('data-label') || 'File';
      const holder = btn.parentElement;
      const card = holder.closest('.doc-card');
      const count = card && $('[data-series-count]', card);
      const look = (n) => {
        if (n > 30) return;
        const href = `${prefix}${n}.pdf`;
        fileExists(href).then((ok) => {
          if (!ok) return;
          if (n === 1) btn.remove();
          holder.appendChild(fileLink(href, `${label} ${n}`));
          if (count) count.textContent = n === 1 ? '1 file' : `${n} files`;
          look(n + 1);
        });
      };
      look(1);
    });
  }

  // Transitions: headings, text, cards and timeline items rise in as they
  // scroll into view; cards in one row arrive one after another. Motion only,
  // and the classes are added here, so without JavaScript everything shows.
  const REVEAL = [
    '.section > .wrap > .eyebrow', '.section__title', '.section__lead', '.sub__title', '.sub__label',
    '.prose p', '.table-wrap', '.statement-block', '.main-objective', '.comp-card', '.method-step',
    '.finding', '.tags', '.tl-tabs', '.tl-item', '.doc-card', '.slide-card', '.person',
    '.contact__row', '.field', '.contact-form__send',
  ].join(',');
  function initReveal(motion) {
    if (!motion || !('IntersectionObserver' in window)) return;
    const els = $$(REVEAL).filter((el) => !el.closest('.deck'));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        e.target.classList.add('is-in');
        io.unobserve(e.target);
      });
    }, { rootMargin: '0px 0px -8% 0px' });
    els.forEach((el) => {
      // Stagger siblings of the same kind (cards in a grid, items in a list).
      const kin = [...el.parentElement.children].filter((c) => c.className === el.className);
      el.style.setProperty('--d', `${Math.min(kin.indexOf(el), 6) * 0.08}s`);
      el.classList.add('reveal');
      io.observe(el);
    });
  }

  Object.assign(M, { initCounters, initActiveNav, initPipeline, initFiles, initReveal });
})();
