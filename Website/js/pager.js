// The chapter bar at the bottom of the story: which chapter you are in, how
// far through it, and links to jump. The timeline registers chapters.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { $ } = M;

  function createPager(motion) {
    const pager = $('.pager');
    const chapters = [];
    let activeChapter = -1;
    const renderPager = (time) => {
      if (!motion || !chapters.length || !chapters[0].bar) return;
      let active = -1;
      chapters.forEach((c, i) => {
        const end = i < chapters.length - 1 ? chapters[i + 1].start : c.end;
        const p = Math.min(1, Math.max(0, (time - c.start) / (end - c.start)));
        if (p !== c.p) { c.p = p; c.bar.style.transform = `scaleX(${p.toFixed(3)})`; }
        if (time >= c.start - 0.01) active = i;
      });
      if (active !== activeChapter) {
        activeChapter = active;
        chapters.forEach((c, i) => {
          c.link.classList.toggle('is-active', i === active);
          if (i === active) c.link.setAttribute('aria-current', 'step');
          else c.link.removeAttribute('aria-current');
        });
      }
    };

    // Once the timeline is built: find each chapter's link and where it ends.
    const bind = (storyEnd) => {
      chapters.forEach((c) => {
        c.link = $(`a[href="#${c.id}"]`, pager);
        c.bar = $('.pager__bar i', c.link);
        c.end = storyEnd;
      });
    };
    return { pager, chapters, render: renderPager, bind };
  }

  Object.assign(M, { createPager });
})();
