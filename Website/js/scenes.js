// The details each scene draws around the line: people on the road and a gate, labeled
// cars, the car's wheels and box label, the scenario picker, the lock, the privacy dots,
// the quality chart, the seal and the route's nodes. Colors come from classes
// in css/lineart.css: accent = protection, hot = people and risk, cool = data.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { f1, rad, rand, scatter, SHAPES, NODES } = M;

  function buildScenes(stage) {
    const ringD = (cx, cy, r) => `M${cx - r} ${cy}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
    const rectD = (x0, y0, x1, y1) => `M${x0} ${y0}H${x1}V${y1}H${x0}Z`;
    const HIDDEN_DASH = 'stroke-dasharray:1 1.02;stroke-dashoffset:1.01';
    const drawable = (d, parent, cls = '') =>
      make('path', { d, pathLength: 1, class: cls, style: HIDDEN_DASH }, parent);
    const group = (name) => make('g', { class: `orn orn--${name}`, opacity: 0 }, stage);
    const dot = (cx, cy, r, parent, cls = 'fill') =>
      make('circle', { cx: f1(cx), cy: f1(cy), r, class: cls }, parent);

    // Problem: horizon, lane marks, people, one of them circled.
    const gProblem = group('problem');
    const horizon = [drawable('M40 452H476', gProblem, 'thin'), drawable('M524 452H960', gProblem, 'thin')];
    const lanes = [[952, 896], [858, 818], [784, 754], [724, 700], [676, 658], [638, 624], [608, 598], [584, 576]]
      .map(([a, b]) => make('path', { d: `M500 ${a}V${b}`, class: 'thin cool', opacity: 0 }, gProblem));
    const people = [[150, 862, 9], [214, 736, 8], [300, 612, 6.5], [362, 532, 5], [862, 880, 9], [792, 748, 8], [702, 624, 6.5], [640, 538, 5]]
      .map(([x, y, r]) => dot(x, y, r, gProblem, 'hot-fill'));
    // Tracking ripples pulse out from each person (see the life loop).
    const ripples = people.map((p) => make('circle', { cx: p.getAttribute('cx'), cy: p.getAttribute('cy'), r: 8, class: 'thin hot', opacity: 0 }, gProblem));
    const circled = drawable(ringD(792, 748, 34), gProblem, 'hot glow');
    // Privacy law: a striped barrier swings down across the road (see the timeline).
    const gate = make('g', { opacity: 0 }, gProblem);
    make('path', { d: 'M250 690V790M232 790H268' }, gate);
    const gateArm = make('g', {}, gate);
    make('path', {
      class: 'hot',
      d: 'M250 680H770V700H250Z' + Array.from({ length: 12 }, (_, i) => `M${284 + i * 40} 700L${300 + i * 40} 680`).join(''),
    }, gateArm);
    dot(250, 690, 9, gate, 'knock');
    const gateState = { a: -80 };
    const renderGate = () => gateArm.setAttribute('transform', `rotate(${gateState.a.toFixed(2)} 250 690)`);
    renderGate();

    // Labeled frames: two cars on the road, each with a tight bounding box and class tag.
    const carRear = (x0, y0, x1, y1) => {
      const w = x1 - x0, h = y1 - y0, f = (v) => f1(v);
      return `M${f(x0)} ${f(y0 + 0.82 * h)}V${f(y0 + 0.46 * h)}L${f(x0 + 0.16 * w)} ${f(y0 + 0.4 * h)}L${f(x0 + 0.26 * w)} ${f(y0)}`
        + `H${f(x1 - 0.26 * w)}L${f(x1 - 0.16 * w)} ${f(y0 + 0.4 * h)}L${f(x1)} ${f(y0 + 0.46 * h)}V${f(y0 + 0.82 * h)}Z`
        + `M${f(x0 + 0.1 * w)} ${f(y0 + 0.82 * h)}V${f(y1)}M${f(x1 - 0.1 * w)} ${f(y0 + 0.82 * h)}V${f(y1)}`;
    };
    const gFrames = group('frames');
    const FRAMED = [[452, 502, 484, 528], [550, 618, 636, 674]]; // far car (left lane), near car (right lane)
    const smallCars = FRAMED.map((c) => drawable(carRear(...c), gFrames, 'thin'));
    const boxes = FRAMED.map(([x0, y0, x1, y1]) => drawable(rectD(x0 - 6, y0 - 6, x1 + 6, y1 + 6), gFrames, 'thin cool'));
    const boxTags = FRAMED.map(([x0, y0, x1]) =>
      make('rect', { x: x0 - 6, y: y0 - 18, width: Math.round((x1 - x0 + 12) * 0.6), height: 12, class: 'cool-fill', opacity: 0 }, gFrames));
    // A detection scan sweeps down the road while frames are being labeled.
    // Drawn at the horizon (not at 0) so it doesn't stretch the scene's measured size.
    const roadScan = make('path', { d: 'M150 470H850', class: 'thin cool', opacity: 0 }, gFrames);

    // Car: wheels, windows, a YOLO box with its class tag.
    const gCar = group('car');
    const wheels = [ringD(300, 704, 56), ringD(720, 704, 56)].map((d) => drawable(d, gCar));
    const hubs = [ringD(300, 704, 14), ringD(720, 704, 14)].map((d) => drawable(d, gCar, 'thin'));
    const windows = drawable('M318 584L700 566M512 462V572', gCar, 'thin');
    // Tight around the body (x 140 to 906, y 448 to 762).
    const yoloBox = drawable(rectD(130, 436, 916, 772), gCar, 'thin accent glow');
    const yoloTag = make('g', { opacity: 0 }, gCar);
    make('rect', { x: 130, y: 402, width: 76, height: 34, class: 'accent-fill' }, yoloTag);
    make('text', { x: 146, y: 426, 'font-size': 22, class: 'tag-text' }, yoloTag).textContent = 'car';
    // Spokes that turn and speed lines that stream behind (see the life loop).
    const WHEELS = [[300, 704], [720, 704]];
    const spokes = WHEELS.map(([cx, cy]) => {
      const g = make('g', { opacity: 0 }, gCar);
      make('path', { class: 'thin', d: `M${cx - 44} ${cy}H${cx - 16}M${cx + 16} ${cy}H${cx + 44}M${cx} ${cy - 44}V${cy - 16}M${cx} ${cy + 16}V${cy + 44}` }, g);
      return g;
    });
    const speed = make('path', { class: 'thin cool', d: 'M24 596H116M6 636H112M36 676H118', style: 'stroke-dasharray:22 18' }, gCar);

    // Scenario picker: the line is a road bending into the hills (your roads),
    // and the sky offers day, overcast and night (your weather, your hours).
    // A volt ring keeps moving between them (see the life loop).
    const gScenario = group('scenario');
    const hills = [
      drawable('M70 470C150 452 220 436 300 446S460 470 552 470', gScenario, 'thin'),
      drawable('M568 470C640 470 700 442 790 438S900 456 930 470', gScenario, 'thin'),
    ];
    const roadDashes = make('path', { d: 'M500 960C500 760 660 640 560 470', class: 'thin cool', style: 'stroke-dasharray:18 22', opacity: 0 }, gScenario);
    const SKY_X = [300, 500, 700];
    const SKY_Y = 260; // high enough that the labels clear the hills
    const sunIcon = make('g', {}, gScenario);
    dot(SKY_X[0], SKY_Y, 22, sunIcon, 'knock hot');
    make('path', {
      class: 'thin hot',
      d: Array.from({ length: 8 }, (_, i) => {
        const a = rad(i * 45);
        return `M${f1(SKY_X[0] + 32 * Math.cos(a))} ${f1(SKY_Y + 32 * Math.sin(a))}L${f1(SKY_X[0] + 44 * Math.cos(a))} ${f1(SKY_Y + 44 * Math.sin(a))}`;
      }).join(''),
    }, sunIcon);
    const cloudIcon = make('g', {}, gScenario);
    make('path', { d: `M${SKY_X[1] - 36} ${SKY_Y + 20}a16 16 0 0 1 4-30a28 28 0 0 1 54-4a18 18 0 0 1 14 34Z` }, cloudIcon);
    const moonIcon = make('g', {}, gScenario);
    make('path', { class: 'cool-fill', d: `M${SKY_X[2] + 10} ${SKY_Y - 26}A28 28 0 1 0 ${SKY_X[2] + 10} ${SKY_Y + 26}A22 22 0 0 1 ${SKY_X[2] + 10} ${SKY_Y - 26}Z` }, moonIcon);
    const skyIcons = [sunIcon, cloudIcon, moonIcon];
    const skyLabels = make('g', { opacity: 0 }, gScenario);
    ['Day', 'Overcast', 'Night'].forEach((label, i) => {
      make('text', { x: SKY_X[i], y: SKY_Y + 92, 'font-size': 26, 'text-anchor': 'middle' }, skyLabels).textContent = label;
    });
    const picker = make('circle', { cx: SKY_X[0], cy: SKY_Y, r: 58, class: 'thin accent glow', opacity: 0 }, gScenario);

    // Lock: keyhole, and files that each get a check mark.
    const gLock = group('lock');
    const keyhole = drawable(`${ringD(500, 628, 24)}M500 652V712`, gLock, 'accent');
    const files = [336, 408, 480, 552, 624].map((x) =>
      drawable(`${rectD(x, 850, x + 40, 890)}M${x + 10} 870L${x + 18} 878L${x + 31} 862`, gLock, 'thin'));

    // Privacy: members and non-members of a training set, then one mixed cloud.
    const gPrivacy = group('privacy');
    const COUNT = 18;
    const mergedPts = scatter(500, 520, 255, COUNT * 2, 27);
    const memberPts = scatter(300, 520, 132, COUNT, 26);
    const otherPts = scatter(700, 520, 132, COUNT, 26);
    const members = memberPts.map(([x, y]) => dot(x, y, 9, gPrivacy, 'hot-fill'));
    const others = otherPts.map(([x, y]) => dot(x, y, 9, gPrivacy, 'knock cool'));
    const groupLabels = make('g', { opacity: 0 }, gPrivacy);
    make('text', { x: 296, y: 348, 'font-size': 34, 'text-anchor': 'middle', class: 'hot-text' }, groupLabels).textContent = 'In training data';
    make('text', { x: 704, y: 348, 'font-size': 34, 'text-anchor': 'middle', class: 'cool-text' }, groupLabels).textContent = 'Not in training data';
    // Interleave so filled and hollow dots land mixed together.
    const mixedOrder = mergedPts.map((p, i) => ({ p, k: rand() + (i % 2) * 0.0001 })).sort((a, b) => a.k - b.k).map((o) => o.p);
    const privacyDots = [...members, ...others];

    // Quality: axis ticks, scattered results, bad labels caught by a scan.
    const gQuality = group('quality');
    const ticks = make('path', {
      class: 'thin',
      opacity: 0,
      d: [290, 410, 530, 650, 770].map((x) => `M${x} 820V838`).join('') + [700, 580, 460, 340].map((y) => `M152 ${y}H170`).join(''),
    }, gQuality);
    const goodPts = [];
    for (let i = 0; i < 24; i++) {
      const x = 215 + i * 27 + (rand() - 0.5) * 16;
      goodPts.push([x, 742 - (x - 215) * 0.52 + (rand() - 0.5) * 80]);
    }
    const good = goodPts.map(([x, y]) => dot(x, y, 7, gQuality, 'cool-fill'));
    const BAD = [[300, 392], [452, 752], [560, 302], [690, 760], [806, 600]];
    const bad = BAD.map(([x, y]) => dot(x, y, 11, gQuality, 'knock hot'));
    const SCAN_FROM = 180, SCAN_TO = 870;
    const scan = make('g', { opacity: 0 }, gQuality);
    make('path', { d: `M${SCAN_FROM} 186V820`, class: 'accent' }, scan);
    dot(SCAN_FROM, 176, 7, scan, 'accent-fill');

    // Legal: inner ring, check mark, twelve points in a circle.
    const gLegal = group('legal');
    const sealInner = drawable(ringD(500, 430, 168), gLegal, 'thin');
    const sealCheck = drawable('M428 436L482 490L582 382', gLegal, 'accent glow');
    const stars = make('g', {}, gLegal);
    const sealDots = Array.from({ length: 12 }, (_, i) => {
      const a = rad(i * 30 - 90);
      return dot(500 + 194 * Math.cos(a), 430 + 194 * Math.sin(a), 5, stars, 'accent-fill');
    });

    // How it works: four nodes on the route.
    const gHow = group('how');
    const nodes = NODES.map(([x, y], i) => {
      const c = dot(x, y, 18, gHow, 'knock');
      const below = i % 2 === 0;
      make('text', { x, y: below ? y + 66 : y - 40, 'font-size': 30, 'text-anchor': 'middle' }, gHow).textContent = String(i + 1);
      return c;
    });

    return {
      gProblem, horizon, lanes, people, ripples, circled, gate, gateState, renderGate,
      gFrames, smallCars, boxes, boxTags, roadScan,
      gCar, wheels, hubs, windows, yoloBox, yoloTag, WHEELS, spokes, speed,
      gScenario, hills, roadDashes, skyIcons, skyLabels, picker, SKY_X,
      gLock, keyhole, files,
      gPrivacy, members, others, groupLabels, mixedOrder, privacyDots,
      gQuality, ticks, good, bad, BAD, SCAN_FROM, SCAN_TO, scan,
      gLegal, sealInner, sealCheck, stars, sealDots,
      gHow, nodes,
    };
  }

  // Center every scene's drawing in the 1000 x 1000 space and scale it to a
  // common size, so each one sits in the middle of the stage, close to its
  // words, no matter where or how large it was drawn. Measured before any dot
  // is scaled down, so the boxes are true.
  function centerScenes(stage, art) {
    const { gProblem, gFrames, gCar, gScenario, gLock, gPrivacy, gQuality, gLegal, gHow } = art;
    const prevStyle = stage.getAttribute('style') || '';
    const hidden = getComputedStyle(stage).display === 'none';
    if (hidden) stage.setAttribute('style', `${prevStyle};display:block;position:absolute;visibility:hidden;width:1000px;height:1000px`);
    [
      { shapes: [0], groups: [gProblem, gFrames] },
      { shapes: [1], groups: [gCar] },
      { shapes: [2], groups: [gScenario] },
      { shapes: [3], groups: [gLock] },
      { shapes: [4, 5], groups: [gPrivacy] },
      { shapes: [6], groups: [gQuality] },
      { shapes: [7], groups: [gLegal] },
      { shapes: [8], groups: [gHow] },
    ].forEach(({ shapes, groups }) => {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      shapes.forEach((i) => {
        const a = SHAPES[i];
        for (let k = 0; k < a.length; k += 2) {
          x0 = Math.min(x0, a[k]); x1 = Math.max(x1, a[k]);
          y0 = Math.min(y0, a[k + 1]); y1 = Math.max(y1, a[k + 1]);
        }
      });
      // Text labels are left out: their size depends on whether the font has
      // loaded yet, which would make the centering shift from load to load.
      groups.forEach((g) => {
        g.querySelectorAll('path, circle, rect').forEach((el) => {
          const b = el.getBBox();
          if (!b.width && !b.height) return;
          x0 = Math.min(x0, b.x); x1 = Math.max(x1, b.x + b.width);
          y0 = Math.min(y0, b.y); y1 = Math.max(y1, b.y + b.height);
        });
      });
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const s = Math.min(1.45, Math.max(0.85, Math.min(880 / (x1 - x0), 840 / (y1 - y0))));
      shapes.forEach((i) => {
        const a = SHAPES[i];
        for (let k = 0; k < a.length; k += 2) {
          a[k] = 500 + (a[k] - cx) * s;
          a[k + 1] = 500 + (a[k + 1] - cy) * s;
        }
      });
      groups.forEach((g) => {
        g.setAttribute('transform', `translate(500 500) scale(${s.toFixed(3)}) translate(${f1(-cx)} ${f1(-cy)})`);
        g.style.setProperty('--k', s.toFixed(3));
      });
    });
    if (hidden) stage.setAttribute('style', prevStyle);
  }

  // After centering: the route's pen, and the dots and icons that pop in.
  function finishScenes(stage, art) {
    const gsap = window.gsap;
    const { people, privacyDots, good, bad, sealDots, skyIcons } = art;
    art.pen = make('circle', { r: 9, class: 'accent-fill glow', opacity: 0 }, stage);
    const scaleIn = [...people, ...privacyDots, ...good, ...bad, ...sealDots, ...skyIcons];
    gsap.set(scaleIn, { scale: 0, transformOrigin: '50% 50%' });
  }

  const NS = 'http://www.w3.org/2000/svg';
  function make(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    Object.keys(attrs).forEach((k) => n.setAttribute(k, attrs[k]));
    if (parent) parent.appendChild(n);
    return n;
  }

  Object.assign(M, { buildScenes, centerScenes, finishScenes });
})();
