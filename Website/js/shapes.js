// The shapes the one line takes, in a 1000 x 1000 drawing space:
// 0 road, 1 car, 2 curving road, 3 padlock, 4 boundary, 5 circle, 6 chart axis,
// 7 seal, 8 route. Each is resampled to N points so any two can blend.
(() => {
  'use strict';
  const M = (window.MURAGALA = window.MURAGALA || {});
  const { N, lerp, rad, seg, cubic, arc, join, through, resample } = M;

  const sealPt = (d) => {
    const r = 220 + 8 * Math.cos(rad(28 * d));
    return [500 + r * Math.cos(rad(d)), 430 + r * Math.sin(rad(d))];
  };
  const sealRing = [];
  for (let d = 112; d <= 428; d += 1) sealRing.push(sealPt(d));

  // Both edges of one lane that curves into the distance, as a single line:
  // up the left edge, across the far end, down the right edge.
  const ROAD_CURVE = [[500, 960], [500, 760], [660, 640], [560, 470]]; // centerline, near to far
  const curvedRoad = () => {
    const [p0, p1, p2, p3] = ROAD_CURVE;
    const left = [], right = [];
    for (let i = 0; i <= 60; i++) {
      const t = i / 60, u = 1 - t;
      const x = u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0];
      const y = u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1];
      const dx = 3 * u * u * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0]);
      const dy = 3 * u * u * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1]);
      const len = Math.hypot(dx, dy) || 1;
      const hw = lerp(210, 6, Math.pow(t, 0.75)); // half the road's width, narrowing with distance
      left.push([x + (dy / len) * hw, y - (dx / len) * hw]);
      right.push([x - (dy / len) * hw, y + (dx / len) * hw]);
    }
    return join(left, right.reverse());
  };

  const NODES = [[170, 700], [390, 410], [610, 650], [830, 360]];

  const SHAPES = [
    // 0 road in perspective
    join(
      seg([110, 960], [484, 452], 48),
      seg([484, 452], [516, 452], 4),
      seg([516, 452], [890, 960], 48)
    ),
    // 1 car in profile
    join(
      seg([40, 762], [170, 762]),
      cubic([170, 762], [150, 730], [140, 670], [150, 640]),
      cubic([150, 640], [160, 605], [200, 592], [300, 586]),
      cubic([300, 586], [340, 540], [370, 490], [420, 468]),
      cubic([420, 468], [480, 450], [560, 448], [610, 462]),
      cubic([610, 462], [650, 480], [690, 540], [720, 566]),
      cubic([720, 566], [790, 574], [860, 584], [888, 612]),
      cubic([888, 612], [906, 640], [906, 700], [880, 762]),
      seg([880, 762], [960, 762])
    ),
    // 2 a road that bends away to the horizon (your roads)
    curvedRoad(),
    // 3 padlock
    join(
      seg([500, 800], [332, 800]), arc(332, 768, 32, 90, 180),
      seg([300, 768], [300, 532]), arc(332, 532, 32, 180, 270),
      seg([332, 500], [372, 500]), seg([372, 500], [372, 380]),
      arc(500, 380, 128, 180, 360),
      seg([628, 380], [628, 500]), seg([628, 500], [668, 500]), arc(668, 532, 32, 270, 360),
      seg([700, 532], [700, 768]), arc(668, 768, 32, 0, 90),
      seg([668, 800], [500, 800])
    ),
    // 4 the attacker's boundary between two groups
    Array.from({ length: 121 }, (_, i) => [500 + 22 * Math.sin((i / 120) * Math.PI * 4), lerp(150, 890, i / 120)]),
    // 5 a protective circle around one mixed group
    arc(500, 520, 330, -90, 270),
    // 6 chart axis
    join(seg([170, 170], [170, 820]), seg([170, 820], [880, 820])),
    // 7 seal with ribbon tails
    join(
      seg([468, 664], [448, 905]), seg([448, 905], [418, 876]), seg([418, 876], [382, 902]),
      seg([382, 902], sealRing[0]),
      sealRing,
      seg(sealRing[sealRing.length - 1], [618, 902]),
      seg([618, 902], [582, 876]), seg([582, 876], [552, 905]), seg([552, 905], [532, 664])
    ),
    // 8 route through four steps
    through([[30, 560], ...NODES, [970, 470]]),
  ].map(resample);

  const lastShape = SHAPES.length - 1;
  const wave = SHAPES[8];
  const nodeFrac = NODES.map(([x, y]) => {
    let best = 0, bestD = Infinity;
    for (let k = 0; k < N; k++) {
      const d = Math.hypot(wave[2 * k] - x, wave[2 * k + 1] - y);
      if (d < bestD) { bestD = d; best = k; }
    }
    return best / (N - 1);
  });

  Object.assign(M, { NODES, SHAPES, lastShape, nodeFrac });
})();
