// 万卡集群模拟器 · 等距机房渲染（canvas 2D）
// 世界坐标：x 沿机柜行，y 跨行，单位是一块 600mm 地板；z 是像素高度。
// 几何都按“设计像素”（一块地板宽 24px）计算，画的时候整体缩放。
// 机房本体只画一次缓存起来，每帧只画节点灯、高亮和人。

export const NODES_PER_RACK = 4;
const HW = 12, HH = 6;                       // 一块地板在屏幕上的半宽、半高
const RH = 24, WH = 44, SLAB = 16, PLINTH = 8; // 机柜高、墙高、架空地板厚、底座厚

export const COLORS = {
  shadow: 'rgba(32,42,54,0.24)',
  plinthTop: '#b9c1c9', plinthL: '#a4adb7', plinthR: '#939da8',
  floor: '#f5f7f9', grid: 'rgba(86,102,118,0.11)',
  slabL: '#d2d8de', slabR: '#c1c9d1',
  wallTop: '#ffffff', wallL: '#eaeef1', wallR: '#dde2e7', wallText: 'rgba(28,34,42,0.42)',
  cold: '#d9e8f4', perf: 'rgba(48,96,146,0.30)', walk: '#f2bd1d', ao: 'rgba(24,32,44,0.20)',
  rackTop: '#3b424c', rackL: '#2a2f37', rackR: '#1f232a', rackEdge: 'rgba(0,0,0,0.34)', rackHi: 'rgba(255,255,255,0.10)',
  strip: '#252a32', mesh: 'rgba(255,255,255,0.055)',
  roof: 'rgba(240,245,250,0.94)', roofRib: 'rgba(196,209,221,0.9)', door: 'rgba(206,224,239,0.80)', doorFrame: 'rgba(255,255,255,0.9)',
  trayTop: '#f4c21b', trayL: '#d8a60b', trayR: '#be9006',
  crahTop: '#fbfcfd', crahL: '#eef1f4', crahR: '#e0e5ea', grille: 'rgba(58,70,84,0.22)',
  led: { ok: '#29e08a', slow: '#ffa11c', fail: '#ff3b3b', ckpt: '#4d8bff', off: '#5b636e' },
};

// 每种规模怎么排：[组数（两排背靠背算一组）, 每排柜数]
const PLANS = { 16: [1, 8], 32: [2, 8], 64: [2, 16], 128: [4, 16], 256: [4, 32], 320: [4, 40], 384: [6, 32], 512: [8, 32] };

export function hallLayout(gpus) {
  const racks = Math.ceil(gpus / 8 / NODES_PER_RACK);
  let [pairs, perRow] = PLANS[racks] || [];
  if (!pairs) {
    pairs = Math.max(1, Math.min(8, Math.round(Math.sqrt(racks / 10))));
    perRow = Math.ceil(racks / (2 * pairs));
  }
  const split = perRow > 20;
  const blockLen = split ? Math.ceil(perRow / 2) : perRow;
  const blocks = split ? [[4, 4 + blockLen], [7 + blockLen, 7 + perRow]] : [[4, 4 + perRow]];
  const hallX = blocks[blocks.length - 1][1] + 3;
  const far = (p) => 3 + 8 * p;
  const hallY = far(pairs - 1) + 8;
  const rows = [];
  for (let p = 0; p < pairs; p++) {
    rows.push({ letter: String.fromCharCode(65 + 2 * p), y0: far(p), near: false, pair: p });
    rows.push({ letter: String.fromCharCode(66 + 2 * p), y0: far(p) + 3, near: true, pair: p });
  }
  const cold = [0];
  for (let p = 0; p < pairs; p++) cold.push(far(p) + 5);
  const rackList = [];
  for (let r = 0; r < rows.length && rackList.length < racks; r++) {
    for (let n = 1; n <= perRow && rackList.length < racks; n++) {
      const b = Math.floor((n - 1) / blockLen);
      rackList.push({ row: r, n, x: blocks[b][0] + (n - 1) % blockLen, y: rows[r].y0, near: rows[r].near });
    }
  }
  return { gpus, racks: rackList.length, nodes: rackList.length * NODES_PER_RACK, pairs, perRow, blocks, hallX, hallY, rows, cold, rackList, far };
}

export function rackName(L, r) {
  const k = L.rackList[r];
  return `${L.rows[k.row].letter}${String(k.n).padStart(2, '0')}`;
}
export const nodeHost = (L, i) => `${rackName(L, Math.floor(i / NODES_PER_RACK))}-${i % NODES_PER_RACK + 1}`;

const P = (x, y, z = 0) => [(x - y) * HW, (x + y) * HH - z];

export class HallView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.L = null;
    this.labels = { title: '', sub: '' };
    this.s = 1; this.dpr = 1; this.offX = 0; this.offY = 0;
    this.cssW = 0; this.cssH = 0;
    this.layer = null;
    this.sprites = {};
  }

  setLayout(L, labels) {
    this.L = L;
    this.labels = labels;
    // 场景包围盒（设计像素，原点是世界原点）
    const pts = [P(-1, -1, WH + 8), P(L.hallX + 1, -1, 0), P(-1, L.hallY + 1, 0), P(L.hallX + 1, L.hallY + 1, -SLAB - PLINTH - 40), P(-1, L.hallY + 1, -SLAB - PLINTH - 30), P(L.hallX + 1, -1, -SLAB - PLINTH - 30)];
    this.bounds = {
      minX: Math.min(...pts.map((p) => p[0])) - 6, maxX: Math.max(...pts.map((p) => p[0])) + 6,
      minY: Math.min(...pts.map((p) => p[1])) - 4, maxY: Math.max(...pts.map((p) => p[1])) + 4,
    };
    this.racks = L.rackList.map((k, r) => {
      const { x, y } = k;
      return {
        top: [P(x, y, RH), P(x + 1, y, RH), P(x + 1, y + 2, RH), P(x, y + 2, RH)],
        hull: [P(x, y, RH), P(x + 1, y, RH), P(x + 1, y, 0), P(x + 1, y + 2, 0), P(x, y + 2, 0), P(x, y + 2, RH)],
        anchor: P(x + 0.5, y + 1, RH),
        ledTop: [0, 1, 2, 3].map((s) => P(x + 0.125 + 0.25 * s, y + 1, RH)),
        ledFront: k.near ? [0, 1, 2, 3].map((s) => [P(x + 0.18, y + 2, RH - 3 - s * 5.4), P(x + 0.4, y + 2, RH - 3 - s * 5.4)]) : null,
        rowFrac: L.rows.length > 1 ? k.row / (L.rows.length - 1) : 0,
        x, y, near: k.near, row: k.row,
      };
    });
    this.phase = new Float32Array(L.nodes).map(() => Math.random() * Math.PI * 2);
    this.layer = null;
  }

  // free: 场景可用的矩形（CSS 像素，相对画布左上角）
  resize(cssW, cssH, dpr, free, maxScale = 1.5) {
    this.cssW = cssW; this.cssH = cssH; this.dpr = dpr;
    this.canvas.width = Math.max(1, Math.round(cssW * dpr));
    this.canvas.height = Math.max(1, Math.round(cssH * dpr));
    const b = this.bounds, bw = b.maxX - b.minX, bh = b.maxY - b.minY;
    const f = free || { x: 0, y: 0, w: cssW, h: cssH };
    this.s = Math.max(0.1, Math.min(f.w / bw, f.h / bh, maxScale));
    this.offX = f.x + (f.w - bw * this.s) / 2 - b.minX * this.s;
    this.offY = f.y + (f.h - bh * this.s) / 2 - b.minY * this.s;
    this.renderLayer();
  }

  renderLayer() {
    const b = this.bounds, k = this.s * this.dpr;
    const layer = document.createElement('canvas');
    layer.width = Math.max(1, Math.ceil((b.maxX - b.minX) * k));
    layer.height = Math.max(1, Math.ceil((b.maxY - b.minY) * k));
    const ctx = layer.getContext('2d');
    ctx.setTransform(k, 0, 0, k, -b.minX * k, -b.minY * k);
    drawStatic(ctx, this.L, this.labels, k);
    this.layer = layer;
    this.sprites = {};
    for (const [name, col] of Object.entries(COLORS.led)) this.sprites[name] = glowSprite(col, k);
  }

  toScreen([x, y]) { return { x: this.offX + x * this.s, y: this.offY + y * this.s }; }
  worldToScreen(x, y, z = 0) { return this.toScreen(P(x, y, z)); }
  anchor(r, lift = 0) { const a = this.racks[r].anchor; return this.toScreen([a[0], a[1] - lift]); }
  floorPoint(x, y) { return this.toScreen(P(x, y, 0)); }

  pick(cx, cy) {
    if (!this.racks) return null;
    const x = (cx - this.offX) / this.s, y = (cy - this.offY) / this.s;
    for (let r = this.racks.length - 1; r >= 0; r--) if (inPoly(x, y, this.racks[r].hull)) return r;
    return null;
  }

  // st: { nodes, phase, progress, paused, selected, hover, techs, now }
  draw(st) {
    const ctx = this.ctx, dpr = this.dpr, s = this.s, k = s * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.layer) return;
    const b = this.bounds;
    ctx.drawImage(this.layer, Math.round((this.offX + b.minX * s) * dpr), Math.round((this.offY + b.minY * s) * dpr));
    ctx.setTransform(k, 0, 0, k, this.offX * dpr, this.offY * dpr);

    const t = st.now / 1000;
    const blink = Math.sin(t * 6) > 0;
    const color = (i, rk) => {
      const v = st.nodes[i];
      if (v === 1) return st.phase === 'run' || blink ? 'fail' : 'off';
      if (v === 3) return 'off';
      if (v === 2) return 'slow';
      if (st.phase === 'ckpt') return rk.rowFrac <= st.progress ? 'ckpt' : 'ok';
      if (st.phase === 'restart') return rk.rowFrac <= st.progress ? 'ok' : 'off';
      if (st.phase === 'repair') return 'off';
      return 'ok';
    };

    // 出问题的机柜：顶面染色 + 光晕
    for (let r = 0; r < this.racks.length; r++) {
      const rk = this.racks[r];
      let bad = null;
      for (let j = 0; j < NODES_PER_RACK; j++) {
        const v = st.nodes[r * NODES_PER_RACK + j];
        if (v === 1) bad = 'fail'; else if (v === 2 && !bad) bad = 'slow';
      }
      if (!bad) continue;
      const col = COLORS.led[bad], [ax, ay] = rk.anchor;
      const g = ctx.createRadialGradient(ax, ay, 0, ax, ay, 46);
      g.addColorStop(0, col + '88'); g.addColorStop(0.45, col + '33'); g.addColorStop(1, col + '00');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(ax, ay, 46, 0, Math.PI * 2); ctx.fill();
      poly(ctx, rk.top, col + 'b0', col, 0.8);
    }

    // 节点灯：按颜色和亮度分桶，一桶一次 fill
    const buckets = {};
    const add = (key, fn) => { (buckets[key] || (buckets[key] = new Path2D()), fn(buckets[key])); };
    const twinkle = !st.paused && st.phase === 'run';
    for (let r = 0; r < this.racks.length; r++) {
      const rk = this.racks[r];
      for (let j = 0; j < NODES_PER_RACK; j++) {
        const i = r * NODES_PER_RACK + j, c = color(i, rk);
        const lvl = twinkle && c === 'ok' ? Math.floor((Math.sin(t * 2.2 + this.phase[i]) + 1) * 1.5) : 2;
        const key = `${c}|${Math.min(2, lvl)}`;
        const [lx, ly] = rk.ledTop[j];
        const big = c === 'fail' || c === 'slow';
        add(key, (p) => ledShape(p, lx, ly, big ? 0.09 : 0.06));
        if (rk.ledFront) add(key + '|f', (p) => { const [a, b2] = rk.ledFront[j]; p.moveTo(a[0], a[1]); p.lineTo(b2[0], b2[1]); });
        if (big && this.sprites[c]) {
          const sp = this.sprites[c];
          ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.drawImage(sp, (this.offX + lx * s) * dpr - sp.width / 2, (this.offY + ly * s) * dpr - sp.height / 2);
          ctx.restore();
        }
      }
    }
    const dimA = st.paused ? 0.5 : 1;
    for (const [key, path] of Object.entries(buckets)) {
      const [c, lvl, f] = key.split('|');
      const alpha = (c === 'ok' ? [0.62, 0.8, 1][+lvl] : 1) * (c === 'off' ? 0.9 : dimA);
      ctx.globalAlpha = alpha;
      if (f) { ctx.strokeStyle = COLORS.led[c]; ctx.lineWidth = 1.3; ctx.stroke(path); }
      else {
        ctx.fillStyle = c === 'fail' || c === 'slow' ? '#ffffff' : COLORS.led[c];
        ctx.fill(path);
        if (c !== 'off' && c !== 'fail' && c !== 'slow') { ctx.globalAlpha = alpha * 0.28; ctx.fillStyle = COLORS.led[c]; ctx.lineWidth = 2.2; ctx.strokeStyle = COLORS.led[c]; ctx.stroke(path); }
      }
    }
    ctx.globalAlpha = 1;

    // 悬停、选中
    const outline = (r, w, col) => { if (r == null || !this.racks[r]) return; poly(ctx, this.racks[r].hull, null, col, w); };
    if (st.hover != null && st.hover !== st.selected) outline(st.hover, 1.6, 'rgba(255,255,255,0.95)');
    if (st.selected != null) { outline(st.selected, 3.2, 'rgba(20,24,29,0.85)'); outline(st.selected, 1.6, '#ffffff'); }

    // 现场工程师（只画在看得见的冷通道里）
    for (const tch of st.techs || []) {
      const rk = this.racks[tch.rack];
      if (rk && rk.near) drawPerson(ctx, rk.x + 0.55, rk.y + 2.55);
    }
  }
}

function inPoly(x, y, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function ledShape(p, cx, cy, hx) {
  // 顶面上一小块等距矩形：沿 x 方向 2·hx，沿 y 方向 0.16
  const ex = [hx * HW, hx * HH], ey = [-0.08 * HW, 0.08 * HH];
  p.moveTo(cx - ex[0] - ey[0], cy - ex[1] - ey[1]);
  p.lineTo(cx + ex[0] - ey[0], cy + ex[1] - ey[1]);
  p.lineTo(cx + ex[0] + ey[0], cy + ex[1] + ey[1]);
  p.lineTo(cx - ex[0] + ey[0], cy - ex[1] + ey[1]);
  p.closePath();
}

function glowSprite(color, k) {
  const r = Math.ceil(9 * k), c = document.createElement('canvas');
  c.width = c.height = r * 2;
  const g = c.getContext('2d'), grad = g.createRadialGradient(r, r, 0, r, r, r);
  grad.addColorStop(0, color + 'ff'); grad.addColorStop(0.35, color + '99'); grad.addColorStop(1, color + '00');
  g.fillStyle = grad; g.fillRect(0, 0, r * 2, r * 2);
  return c;
}

function poly(ctx, pts, fill, stroke, lw = 1) {
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
}

function drawPerson(ctx, x, y) {
  const [sx, sy] = P(x, y, 0);
  ctx.fillStyle = 'rgba(20,28,38,0.22)'; ctx.beginPath(); ctx.ellipse(sx, sy, 3.6, 1.8, 0, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#2b3240'; ctx.fillRect(sx - 1.7, sy - 6, 1.4, 6); ctx.fillRect(sx + 0.3, sy - 6, 1.4, 6);
  ctx.fillStyle = '#ff7417'; roundRect(ctx, sx - 2.4, sy - 12, 4.8, 6.8, 1.6); ctx.fill();
  ctx.fillStyle = '#e9c09a'; ctx.beginPath(); ctx.arc(sx, sy - 14.2, 2.1, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(sx, sy - 14.6, 2.3, Math.PI, 0); ctx.fill();
  box(ctx, x + 0.45, y - 0.3, 0, 0.8, 0.6, 7, { top: '#cfd6dd', left: '#b8c0c8', right: '#a5aeb8' });
  box(ctx, x + 0.5, y - 0.25, 7, 0.7, 0.5, 2.4, { top: '#3b424c', left: '#2a2f37', right: '#1f232a' });
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

const rectXY = (x, y, dx, dy, z = 0) => [P(x, y, z), P(x + dx, y, z), P(x + dx, y + dy, z), P(x, y + dy, z)];

function box(ctx, x, y, z, dx, dy, dz, col) {
  if (col.left) poly(ctx, [P(x, y + dy, z + dz), P(x + dx, y + dy, z + dz), P(x + dx, y + dy, z), P(x, y + dy, z)], col.left);
  if (col.right) poly(ctx, [P(x + dx, y, z + dz), P(x + dx, y + dy, z + dz), P(x + dx, y + dy, z), P(x + dx, y, z)], col.right);
  if (col.top) poly(ctx, rectXY(x, y, dx, dy, z + dz), col.top, col.edge, col.edgeW || 0.6);
}

function line(ctx, a, b, color, lw = 1, dash = null) {
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]);
}

// 模糊的多边形：画到很远的地方，靠阴影偏移把模糊的影子挪回原处（所有浏览器都支持）
function blurredPoly(ctx, pts, color, blur, k) {
  ctx.save();
  ctx.shadowColor = color; ctx.shadowBlur = blur * k; ctx.shadowOffsetX = 10000 * k; ctx.shadowOffsetY = 0;
  ctx.fillStyle = '#000';
  ctx.beginPath(); ctx.moveTo(pts[0][0] - 10000, pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] - 10000, pts[i][1]);
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawStatic(ctx, L, labels, k) {
  const C = COLORS, HX = L.hallX, HY = L.hallY;

  // 底座、投影、架空地板
  blurredPoly(ctx, rectXY(-1, -1, HX + 2, HY + 2, -SLAB - PLINTH - 22), C.shadow, 16, k);
  box(ctx, -1, -1, -SLAB - PLINTH, HX + 2, HY + 2, PLINTH, { top: C.plinthTop, left: C.plinthL, right: C.plinthR });
  box(ctx, -0.5, -0.5, -SLAB, HX + 0.5, HY + 0.5, SLAB, { top: C.floor, left: C.slabL, right: C.slabR });
  line(ctx, P(-0.5, HY, 0), P(HX, HY, 0), 'rgba(255,255,255,0.9)', 1);
  line(ctx, P(HX, -0.5, 0), P(HX, HY, 0), 'rgba(255,255,255,0.7)', 1);

  // 冷通道的通风地板
  for (const cy of L.cold) for (const [bx0, bx1] of L.blocks) {
    poly(ctx, rectXY(bx0, cy, bx1 - bx0, 3), C.cold);
    ctx.fillStyle = C.perf;
    for (let ix = bx0; ix < bx1; ix++) for (let iy = cy; iy < cy + 3; iy++) {
      ctx.beginPath();
      for (let u = 0; u < 4; u++) for (let v = 0; v < 4; v++) {
        const [sx, sy] = P(ix + 0.2 + 0.2 * u, iy + 0.2 + 0.2 * v, 0);
        ctx.moveTo(sx + 0.95, sy); ctx.ellipse(sx, sy, 0.95, 0.55, 0, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }
  for (let i = 0; i <= HX; i++) line(ctx, P(i, 0, 0), P(i, HY, 0), C.grid, 0.7);
  for (let j = 0; j <= HY; j++) line(ctx, P(0, j, 0), P(HX, j, 0), C.grid, 0.7);

  // 主通道和前沿的黄线
  if (L.blocks.length > 1) {
    const [a0, a1] = [L.blocks[0][1], L.blocks[1][0]];
    for (const x of [a0 + 0.3, a1 - 0.3]) line(ctx, P(x, 0.4, 0), P(x, HY - 0.4, 0), C.walk, 1.6, [7, 5]);
  }
  line(ctx, P(0.4, HY - 0.45, 0), P(HX - 0.4, HY - 0.45, 0), C.walk, 1.6, [7, 5]);

  // 每排右端地上的行号
  ctx.fillStyle = 'rgba(28,34,42,0.30)';
  const lastX = L.blocks[L.blocks.length - 1][1];
  for (const r of L.rows) {
    const [sx, sy] = P(lastX + 1.55, r.y0 + 1, 0);
    ctx.save(); ctx.translate(sx, sy); ctx.transform(1, 0, 0, 0.5, 0, 0);
    ctx.font = '700 30px "Barlow Condensed", "Arial Narrow", sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(r.letter, 0, 2); ctx.restore();
  }

  // 后墙、门、安全出口、字
  poly(ctx, [P(0, 0, 0), P(HX, 0, 0), P(HX, 0.7, 0), P(0.7, 0.7, 0), P(0.7, HY, 0), P(0, HY, 0)], 'rgba(30,40,52,0.06)');
  box(ctx, -0.5, -0.5, 0, HX + 0.5, 0.5, WH, { top: C.wallTop, left: C.wallL, right: C.wallR });
  box(ctx, -0.5, 0, 0, 0.5, HY, WH, { top: C.wallTop, left: C.wallL, right: C.wallR });
  const doorC = L.blocks.length > 1 ? (L.blocks[0][1] + L.blocks[1][0]) / 2 : HX - 1.6;
  const dx0 = doorC - 0.8, dx1 = doorC + 0.8;
  poly(ctx, [P(dx0, 0, 31), P(dx1, 0, 31), P(dx1, 0, 0), P(dx0, 0, 0)], '#d3dae1');
  line(ctx, P(doorC, 0, 31), P(doorC, 0, 0), 'rgba(80,92,106,0.35)', 1);
  poly(ctx, [P(dx0 + 0.35, 0, 38), P(dx1 - 0.35, 0, 38), P(dx1 - 0.35, 0, 34), P(dx0 + 0.35, 0, 34)], '#16a05c');
  const wallText = (txt, x, z, size, weight) => {
    const [sx, sy] = P(x, 0, z);
    ctx.save(); ctx.translate(sx, sy); ctx.transform(1, 0.5, 0, 1, 0, 0);
    ctx.font = `${weight} ${size}px "Noto Sans SC", "PingFang SC", sans-serif`; ctx.fillStyle = C.wallText;
    ctx.fillText(txt, 0, 0); ctx.restore();
  };
  wallText(labels.title || '', 2.6, 22, 15, 900);
  wallText(labels.sub || '', 2.6, 9, 9, 500);

  for (let p = 0; p < L.pairs; p++) {
    const far = L.far(p), near = far + 3;
    // 精密空调
    const cy = far + 1.3;
    box(ctx, 0.25, cy, 0, 1.3, 2.4, 32, { top: C.crahTop, left: C.crahL, right: C.crahR, edge: 'rgba(0,0,0,0.06)' });
    for (let z = 4; z <= 21; z += 2.4) line(ctx, P(1.55, cy + 0.25, z), P(1.55, cy + 2.15, z), C.grille, 0.8);
    poly(ctx, [P(1.55, cy + 0.3, 29), P(1.55, cy + 0.95, 29), P(1.55, cy + 0.95, 25.5), P(1.55, cy + 0.3, 25.5)], '#2a3038');
    { const [sx, sy] = P(1.55, cy + 0.5, 27.2); ctx.fillStyle = '#5aa9ff'; ctx.beginPath(); ctx.arc(sx, sy, 0.9, 0, Math.PI * 2); ctx.fill(); }

    const inPair = L.rackList.filter((rk) => L.rows[rk.row].pair === p);
    if (!inPair.length) continue;
    for (const [bx0, bx1] of L.blocks) blurredPoly(ctx, rectXY(bx0 - 0.15, far - 0.15, bx1 - bx0 + 0.45, 5.45), C.ao, 4, k);
    for (const rk of inPair) if (!rk.near) staticRack(ctx, rk);
    // 热通道封闭：顶板、端门、走线槽
    for (const [bx0, bx1] of L.blocks) {
      poly(ctx, rectXY(bx0, far + 2, bx1 - bx0, 1, RH + 0.5), C.roof, 'rgba(255,255,255,0.95)', 0.8);
      for (let x = bx0 + 1; x < bx1; x += 2) line(ctx, P(x, far + 2, RH + 0.5), P(x, far + 3, RH + 0.5), C.roofRib, 0.7);
      poly(ctx, [P(bx1, far + 2, RH), P(bx1, far + 3, RH), P(bx1, far + 3, 0), P(bx1, far + 2, 0)], C.door, C.doorFrame, 1);
      line(ctx, P(bx1, far + 2.5, RH - 1), P(bx1, far + 2.5, 1), 'rgba(255,255,255,0.8)', 0.8);
      box(ctx, bx0 + 0.1, far + 2.28, RH + 0.5, bx1 - bx0 - 0.2, 0.44, 3, { top: C.trayTop, left: C.trayL, right: C.trayR });
    }
    for (const rk of inPair) if (rk.near) staticRack(ctx, rk);
  }

  function staticRack(ctx2, rk) {
    const { x: x0, y: y0 } = rk;
    box(ctx2, x0, y0, 0, 1, 2, RH, { left: C.rackL, right: C.rackR });
    poly(ctx2, rectXY(x0, y0, 1, 2, RH), C.rackTop, C.rackEdge, 0.6);
    line(ctx2, P(x0, y0, RH), P(x0 + 1, y0, RH), C.rackHi, 0.6);
    if (rk.near) for (let u = 0.14; u < 0.9; u += 0.09) line(ctx2, P(x0 + u, y0 + 2, RH - 2), P(x0 + u, y0 + 2, 2), C.mesh, 0.6);
    poly(ctx2, rectXY(x0 + 0.04, y0 + 0.86, 0.92, 0.28, RH), C.strip);
  }
}
