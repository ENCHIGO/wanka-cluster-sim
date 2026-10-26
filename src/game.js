// 万卡集群模拟器 · 游戏循环与界面
import {
  GPUS, NETWORKS, STORAGE, MODELS, SCENARIOS, GPU_COUNTS, TP_OPTIONS, PP_OPTIONS, EP_OPTIONS,
  GBS_OPTIONS, CKPT_OPTIONS, NODE_GPUS, REPAIR_WAIT_SEC, SEQ_LEN, evaluate, lossAt,
  REPAIR_HOURS, SWAP_REPAIR_HOURS, STRAGGLER_SLOWDOWN, STRAGGLER_RECOVER,
} from './sim.js';
import { HallView, hallLayout, nodeHost, rackName, NODES_PER_RACK } from './scene.js';
import { $, STROKE, icon, fillIcons, fmtInt, fmtMoney, pct, pad2, segment, stepper, showOnly, registerMode } from './ui.js';
import { serveSnapshot, serveBoot } from './serve-ui.js';

const SPEEDS = [2, 8, 24];               // 每真实秒推进的模拟小时
const SPIKE_PER_HOUR = 1 / 70;
const NODE_OK = 0, NODE_FAIL = 1, NODE_SLOW = 2, NODE_SWAP = 3;
const HALL = '训练大厅 A';
const COST_S = 0.9, GOODPUT_S = 0.85;   // 评级：按期交付 + 花费不超过预算 90% + Goodput 不低于 85%
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- 格式化 ----------
const fmtTok = (t) => t >= 1e12 ? (t / 1e12).toFixed(2) + 'T' : t >= 1e9 ? (t / 1e9).toFixed(1) + 'B' : (t / 1e6).toFixed(0) + 'M';
const fmtTokShort = (t) => t % 1e12 === 0 ? t / 1e12 + 'T' : fmtTok(t);
const fmtTokAxis = (t) => t >= 1e11 ? +(t / 1e12).toFixed(2) + 'T' : fmtTok(t);
const fmtDur = (sec) => sec < 90 ? Math.max(1, Math.round(sec)) + ' 秒' : sec < 5400 ? Math.round(sec / 60) + ' 分钟' : (sec / 3600).toFixed(1) + ' 小时';
const clock = (sec) => `D${Math.floor(sec / 86400) + 1} ${pad2(Math.floor(sec % 86400 / 3600))}:${pad2(Math.floor(sec % 3600 / 60))}`;
const gpuLabel = (n) => n < 1000 ? String(n) : n / 1024 + 'K';

// ---------- 状态 ----------
let scIdx = 0;
let cfg = { ...SCENARIOS[0].defaults };
let ev = null;            // 当前配置的核算
let g = null;             // 进行中的一局
let L = null;             // 机房排布
let speedIdx = 1;
let reconfig = false;
let selected = null, hover = null;
const view = new HallView($('scene'));
const preview = new HallView($('previewCanvas'));

// ---------- 开局 ----------
function buildSetup() {
  $('missions').onclick = (e) => {
    const b = e.target.closest('.mission');
    if (!b || b.disabled || reconfig) return;
    scIdx = +b.dataset.i;
    cfg = { ...SCENARIOS[scIdx].defaults };
    syncSetup(true);
  };
  $('gpuTiles').onclick = (e) => { const b = e.target.closest('.tile'); if (b) { cfg.gpu = b.dataset.v; syncSetup(); } };
  const set = (k) => (v) => { cfg[k] = v; syncSetup(); };
  stepper($('stTp'), 'TP ', TP_OPTIONS, () => cfg.tp, set('tp'));
  stepper($('stPp'), 'PP ', PP_OPTIONS, () => cfg.pp, set('pp'));
  stepper($('stEp'), 'EP ', EP_OPTIONS, () => cfg.ep, set('ep'));
  stepper($('stSpares'), '备机', Array.from({ length: 65 }, (_, i) => i), () => cfg.spares, set('spares'));
  $('swRecompute').onclick = () => { cfg.recompute = !cfg.recompute; syncSetup(); };
  $('startBtn').onclick = () => (reconfig ? applyReconfig() : startGame());
  $('cancelBtn').onclick = () => { reconfig = false; g.paused = false; showOps(); };
}

function syncSetup(missionChanged = false) {
  const sc = SCENARIOS[scIdx], model = MODELS[sc.model];
  ev = evaluate(cfg, sc);
  const gpu = GPUS[cfg.gpu], net = NETWORKS[cfg.net], sto = STORAGE[cfg.storage];

  // 任务卡
  $('missions').innerHTML = SCENARIOS.map((s, i) => {
    const sel = i === scIdx, lvl = s.level === '入门' ? 'ok' : s.level === '困难' ? 'hard' : 'bad';
    return `<button type="button" class="mission${sel ? ' sel' : ''}" data-i="${i}" aria-pressed="${sel}"${reconfig && !sel ? ' disabled' : ''}>
      <span class="row1"><span class="mono small muted">${pad2(i + 1)}</span><span style="display: flex; align-items: center; gap: 8px"><span class="chip ${lvl}">${s.level}</span><span class="check">${icon('check', 13, 3)}</span></span></span>
      <span class="big"><span class="size">${s.size}</span><span class="muted" style="font-size: 13px">${s.kind} · 训 ${fmtTokShort(s.tokens)} token</span></span>
      <span class="brief">${s.brief}</span>
      <span class="facts"><span>${s.deadlineDays} 天</span><span>${fmtMoney(s.budget)}</span><span>最多 ${fmtInt(s.maxGpus)} 卡</span></span>
    </button>`;
  }).join('');

  // 硬件
  $('gpuTiles').innerHTML = Object.values(GPUS).map((x) => `<button type="button" class="tile" data-v="${x.id}" aria-pressed="${cfg.gpu === x.id}">
      <span style="display: flex; align-items: baseline; gap: 8px"><span style="font-size: 17px; font-weight: 700">${x.id}</span><span class="small muted">${x.hbm} GB</span></span>
      <span class="small muted">${fmtInt(x.tflops)} TFLOPS · $${x.price}/卡时</span>
      <span class="tick">${icon('check', 12, 3)}</span></button>`).join('');
  segment($('segGpus'), GPU_COUNTS.map((n) => ({ v: n, label: gpuLabel(n), disabled: n > sc.maxGpus, title: n > sc.maxGpus ? `这单最多批 ${fmtInt(sc.maxGpus)} 卡` : `${fmtInt(n)} 卡` })), cfg.gpus, (v) => { cfg.gpus = +v; syncSetup(); }, true);
  $('gpusInfo').innerHTML = `<b class="num" style="font-size: 20px; color: var(--ink)">${fmtInt(cfg.gpus)}</b> 卡 · ${fmtInt(cfg.gpus / NODE_GPUS)} 台 · ${fmtInt(cfg.gpus / 32)} 柜`;
  segment($('segNet'), [['roce200', 'RoCE 200G'], ['ib400', 'IB 400G'], ['ib800', 'IB 800G']].map(([v, label]) => ({ v, label })), cfg.net, (v) => { cfg.net = v; syncSetup(); });
  segment($('segSto'), [['std', '标准 100 GB/s'], ['fast', '全闪 600 GB/s']].map(([v, label]) => ({ v, label })), cfg.storage, (v) => { cfg.storage = v; syncSetup(); });
  $('netInfo').textContent = `每卡 ${net.bw.toFixed(0)} GB/s · $${net.price}/卡时`;
  $('stoInfo').textContent = ev.ckptSec ? `存一次 ${fmtDur(ev.ckptSec)} · $${sto.price}/小时` : `$${sto.price}/小时`;

  // 并行
  ['stTp', 'stPp', 'stEp', 'stSpares'].forEach((id) => $(id).sync());
  $('epRow').hidden = !model.moe;
  const mp = cfg.tp * cfg.pp, dp = cfg.gpus / mp;
  $('ppInfo').textContent = cfg.pp === 1 ? `${model.layers} 层不切` : `${model.layers} 层切成 ${cfg.pp} 段`;
  $('epInfo').textContent = `${model.experts || 0} 个专家分成 ${cfg.ep} 份`;
  $('dpInfo').textContent = `${fmtInt(cfg.gpus)} ÷ ${mp}，自动算出`;
  $('dpOut').textContent = Number.isInteger(dp) ? fmtInt(dp) : '除不尽';
  renderDiagram(dp, model);

  // 训练与容错
  segment($('segGbs'), GBS_OPTIONS.map((v) => ({ v, label: fmtInt(v) })), cfg.gbs, (v) => { cfg.gbs = +v; syncSetup(); }, true);
  const tokM = cfg.gbs * SEQ_LEN / 2 ** 20, critM = model.critBatch / 2 ** 20;
  $('gbsInfo').textContent = `每步 ${tokM}M token，` + (tokM < critM ? `临界批次是 ${critM}M` : tokM === critM ? '正好卡在临界批次上' : `超过了 ${critM}M 的临界批次`);
  $('swRecompute').setAttribute('aria-checked', String(cfg.recompute));
  $('recomputeTxt').textContent = cfg.recompute ? '开' : '关';
  segment($('segCkpt'), CKPT_OPTIONS.map((v) => ({ v, label: v })), cfg.ckptMin, (v) => { cfg.ckptMin = +v; syncSetup(); }, true);
  $('ckptInfo').textContent = ev.ok ? `最优约 ${Math.round(ev.youngMin)} 分钟：全集群 ${ev.mtbfHours.toFixed(1)} 小时坏一次` : '';
  const perSpare = `每台 $${Math.round(NODE_GPUS * (gpu.price + net.price))}/小时`;
  $('sparesInfo').textContent = ev.ok ? `平均 ${ev.inRepair.toFixed(1)} 台在送修，至少留 ${ev.sparesNeeded} 台 · ${perSpare}` : `闲着也计费，${perSpare}`;
  $('cfgSummary').textContent = `${cfg.gpu} × ${fmtInt(cfg.gpus)} · TP${cfg.tp} PP${cfg.pp} DP${Number.isInteger(dp) ? dp : '?'}${model.moe ? ' EP' + cfg.ep : ''}`;

  renderForecast(sc);
  if (missionChanged || preview.L?.gpus !== cfg.gpus) layoutPreview();
}

function renderDiagram(dp, model) {
  const { tp, pp } = cfg;
  const ch = Math.max(2, Math.min(9, Math.floor(120 / pp) - 2)), cw = Math.max(2, Math.min(9, Math.floor(150 / tp) - 2));
  const labels = ch >= 7;
  const firstLabel = tp === NODE_GPUS ? '第 1 段 · 一台机器' : tp > NODE_GPUS ? `第 1 段 · 跨 ${tp / NODE_GPUS} 台` : '第 1 段';
  let html = '';
  for (let r = 0; r < pp; r++) {
    html += '<div>' + `<i class="${r === 0 ? 'first' : ''}" style="width: ${cw}px; height: ${ch}px"></i>`.repeat(tp);
    if (labels && r === 0) html += `<span>${firstLabel}</span>`;
    if (labels && r === pp - 1 && pp > 1) html += `<span>第 ${pp} 段</span>`;
    html += '</div>';
  }
  $('cells').innerHTML = html;
  $('cells').setAttribute('aria-label', `一份模型：${pp} 段流水线，每段 ${tp} 张卡做张量并行`);
  $('diagTitle').textContent = `一份模型 = ${tp * pp} 卡`;
  $('diagReps').textContent = Number.isInteger(dp) ? `× ${fmtInt(dp)} 份` : '';
  const machines = tp * pp / NODE_GPUS;
  let note = tp > NODE_GPUS ? `TP${tp} 一行要跨 ${tp / NODE_GPUS} 台机器，张量并行走 IB。`
    : tp === NODE_GPUS ? `TP8 在一台机器里，PP${pp} 把 ${pp} 台串起来${pp % 4 === 0 ? `，正好 ${pp / 4} 个机柜` : ''}。`
      : machines < 1 ? `一台机器能放 ${1 / machines} 份模型。` : `每台机器放 ${NODE_GPUS / tp} 段，一份模型占 ${machines} 台。`;
  if (model.moe) note += ` EP${cfg.ep}：每张卡放 ${model.experts / cfg.ep} 个专家。`;
  $('diagNote').textContent = note;
}

function renderForecast(sc) {
  const e = ev;
  let chip;
  if (!e.ok) chip = ['bad', '配置不成立'];
  else {
    const d = e.etaDays / sc.deadlineDays, c = e.estCost / sc.budget;
    chip = d > 1 && c > 1 ? ['bad', '会超期也会超支'] : d > 1 ? ['bad', '会超期'] : c > 1 ? ['bad', '会超支']
      : c > 0.95 ? ['warn', '能交，预算吃紧'] : d > 0.92 ? ['warn', '能交，时间吃紧'] : ['ok', '能按期交付'];
  }
  const metric = (label, big, unit, limit, frac, notes) => {
    const col = frac > 1 ? 'var(--bad)' : frac > notes.warnAt ? 'var(--warn)' : 'var(--ink)';
    return `<div class="metric"><div class="small muted">${label}</div>
      <div class="line"><span class="big">${big}${unit ? `<span class="unit" style="font-size: 15px; font-weight: 500; color: var(--ink)">${unit}</span>` : ''}</span><span class="muted" style="font-size: 13px">${limit}</span></div>
      <div class="bar"><i style="width: ${Math.min(100, frac * 100).toFixed(1)}%; background: ${col}"></i></div>
      <div class="small ${notes.cls}" style="font-weight: 500">${notes.text}</div></div>`;
  };
  let body;
  if (e.ok) {
    const d = e.etaDays / sc.deadlineDays, c = e.estCost / sc.budget;
    const slack = sc.deadlineDays - e.etaDays, left = sc.budget - e.estCost;
    body = metric('工期', e.etaDays.toFixed(1), '天', `期限 ${sc.deadlineDays} 天`, d,
      { warnAt: 0.92, cls: d > 1 ? 't-bad' : d > 0.92 ? 't-warn' : 't-ok', text: d > 1 ? `超期 ${(-slack).toFixed(1)} 天` : d > 0.92 ? `只富余 ${slack.toFixed(1)} 天` : `富余 ${slack.toFixed(1)} 天` })
      + metric('花费', fmtMoney(e.estCost), '', `预算 ${fmtMoney(sc.budget)}`, c,
        { warnAt: 0.95, cls: c > 1 ? 't-bad' : c > 0.95 ? 't-warn' : 't-ok', text: c > 1 ? `超支 ${fmtMoney(-left)}` : c > 0.95 ? `只剩 ${fmtMoney(left)} 余量` : `还能剩 ${fmtMoney(left)}` });
  } else {
    body = `<div class="metric"><div class="small muted">工期 / 花费</div><div class="line"><span class="big" style="color: var(--ink-2)">—</span></div><div class="small t-bad" style="font-weight: 500">先把配置改到能跑起来</div></div>`;
  }
  $('fcMain').innerHTML = `<div class="fc-head"><h2>开跑前预估</h2><span class="chip ${chip[0]}">${chip[1]}</span></div>${body}`;

  // 单步和显存
  let step = '';
  if (e.breakdown) {
    const b = e.breakdown, tot = e.stepSec;
    const segs = [['计算', b.comp, 'var(--ink)'], ['流水线气泡', b.bubble, 'var(--yellow)'], ['TP 通信', b.tp, 'var(--gray)'], ['EP all-to-all', b.ep, '#5b6470'], ['梯度同步', b.dp, 'var(--info)']].filter((x) => x[1] > 0.0005);
    step += `<div class="fc-head" style="align-items: baseline"><span style="font-size: 13px; font-weight: 700">每步 <span class="num" style="font-size: 22px">${tot.toFixed(2)}</span> 秒</span><span class="small muted">MFU ${pct(e.mfu)} · ${(e.rate / 1e6).toFixed(2)}M token/秒</span></div>
      <div class="stack" role="img" aria-label="每步耗时：${segs.map((x) => `${x[0]} ${x[1].toFixed(2)} 秒`).join('，')}" style="background: none">${segs.map((x) => `<i style="width: ${(x[1] / tot * 100).toFixed(2)}%; background: ${x[2]}"></i>`).join('')}</div>
      <div class="legend2">${segs.map((x) => `<span><b style="background: ${x[2]}"></b>${x[0]} ${x[1].toFixed(2)} 秒</span>`).join('')}</div>`;
  } else {
    step += `<div class="fc-head"><span style="font-size: 13px; font-weight: 700">每步耗时</span><span class="small muted">配置不成立，算不出来</span></div>`;
  }
  if (e.mem) {
    const m = e.mem, over = m.total > m.cap, scale = Math.max(m.cap, m.total);
    const ms = [[m.weights, 'var(--ink)'], [m.act, 'var(--yellow)'], [m.opt + m.other, 'var(--gray)']];
    step += `<div class="divider" style="margin: 2px 0"></div>
      <div class="fc-head" style="align-items: baseline"><span style="font-size: 13px; font-weight: 700">每卡显存</span><span class="small ${over ? 't-bad' : 'muted'}"><b class="num" style="font-size: 18px; color: ${over ? 'var(--bad-ink)' : 'var(--ink)'}">${m.total.toFixed(1)}</b> / ${m.cap} GB</span></div>
      <div class="stack" role="img" aria-label="每卡显存 ${m.total.toFixed(1)} GB，上限 ${m.cap} GB" style="position: relative">${ms.map(([v, c]) => `<i style="width: ${(v / scale * 100).toFixed(2)}%; background: ${over ? 'var(--bad)' : c}"></i>`).join('')}${over ? `<span style="position: absolute; left: ${(m.cap / scale * 100).toFixed(1)}%; top: -3px; bottom: -3px; width: 2px; background: var(--ink)"></span>` : ''}</div>
      <div class="small muted">权重和梯度 ${m.weights.toFixed(1)} · 激活 ${m.act.toFixed(1)} · 其他 ${(m.opt + m.other).toFixed(1)} GB${over ? `，超了 ${(m.total - m.cap).toFixed(1)} GB` : ''}</div>`;
  }
  $('fcStep').innerHTML = step;
  $('fcStep').style.gap = '10px';

  const notes = [...e.errors.map((x) => `<li class="err">${x}</li>`), ...e.warnings.map((x) => `<li>${x}</li>`)];
  const slowEvery = e.ok ? 1 / (cfg.gpus / (e.gpu.mtbf * 4)) : Infinity;
  if (slowEvery < 30) notes.push(`<li>平均 ${slowEvery.toFixed(0)} 小时就有一台降频。手动换不过来的话，开跑后在底栏打开“降频自动换下”。</li>`);
  $('notes').innerHTML = notes.length ? notes.join('') : '<li>配置没问题，开跑吧。</li>';

  $('startBtn').disabled = !e.ok;
  $('cancelBtn').hidden = !reconfig;
  $('startTitle').textContent = reconfig ? '按新配置重启' : '开始训练';
  $('startSub').textContent = reconfig && g ? `从 ${fmtTok(g.ckptTokens)} token 的存盘重启` : `提交到${HALL} · ${fmtInt(cfg.gpus)} × ${cfg.gpu}`;
}

function layoutPreview() {
  const c = $('previewCanvas');
  const w = c.clientWidth || 284, h = c.clientHeight || 176;
  const PL = hallLayout(cfg.gpus);
  preview.setLayout(PL, { title: HALL, sub: `${SCENARIOS[scIdx].hall} · ${PL.racks} 柜` });
  preview.resize(w, h, window.devicePixelRatio || 1, { x: 0, y: 0, w, h }, 1);
  preview.draw({ nodes: new Uint8Array(PL.nodes), phase: 'run', progress: 0, paused: false, now: 0 });
  $('previewInfo').textContent = `${fmtInt(PL.racks)} 柜 · ${fmtInt(PL.nodes)} 台`;
}

// ---------- 一局 ----------
function newGame() {
  return {
    scIdx, cfg: { ...cfg }, t: 0, cost: 0, tokens: 0, ckptTokens: 0, sinceCkpt: 0,
    phase: 'restart', busyLeft: ev.restartSec, busyTotal: ev.restartSec, pendingEvict: null, failedNode: null, manualCkpt: false,
    paused: false, over: null,
    nodes: new Array(cfg.gpus / NODE_GPUS).fill(NODE_OK),
    spares: cfg.spares, repairs: [],
    log: [], hist: [[0, 0]], lossHist: [], recent: [],
    stats: { failures: 0, flaps: 0, ckpts: 0, lost: 0, stragglers: 0, spikes: 0 },
    dayFails: [], spikes: [], worst: null, tech: null, slowUntil: {}, slowSince: {},
    spent: { run: 0, slow: 0, ckpt: 0, restart: 0, repair: 0, rolled: 0 },
    event: null, spikeWatch: null, lastLossTok: 0,
  };
}

const stepOf = (tok) => fmtInt(tok / ev.tokensPerStep);
const jobDesc = (c) => `${fmtInt(c.gpus)} × ${c.gpu} · TP${c.tp} PP${c.pp} DP${ev.dp}${ev.model.moe ? ' EP' + c.ep : ''}`;
function setBusy(phase, sec) { g.phase = phase; g.busyLeft = sec; g.busyTotal = sec; }

function log(kind, ico, title, sub = '') {
  g.log.unshift({ t: g.t, kind, ico, title, sub });
  if (g.log.length > 160) g.log.pop();
  dirtyAlerts = true;
}

function startGame() {
  g = newGame();
  selected = null; hover = null;
  log('info', 'flag', '作业提交', jobDesc(cfg));
  showOps();
}

function applyReconfig() {
  const old = g.cfg;
  g.cfg = { ...cfg };
  g.cfg.autoSwap = old.autoSwap || 0;
  if (old.gpus !== cfg.gpus) { g.nodes = new Array(cfg.gpus / NODE_GPUS).fill(NODE_OK); g.slowUntil = {}; g.slowSince = {}; selected = null; }
  g.spares = Math.max(0, g.spares + (cfg.spares - old.spares));
  rollback('改配置');
  g.failedNode = null; g.pendingEvict = null; g.tech = null;
  for (let i = 0; i < g.nodes.length; i++) if (g.nodes[i] === NODE_SWAP || g.nodes[i] === NODE_FAIL) g.nodes[i] = NODE_OK;
  setBusy('restart', ev.restartSec);
  log('info', 'sliders', '按新配置重启', jobDesc(cfg));
  reconfig = false;
  g.paused = false;
  showOps();
}

function rollback(reason) {
  const lost = g.tokens - g.ckptTokens, sec = g.sinceCkpt;
  if (lost > 0) {
    g.stats.lost += lost;
    g.spent.rolled += lost / ev.rate;
    if (!g.worst || lost > g.worst.lost) g.worst = { t: g.t, lost, reason, tokens: g.ckptTokens };
  }
  g.tokens = g.ckptTokens;
  g.sinceCkpt = 0;
  g.lossHist = g.lossHist.filter((p) => p[0] <= g.tokens);
  return { lost: Math.max(0, lost), sec };
}

const rnd = Math.random;
const slowCount = () => { let n = 0; for (const s of g.nodes) if (s === NODE_SLOW) n++; return n; };
function pickHealthyNode() {
  for (let k = 0; k < 50; k++) { const i = Math.floor(rnd() * g.nodes.length); if (g.nodes[i] === NODE_OK || g.nodes[i] === NODE_SLOW) return i; }
  return 0;
}
const repairHours = ([a, b]) => a + rnd() * (b - a);
const dayFail = () => { const d = Math.floor(g.t / 86400); g.dayFails[d] = (g.dayFails[d] || 0) + 1; };

function currentLoss() {
  const sc = SCENARIOS[g.scIdx];
  let l = lossAt(MODELS[sc.model].active, g.tokens) + (rnd() - 0.5) * 0.008;
  const w = g.spikeWatch;
  if (w) {
    const k = w.until === Infinity ? 1 : Math.max(0, Math.min(1, (w.until - g.t) / w.span));
    l += w.diverge ? w.offset * (1.6 - 0.6 * k) : w.offset * k;
  }
  return l;
}

// 推进一分钟模拟时间
function tickMinute() {
  const sc = SCENARIOS[g.scIdx];
  g.t += 60;
  g.cost += ev.costPerHour / 60;

  for (let i = g.repairs.length - 1; i >= 0; i--) {
    if (g.repairs[i] <= g.t) { g.repairs.splice(i, 1); g.spares++; log('ok', 'wrench', '送修的机器修好了', `回到备机池，现在 ${g.spares} 台`); }
  }
  for (const key of Object.keys(g.slowUntil)) {
    const i = +key, until = g.slowUntil[key];
    if (g.nodes[i] !== NODE_SLOW) { delete g.slowUntil[key]; continue; }
    if (until > 0 && until <= g.t) {
      g.nodes[i] = NODE_OK; delete g.slowUntil[key];
      log('ok', 'thermo', `${nodeHost(L, i)} 温度降下来了`, slowCount() ? '这台恢复了，还有别的慢机器' : '降频解除，速度恢复');
    }
  }
  const auto = g.cfg.autoSwap || 0;
  if (auto && g.phase === 'run' && g.pendingEvict == null && g.spares > 0 && !g.event) {
    const due = Object.keys(g.slowUntil).map(Number).find((i) => g.nodes[i] === NODE_SLOW && g.t - (g.slowSince[i] ?? g.t) >= auto * 3600);
    if (due != null) evict(due, true);
  }

  const throttled = slowCount() > 0;
  const rate = ev.rate * (throttled ? STRAGGLER_SLOWDOWN : 1);
  let avail = 60;
  while (avail > 0 && !g.over) {
    if (g.phase === 'run') {
      const run = Math.min(avail, g.cfg.ckptMin * 60 - g.sinceCkpt);
      g.tokens += run * rate; g.sinceCkpt += run; avail -= run;
      g.spent.run += run; if (throttled) g.spent.slow += run;
      if (g.sinceCkpt >= g.cfg.ckptMin * 60) setBusy('ckpt', ev.ckptSec);
    } else {
      const d = Math.min(avail, g.busyLeft);
      g.busyLeft -= d; avail -= d;
      g.spent[g.phase] += d;
      if (g.busyLeft <= 1e-6) finishBusy();
    }
  }

  if (g.phase === 'run' || g.phase === 'ckpt') rollEvents();

  const l = currentLoss();
  g.recent.push([g.tokens / ev.tokensPerStep, l]);
  if (g.recent.length > 240) g.recent.shift();
  if (g.tokens - g.lastLossTok > sc.tokens / 400 || g.lossHist.length === 0) { g.lastLossTok = g.tokens; g.lossHist.push([g.tokens, l]); }
  if (g.t % 1800 === 0) g.hist.push([g.t, g.tokens]);

  if (g.tokens >= sc.tokens) endGame('win');
  else if (g.cost >= sc.budget) endGame('budget');
  else if (g.t >= sc.deadlineDays * 86400) endGame('late');
}

function finishBusy() {
  if (g.phase === 'ckpt') {
    g.ckptTokens = g.tokens; g.sinceCkpt = 0; g.stats.ckpts++;
    if (g.manualCkpt) { g.manualCkpt = false; log('info', 'save', '手动存盘完成', `step ${stepOf(g.tokens)}`); }
    if (g.pendingEvict != null) {
      const i = g.pendingEvict, h = nodeHost(L, i); g.pendingEvict = null;
      g.spares--; g.repairs.push(g.t + repairHours(SWAP_REPAIR_HOURS) * 3600);
      g.nodes[i] = NODE_SWAP; g.failedNode = i; delete g.slowUntil[i]; delete g.slowSince[i];
      g.tech = { node: i, label: `${h} 下线送修` };
      log('info', 'server', `${h} 下线送修`, '备机顶上，作业重启中');
      setBusy('restart', ev.restartSec);
      return;
    }
    g.phase = 'run';
  } else if (g.phase === 'repair') {
    const h = g.failedNode != null ? nodeHost(L, g.failedNode) : '故障机';
    if (g.failedNode != null) g.nodes[g.failedNode] = NODE_OK;
    g.failedNode = null; g.tech = null;
    log('ok', 'wrench', `${h} 现场修好了`, '重新拉起作业');
    setBusy('restart', ev.restartSec);
  } else if (g.phase === 'restart') {
    if (g.failedNode != null) g.nodes[g.failedNode] = NODE_OK;
    g.failedNode = null; g.tech = null;
    g.phase = 'run';
    log('ok', 'refresh', '作业恢复', `从 step ${stepOf(g.tokens)} 继续`);
  }
}

function rollEvents() {
  const N = g.cfg.gpus;
  const pFail = N / ev.gpu.mtbf / 60, pFlap = N / ev.net.flapMtbf / 60, pSlow = N / (ev.gpu.mtbf * 4) / 60;
  const r = rnd();
  if (r < pFail) return hardwareFailure();
  if (r < pFail + pFlap) return linkFlap();
  if (r < pFail + pFlap + pSlow) {
    const i = pickHealthyNode();
    if (g.nodes[i] === NODE_OK) {
      g.nodes[i] = NODE_SLOW; g.stats.stragglers++;
      g.slowUntil[i] = rnd() < STRAGGLER_RECOVER.chance ? g.t + repairHours(STRAGGLER_RECOVER.hours) * 3600 : -1;
      g.slowSince[i] = g.t;
      log('warn', 'thermo', `${nodeHost(L, i)} 过热降频`, `全集群步时 ${ev.stepSec.toFixed(2)} → ${(ev.stepSec / STRAGGLER_SLOWDOWN).toFixed(2)} 秒`);
    }
    return;
  }
  if (g.phase === 'run' && !g.spikeWatch && !g.event && rnd() < SPIKE_PER_HOUR / 60) return lossSpike();

  if (g.spikeWatch && g.t >= g.spikeWatch.until) {
    if (g.spikeWatch.diverge) {
      g.ckptTokens = Math.min(g.ckptTokens, g.spikeWatch.safeTokens);
      log('bad', 'spike', 'loss 发散成 NaN', `尖峰之后的存盘作废，回到 step ${stepOf(g.ckptTokens)}`);
      rollback('loss 发散');
      setBusy('restart', ev.restartSec);
    } else log('ok', 'check', 'loss 回落了', '继续训练');
    g.spikeWatch = null;
  }
}

const FAULTS = [['GPU 掉卡（Xid 79）', 'fail'], ['HBM 坏块（ECC）', 'fail'], ['NVLink 报错', 'fail'], ['整机掉电', 'fail'], ['网卡掉线', 'plug']];

function hardwareFailure() {
  const i = pickHealthyNode(), h = nodeHost(L, i);
  const [fault, ico] = FAULTS[Math.floor(rnd() * FAULTS.length)];
  g.stats.failures++; dayFail();
  g.nodes[i] = NODE_FAIL; g.failedNode = i; g.pendingEvict = null; g.manualCkpt = false; delete g.slowUntil[i]; delete g.slowSince[i];
  const { lost, sec } = rollback('坏机');
  log('bad', ico, `${h} ${fault}`, `回滚 ${fmtDur(sec)}，白算 ${fmtTok(lost)} token`);
  if (g.spares > 0) {
    g.spares--; g.repairs.push(g.t + repairHours(REPAIR_HOURS) * 3600);
    g.tech = { node: i, label: `${h} 拆下送修` };
    setBusy('restart', ev.restartSec);
  } else {
    setBusy('repair', REPAIR_WAIT_SEC * (0.7 + rnd() * 0.6));
    g.tech = { node: i, label: `${h} 现场维修` };
    log('bad', 'wrench', '没有备机了', `等现场工程师修 ${h}，大约 ${fmtDur(g.busyLeft)}`);
  }
}

function linkFlap() {
  g.stats.flaps++; dayFail(); g.manualCkpt = false;
  const { lost, sec } = rollback('链路抖动');
  log('bad', 'plug', 'IB 交换机端口抖动', `NCCL 超时，回滚 ${fmtDur(sec)}，白算 ${fmtTok(lost)} token`);
  setBusy('restart', ev.restartSec);
}

function lossSpike() {
  g.stats.spikes++;
  const from = currentLoss(), offset = 0.5 + rnd() * 0.35;
  g.spikes.push({ t: g.t, tokens: g.tokens });
  g.spikeWatch = { until: Infinity, span: 1, diverge: false, safeTokens: g.ckptTokens, offset };
  g.event = { t: g.t, step: g.tokens / ev.tokensPerStep, from, to: from + offset };
  g.recent.push([g.event.step, g.event.to]);
  g.paused = true;
  log('warn', 'spike', 'loss 尖峰', `step ${stepOf(g.tokens)}：${from.toFixed(2)} → ${(from + offset).toFixed(2)}`);
  renderEvent();
}

function resolveSpike(choice) {
  if (!g || !g.event) return;
  if (choice === 'rollback') {
    g.spikeWatch = null;
    log('info', 'rollback', '回滚，跳过这批数据', `回到 step ${stepOf(g.ckptTokens)}`);
    const { lost } = rollback('尖峰回滚');
    if (lost > 0 && g.worst && g.worst.reason === '尖峰回滚') g.worst.reason = 'loss 尖峰';
    setBusy('restart', ev.restartSec);
  } else {
    const span = (30 + rnd() * 60) * 60;
    Object.assign(g.spikeWatch, { until: g.t + span, span, diverge: rnd() < 0.4 });
    log('info', 'eye', '先不回滚，盯着 loss', '30 到 90 分钟见分晓');
  }
  g.event = null; g.paused = false;
  renderEvent();
}

function evict(i, auto = false) {
  if (!g || g.over || g.nodes[i] !== NODE_SLOW || g.phase !== 'run' || g.spares <= 0) return;
  g.pendingEvict = i;
  setBusy('ckpt', ev.ckptSec);
  log('info', 'save', `${auto ? '自动' : ''}换下 ${nodeHost(L, i)}`, auto ? `降频超过 ${g.cfg.autoSwap} 小时，先存盘再换` : '先存盘再重启');
}

function endGame(how) {
  g.over = { how };
  selected = null;
  log(how === 'win' ? 'ok' : 'bad', 'flag', how === 'win' ? '训练完成，交付' : how === 'late' ? '到期没训完，任务终止' : '预算用完，任务终止');
  showResult();
}

// ---------- 值班界面 ----------
let dirtyAlerts = true;
let alertsKey = '';
let sparkKey = '';
let pinsKey = '';

function showOps() {
  showOnly('ops');
  const sc = SCENARIOS[g.scIdx];
  ev = evaluate(g.cfg, sc);
  L = hallLayout(g.cfg.gpus);
  view.setLayout(L, { title: HALL, sub: `${sc.hall} · ${fmtInt(L.racks)} 柜 · ${fmtInt(g.cfg.gpus)} GPU` });
  $('opsSub').textContent = `${sc.title} · ${HALL}`;
  fillSelect($('opsCkpt'), g.cfg.ckptMin);
  dirtyAlerts = true; sparkKey = ''; pinsKey = '';
  layoutOps();
  renderEvent();
  renderHud();
}

function fillSelect(el, value) {
  el.innerHTML = CKPT_OPTIONS.map((v) => `<option value="${v}"${v === value ? ' selected' : ''}>${v}</option>`).join('');
}

function showSetup(isReconfig) {
  reconfig = isReconfig;
  if (isReconfig) { cfg = { ...g.cfg }; scIdx = g.scIdx; g.paused = true; }
  showOnly('setup');
  document.querySelector('#setup .modes').hidden = isReconfig;
  syncSetup(true);
}

const narrowQuery = window.matchMedia('(max-width: 860px)');
function layoutOps() {
  if (!g || $('ops').hidden) return;
  const stage = $('stage'), sr = stage.getBoundingClientRect();
  const w = sr.width, h = sr.height;
  let free;
  if (narrowQuery.matches) free = { x: 6, y: 6, w: w - 12, h: h - 12 };
  else {
    const rail = document.querySelector('.rail').getBoundingClientRect(), dock = document.querySelector('.dock').getBoundingClientRect();
    const x = 12, y = 112;
    free = { x, y, w: rail.left - sr.left - 20 - x, h: dock.top - sr.top - 14 - y };
  }
  view.resize(w, h, window.devicePixelRatio || 1, free, narrowQuery.matches ? 2 : 1.6);
  pinsKey = '';
}

function buildOps() {
  fillIcons();
  $('speed').innerHTML = [['pause', '暂停'], ['play1', '每秒 2 小时'], ['play2', '每秒 8 小时'], ['play3', '每秒 24 小时']]
    .map(([ic, label], i) => `<button type="button" data-i="${i - 1}" aria-label="${label}">${icon(ic, ic === 'play3' ? 20 : ic === 'play2' ? 18 : 16)}</button>`).join('');
  $('speed').onclick = (e) => {
    const b = e.target.closest('button'); if (!b || !g || g.over || g.event) return;
    const i = +b.dataset.i;
    if (i < 0) g.paused = !g.paused; else { speedIdx = i; g.paused = false; }
    renderHud();
  };
  $('ckptBtn').onclick = manualCkpt;
  $('spareAdd').onclick = addSpare;
  $('spareDel').onclick = () => {
    if (!g || g.spares <= 0) return;
    g.spares--; g.cfg.spares--; ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
    log('info', 'server', '退掉一台备机', `现在每小时 ${fmtMoney(ev.costPerHour)}`);
  };
  $('opsAuto').onchange = (e) => {
    g.cfg.autoSwap = +e.target.value;
    log('info', 'server', g.cfg.autoSwap ? `降频超过 ${g.cfg.autoSwap} 小时就自动换下` : '关掉了自动换下', g.cfg.autoSwap ? '每换一台用掉一台备机' : '降频的机器要自己点着换');
  };
  $('opsCkpt').onchange = (e) => {
    g.cfg.ckptMin = +e.target.value; ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
    log('info', 'save', `存盘间隔改成 ${g.cfg.ckptMin} 分钟`, ev.ok ? `最优约 ${Math.round(ev.youngMin)} 分钟` : '');
  };
  $('reconfBtn').onclick = () => { if (g && !g.over && !g.event) showSetup(true); };

  const canvas = $('scene');
  canvas.addEventListener('mousemove', (e) => {
    const r = canvas.getBoundingClientRect();
    const h = view.pick(e.clientX - r.left, e.clientY - r.top);
    if (h !== hover) { hover = h; canvas.style.cursor = h == null ? 'default' : 'pointer'; renderHoverTag(); }
  });
  canvas.addEventListener('mouseleave', () => { hover = null; renderHoverTag(); });
  canvas.addEventListener('click', (e) => {
    const r = canvas.getBoundingClientRect();
    const h = view.pick(e.clientX - r.left, e.clientY - r.top);
    selected = h == null || h === selected ? null : h;
    renderInspector(true);
  });
  $('pins').onclick = (e) => {
    const b = e.target.closest('.pin'); if (!b) return;
    const r = +b.dataset.r; selected = selected === r ? null : r;
    renderInspector(true);
  };
  $('inspector').onclick = (e) => {
    if (e.target.closest('#insClose')) { selected = null; renderInspector(true); }
    const act = e.target.closest('#insAct'); if (act && !act.disabled) { evict(+act.dataset.node); renderInspector(true); }
  };
  $('eventModal').onclick = (e) => { const b = e.target.closest('[data-choice]'); if (b) resolveSpike(b.dataset.choice); };

  document.addEventListener('keydown', (e) => {
    if (!g || $('ops').hidden || e.target.closest?.('input, select, textarea')) return;
    if (g.event) {
      if (e.key === '1') resolveSpike('rollback');
      else if (e.key === '2') resolveSpike('continue');
      return;
    }
    if (g.over) return;
    if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); g.paused = !g.paused; }
    else if (['1', '2', '3'].includes(e.key)) { speedIdx = +e.key - 1; g.paused = false; }
    else if (e.key === 's' || e.key === 'S') manualCkpt();
    else if (e.key === 'b' || e.key === 'B') addSpare();
    else if (e.key === 'c' || e.key === 'C') showSetup(true);
    else if (e.key === 'Escape') { selected = null; renderInspector(true); }
    else return;
    renderHud();
  });
}

function manualCkpt() {
  if (!g || g.over || g.phase !== 'run') return;
  g.manualCkpt = true;
  setBusy('ckpt', ev.ckptSec);
}
function addSpare() {
  if (!g || g.over) return;
  g.spares++; g.cfg.spares++; ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
  log('info', 'server', '加租一台备机', `现在每小时 ${fmtMoney(ev.costPerHour)}`);
}

const PHASE = { run: ['运行中', 'var(--ok)'], ckpt: ['存盘中', 'var(--info)'], restart: ['重启中', 'var(--bad)'], repair: ['停机待修', 'var(--bad)'] };

function renderHud() {
  if (!g) return;
  const sc = SCENARIOS[g.scIdx];
  const slow = slowCount(), k = slow ? STRAGGLER_SLOWDOWN : 1;
  const step = g.tokens / ev.tokensPerStep;

  // 时钟
  $('clockTime').textContent = clock(g.t);
  const sub = $('clockSub');
  sub.textContent = g.event ? '已暂停 · 等你决定' : g.paused ? '已暂停 · 空格继续' : `第 ${Math.floor(g.t / 86400) + 1} 天 · 每秒 ${SPEEDS[speedIdx]} 小时`;
  sub.style.color = g.event ? 'var(--warn-ink)' : '';
  sub.style.fontWeight = g.event ? '500' : '';
  $('speed').querySelectorAll('button').forEach((b) => {
    const i = +b.dataset.i;
    b.setAttribute('aria-pressed', String(i < 0 ? g.paused : !g.paused && i === speedIdx));
  });

  // 交付进度
  const frac = g.tokens / sc.tokens, timeFrac = g.t / (sc.deadlineDays * 86400);
  const remain = (sc.tokens - g.tokens) / (ev.rate * k * (1 - ev.overhead));
  const finishDay = Math.floor((g.t + remain) / 86400) + 1, projCost = g.cost + remain / 3600 * ev.costPerHour;
  const chip = projCost > sc.budget ? ['bad', `照这样会超支 ${fmtMoney(projCost - sc.budget)}`]
    : finishDay > sc.deadlineDays ? ['bad', `预计 D${finishDay} 完成，会超期`] : ['ok', `预计 D${finishDay} 完成，期限 D${sc.deadlineDays}`];
  $('progCard').innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; gap: 8px"><span class="small muted" style="font-weight: 500">交付进度</span><span class="chip ${chip[0]}">${chip[1]}</span></div>
    <div style="display: flex; align-items: baseline; gap: 10px"><span class="big">${pct(frac)}</span><span class="muted" style="font-size: 13px">${fmtTok(g.tokens)} / ${fmtTok(sc.tokens)} token</span></div>
    <div class="track"><i style="width: ${Math.min(100, frac * 100).toFixed(2)}%"></i><b style="left: calc(${Math.min(100, timeFrac * 100).toFixed(2)}% - 1.5px)"></b></div>
    <div class="small muted" style="display: flex; align-items: center; gap: 6px"><i style="width: 3px; height: 10px; border-radius: 2px; background: var(--yellow)"></i>时间已过 ${pct(timeFrac)}</div>
    <div class="minis">
      <div><div class="small muted">预算剩余</div><div class="num">${fmtMoney(sc.budget - g.cost)}</div><div class="small muted">已花 ${fmtMoney(g.cost)} / ${fmtMoney(sc.budget)}</div></div>
      <div><div class="small muted">距期限</div><div class="num">${Math.max(0, sc.deadlineDays - g.t / 86400).toFixed(1)}<span class="unit">天</span></div><div class="small muted">每小时 ${fmtMoney(ev.costPerHour)}</div></div>
    </div>`;

  // 训练
  const lastLoss = g.recent.length ? g.recent[g.recent.length - 1][1] : lossAt(MODELS[sc.model].active, 0);
  const sk = `${g.lossHist.length}|${g.spikes.length}`;
  if (sk !== sparkKey) { sparkKey = sk; $('trainSpark') ? updateSpark() : null; }
  const gp = g.t > 3600 ? g.tokens / (ev.rate * g.t) : null;
  if (!$('trainSpark')) {
    $('trainCard').innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: flex-end; gap: 8px"><div><div class="small muted" style="font-weight: 500">训练 loss</div><div class="num" style="font-size: 36px" id="tLoss"></div></div><div class="small muted" style="text-align: right" id="tStep"></div></div>
      <svg id="trainSpark" viewBox="0 0 288 56" preserveAspectRatio="none" role="img" aria-label="loss 曲线"></svg>
      <div class="stats3">
        <div><div class="small muted">步时</div><div class="num" id="tStepT"></div><div class="small muted" id="tStepN"></div></div>
        <div><div class="small muted">MFU</div><div class="num" id="tMfu"></div><div class="small muted" id="tMfuN"></div></div>
        <div><div class="small muted">Goodput</div><div class="num" id="tGp"></div><div class="small muted">≥85% 加分</div></div>
      </div>`;
    updateSpark();
  }
  $('tLoss').textContent = lastLoss.toFixed(3);
  $('tStep').innerHTML = `第 ${fmtInt(step)} 步<br>每步 ${(ev.tokensPerStep / 2 ** 20).toFixed(0)}M token`;
  $('tStepT').innerHTML = `${(ev.stepSec / k).toFixed(2)}<span class="unit">s</span>`;
  $('tStepT').style.color = slow ? 'var(--warn-ink)' : '';
  $('tStepN').textContent = slow ? `平时 ${ev.stepSec.toFixed(2)}` : '没有慢机器';
  $('tMfu').textContent = pct(ev.mfu * k);
  $('tMfu').style.color = slow ? 'var(--warn-ink)' : '';
  $('tMfuN').textContent = slow ? `平时 ${pct(ev.mfu)}` : `B=${g.cfg.gbs}`;
  $('tGp').textContent = gp == null ? '—' : pct(gp);

  // 告警
  const day = Math.floor(g.t / 86400);
  const ak = `${g.log.length}|${g.log[0]?.t}|${day}`;
  if (dirtyAlerts || ak !== alertsKey) {
    dirtyAlerts = false; alertsKey = ak;
    const icoCls = { bad: 'bad', warn: 'warn', ok: 'ok', info: 'info' };
    $('alerts').innerHTML = g.log.map((x, i) => {
      const sameDay = Math.floor(x.t / 86400) === day;
      const hot = i === 0 && (x.kind === 'warn' || x.kind === 'bad') && g.t - x.t < 3600;
      return `<li class="${hot ? 'hot' : ''}"><span class="ico ${icoCls[x.kind] || 'plain'}">${icon(x.ico || 'info', 17, 2)}</span><div style="min-width: 0"><div class="ttl"><span>${x.title}</span><time>${sameDay ? clock(x.t).slice(-5) : clock(x.t)}</time></div>${x.sub ? `<div class="sub">${x.sub}</div>` : ''}</div></li>`;
    }).join('');
    const today = g.log.filter((x) => Math.floor(x.t / 86400) === day).length;
    $('alertsCount').textContent = `今天 ${today} 条`;
  }

  // 底栏
  const [pLabel, pColor] = g.event ? ['等你决定', 'var(--warn)'] : g.paused ? ['已暂停', 'var(--gray)'] : PHASE[g.phase];
  $('statusDot').style.background = pColor;
  $('statusDot').style.boxShadow = `0 0 0 4px color-mix(in srgb, ${pColor} 18%, transparent)`;
  $('statusT').textContent = pLabel;
  const left = fmtDur(g.busyLeft);
  $('statusSub').textContent = g.event ? `step ${fmtInt(step)} · loss 尖峰`
    : g.phase === 'run' ? `step ${fmtInt(step)} · ${fmtDur(g.sinceCkpt)}前存盘`
      : g.phase === 'ckpt' ? `正在存 step ${fmtInt(step)} · 还要 ${left}`
        : g.phase === 'restart' ? `从 step ${fmtInt(step)} 恢复 · 还要 ${left}` : `等现场维修 · 还要 ${left}`;
  $('cSpares').innerHTML = `${g.spares}<span class="unit">台</span>`;
  $('cSpares').style.color = g.spares === 0 ? 'var(--warn-ink)' : '';
  $('cRma').innerHTML = `${g.repairs.length}<span class="unit">台</span>`;
  $('cSlow').innerHTML = `${slow}<span class="unit">台</span>`;
  $('cSlow').style.color = slow ? 'var(--warn-ink)' : '';
  $('cMtbf').innerHTML = `${ev.mtbfHours.toFixed(1)}<span class="unit">小时</span>`;
  if (+$('opsCkpt').value !== g.cfg.ckptMin) $('opsCkpt').value = g.cfg.ckptMin;
  if (+$('opsAuto').value !== (g.cfg.autoSwap || 0)) $('opsAuto').value = g.cfg.autoSwap || 0;
  const busy = !!g.over || !!g.event;
  $('ckptBtn').disabled = busy || g.phase !== 'run';
  $('spareAdd').disabled = busy;
  $('spareDel').disabled = busy || g.spares <= 0;
  $('reconfBtn').disabled = busy;

  renderPins();
  renderInspector(false);
}

function updateSpark() {
  const sc = SCENARIOS[g.scIdx], active = MODELS[sc.model].active;
  const hi = lossAt(active, sc.tokens / 400) + 0.3, lo = lossAt(active, sc.tokens) - 0.04;
  const pts = g.lossHist, n = Math.max(1, pts.length - 1);
  const Y = (l) => (Math.max(0, Math.min(1, (hi - l) / (hi - lo))) * 54 + 1).toFixed(1);
  // 横轴按已经训过的 token 摊满整个宽度
  const span = Math.max(1, pts.length ? pts[pts.length - 1][0] : 1);
  const X = (tok) => (tok / span * 284 + 2).toFixed(1);
  let html = '';
  if (pts.length > 1) {
    const step = Math.max(1, Math.ceil(pts.length / 140));
    const sel = pts.filter((_, i) => i % step === 0 || i === n);
    const d = sel.map(([t, l], i) => `${i ? 'L' : 'M'}${X(t)} ${Y(l)}`).join(' ');
    html += `<path d="${d} L${X(sel[sel.length - 1][0])} 56 L${X(sel[0][0])} 56 Z" fill="rgba(20,24,29,0.06)"></path>`;
    html += `<path d="${d}" fill="none" stroke="#14181d" stroke-width="1.75" stroke-linejoin="round" vector-effect="non-scaling-stroke"></path>`;
    const [lt, ll] = pts[n];
    html += `<circle cx="${X(lt)}" cy="${Y(ll)}" r="2.6" fill="#14181d"></circle>`;
  }
  for (const s of g.spikes) if (s.tokens <= span && pts.length) {
    const near = pts.reduce((a, p) => (Math.abs(p[0] - s.tokens) < Math.abs(a[0] - s.tokens) ? p : a), pts[0]);
    html += `<circle cx="${X(s.tokens)}" cy="${Y(near[1])}" r="3" fill="#f59300"></circle>`;
  }
  $('trainSpark').innerHTML = html;
}

function rackState(r) {
  let bad = null, node = null;
  for (let j = 0; j < NODES_PER_RACK; j++) {
    const i = r * NODES_PER_RACK + j, v = g.nodes[i];
    if (v === NODE_FAIL) { bad = 'fail'; node = i; } else if (v === NODE_SLOW && bad !== 'fail') { bad = 'slow'; node ??= i; }
  }
  return { bad, node };
}

function renderPins() {
  const items = [];
  for (let r = 0; r < L.racks; r++) { const s = rackState(r); if (s.bad) items.push([r, s.bad]); }
  const tech = g.tech ? `${g.tech.node}|${g.tech.label}` : '';
  const key = `${items.map((x) => x.join(':')).join(',')}|${selected}|${tech}|${view.s}|${view.offX}|${view.offY}`;
  if (key === pinsKey) return;
  pinsKey = key;
  let html = '';
  for (const [r, bad] of items) {
    const sel = r === selected, stem = sel ? 30 : 20, a = view.anchor(r);
    const size = sel ? 36 : 28;
    html += `<button type="button" class="pin ${bad}${sel ? ' sel' : ' small'}" data-r="${r}" aria-label="${rackName(L, r)} ${bad === 'fail' ? '故障' : '降频'}，查看详情" style="left: ${a.x.toFixed(1)}px; top: ${(a.y - stem - size / 2).toFixed(1)}px; --stem: ${stem}px">${icon(bad === 'fail' ? 'x' : 'thermo', sel ? 17 : 14, 2.4)}</button>`;
  }
  if (g.tech) {
    const r = Math.floor(g.tech.node / NODES_PER_RACK), rk = L.rackList[r];
    const p = rk.near ? view.worldToScreen(rk.x + 0.55, rk.y + 2.55, 22) : view.anchor(r);
    const dx = rk.near ? 0 : 70;
    html += `<div class="tech" style="left: ${(p.x + dx).toFixed(1)}px; top: ${(p.y - (rk.near ? 0 : 8)).toFixed(1)}px">${icon('wrench', 14, 2)}<span>${g.tech.label}</span></div>`;
  }
  $('pins').innerHTML = html;
}

function renderHoverTag() {
  const tag = $('hoverTag');
  if (hover == null || !L || hover === selected) { tag.hidden = true; return; }
  const a = view.anchor(hover);
  tag.hidden = false;
  tag.textContent = rackName(L, hover);
  tag.style.left = a.x + 'px';
  tag.style.top = (a.y - (rackState(hover).bad ? 58 : 10)) + 'px';
}

let insKey = '';
function temp(i) {
  const h = Math.imul(i + 1, 2654435761) >>> 0, v = g.nodes[i];
  const wob = Math.sin(g.t / 5400 + (h % 97)) * 1.3;
  if (v === NODE_FAIL || v === NODE_SWAP) return null;
  if (v === NODE_SLOW) return Math.round(92 + (h % 4) + wob);
  return Math.round(66 + (h % 8) + wob + (g.phase === 'run' ? 2 : -6));
}

function renderInspector(force) {
  const box = $('inspector');
  if (selected == null || !g || !L || selected >= L.racks) { box.hidden = true; insKey = ''; renderHoverTag(); return; }
  const r = selected, name = rackName(L, r), rk = L.rackList[r], st = rackState(r);
  const nodes = [0, 1, 2, 3].map((j) => r * NODES_PER_RACK + j);
  const key = `${r}|${nodes.map((i) => g.nodes[i]).join('')}|${g.phase}|${g.spares}|${Math.floor(g.busyLeft / 30)}|${Math.floor(g.t / 600)}`;
  if (!force && key === insKey) { positionInspector(); return; }
  insKey = key;
  const rowsHtml = nodes.map((i) => {
    const v = g.nodes[i], t = temp(i);
    const [cls, dot, txt] = v === NODE_SLOW ? ['slow', 'var(--warn)', '过热降频'] : v === NODE_FAIL ? ['fail', 'var(--bad)', '故障'] : v === NODE_SWAP ? ['', 'var(--gray)', '下线中'] : ['', g.phase === 'run' || g.phase === 'ckpt' ? 'var(--ok)' : 'var(--gray)', '正常'];
    return `<div class="${cls}"><span class="dot" style="background: ${dot}"></span><span class="mono">${nodeHost(L, i)}</span><span class="${cls ? '' : 'muted'}">${txt}</span><span class="mono small ${cls ? '' : 'muted'}">${t == null ? '—' : t + '°C'}</span></div>`;
  }).join('');
  const icoCls = st.bad === 'fail' ? 'bad' : st.bad === 'slow' ? 'warn' : 'plain';
  let msg, act = '';
  if (st.bad === 'slow') {
    const i = nodes.find((x) => g.nodes[x] === NODE_SLOW), h = nodeHost(L, i);
    const perHour = ev.rate * (1 - STRAGGLER_SLOWDOWN) * 3600;
    msg = `同步训练要等最慢的那台。步时从 ${ev.stepSec.toFixed(2)} 秒变成 <b class="t-warn">${(ev.stepSec / STRAGGLER_SLOWDOWN).toFixed(2)} 秒</b>，每小时少训 ${fmtTok(perHour)} token。有的过几个小时会自己降下来，有的不会。`;
    const swapping = g.pendingEvict === i;
    const can = g.phase === 'run' && g.spares > 0 && !g.event && !swapping;
    const note = swapping ? '正在存盘，存完就换上备机。' : g.spares <= 0 ? '没有备机可换。先加一台备机。' : g.phase !== 'run' ? '作业没在跑，等它起来再换。' : `先存盘再重启，大约 ${fmtDur(ev.ckptSec + ev.restartSec)}。还剩 ${g.spares} 台备机。`;
    act = `<div style="display: flex; flex-direction: column; gap: 8px"><button type="button" class="ins-act" id="insAct" data-node="${i}"${can ? '' : ' disabled'}><span>${swapping ? `正在换下 ${h}` : `换下 ${h}`}</span><span>用 1 台备机</span></button><div class="small muted">${note}</div></div>`;
  } else if (st.bad === 'fail') {
    const h = nodeHost(L, st.node);
    msg = g.phase === 'repair' ? `${h} 坏了，没有备机，在等现场修，还要 ${fmtDur(g.busyLeft)}。` : `${h} 坏了，已经换上备机，作业还要 ${fmtDur(g.busyLeft)} 才能起来。`;
  } else if (nodes.some((i) => g.nodes[i] === NODE_SWAP)) {
    msg = `换下来的机器正在送修，备机已经顶上，作业还要 ${fmtDur(g.busyLeft)} 才能起来。`;
  } else {
    msg = g.phase === 'run' ? '四台都在正常训练。' : g.phase === 'ckpt' ? '四台都正常，正在一起存盘。' : '四台都正常，在等作业重新拉起。';
  }
  box.innerHTML = `<span class="tip-arrow"></span>
    <div class="ins-head"><div class="ins-ico ico ${icoCls}">${icon(st.bad === 'fail' ? 'fail' : st.bad === 'slow' ? 'thermo' : 'server', 18, 2)}</div>
      <div style="flex: 1; min-width: 0"><div style="font-size: 16px; font-weight: 700; line-height: 1.3">${name} 机柜</div><div class="small muted">${L.rows[rk.row].letter} 行 ${rk.n} 号 · 4 台 · 32 卡</div></div>
      <button type="button" class="icon-btn" id="insClose" aria-label="关闭">${icon('x', 16, 2)}</button></div>
    <div class="nodes">${rowsHtml}</div>
    <p style="font-size: 13px; line-height: 1.6">${msg}</p>${act}`;
  box.hidden = false;
  positionInspector();
  renderHoverTag();
}

function positionInspector() {
  const box = $('inspector');
  if (box.hidden || narrowQuery.matches || selected == null) { box.style.left = ''; box.style.top = ''; return; }
  const a = view.anchor(selected), stage = $('stage').getBoundingClientRect();
  const rail = document.querySelector('.rail').getBoundingClientRect();
  const w = 300, h = box.offsetHeight || 340, pinY = a.y - (rackState(selected).bad ? 48 : 20);
  let left = a.x + 38, side = 'left';
  if (left + w > rail.left - stage.left - 12) { left = a.x - 38 - w; side = 'right'; }
  left = Math.max(12, left);
  let top = pinY - 76;
  top = Math.max(100, Math.min(top, stage.height - 116 - h));
  box.style.left = left + 'px';
  box.style.top = top + 'px';
  const arrow = box.querySelector('.tip-arrow');
  if (arrow) {
    arrow.style.left = side === 'left' ? '-6px' : '';
    arrow.style.right = side === 'right' ? '-6px' : '';
    arrow.style.top = Math.max(16, Math.min(h - 28, pinY - top - 6)) + 'px';
  }
}

function renderEvent() {
  const shade = $('eventShade');
  if (!g || !g.event) { shade.hidden = true; return; }
  const e = g.event;
  const lost = g.tokens - g.ckptTokens;
  shade.hidden = false;
  $('eventModal').innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap">
      <span class="chip warn" style="font-weight: 700; padding: 3px 10px 3px 8px">${icon('pause', 13)}已自动暂停</span>
      <span class="mono small muted">${clock(e.t)} · step ${fmtInt(e.step)}</span>
    </div>
    <div style="display: flex; flex-direction: column; gap: 8px">
      <h1 id="eventTitle">loss 尖峰</h1>
      <p style="font-size: 15px; line-height: 1.65; color: var(--ink-2)">loss 从 ${e.from.toFixed(2)} 跳到 ${e.to.toFixed(2)}。多半是混进了一批脏数据，也可能是优化器状态出了问题。</p>
    </div>
    <div class="chart-box">${spikeChart()}</div>
    <div class="choices">
      <button type="button" class="choice" data-choice="rollback">
        <span class="hd"><span style="display: flex; align-items: center; gap: 8px"><kbd style="color: var(--ink); font-size: 12px">1</kbd><span class="chip ok">稳妥</span></span><span class="muted">${icon('rollback', 20, 2)}</span></span>
        <span class="ttl">回滚，跳过这批数据</span>
        <ul><li>回到 step ${stepOf(g.ckptTokens)}，${fmtDur(g.sinceCkpt)}前存的盘</li><li>重启大约 ${fmtDur(ev.restartSec)}</li><li>白算 <b style="color: var(--ink)">${fmtTok(lost)} token</b></li></ul>
      </button>
      <button type="button" class="choice" data-choice="continue">
        <span class="hd"><span style="display: flex; align-items: center; gap: 8px"><kbd style="color: var(--ink); font-size: 12px">2</kbd><span class="chip warn">赌一把</span></span><span class="muted">${icon('eye', 20, 2)}</span></span>
        <span class="ttl">先不动，观察 30 到 90 分钟</span>
        <span style="display: flex; flex-direction: column; gap: 6px; width: 100%">
          <span class="odds" role="img" aria-label="六成会自己回落，四成会发散"><i style="width: 60%; background: var(--ok)"></i><i style="width: 40%; background: var(--bad)"></i></span>
          <span style="display: flex; justify-content: space-between; gap: 8px; font-size: 12px; font-weight: 500"><span class="t-ok">60% 自己回落，什么都不丢</span><span class="t-bad">40% 发散</span></span>
        </span>
        <span style="font-size: 13px; color: var(--ink-2); line-height: 1.55">发散的话，尖峰之后存的盘都作废，要退回更早的存盘。</span>
      </button>
    </div>
    <div class="small muted">按 1 或 2 选。暂停期间时间不走，也不花钱。</div>`;
  shade.querySelector('[data-choice="rollback"]').focus({ preventScroll: true });
}

function spikeChart() {
  const pts = g.recent.slice(-200);
  if (pts.length < 2) return '';
  const s0 = pts[0][0], s1 = pts[pts.length - 1][0], W = 656, H = 170;
  const ls = pts.map((p) => p[1]);
  const lo = Math.floor((Math.min(...ls) - 0.06) * 10) / 10, hi = Math.ceil((Math.max(...ls) + 0.05) * 10) / 10;
  const X = (s) => ((s - s0) / Math.max(1, s1 - s0) * W).toFixed(1), Y = (l) => ((hi - l) / (hi - lo) * H).toFixed(1);
  let grid = '';
  for (let v = lo; v <= hi + 1e-9; v += (hi - lo) > 0.6 ? 0.2 : 0.1) {
    grid += `<line x1="0" y1="${Y(v)}" x2="${W}" y2="${Y(v)}" stroke="${Math.abs(v - lo) < 1e-6 ? '#cfd5db' : '#e3e7eb'}"></line><text x="-10" y="${(+Y(v) + 4).toFixed(1)}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="12" fill="#59636e">${v.toFixed(1)}</text>`;
  }
  const ck = g.ckptTokens / ev.tokensPerStep;
  const ckLine = ck >= s0 ? `<line x1="${X(ck)}" y1="-14" x2="${X(ck)}" y2="${H}" stroke="#3b7cf3" stroke-width="1.5" stroke-dasharray="4 3"></line><text x="${(+X(ck) - 7).toFixed(1)}" y="-6" text-anchor="end" font-family="Noto Sans SC, sans-serif" font-size="12" font-weight="500" fill="#2457c5">上次存盘 · ${fmtDur(g.sinceCkpt)}前</text>` : '';
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0])} ${Y(p[1])}`).join(' ');
  const last = pts[pts.length - 1];
  return `<svg viewBox="-40 -24 706 214" role="img" aria-label="最近的 loss：前面很平，最后一步跳到 ${last[1].toFixed(2)}">${grid}
    <text x="0" y="188" font-family="IBM Plex Mono, monospace" font-size="12" fill="#59636e">step ${fmtInt(s0)}</text>
    <text x="${W}" y="188" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="12" fill="#59636e">${fmtInt(s1)}</text>
    ${ckLine}
    <path d="${d}" fill="none" stroke="#14181d" stroke-width="2" stroke-linejoin="round"></path>
    <circle cx="${X(last[0])}" cy="${Y(last[1])}" r="5" fill="#e5383b" stroke="#ffffff" stroke-width="2"></circle>
    <text x="${(+X(last[0]) - 12).toFixed(1)}" y="${(+Y(last[1]) + 18).toFixed(1)}" text-anchor="end" font-family="Barlow Condensed, sans-serif" font-size="18" font-weight="600" fill="#bf2629">${last[1].toFixed(2)}</text>
  </svg>`;
}

// ---------- 验收 ----------
function showResult() {
  const sc = SCENARIOS[g.scIdx], how = g.over.how, win = how === 'win';
  showOnly('result');
  const goodput = g.tokens / (ev.rate * Math.max(1, g.t));
  const costFrac = g.cost / sc.budget, cheap = costFrac <= COST_S, smooth = goodput >= GOODPUT_S;
  const grade = win ? (cheap && smooth ? 'S' : cheap || smooth ? 'A' : 'B') : '—';
  const days = g.t / 86400;
  const tick = (ok, txt) => `<span class="${ok ? '' : 'muted'}">${ok ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0e7a47" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${STROKE.check}</svg>` : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#bf2629" stroke-width="2.6" stroke-linecap="round" aria-hidden="true"><path d="M7 7l10 10M17 7L7 17"/></svg>`}${txt}</span>`;
  const missing = [];
  if (!cheap) missing.push(`花费得压到 ${fmtMoney(sc.budget * COST_S)} 以内，也就是预算的 ${COST_S * 100}%`);
  if (!smooth) missing.push('Goodput 要到 85%，存盘间隔、备机和慢机器都会影响它');
  const hint = !win ? '这一局没交付，不评级。' : grade === 'S' ? '三条全中。' : `要拿 S，${missing.join('；')}。`;
  const title = win ? '按期交付' : how === 'late' ? '逾期' : '预算用完';
  const lead = win ? `${clock(g.t)} 训满 ${fmtTok(sc.tokens)} token，比期限早 ${(sc.deadlineDays - days).toFixed(1)} 天。`
    : how === 'late' ? `到期只训了 ${pct(g.tokens / sc.tokens)}（${fmtTok(g.tokens)} / ${fmtTok(sc.tokens)}）。` : `钱花完时训了 ${pct(g.tokens / sc.tokens)}，离期限还有 ${(sc.deadlineDays - days).toFixed(1)} 天。`;
  $('resultSub').textContent = `${sc.title} · 验收`;
  $('verdict').innerHTML = `
    <span class="small muted" style="font-weight: 500">任务 ${pad2(g.scIdx + 1)} · ${HALL}</span>
    <div style="display: flex; align-items: center; gap: 14px; margin-top: 14px">
      <span class="vmark" style="background: ${win ? 'var(--ok-bg)' : 'var(--bad-bg)'}; color: ${win ? 'var(--ok-ink)' : 'var(--bad-ink)'}">${win ? icon('check', 26, 2.6) : icon('x', 24, 2.6)}</span>
      <h1>${title}</h1>
    </div>
    <p style="margin-top: 12px; font-size: 15px; line-height: 1.65; color: var(--ink-2)">${lead}</p>
    <div style="display: flex; align-items: center; gap: 22px; margin-top: 28px; flex-wrap: wrap">
      <div class="grade${win ? '' : ' none'}" role="img" aria-label="评级 ${win ? grade : '无'}">${grade}</div>
      <div class="crit"><span class="small muted">评级</span>${tick(win, '按期交付')}${tick(smooth, `Goodput ${pct(goodput)}，${smooth ? '过了' : '没到'} 85%`)}${tick(cheap, `花了 ${pct(costFrac)} 的预算，${cheap ? '压在' : '没压到'} ${COST_S * 100}%${cheap ? ' 以内' : ''}`)}</div>
    </div>
    <div style="margin-top: 22px; padding: 12px 14px; border-radius: 12px; background: var(--fill); font-size: 13px; line-height: 1.6">${hint}</div>
    <div class="mono small muted" style="margin-top: 22px; line-height: 1.7">${g.cfg.gpu} × ${fmtInt(g.cfg.gpus)} · TP${g.cfg.tp} PP${g.cfg.pp} DP${ev.dp}${ev.model.moe ? ' EP' + g.cfg.ep : ''}<br>存盘 ${g.cfg.ckptMin} 分钟 · 备机 ${g.cfg.spares} 台 · ${NETWORKS[g.cfg.net].name}</div>
    <div style="flex: 1; min-height: 24px"></div>
    <div style="display: flex; flex-direction: column; gap: 10px">
      <button type="button" class="btn primary" style="height: 52px; justify-content: center; font-size: 16px; border-radius: 14px" id="againBtn">再来一局</button>
      <button type="button" class="btn" style="height: 52px; justify-content: center; font-size: 16px; border-radius: 14px" id="menuBtn">换个任务</button>
    </div>`;
  const lossNow = g.lossHist.length ? g.lossHist[g.lossHist.length - 1][1] : 0;
  const tile = (label, big, sub) => `<div class="card tile4"><span class="small muted" style="font-weight: 500">${label}</span><span class="num">${big}</span><span class="small muted">${sub}</span></div>`;
  $('tiles').innerHTML = tile('用时', `${days.toFixed(1)}<span class="unit" style="font-size: 14px; font-weight: 500; color: var(--ink)">天</span>`, `期限 ${sc.deadlineDays} 天`)
    + tile('花费', fmtMoney(g.cost), `预算 ${fmtMoney(sc.budget)} 的 ${pct(costFrac)}`)
    + tile('Goodput', pct(goodput), '真正在训练的时间占比')
    + tile(win ? '最终 loss' : '停下时的 loss', lossNow.toFixed(3), `${win ? '训满' : '训到'} ${fmtTok(g.tokens)} token`);
  $('timelineSvg').innerHTML = timelineSvg(sc, win) + spentBar(goodput);
  const inc = (n, label, color) => `<div><div class="num">${n}</div><div class="small"><span class="dot" style="width: 7px; height: 7px; background: ${color}"></span>${label}</div></div>`;
  $('incidents').innerHTML = inc(g.stats.failures, '硬件故障', 'var(--bad)') + inc(g.stats.flaps, '链路抖动', 'var(--bad)')
    + inc(g.stats.stragglers, '过热降频', 'var(--warn)') + inc(g.stats.spikes, 'loss 尖峰', 'var(--warn)')
    + inc(fmtTok(g.stats.lost), '回滚白算的 token', 'var(--gray)') + inc(fmtInt(g.stats.ckpts), '存盘次数', 'var(--info)');
  $('againBtn').onclick = () => { scIdx = g.scIdx; cfg = { ...g.cfg }; ev = evaluate(cfg, SCENARIOS[scIdx]); startGame(); };
  $('menuBtn').onclick = () => { scIdx = g.scIdx; cfg = { ...g.cfg }; g = null; showSetup(false); };
}

function spentBar(goodput) {
  const T = Math.max(1, g.t), sp = g.spent;
  const parts = [
    ['有效训练', goodput, 'var(--ink)'],
    ['降频拖慢', sp.slow * (1 - STRAGGLER_SLOWDOWN) / T, 'var(--warn)'],
    ['重启', sp.restart / T, 'var(--bad)'],
    ['等维修', sp.repair / T, '#8f1f22'],
    ['存盘', sp.ckpt / T, 'var(--info)'],
    ['回滚白算', sp.rolled / T, 'var(--gray)'],
  ].filter((x) => x[1] > 0.0005);
  return `<div style="display: flex; flex-direction: column; gap: 8px; margin-top: 6px">
    <div style="display: flex; justify-content: space-between; align-items: baseline; gap: 8px"><span style="font-size: 13px; font-weight: 700">时间都花在哪了</span><span class="small muted">按满速训练折算</span></div>
    <div class="stack" role="img" aria-label="${parts.map((x) => `${x[0]} ${pct(x[1])}`).join('，')}" style="background: none">${parts.map((x) => `<i style="width: ${(x[1] * 100).toFixed(2)}%; background: ${x[2]}"></i>`).join('')}</div>
    <div class="tl-legend">${parts.map((x) => `<span><i class="dot" style="background: ${x[2]}"></i>${x[0]} ${pct(x[1])}</span>`).join('')}</div>
  </div>`;
}

function timelineSvg(sc, win) {
  const D = sc.deadlineDays, W = 640, H = 200;
  const X = (t) => Math.min(W, t / (D * 86400) * W), Y = (tok) => H - Math.min(1, tok / sc.tokens) * H;
  const f1 = (v) => v.toFixed(1);
  let s = '';
  for (let q = 0; q <= 4; q++) {
    const y = H - q * H / 4;
    s += `<line x1="0" y1="${y}" x2="${W}" y2="${y}" stroke="${q ? '#eceff2' : '#cfd5db'}"></line><text x="-10" y="${y + 4}" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="11" fill="#59636e">${q ? fmtTokAxis(sc.tokens * q / 4) : 0}</text>`;
  }
  s += `<line x1="${W}" y1="-14" x2="${W}" y2="288" stroke="#e5383b" stroke-width="1.2" stroke-dasharray="4 3"></line><text x="${W - 4}" y="-12" text-anchor="end" font-family="Noto Sans SC, sans-serif" font-size="11" font-weight="500" fill="#bf2629">期限</text>`;
  const all = [...g.hist, [g.t, g.tokens]];
  const step = Math.max(1, Math.ceil(all.length / 400));
  const pts = all.filter((_, i) => i % step === 0 || i === all.length - 1);
  const d = pts.map(([t, k], i) => `${i ? 'L' : 'M'}${f1(X(t))} ${f1(Y(k))}`).join(' ');
  const ex = X(g.t), ey = Y(g.tokens);
  s += `<path d="${d} L${f1(ex)} ${H} L0 ${H} Z" fill="rgba(20,24,29,0.05)"></path><path d="${d}" fill="none" stroke="#14181d" stroke-width="2" stroke-linejoin="round"></path>`;
  s += `<circle cx="${f1(ex)}" cy="${f1(ey)}" r="5" fill="${win ? '#19b26b' : '#e5383b'}" stroke="#ffffff" stroke-width="2"></circle>`;
  const endTxt = win ? `第 ${(g.t / 86400).toFixed(1)} 天交付` : `停在 ${pct(g.tokens / sc.tokens)}`;
  const halo = 'paint-order="stroke" stroke="#ffffff" stroke-width="4" stroke-linejoin="round"';
  s += W - ex >= 100 ? `<text x="${f1(ex + 10)}" y="${f1(ey + 5)}" ${halo} font-family="Noto Sans SC, sans-serif" font-size="12" font-weight="700" fill="${win ? '#0e7a47' : '#bf2629'}">${endTxt}</text>`
    : `<text x="${f1(ex - 10)}" y="${f1(Math.max(-8, ey - 10))}" text-anchor="end" ${halo} font-family="Noto Sans SC, sans-serif" font-size="12" font-weight="700" fill="${win ? '#0e7a47' : '#bf2629'}">${endTxt}</text>`;
  const tokAt = (t) => { let best = all[0]; for (const p of all) if (p[0] <= t) best = p; return best[1]; };
  for (const sp of g.spikes) {
    const x = X(sp.t), y = Y(sp.tokens);
    s += `<rect x="${f1(x - 3.5)}" y="${f1(y - 3.5)}" width="7" height="7" transform="rotate(45 ${f1(x)} ${f1(y)})" fill="#f59300" stroke="#ffffff" stroke-width="1.5"></rect>`;
  }
  if (g.worst && g.worst.lost > 0) {
    const x = X(g.worst.t), y = Y(tokAt(g.worst.t));
    const right = x < W - 260, below = y < 150, dy = below ? 1 : -1;
    s += `<line x1="${f1(x + (right ? 4 : -4))}" y1="${f1(y + 5 * dy)}" x2="${f1(x + (right ? 18 : -18))}" y2="${f1(y + 20 * dy)}" stroke="#9aa4ae"></line>`;
    const why = g.worst.reason, gap = /^[A-Za-z]/.test(why) ? ' ' : '';
    s += `<text x="${f1(x + (right ? 22 : -22))}" y="${f1(below ? y + 32 : y - 24)}" text-anchor="${right ? 'start' : 'end'}" ${halo} font-family="Noto Sans SC, sans-serif" font-size="12" fill="#14181d">损失最大的一次：第 ${Math.floor(g.worst.t / 86400) + 1} 天${gap}${why}，白算 ${fmtTok(g.worst.lost)} token</text>`;
  }
  const nDays = Math.max(1, Math.ceil(g.t / 86400));
  const counts = Array.from({ length: nDays }, (_, i) => g.dayFails[i] || 0), max = Math.max(1, ...counts);
  const total = counts.reduce((a, b) => a + b, 0), bw = W / D;
  s += `<text x="0" y="226" font-family="Noto Sans SC, sans-serif" font-size="11" fill="#59636e">当天坏机次数（含链路抖动），一共 ${total} 次</text><line x1="0" y1="288" x2="${W}" y2="288" stroke="#cfd5db"></line>`;
  counts.forEach((c, i) => {
    if (!c) return;
    const h = c / max * 56, partial = i === nDays - 1 ? Math.max(0.12, (g.t / 86400) - i) : 1;
    s += `<rect x="${f1(i * bw + 1.5)}" y="${f1(288 - h)}" width="${f1(Math.max(2, (bw - 3) * Math.min(1, partial)))}" height="${f1(h)}" rx="1.5" fill="#e5383b"></rect>`;
  });
  const ticks = [];
  for (let dd = 0; dd < D; dd += D > 24 ? 10 : 5) if (D - dd >= 3) ticks.push(dd);
  ticks.forEach((dd) => { s += `<text x="${f1(dd / D * W)}" y="306" text-anchor="${dd ? 'middle' : 'start'}" font-family="IBM Plex Mono, monospace" font-size="11" fill="#59636e">${dd}</text>`; });
  s += `<text x="${W}" y="306" text-anchor="end" font-family="IBM Plex Mono, monospace" font-size="11" fill="#59636e">${D} 天</text>`;
  return `<svg viewBox="-52 -24 712 342" role="img" aria-label="训练时间线：${win ? `第 ${(g.t / 86400).toFixed(1)} 天训满` : `停在 ${pct(g.tokens / sc.tokens)}`}，期限第 ${D} 天；${g.spikes.length} 次 loss 尖峰，一共坏机 ${total} 次">${s}</svg>`;
}

// ---------- 主循环 ----------
let last = performance.now(), acc = 0, uiAcc = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (g && !$('ops').hidden) {
    if (!g.paused && !g.over) {
      acc += dt * SPEEDS[speedIdx] * 60;
      let n = Math.floor(acc);
      acc -= n;
      while (n-- > 0 && !g.paused && !g.over) tickMinute();
    }
    if (g && !g.over && !$('ops').hidden) {
      view.draw({
        nodes: g.nodes, phase: g.phase, progress: g.phase === 'run' ? 0 : 1 - g.busyLeft / Math.max(1, g.busyTotal),
        paused: g.paused, selected, hover, techs: g.tech ? [{ rack: Math.floor(g.tech.node / NODES_PER_RACK) }] : [],
        now: reduceMotion ? 0 : now,
      });
      uiAcc += dt;
      if (uiAcc > 0.15) { uiAcc = 0; renderHud(); }
    }
  }
  requestAnimationFrame(frame);
}

let resizeRaf = 0;
window.addEventListener('resize', () => {
  cancelAnimationFrame(resizeRaf);
  resizeRaf = requestAnimationFrame(() => {
    if (g && !$('ops').hidden) { layoutOps(); renderHud(); }
    if (!$('setup').hidden) layoutPreview();
  });
});

// 页面热更新时保留这一局
window.claude?.hot?.snapshot?.(() => ({ g, scIdx, cfg, speedIdx, reconfig, serve: serveSnapshot() }));

// 从推理部署切回训练：回到开局页
registerMode('train', () => { if (g && !g.over) showSetup(true); else { g = null; showSetup(false); } });

function boot(data) {
  buildSetup(); buildOps();
  if (serveBoot(data?.serve)) {
    if (data?.cfg && SCENARIOS[data.scIdx]) { scIdx = data.scIdx; cfg = data.cfg; }
  } else if (data && data.g && data.g.nodes && data.g.recent) {
    ({ g, scIdx, cfg, speedIdx } = data);
    ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
    L = hallLayout(g.cfg.gpus);
    if (g.over) { view.setLayout(L, { title: HALL, sub: '' }); showResult(); }
    else if (data.reconfig) showSetup(true);
    else showOps();
  } else {
    if (data && data.cfg && SCENARIOS[data.scIdx]) { scIdx = data.scIdx; cfg = data.cfg; }
    showOnly('setup');
    syncSetup(true);
  }
  // 字体到了以后重画一遍机房（墙上和地上的字）
  document.fonts?.ready?.then(() => {
    if (preview.L) layoutPreview();
    if (g && !$('ops').hidden) layoutOps();
  });
  requestAnimationFrame(frame);
}
window.claude?.hot?.ready ? window.claude.hot.ready(boot) : boot(window.claude?.hot?.data ?? {});
