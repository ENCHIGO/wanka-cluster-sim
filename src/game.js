// 万卡集群模拟器 · 游戏循环与界面
import {
  GPUS, NETWORKS, STORAGE, MODELS, SCENARIOS, GPU_COUNTS, TP_OPTIONS, PP_OPTIONS, EP_OPTIONS,
  GBS_OPTIONS, CKPT_OPTIONS, NODE_GPUS, REPAIR_WAIT_SEC, SEQ_LEN, evaluate, lossAt,
} from './sim.js';

const $ = (id) => document.getElementById(id);
const SPEEDS = [0.5, 2, 8, 24];          // 每真实秒推进的模拟小时
const STRAGGLER_SLOWDOWN = 0.62;
const SPIKE_PER_HOUR = 1 / 70;
const NODES_PER_RACK = 4;
const NODE_OK = 0, NODE_FAIL = 1, NODE_SLOW = 2;

// ---------- 格式化 ----------
const fmtTok = (t) => t >= 1e12 ? (t / 1e12).toFixed(2) + 'T' : t >= 1e9 ? (t / 1e9).toFixed(1) + 'B' : (t / 1e6).toFixed(0) + 'M';
const fmtMoney = (d) => d >= 1e6 ? '$' + (d / 1e6).toFixed(2) + 'M' : '$' + (d / 1e3).toFixed(0) + 'K';
const fmtDur = (sec) => sec < 90 ? sec.toFixed(0) + ' 秒' : sec < 5400 ? (sec / 60).toFixed(0) + ' 分钟' : (sec / 3600).toFixed(1) + ' 小时';
const fmtClock = (sec) => {
  const d = Math.floor(sec / 86400), h = Math.floor(sec % 86400 / 3600), m = Math.floor(sec % 3600 / 60);
  return `D${d + 1} ${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
};
const fmtInt = (n) => Math.floor(n).toLocaleString('en-US');
const pct = (x) => (x * 100).toFixed(1) + '%';

// ---------- 状态 ----------
let scIdx = 0;
let cfg = { ...SCENARIOS[0].defaults };
let ev = null;          // 当前配置的核算结果
let g = null;           // 进行中的一局
let speedIdx = 1;
let reconfig = false;   // 跑到一半回来改申请单
const formNo = `GPU-20261026-${String(Math.floor(Math.random() * 9000) + 1000)}`;

// ---------- 机柜命名：行字母 + 柜号 + 槽位，和平面图一致 ----------
const racksPerRow = (nodes) => { const r = Math.ceil(nodes / NODES_PER_RACK); return r <= 32 ? 8 : r <= 256 ? 16 : 20; };
function host(i) {
  const rack = Math.floor(i / NODES_PER_RACK), per = racksPerRow(g.nodes.length);
  return `${String.fromCharCode(65 + Math.floor(rack / per))}${String(rack % per + 1).padStart(2, '0')}-${i % NODES_PER_RACK + 1}`;
}

// ---------- 申请单 ----------
function fillSelect(el, items) {
  el.innerHTML = items.map(([v, label]) => `<option value="${v}">${label}</option>`).join('');
}

function buildSetup() {
  $('formNo').textContent = `编号 ${formNo}`;
  $('levels').innerHTML = SCENARIOS.map((sc, i) => `
    <label class="check"><input type="radio" name="task" value="${i}" ${i === scIdx ? 'checked' : ''}> ${sc.title}</label>`).join('');
  $('levels').querySelectorAll('input').forEach((r) => r.addEventListener('change', () => {
    scIdx = +r.value;
    cfg = { ...SCENARIOS[scIdx].defaults };
    syncForm();
  }));

  fillSelect($('fGpu'), Object.values(GPUS).map((x) => [x.id, x.name]));
  fillSelect($('fNet'), Object.values(NETWORKS).map((x) => [x.id, x.name]));
  fillSelect($('fStorage'), Object.values(STORAGE).map((x) => [x.id, x.name]));
  fillSelect($('fTp'), TP_OPTIONS.map((x) => [x, x]));
  fillSelect($('fPp'), PP_OPTIONS.map((x) => [x, x]));
  fillSelect($('fEp'), EP_OPTIONS.map((x) => [x, x]));
  fillSelect($('fGbs'), GBS_OPTIONS.map((x) => [x, `${x} 条 × 4096 = ${(x * SEQ_LEN / 2 ** 20).toFixed(0)}M token`]));
  fillSelect($('fCkpt'), CKPT_OPTIONS.map((x) => [x, x + ' 分钟']));
  fillSelect($('opsCkpt'), CKPT_OPTIONS.map((x) => [x, x + ' 分钟']));

  const bind = (id, key, conv) => $(id).addEventListener('change', (e) => { cfg[key] = conv(e.target); refreshForecast(); });
  bind('fGpu', 'gpu', (t) => t.value);
  bind('fGpus', 'gpus', (t) => +t.value);
  bind('fNet', 'net', (t) => t.value);
  bind('fStorage', 'storage', (t) => t.value);
  bind('fTp', 'tp', (t) => +t.value);
  bind('fPp', 'pp', (t) => +t.value);
  bind('fEp', 'ep', (t) => +t.value);
  bind('fGbs', 'gbs', (t) => +t.value);
  bind('fRecompute', 'recompute', (t) => t.checked);
  bind('fCkpt', 'ckptMin', (t) => +t.value);
  bind('fSpares', 'spares', (t) => Math.max(0, Math.min(64, Math.round(+t.value || 0))));
  $('startBtn').addEventListener('click', () => (reconfig ? applyReconfig() : startGame()));
  $('cancelBtn').addEventListener('click', () => { reconfig = false; g.paused = false; showOps(); });
}

function syncForm() {
  const sc = SCENARIOS[scIdx], model = MODELS[sc.model];
  $('levels').querySelectorAll('input').forEach((r) => { r.checked = +r.value === scIdx; r.disabled = reconfig && +r.value !== scIdx; });
  $('req').textContent = sc.requester;
  $('deliv').textContent = `交付 ${model.name}，训满 ${fmtTok(sc.tokens)} token`;
  $('limits').textContent = `${sc.deadlineDays} 天 / ${fmtMoney(sc.budget)}`;
  $('maxCards').textContent = `${sc.hall}，最多批 ${sc.maxGpus} 卡`;
  $('brief').textContent = sc.brief;
  fillSelect($('fGpus'), GPU_COUNTS.filter((n) => n <= sc.maxGpus).map((n) => [n, `${n} 卡 · ${n / NODE_GPUS} 台`]));
  $('epRow').hidden = !model.moe;
  $('fGpu').value = cfg.gpu; $('fGpus').value = cfg.gpus; $('fNet').value = cfg.net; $('fStorage').value = cfg.storage;
  $('fTp').value = cfg.tp; $('fPp').value = cfg.pp; $('fEp').value = cfg.ep; $('fGbs').value = cfg.gbs;
  $('fRecompute').checked = cfg.recompute; $('fCkpt').value = cfg.ckptMin; $('fSpares').value = cfg.spares;
  refreshForecast();
}

function refreshForecast() {
  const sc = SCENARIOS[scIdx];
  ev = evaluate(cfg, sc);
  const gpu = GPUS[cfg.gpu], net = NETWORKS[cfg.net], sto = STORAGE[cfg.storage];
  $('hGpu').textContent = `${gpu.tflops} TFLOPS，${gpu.hbm} GB，$${gpu.price}/卡时`;
  $('hGpus').textContent = `单卡约 ${(gpu.mtbf / 1000).toFixed(0)}k 小时坏一次`;
  $('hNet').textContent = `每卡实测 ${net.bw.toFixed(0)} GB/s，$${net.price}/卡时`;
  $('hStorage').textContent = `写 ${sto.bw} GB/s，$${sto.price}/小时`;
  $('hGbs').textContent = `这个模型的临界批次约 ${(MODELS[sc.model].critBatch / 2 ** 20).toFixed(0)}M token`;

  const remarks = [];
  const stamp = $('stamp');
  if (!ev.ok) {
    ev.errors.forEach((e) => remarks.push(`<li>${e}</li>`));
    $('fcKv').innerHTML = '<dt>核算</dt><dd class="over">配置不成立，算不出来</dd>';
    for (const id of ['stepBar', 'memBar']) $(id).innerHTML = '';
    $('stepTotal').textContent = ''; $('memTotal').textContent = ''; $('hCkpt').textContent = '';
    stamp.textContent = '驳回'; stamp.className = 'stamp';
  } else {
    const overDays = ev.etaDays > sc.deadlineDays, overCost = ev.estCost > sc.budget;
    $('fcKv').innerHTML = `
      <dt>切分</dt><dd>TP${cfg.tp} × PP${cfg.pp} × DP${ev.dp}${ev.model.moe ? `，EP${cfg.ep}` : ''}</dd>
      <dt>每步</dt><dd>${(ev.tokensPerStep / 2 ** 20).toFixed(0)}M token，${ev.m.toFixed(1)} 个微批</dd>
      <dt>MFU</dt><dd>${pct(ev.mfu)}</dd>
      <dt>吞吐</dt><dd>${(ev.rate / 1e6).toFixed(2)}M token/s</dd>
      <dt>全集群多久坏一次</dt><dd>${ev.mtbfHours.toFixed(1)} 小时</dd>
      <dt>存一次盘</dt><dd>${fmtDur(ev.ckptSec)}</dd>
      <dt>存盘和故障吃掉</dt><dd>${pct(ev.overhead)}</dd>
      <dt>工期</dt><dd class="${overDays ? 'over' : 'fine'}">${ev.etaDays.toFixed(1)} 天（限 ${sc.deadlineDays}）</dd>
      <dt>花费</dt><dd class="${overCost ? 'over' : 'fine'}">${fmtMoney(ev.estCost)}（限 ${fmtMoney(sc.budget)}）</dd>`;
    const b = ev.breakdown, total = ev.stepSec;
    const seg = (v, cls) => v / total > 0.004 ? `<i class="${cls}" style="width:${(v / total * 100).toFixed(2)}%" title="${v.toFixed(2)} 秒"></i>` : '';
    $('stepBar').innerHTML = seg(b.comp, 'p-solid') + seg(b.tp, 'p-hatch') + seg(b.ep, 'p-cross') + seg(b.bubble, 'p-hl') + seg(b.dp, 'p-blue');
    $('stepTotal').textContent = `${total.toFixed(2)} 秒`;
    const mm = ev.mem, cap = Math.max(mm.cap, mm.total);
    const ms = (v, cls) => `<i class="${cls}" style="width:${(v / cap * 100).toFixed(2)}%"></i>`;
    $('memBar').innerHTML = ms(mm.weights, 'p-solid') + ms(mm.opt, 'p-blue') + ms(mm.act, 'p-hl') + ms(mm.other, 'p-dot');
    $('memTotal').textContent = `${mm.total.toFixed(1)} / ${mm.cap} GB`;
    $('hCkpt').textContent = `按 Young 公式，${Math.round(ev.youngMin)} 分钟左右最划算`;
    ev.warnings.forEach((w) => remarks.push(`<li>${w}</li>`));
    if (!remarks.length) remarks.push('<li class="ok">无意见。</li>');
    const pending = overDays || overCost;
    stamp.textContent = pending ? '待定' : '同意';
    stamp.className = 'stamp ' + (pending ? 'pending' : 'ok');
  }
  $('remarks').innerHTML = remarks.join('');
  $('startBtn').disabled = !ev.ok;
  $('cancelBtn').hidden = !reconfig;
  $('startBtn').textContent = reconfig ? '按新配置重启' : '提交，开始训练';
}

// ---------- 一局 ----------
function newGame() {
  return {
    scIdx, cfg: { ...cfg }, t: 0, cost: 0, tokens: 0, ckptTokens: 0, sinceCkpt: 0,
    phase: 'restart', busyLeft: ev.restartSec, busyTotal: ev.restartSec, pendingEvict: null, failedNode: null,
    paused: false, over: null,
    nodes: new Array(cfg.gpus / NODE_GPUS).fill(NODE_OK),
    spares: cfg.spares, repairs: [],
    log: [], hist: [[0, 0]], lossHist: [],
    stats: { failures: 0, flaps: 0, ckpts: 0, lost: 0, stragglers: 0, spikes: 0 },
    event: null, spikeWatch: null, lastLossTok: 0,
  };
}

const step = () => fmtInt(g.tokens / ev.tokensPerStep);
const ckptStep = () => fmtInt(g.ckptTokens / ev.tokensPerStep);
const jobDesc = (c) => `${c.gpus} × ${c.gpu}，TP${c.tp} PP${c.pp} DP${ev.dp}${ev.model.moe ? ' EP' + c.ep : ''}`;

function setBusy(phase, sec) { g.phase = phase; g.busyLeft = sec; g.busyTotal = sec; }

function startGame() {
  g = newGame();
  log('info', `提交作业：${jobDesc(cfg)}。拉起进程组，加载数据。`);
  showOps();
}

function applyReconfig() {
  const old = g.cfg;
  g.cfg = { ...cfg };
  if (old.gpus !== cfg.gpus) g.nodes = new Array(cfg.gpus / NODE_GPUS).fill(NODE_OK);
  g.spares = Math.max(0, g.spares + (cfg.spares - old.spares));
  rollback();
  g.failedNode = null; g.pendingEvict = null;
  setBusy('restart', ev.restartSec);
  log('info', `改配置：${jobDesc(cfg)}。从 step ${ckptStep()} 重启。`);
  reconfig = false;
  g.paused = false;
  showOps();
}

function rollback() {
  const lost = g.tokens - g.ckptTokens;
  if (lost > 0) { g.stats.lost += lost; log('warn', `回滚到 step ${ckptStep()}，白算了 ${fmtTok(lost)} token。`); }
  g.tokens = g.ckptTokens;
  g.sinceCkpt = 0;
  g.lossHist = g.lossHist.filter((p) => p[0] <= g.tokens);
}

function log(kind, text) {
  g.log.unshift({ t: g.t, kind, text });
  if (g.log.length > 120) g.log.pop();
  dirtyLog = true;
}

const rnd = Math.random;
function stragglerCount() { let n = 0; for (const s of g.nodes) if (s === NODE_SLOW) n++; return n; }
function pickHealthyNode() {
  for (let k = 0; k < 50; k++) { const i = Math.floor(rnd() * g.nodes.length); if (g.nodes[i] !== NODE_FAIL) return i; }
  return 0;
}

// 推进一分钟模拟时间
function tickMinute() {
  const sc = SCENARIOS[g.scIdx];
  g.t += 60;
  g.cost += ev.costPerHour / 60;

  for (let i = g.repairs.length - 1; i >= 0; i--) {
    if (g.repairs[i] <= g.t) { g.repairs.splice(i, 1); g.spares++; log('good', '一台送修的机器修好了，回到备机池。'); }
  }

  const rate = ev.rate * (stragglerCount() > 0 ? STRAGGLER_SLOWDOWN : 1);
  let avail = 60;
  while (avail > 0 && !g.over) {
    if (g.phase === 'run') {
      const run = Math.min(avail, g.cfg.ckptMin * 60 - g.sinceCkpt);
      g.tokens += run * rate; g.sinceCkpt += run; avail -= run;
      if (g.sinceCkpt >= g.cfg.ckptMin * 60) setBusy('ckpt', ev.ckptSec);
    } else {
      const d = Math.min(avail, g.busyLeft);
      g.busyLeft -= d; avail -= d;
      if (g.busyLeft <= 1e-6) finishBusy();
    }
  }

  if (g.phase === 'run' || g.phase === 'ckpt') rollEvents();

  if (g.tokens - g.lastLossTok > sc.tokens / 400 || g.lossHist.length === 0) {
    g.lastLossTok = g.tokens;
    let l = lossAt(MODELS[sc.model].active, g.tokens) + (rnd() - 0.5) * 0.03;
    if (g.spikeWatch) l += 0.5 + rnd() * 0.4;
    g.lossHist.push([g.tokens, l]);
  }
  if (g.t % 1800 === 0) g.hist.push([g.t, g.tokens]);

  if (g.tokens >= sc.tokens) endGame('win');
  else if (g.cost >= sc.budget) endGame('budget');
  else if (g.t >= sc.deadlineDays * 86400) endGame('late');
}

function finishBusy() {
  if (g.phase === 'ckpt') {
    g.ckptTokens = g.tokens; g.sinceCkpt = 0; g.stats.ckpts++;
    if (g.pendingEvict != null) {
      const i = g.pendingEvict; g.pendingEvict = null;
      g.spares--; g.repairs.push(g.t + (24 + rnd() * 24) * 3600);
      g.nodes[i] = NODE_FAIL; g.failedNode = i;
      log('info', `存好了。${host(i)} 下线送修，备机顶上。`);
      setBusy('restart', ev.restartSec);
      return;
    }
    g.phase = 'run';
  } else if (g.phase === 'repair') {
    const name = g.failedNode != null ? host(g.failedNode) : '故障机';
    if (g.failedNode != null) g.nodes[g.failedNode] = NODE_OK;
    g.failedNode = null;
    log('info', `${name} 现场修好，重新拉起作业。`);
    setBusy('restart', ev.restartSec);
  } else if (g.phase === 'restart') {
    if (g.failedNode != null) g.nodes[g.failedNode] = NODE_OK;
    g.failedNode = null;
    g.phase = 'run';
    log('good', `作业起来了，从 step ${step()} 继续。`);
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
      log('warn', `${host(i)} 过热降频。同步训练要等最慢的那台，整体步时变成 ${(1 / STRAGGLER_SLOWDOWN).toFixed(1)} 倍。`);
    }
    return;
  }
  if (g.phase === 'run' && !g.spikeWatch && !g.event && rnd() < SPIKE_PER_HOUR / 60) return lossSpike();

  if (g.spikeWatch && g.t >= g.spikeWatch.until) {
    if (g.spikeWatch.diverge) {
      log('bad', 'loss 变成 NaN。尖峰之后存的 checkpoint 不能用。');
      g.ckptTokens = Math.min(g.ckptTokens, g.spikeWatch.safeTokens);
      rollback();
      setBusy('restart', ev.restartSec);
    } else log('good', 'loss 回落，继续。');
    g.spikeWatch = null;
  }
}

const FAULTS = ['Xid 79，GPU 掉卡', 'HBM 双比特 ECC 错误', 'NVLink 错误', '电源故障，整机掉电', '网卡 down'];

function hardwareFailure() {
  const i = pickHealthyNode();
  g.stats.failures++;
  g.nodes[i] = NODE_FAIL; g.failedNode = i; g.pendingEvict = null;
  log('bad', `${host(i)} ${FAULTS[Math.floor(rnd() * FAULTS.length)]}。NCCL 超时，作业退出。`);
  rollback();
  if (g.spares > 0) {
    g.spares--; g.repairs.push(g.t + (24 + rnd() * 48) * 3600);
    setBusy('restart', ev.restartSec);
    log('info', `换上备机，${host(i)} 送修。大约 ${fmtDur(ev.restartSec)} 后恢复。`);
  } else {
    setBusy('repair', REPAIR_WAIT_SEC * (0.7 + rnd() * 0.6));
    log('bad', `没有备机了。等现场工程师处理 ${host(i)}，大约 ${fmtDur(g.busyLeft)}。`);
  }
}

function linkFlap() {
  g.stats.flaps++;
  log('bad', 'IB 交换机端口 flap，NCCL 超时，作业退出。');
  rollback();
  setBusy('restart', ev.restartSec);
}

function lossSpike() {
  g.stats.spikes++;
  const before = g.lossHist.length ? g.lossHist[g.lossHist.length - 1][1] : 0;
  g.spikeWatch = { until: Infinity, diverge: false, safeTokens: g.ckptTokens };
  g.event = { kind: 'spike', step: step(), from: before, to: before + 0.6 + rnd() * 0.3 };
  g.paused = true;
  log('warn', `step ${g.event.step} loss 尖峰（${before.toFixed(2)} → ${g.event.to.toFixed(2)}）。暂停，等处理。`);
  renderEvent();
}

function resolveSpike(choice) {
  if (choice === 'rollback') {
    g.spikeWatch = null;
    log('info', '回滚，跳过这批数据。');
    rollback();
    setBusy('restart', ev.restartSec);
  } else {
    g.spikeWatch.until = g.t + (30 + rnd() * 60) * 60;
    g.spikeWatch.diverge = rnd() < 0.4;
    log('info', '先不回滚，盯着 loss。');
  }
  g.event = null; g.paused = false;
  renderEvent();
}

function evict(i) {
  if (!g || g.over || g.nodes[i] !== NODE_SLOW) return;
  if (g.phase !== 'run') { flashTip('作业没在跑，等它起来再换。'); return; }
  if (g.spares <= 0) { flashTip('没有备机可换。先加一台备机。'); return; }
  g.pendingEvict = i;
  setBusy('ckpt', ev.ckptSec);
  log('info', `${host(i)} 计划下线，先存盘。`);
}

function endGame(how) {
  const sc = SCENARIOS[g.scIdx];
  const win = how === 'win';
  g.over = { how };
  const goodput = g.tokens / (ev.rate * g.t);
  const cheap = g.cost <= sc.budget * 0.85, smooth = goodput >= 0.85;
  $('endNo').textContent = `${formNo} · ${sc.title}`;
  $('endStamp').textContent = win ? '按期交付' : how === 'late' ? '逾期' : '超支';
  $('endGrade').textContent = win ? (cheap && smooth ? '甲' : cheap || smooth ? '乙' : '丙') : '—';
  $('endGradeNote').textContent = win ? `省钱（≤ 预算 85%）${cheap ? '✓' : '✗'}　稳定（Goodput ≥ 85%）${smooth ? '✓' : '✗'}` : '未交付，不评级';
  $('endKv').innerHTML = `
    <dt>用时</dt><dd>${(g.t / 86400).toFixed(1)} 天（限 ${sc.deadlineDays}）</dd>
    <dt>花费</dt><dd>${fmtMoney(g.cost)}（限 ${fmtMoney(sc.budget)}）</dd>
    <dt>训练 token</dt><dd>${fmtTok(g.tokens)} / ${fmtTok(sc.tokens)}</dd>
    <dt>Goodput</dt><dd>${pct(goodput)}</dd>
    <dt>坏机 / 链路抖动</dt><dd>${g.stats.failures} / ${g.stats.flaps}</dd>
    <dt>降频 / loss 尖峰</dt><dd>${g.stats.stragglers} / ${g.stats.spikes}</dd>
    <dt>回滚白算的 token</dt><dd>${fmtTok(g.stats.lost)}</dd>
    <dt>存盘次数</dt><dd>${g.stats.ckpts}</dd>`;
  $('endShade').hidden = false;
  log(win ? 'good' : 'bad', win ? '训练完成，交付。' : how === 'late' ? '到期没训完，任务终止。' : '预算用完，任务终止。');
  renderOps();
}

// ---------- 值班界面 ----------
let dirtyLog = true;
let tipUntil = 0;
const DEFAULT_TIP = '鼠标移到机器上看编号。黄色是降频的机器，点它换下（要有备机）。';

function showOps() {
  $('setup').hidden = true; $('ops').hidden = false; $('endShade').hidden = true;
  $('opsCkpt').value = g.cfg.ckptMin;
  ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
  layoutPlan();
  renderEvent();
  dirtyLog = true;
  setTip('');
  renderOps();
}

function showSetup(isReconfig) {
  reconfig = isReconfig;
  if (isReconfig) { cfg = { ...g.cfg }; scIdx = g.scIdx; g.paused = true; }
  $('ops').hidden = true; $('setup').hidden = false; $('endShade').hidden = true;
  syncForm();
}

function buildOps() {
  $('speed').innerHTML = SPEEDS.map((s, i) => `<button data-i="${i}" aria-pressed="${i === speedIdx}" title="每秒 ${s} 模拟小时">${s < 1 ? '½' : s}h/s</button>`).join('');
  $('speed').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    speedIdx = +b.dataset.i;
    $('speed').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  }));
  $('pauseBtn').addEventListener('click', togglePause);
  $('ckptBtn').addEventListener('click', () => {
    if (g.phase === 'run') { setBusy('ckpt', ev.ckptSec); log('info', '手动存盘。'); }
  });
  $('spareAdd').addEventListener('click', () => {
    g.spares++; g.cfg.spares++; ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
    log('info', `加租一台备机，现在每小时 ${fmtMoney(ev.costPerHour)}。`);
  });
  $('spareDel').addEventListener('click', () => {
    if (g.spares <= 0) return;
    g.spares--; g.cfg.spares--; ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
    log('info', `退掉一台备机，现在每小时 ${fmtMoney(ev.costPerHour)}。`);
  });
  $('opsCkpt').addEventListener('change', (e) => {
    g.cfg.ckptMin = +e.target.value; ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
    log('info', `存盘间隔改成 ${g.cfg.ckptMin} 分钟。`);
  });
  $('reconfBtn').addEventListener('click', () => showSetup(true));
  $('againBtn').addEventListener('click', () => { scIdx = g.scIdx; cfg = { ...g.cfg }; ev = evaluate(cfg, SCENARIOS[scIdx]); startGame(); });
  $('menuBtn').addEventListener('click', () => { const keep = g; g = null; scIdx = keep.scIdx; cfg = { ...keep.cfg }; showSetup(false); renderTop(); });

  const grid = $('grid');
  grid.addEventListener('mousemove', (e) => {
    const i = nodeAt(e);
    if (i == null || !g) { setTip(''); return; }
    const s = g.nodes[i];
    setTip(`${host(i)}　GPU ${i * 8}–${i * 8 + 7}　${s === NODE_FAIL ? '故障' : s === NODE_SLOW ? '降频，点击换下' : '正常'}`);
  });
  grid.addEventListener('mouseleave', () => setTip(''));
  grid.addEventListener('click', (e) => { const i = nodeAt(e); if (i != null) evict(i); });
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && g && !$('ops').hidden && e.target === document.body) { e.preventDefault(); togglePause(); }
  });
}

function togglePause() {
  if (!g || g.over || g.event) return;
  g.paused = !g.paused;
}

function setTip(text) {
  if (Date.now() < tipUntil) return;
  $('tip').textContent = text || DEFAULT_TIP;
}
function flashTip(text) { tipUntil = 0; setTip(text); tipUntil = Date.now() + 2500; }

function renderEvent() {
  const box = $('eventBox');
  if (!g || !g.event) { box.hidden = true; box.innerHTML = ''; return; }
  const e = g.event;
  box.hidden = false;
  box.innerHTML = `<div class="ask">
    <div class="ask-title">请示</div>
    <p>step ${e.step} loss 从 ${e.from.toFixed(2)} 跳到 ${e.to.toFixed(2)}，可能是一批脏数据。已暂停。</p>
    <div class="opts">
      <button class="opt" id="evRollback"><b>回滚，跳过这批数据</b><span>回到 step ${ckptStep()}，重启要 ${fmtDur(ev.restartSec)}。</span></button>
      <button class="opt" id="evContinue"><b>先不动，看看</b><span>多半会自己回落。回不来就变 NaN，尖峰之后存的盘都作废。</span></button>
    </div>
  </div>`;
  $('evRollback').addEventListener('click', () => resolveSpike('rollback'));
  $('evContinue').addEventListener('click', () => resolveSpike('continue'));
}

const PHASE_LABEL = { run: '运行中', ckpt: '存盘中', restart: '重启中', repair: '停机待修' };

function renderTop() {
  const sc = SCENARIOS[g ? g.scIdx : scIdx];
  const t = g ? g.t : 0, cost = g ? g.cost : 0, tok = g ? g.tokens : 0;
  $('mTime').textContent = `${(t / 86400).toFixed(1)} / ${sc.deadlineDays} 天`;
  $('mCost').textContent = `${fmtMoney(cost)} / ${fmtMoney(sc.budget)}`;
  $('mTok').textContent = `${fmtTok(tok)} / ${fmtTok(sc.tokens)}`;
  $('bTime').style.width = Math.min(100, t / (sc.deadlineDays * 86400) * 100) + '%';
  $('bCost').style.width = Math.min(100, cost / sc.budget * 100) + '%';
  $('bTok').style.width = Math.min(100, tok / sc.tokens * 100) + '%';
}

function renderOps() {
  if (!g) return;
  renderTop();
  const sc = SCENARIOS[g.scIdx];
  const st = $('state');
  st.textContent = g.paused ? (g.event ? '等你决定' : '暂停') : PHASE_LABEL[g.phase];
  st.className = 'state ' + (g.paused ? 'paused' : g.phase);
  const busy = g.phase !== 'run' && !g.paused ? `还要 ${fmtDur(g.busyLeft)}　` : '';
  $('stateSub').textContent = `${busy}${fmtClock(g.t)}　step ${step()}`;
  $('hall').textContent = `${sc.hall} · ${Math.ceil(g.nodes.length / NODES_PER_RACK)} 柜 ${g.nodes.length} 台 ${g.cfg.gpus} 卡 ${g.cfg.gpu}`;
  $('dutyNo').textContent = `${formNo}`;
  $('pauseBtn').textContent = g.paused ? '继续' : '暂停';
  $('pauseBtn').disabled = !!g.over || !!g.event;
  $('ckptBtn').disabled = g.phase !== 'run' || !!g.over;
  $('spareDel').disabled = g.spares <= 0 || !!g.over;
  $('spareAdd').disabled = !!g.over;
  $('reconfBtn').disabled = !!g.over || !!g.event;

  const boxes = (n, cls) => `<span class="boxes ${cls}">${'<i></i>'.repeat(Math.min(n, 16))}</span>${n > 16 ? ' +' + (n - 16) : ''}`;
  $('pool').innerHTML = `<span>备机 ${g.spares} 台${boxes(g.spares, 'spare')}</span><span>送修 ${g.repairs.length} 台${boxes(g.repairs.length, 'rma')}</span>`;

  const slow = stragglerCount();
  const goodput = g.tokens / (ev.rate * Math.max(g.t, 1));
  const etaLeft = (sc.tokens - g.tokens) / (ev.rate * (1 - ev.overhead)) / 86400;
  const k = slow ? STRAGGLER_SLOWDOWN : 1;
  $('liveKv').innerHTML = `
    <dt>步时 / MFU</dt><dd>${(ev.stepSec / k).toFixed(2)} 秒 / ${pct(ev.mfu * k)}</dd>
    <dt>降频机器</dt><dd class="${slow ? 'over' : ''}">${slow ? slow + ' 台' : '无'}</dd>
    <dt>Goodput</dt><dd>${g.t > 3600 ? pct(goodput) : '—'}</dd>
    <dt>上次存盘</dt><dd>${fmtDur(g.sinceCkpt)}前，step ${ckptStep()}</dd>
    <dt>坏机 / 抖动 / 尖峰</dt><dd>${g.stats.failures} / ${g.stats.flaps} / ${g.stats.spikes}</dd>
    <dt>每小时</dt><dd>${fmtMoney(ev.costPerHour)}</dd>
    <dt>照这个速度还要</dt><dd class="${g.t / 86400 + etaLeft > sc.deadlineDays ? 'over' : ''}">${etaLeft.toFixed(1)} 天</dd>`;

  if (dirtyLog) {
    $('log').innerHTML = g.log.map((l) => `<li class="${l.kind}"><time>${fmtClock(l.t)}</time><span>${l.text}</span></li>`).join('');
    dirtyLog = false;
  }
  drawProgress(); drawLoss();
}

// ---------- 机房平面图 ----------
const css = getComputedStyle(document.documentElement);
const C = Object.fromEntries(['ink', 'ink-2', 'red', 'blue', 'hl', 'rule', 'rule-soft', 'field', 'paper', 'umber']
  .map((k) => [k, css.getPropertyValue('--' + k).trim()]));
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

let plan = null;
function layoutPlan() {
  const canvas = $('grid');
  const avail = Math.max(300, canvas.parentElement.clientWidth);
  const nodes = g.nodes.length, racks = Math.ceil(nodes / NODES_PER_RACK), per = racksPerRow(nodes);
  const rows = Math.ceil(racks / per);
  const L = 22, T = 18, R = 8, B = 8;
  let c = 12, pad, gx, rackW;
  for (; c > 1.5; c -= 0.5) {
    pad = Math.max(1, Math.round(c * 0.4)); gx = Math.max(2, Math.round(c * 0.7));
    rackW = 8 * c + 2 * pad;
    if (L + per * rackW + (per - 1) * gx + R <= avail) break;
  }
  const rackH = NODES_PER_RACK * c + 2 * pad;
  const hotGap = Math.max(3, Math.round(c * 0.8)), coldGap = Math.max(12, Math.round(c * 2.6));
  const rowY = [];
  let y = T + coldGap;
  for (let r = 0; r < rows; r++) {
    rowY.push(y);
    y += rackH + (r % 2 === 0 ? hotGap : coldGap);
  }
  const W = Math.ceil(L + per * rackW + (per - 1) * gx + R);
  const H = Math.ceil(y + B - (rows % 2 ? hotGap : coldGap) + coldGap);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const nodeRect = [];
  for (let i = 0; i < nodes; i++) {
    const rack = Math.floor(i / NODES_PER_RACK), slot = i % NODES_PER_RACK;
    const x = L + (rack % per) * (rackW + gx) + pad, yy = rowY[Math.floor(rack / per)] + pad + slot * c;
    nodeRect.push([x, yy]);
  }
  plan = { W, H, L, T, c, pad, gx, rackW, rackH, per, rows, rowY, racks, coldGap, hotGap, dpr, nodeRect,
    noise: Float32Array.from({ length: g.cfg.gpus }, () => rnd() * Math.PI * 2) };
}

function nodeAt(e) {
  if (!plan || !g) return null;
  const r = $('grid').getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const { nodeRect, c } = plan;
  for (let i = 0; i < nodeRect.length; i++) {
    const [nx, ny] = nodeRect[i];
    if (x >= nx - 1 && x < nx + 8 * c + 1 && y >= ny && y < ny + c) return i;
  }
  return null;
}

function drawPlan(now) {
  if (!g || !plan) return;
  const ctx = $('grid').getContext('2d');
  const { W, H, L, T, c, pad, gx, rackW, rackH, per, rows, rowY, racks, coldGap, dpr, nodeRect, noise } = plan;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = C.field; ctx.fillRect(0, 0, W, H);

  // 墙、行号、柜号、通道
  ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
  ctx.strokeRect(L - 6, T, W - L + 2, H - T - 2);
  ctx.font = '10px "IBM Plex Mono", monospace'; ctx.fillStyle = C['ink-2'];
  ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  for (let k = 0; k < per; k++) if (k === 0 || (k + 1) % 5 === 0) ctx.fillText(String(k + 1).padStart(2, '0'), L + k * (rackW + gx) + rackW / 2, T - 2);
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let r = 0; r < rows; r++) ctx.fillText(String.fromCharCode(65 + r), L - 9, rowY[r] + rackH / 2);
  if (coldGap >= 14) {
    ctx.textAlign = 'left'; ctx.fillStyle = C.blue; ctx.font = '10px "Noto Sans SC", sans-serif';
    for (let r = 0; r <= rows; r += 2) {
      const yy = r === 0 ? T + coldGap / 2 : rowY[r - 1] + rackH + coldGap / 2;
      ctx.fillText('冷通道 ↓', L, yy);
    }
  }
  // 机柜外框
  ctx.lineWidth = 1; ctx.strokeStyle = C['ink-2'];
  for (let k = 0; k < racks; k++) {
    const x = L + (k % per) * (rackW + gx), y = rowY[Math.floor(k / per)];
    ctx.strokeRect(x + 0.5, y + 0.5, rackW - 1, rackH - 1);
  }

  const tsec = reduceMotion ? 0 : now / 1000;
  const blink = reduceMotion || Math.sin(tsec * 6) > 0;
  const frac = g.phase === 'run' ? 0 : 1 - g.busyLeft / Math.max(1, g.busyTotal);
  const cell = c >= 3 ? c - 1 : c;
  for (let n = 0; n < nodeRect.length; n++) {
    const [x0, y0] = nodeRect[n];
    const s = g.nodes[n];
    const rowFrac = (y0 - T) / (H - T);
    if (s === NODE_SLOW) { ctx.globalAlpha = 0.9; ctx.fillStyle = C.hl; ctx.fillRect(x0 - pad + 1, y0 - 0.5, 8 * c + 2 * pad - 2, c + 0.5); }
    for (let k = 0; k < NODE_GPUS; k++) {
      let color, alpha = 1;
      if (s === NODE_FAIL) { color = C.red; alpha = blink ? 1 : 0.35; }
      else if (s === NODE_SLOW) { color = C.ink; alpha = 0.5; }
      else if (g.phase === 'run') { color = C.ink; alpha = g.paused ? 0.45 : 0.6 + 0.35 * Math.sin(tsec * 2.4 + noise[n * NODE_GPUS + k]); }
      else if (g.phase === 'ckpt') { color = rowFrac <= frac ? C.blue : C.ink; alpha = 0.8; }
      else { color = rowFrac <= frac ? C['ink-2'] : C['rule-soft']; alpha = rowFrac <= frac ? 0.45 : 1; }
      ctx.globalAlpha = alpha; ctx.fillStyle = color;
      ctx.fillRect(x0 + k * c, y0, cell, cell);
    }
  }
  ctx.globalAlpha = 1;
  // 故障机画个红圈
  if (g.failedNode != null && nodeRect[g.failedNode]) {
    const [x0, y0] = nodeRect[g.failedNode];
    ctx.strokeStyle = C.red; ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.ellipse(x0 + 4 * c, y0 + c / 2, 4 * c + 7, c / 2 + 6, -0.06, 0, Math.PI * 2); ctx.stroke();
  }
}

// ---------- 坐标纸小图 ----------
function prepChart(id) {
  const cv = $(id), dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = C.field; ctx.fillRect(0, 0, w, h);
  ctx.font = '10px "IBM Plex Mono", monospace';
  return { ctx, w, h };
}

function graphPaper(ctx, L, T, pw, ph) {
  ctx.lineWidth = 1;
  for (let x = 0; x <= pw; x += 8) { ctx.strokeStyle = x % 40 === 0 ? C.rule : C['rule-soft']; ctx.globalAlpha = x % 40 === 0 ? 0.7 : 0.5; ctx.beginPath(); ctx.moveTo(L + x + 0.5, T); ctx.lineTo(L + x + 0.5, T + ph); ctx.stroke(); }
  for (let y = 0; y <= ph; y += 8) { ctx.strokeStyle = y % 40 === 0 ? C.rule : C['rule-soft']; ctx.globalAlpha = y % 40 === 0 ? 0.7 : 0.5; ctx.beginPath(); ctx.moveTo(L, T + ph - y - 0.5); ctx.lineTo(L + pw, T + ph - y - 0.5); ctx.stroke(); }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = C.ink; ctx.beginPath(); ctx.moveTo(L + 0.5, T); ctx.lineTo(L + 0.5, T + ph + 0.5); ctx.lineTo(L + pw, T + ph + 0.5); ctx.stroke();
}

function drawProgress() {
  const sc = SCENARIOS[g.scIdx];
  const { ctx, w, h } = prepChart('progChart');
  const L = 38, R = 8, T = 6, B = 16, pw = w - L - R, ph = h - T - B;
  graphPaper(ctx, L, T, pw, ph);
  const X = (t) => L + Math.min(1.2, t / (sc.deadlineDays * 86400)) * pw, Y = (tok) => T + ph - tok / sc.tokens * ph;
  ctx.fillStyle = C['ink-2']; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText(fmtTok(sc.tokens), L - 4, Y(sc.tokens)); ctx.fillText('0', L - 4, Y(0));
  ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left'; ctx.fillText('D1', L, h - 3);
  ctx.textAlign = 'right'; ctx.fillStyle = C.red; ctx.fillText(`期限 D${sc.deadlineDays}`, L + pw, h - 3);
  ctx.strokeStyle = C.red; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(L + pw - 0.5, T); ctx.lineTo(L + pw - 0.5, T + ph); ctx.stroke();
  // 预计线
  const left = sc.tokens - g.tokens, eta = g.t + left / (ev.rate * (1 - ev.overhead));
  ctx.setLineDash([4, 3]); ctx.strokeStyle = C['ink-2'];
  ctx.beginPath(); ctx.moveTo(X(g.t), Y(g.tokens)); ctx.lineTo(X(eta), Y(sc.tokens)); ctx.stroke();
  ctx.setLineDash([]);
  const pts = [...g.hist, [g.t, g.tokens]];
  ctx.beginPath(); pts.forEach(([t, k], i) => (i ? ctx.lineTo(X(t), Y(k)) : ctx.moveTo(X(t), Y(k))));
  ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.stroke();
  ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(X(g.t), Y(g.tokens), 2.6, 0, Math.PI * 2); ctx.fill();
}

function drawLoss() {
  const sc = SCENARIOS[g.scIdx];
  const { ctx, w, h } = prepChart('lossChart');
  const L = 38, R = 8, T = 6, B = 16, pw = w - L - R, ph = h - T - B;
  graphPaper(ctx, L, T, pw, ph);
  const active = MODELS[sc.model].active;
  const hi = lossAt(active, sc.tokens / 400) + 0.3, lo = lossAt(active, sc.tokens) - 0.05;
  const X = (tok) => L + tok / sc.tokens * pw, Y = (l) => T + Math.max(0, Math.min(ph, (hi - l) / (hi - lo) * ph));
  ctx.fillStyle = C['ink-2']; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  [lo, hi].forEach((v) => ctx.fillText(v.toFixed(2), L - 4, Y(v)));
  ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left'; ctx.fillText('0', L, h - 3);
  ctx.textAlign = 'right'; ctx.fillText(fmtTok(sc.tokens) + ' token', L + pw, h - 3);
  if (!g.lossHist.length) return;
  ctx.beginPath(); g.lossHist.forEach(([k, l], i) => (i ? ctx.lineTo(X(k), Y(l)) : ctx.moveTo(X(k), Y(l))));
  ctx.strokeStyle = C.red; ctx.lineWidth = 1.4; ctx.stroke();
  const [k, l] = g.lossHist[g.lossHist.length - 1];
  ctx.fillStyle = C.red; ctx.beginPath(); ctx.arc(X(k), Y(l), 2.6, 0, Math.PI * 2); ctx.fill();
  ctx.textAlign = 'left'; ctx.fillStyle = C.ink; ctx.fillText(l.toFixed(3), Math.min(X(k) + 6, L + pw - 34), Math.max(T + 8, Y(l) - 7));
}

// ---------- 主循环 ----------
let last = performance.now();
let acc = 0, uiAcc = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (g && !$('ops').hidden) {
    if (!g.paused && !g.over) {
      acc += dt * SPEEDS[speedIdx] * 60;            // 模拟分钟
      let n = Math.floor(acc);
      acc -= n;
      while (n-- > 0 && !g.paused && !g.over) tickMinute();
    }
    drawPlan(now);
    uiAcc += dt;
    if (uiAcc > 0.15) { uiAcc = 0; renderOps(); }
  }
  requestAnimationFrame(frame);
}

window.addEventListener('resize', () => { if (g && !$('ops').hidden) layoutPlan(); });

// 页面更新时保留这一局
window.claude?.hot?.snapshot?.(() => ({ g, scIdx, cfg, speedIdx, reconfig }));

function boot(data) {
  buildSetup(); buildOps();
  if (data && data.g && data.g.nodes) {
    ({ g, scIdx, cfg, speedIdx } = data);
    if (g.busyTotal == null) g.busyTotal = g.busyLeft || 1;
    $('speed').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(+x.dataset.i === speedIdx)));
    if (data.reconfig) showSetup(true); else { ev = evaluate(g.cfg, SCENARIOS[g.scIdx]); showOps(); }
  } else {
    if (data && data.cfg) { scIdx = data.scIdx; cfg = data.cfg; }
    $('ops').hidden = true;
    syncForm();
    renderTop();
  }
  requestAnimationFrame(frame);
}
window.claude?.hot?.ready ? window.claude.hot.ready(boot) : boot(window.claude?.hot?.data ?? {});
