// 万卡集群模拟器 · 游戏循环与界面
import {
  GPUS, NETWORKS, STORAGE, MODELS, SCENARIOS, GPU_COUNTS, TP_OPTIONS, PP_OPTIONS, EP_OPTIONS,
  GBS_OPTIONS, CKPT_OPTIONS, NODE_GPUS, REPAIR_WAIT_SEC, SEQ_LEN, evaluate, lossAt,
} from './sim.js';

const $ = (id) => document.getElementById(id);
const SPEEDS = [0.5, 2, 8, 24];          // 每真实秒推进的模拟小时
const STRAGGLER_SLOWDOWN = 0.62;
const SPIKE_PER_HOUR = 1 / 70;
const NODE_OK = 0, NODE_FAIL = 1, NODE_SLOW = 2;

// ---------- 格式化 ----------
const fmtTok = (t) => t >= 1e12 ? (t / 1e12).toFixed(2) + 'T' : t >= 1e9 ? (t / 1e9).toFixed(1) + 'B' : (t / 1e6).toFixed(0) + 'M';
const fmtMoney = (d) => d >= 1e6 ? '$' + (d / 1e6).toFixed(2) + 'M' : '$' + (d / 1e3).toFixed(0) + 'K';
const fmtDur = (sec) => sec < 90 ? sec.toFixed(0) + ' 秒' : sec < 5400 ? (sec / 60).toFixed(0) + ' 分钟' : (sec / 3600).toFixed(1) + ' 小时';
const fmtClock = (sec) => {
  const d = Math.floor(sec / 86400), h = Math.floor(sec % 86400 / 3600), m = Math.floor(sec % 3600 / 60);
  return `D${d + 1} ${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
};
const pct = (x) => (x * 100).toFixed(1) + '%';

// ---------- 状态 ----------
let scIdx = 0;
let cfg = { ...SCENARIOS[0].defaults };
let ev = null;          // 当前配置的评估结果
let g = null;           // 运行中的一局
let speedIdx = 1;
let reconfig = false;   // 运行中回到配置台

// ---------- 配置台 ----------
function fillSelect(el, items, fmt = (x) => x) {
  el.innerHTML = items.map(([v, label]) => `<option value="${v}">${fmt(label)}</option>`).join('');
}

function buildSetup() {
  $('levels').innerHTML = SCENARIOS.map((sc, i) => `
    <button class="level" data-i="${i}" aria-pressed="${i === scIdx}">
      <span class="t">${sc.title}</span>
      <span class="s">${MODELS[sc.model].name} · ${fmtTok(sc.tokens)} tok</span>
      <span class="s">${sc.deadlineDays} 天 · ${fmtMoney(sc.budget)} · ≤${sc.maxGpus} 卡</span>
    </button>`).join('');
  $('levels').querySelectorAll('.level').forEach((b) => b.addEventListener('click', () => {
    if (reconfig) return;
    scIdx = +b.dataset.i;
    cfg = { ...SCENARIOS[scIdx].defaults };
    syncForm();
  }));

  fillSelect($('fGpu'), Object.values(GPUS).map((x) => [x.id, x.name]));
  fillSelect($('fNet'), Object.values(NETWORKS).map((x) => [x.id, x.name]));
  fillSelect($('fStorage'), Object.values(STORAGE).map((x) => [x.id, x.name]));
  fillSelect($('fTp'), TP_OPTIONS.map((x) => [x, x]));
  fillSelect($('fPp'), PP_OPTIONS.map((x) => [x, x]));
  fillSelect($('fEp'), EP_OPTIONS.map((x) => [x, x]));
  fillSelect($('fGbs'), GBS_OPTIONS.map((x) => [x, `${x}（${(x * SEQ_LEN / 2 ** 20).toFixed(0)}M token）`]));
  fillSelect($('fCkpt'), CKPT_OPTIONS.map((x) => [x, x + ' 分钟']));
  fillSelect($('opsCkpt'), CKPT_OPTIONS.map((x) => [x, x + ' 分钟']));

  const bind = (id, key, conv) => $(id).addEventListener('change', (e) => {
    cfg[key] = conv(e.target);
    refreshForecast();
  });
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
  const sc = SCENARIOS[scIdx];
  $('levels').querySelectorAll('.level').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.i === scIdx)));
  $('brief').textContent = sc.brief;
  $('modelName').textContent = `${MODELS[sc.model].name} · 目标 ${fmtTok(sc.tokens)} token`;
  fillSelect($('fGpus'), GPU_COUNTS.filter((n) => n <= sc.maxGpus).map((n) => [n, `${n} 卡（${n / NODE_GPUS} 节点）`]));
  $('epField').hidden = !MODELS[sc.model].moe;
  $('fGpu').value = cfg.gpu; $('fGpus').value = cfg.gpus; $('fNet').value = cfg.net; $('fStorage').value = cfg.storage;
  $('fTp').value = cfg.tp; $('fPp').value = cfg.pp; $('fEp').value = cfg.ep; $('fGbs').value = cfg.gbs;
  $('fRecompute').checked = cfg.recompute; $('fCkpt').value = cfg.ckptMin; $('fSpares').value = cfg.spares;
  refreshForecast();
}

function refreshForecast() {
  const sc = SCENARIOS[scIdx];
  ev = evaluate(cfg, sc);
  const gpu = GPUS[cfg.gpu], net = NETWORKS[cfg.net], sto = STORAGE[cfg.storage];
  $('hGpu').textContent = `${gpu.tflops} TFLOPS · ${gpu.hbm} GB · $${gpu.price}/卡时`;
  $('hGpus').textContent = `单卡 MTBF ≈ ${(gpu.mtbf / 1000).toFixed(0)}k 小时`;
  $('hNet').textContent = `${net.bw.toFixed(0)} GB/s 有效 · $${net.price}/卡时`;
  $('hStorage').textContent = `${sto.bw} GB/s · $${sto.price}/小时`;
  $('hGbs').textContent = `临界批次约 ${(MODELS[sc.model].critBatch / 2 ** 20).toFixed(0)}M token`;

  const notes = [];
  if (!ev.ok) {
    ev.errors.forEach((e) => notes.push(`<li class="err">${e}</li>`));
    $('fcKv').innerHTML = '<dt>配置无效</dt><dd class="bad">无法启动</dd>';
    $('stepStack').innerHTML = ''; $('memStack').innerHTML = '';
    $('stepTotal').textContent = ''; $('memTotal').textContent = '';
    $('hCkpt').textContent = '';
  } else {
    const okDays = ev.etaDays <= sc.deadlineDays, okCost = ev.estCost <= sc.budget;
    $('fcKv').innerHTML = `
      <dt>并行切分</dt><dd>TP${cfg.tp} × PP${cfg.pp} × DP${ev.dp}${ev.model.moe ? ` · EP${cfg.ep}` : ''}</dd>
      <dt>每步 token</dt><dd>${(ev.tokensPerStep / 2 ** 20).toFixed(0)}M · 微批 ${ev.m.toFixed(1)}</dd>
      <dt>MFU</dt><dd class="${ev.mfu > 0.4 ? 'good' : ev.mfu < 0.25 ? 'bad' : ''}">${pct(ev.mfu)}</dd>
      <dt>有效吞吐</dt><dd>${(ev.rate / 1e6).toFixed(2)}M tok/s</dd>
      <dt>集群平均故障间隔</dt><dd class="${ev.mtbfHours < 6 ? 'warn' : ''}">${ev.mtbfHours.toFixed(1)} 小时</dd>
      <dt>存一次 checkpoint</dt><dd>${fmtDur(ev.ckptSec)}</dd>
      <dt>容错开销（预期）</dt><dd>${pct(ev.overhead)}</dd>
      <dt>预计完成</dt><dd class="${okDays ? 'good' : 'bad'}">${ev.etaDays.toFixed(1)} / ${sc.deadlineDays} 天</dd>
      <dt>预计花费</dt><dd class="${okCost ? 'good' : 'bad'}">${fmtMoney(ev.estCost)} / ${fmtMoney(sc.budget)}</dd>`;
    const b = ev.breakdown, total = ev.stepSec;
    const seg = (v, c) => `<i style="width:${(v / total * 100).toFixed(2)}%;background:${c}" title="${v.toFixed(2)}s"></i>`;
    $('stepStack').innerHTML = seg(b.comp, 'var(--ok)') + seg(b.tp, '#9fd3c7') + seg(b.ep, '#c792ea') + seg(b.bubble, 'var(--warn)') + seg(b.dp, 'var(--ckpt)');
    $('stepTotal').textContent = `${total.toFixed(2)} 秒/步`;
    const mm = ev.mem, cap = Math.max(mm.cap, mm.total);
    const ms = (v, c) => `<i style="width:${(v / cap * 100).toFixed(2)}%;background:${c}"></i>`;
    $('memStack').innerHTML = ms(mm.weights, 'var(--ok)') + ms(mm.opt, 'var(--ckpt)') + ms(mm.act, 'var(--warn)') + ms(mm.other, 'var(--muted)');
    $('memTotal').textContent = `${mm.total.toFixed(1)} / ${mm.cap} GB`;
    $('hCkpt').textContent = `按 Young 公式约 ${Math.round(ev.youngMin)} 分钟最省`;
    ev.warnings.forEach((w) => notes.push(`<li>${w}</li>`));
    if (!ev.warnings.length) notes.push('<li class="fine">配置看起来很健康。剩下的交给运气和值班的你。</li>');
  }
  $('notes').innerHTML = notes.join('');
  $('startBtn').disabled = !ev.ok;
  $('cancelBtn').hidden = !reconfig;
  $('startBtn').textContent = reconfig ? `应用新配置（从最近 checkpoint 重启）` : '开始训练';
}

// ---------- 一局 ----------
function newGame(sc, cfg, ev) {
  return {
    scIdx, cfg: { ...cfg }, t: 0, cost: 0, tokens: 0, ckptTokens: 0, sinceCkpt: 0,
    phase: 'restart', busyLeft: ev.restartSec, pendingEvict: null, failedNode: null,
    paused: false, over: null,
    nodes: new Array(cfg.gpus / NODE_GPUS).fill(NODE_OK),
    spares: cfg.spares, repairs: [],
    log: [], hist: [[0, 0]], lossHist: [],
    stats: { failures: 0, flaps: 0, ckpts: 0, lost: 0, stragglers: 0, spikes: 0 },
    event: null, spikeWatch: null, lastLossTok: 0,
  };
}

function startGame() {
  const sc = SCENARIOS[scIdx];
  g = newGame(sc, cfg, ev);
  log('info', `作业提交：${cfg.gpus} 张 ${cfg.gpu}，TP${cfg.tp}/PP${cfg.pp}/DP${ev.dp}${ev.model.moe ? '/EP' + cfg.ep : ''}。正在拉起进程组。`);
  showOps();
}

function applyReconfig() {
  const old = g.cfg;
  g.cfg = { ...cfg };
  // 卡数变化：节点墙按新规模重建，坏掉的节点视为已被调度器剔除
  if (old.gpus !== cfg.gpus) g.nodes = new Array(cfg.gpus / NODE_GPUS).fill(NODE_OK);
  g.spares = Math.max(0, g.spares + (cfg.spares - old.spares));
  rollback('改配置');
  g.phase = 'restart'; g.busyLeft = ev.restartSec; g.failedNode = null; g.pendingEvict = null;
  log('info', `应用新配置：${cfg.gpus} 卡 TP${cfg.tp}/PP${cfg.pp}/DP${ev.dp}，从最近 checkpoint 重启。`);
  reconfig = false;
  g.paused = false;
  showOps();
}

function rollback(reason) {
  const lost = g.tokens - g.ckptTokens;
  if (lost > 0) { g.stats.lost += lost; log('warn', `${reason}：回滚到 ${fmtTok(g.ckptTokens)}，丢失 ${fmtTok(lost)} token 的进度。`); }
  g.tokens = g.ckptTokens;
  g.sinceCkpt = 0;
  g.lossHist = g.lossHist.filter((p) => p[0] <= g.tokens);
}

function log(kind, text) {
  g.log.unshift({ t: g.t, kind, text });
  if (g.log.length > 80) g.log.pop();
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
    if (g.repairs[i] <= g.t) { g.repairs.splice(i, 1); g.spares++; log('good', '一台节点维修完成，回到热备池。'); }
  }

  const slow = stragglerCount() > 0 ? STRAGGLER_SLOWDOWN : 1;
  const rate = ev.rate * slow;
  let avail = 60;
  while (avail > 0 && !g.over) {
    if (g.phase === 'run') {
      const run = Math.min(avail, g.cfg.ckptMin * 60 - g.sinceCkpt);
      g.tokens += run * rate; g.sinceCkpt += run; avail -= run;
      if (g.sinceCkpt >= g.cfg.ckptMin * 60) { g.phase = 'ckpt'; g.busyLeft = ev.ckptSec; }
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

  if (g.tokens >= sc.tokens) endGame(true, '训练完成');
  else if (g.cost >= sc.budget) endGame(false, '预算烧光了');
  else if (g.t >= sc.deadlineDays * 86400) endGame(false, '错过了交付期限');
}

function finishBusy() {
  if (g.phase === 'ckpt') {
    g.ckptTokens = g.tokens; g.sinceCkpt = 0; g.stats.ckpts++;
    if (g.spikeWatch) g.spikeWatch.dirtyCkpt = true;
    if (g.pendingEvict != null) {
      const i = g.pendingEvict; g.pendingEvict = null;
      g.spares--; g.repairs.push(g.t + (24 + rnd() * 24) * 3600);
      g.nodes[i] = NODE_FAIL; g.failedNode = i;
      log('info', `checkpoint 已保存，换下节点 #${i}，用热备顶上。`);
      g.phase = 'restart'; g.busyLeft = ev.restartSec;
      return;
    }
    g.phase = 'run';
  } else if (g.phase === 'repair') {
    if (g.failedNode != null) g.nodes[g.failedNode] = NODE_OK;
    g.failedNode = null;
    log('info', '故障节点原地修好了，重新拉起作业。');
    g.phase = 'restart'; g.busyLeft = ev.restartSec;
  } else if (g.phase === 'restart') {
    if (g.failedNode != null) g.nodes[g.failedNode] = NODE_OK;
    g.failedNode = null;
    g.phase = 'run';
    log('good', `作业恢复运行，从 ${fmtTok(g.tokens)} token 继续。`);
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
    if (g.nodes[i] === NODE_OK) { g.nodes[i] = NODE_SLOW; g.stats.stragglers++; log('warn', `节点 #${i} 过热降频，整个同步训练被它拖慢到 ${pct(STRAGGLER_SLOWDOWN)}。在 GPU 墙上点它。`); }
    return;
  }
  if (g.phase === 'run' && !g.spikeWatch && !g.event && rnd() < SPIKE_PER_HOUR / 60) return lossSpike();

  if (g.spikeWatch && g.t >= g.spikeWatch.until) {
    if (g.spikeWatch.diverge) {
      log('bad', 'loss 发散成 NaN。尖峰之后存的 checkpoint 全部作废。');
      g.ckptTokens = Math.min(g.ckptTokens, g.spikeWatch.safeTokens);
      rollback('loss 发散');
      g.phase = 'restart'; g.busyLeft = ev.restartSec;
    } else log('good', 'loss 尖峰自己回落了，虚惊一场。');
    g.spikeWatch = null;
  }
}

function hardwareFailure() {
  const i = pickHealthyNode();
  const kinds = ['GPU 掉卡（Xid 79）', 'HBM 不可纠正 ECC 错误', 'NVLink 报错', '节点电源故障', '网卡掉线'];
  g.stats.failures++;
  g.nodes[i] = NODE_FAIL; g.failedNode = i; g.pendingEvict = null;
  log('bad', `节点 #${i}：${kinds[Math.floor(rnd() * kinds.length)]}，NCCL 超时，作业挂了。`);
  rollback('故障');
  if (g.spares > 0) {
    g.spares--; g.repairs.push(g.t + (24 + rnd() * 48) * 3600);
    g.phase = 'restart'; g.busyLeft = ev.restartSec;
    log('info', `用热备节点替换 #${i}，约 ${fmtDur(ev.restartSec)} 后恢复。`);
  } else {
    g.phase = 'repair'; g.busyLeft = REPAIR_WAIT_SEC * (0.7 + rnd() * 0.6);
    log('bad', `没有热备了，只能等工程师进机房修 #${i}（${fmtDur(g.busyLeft)}）。`);
  }
}

function linkFlap() {
  g.stats.flaps++;
  log('bad', '交换机端口抖动（link flap），集合通信超时，作业挂了。');
  rollback('链路抖动');
  g.phase = 'restart'; g.busyLeft = ev.restartSec;
}

function lossSpike() {
  g.stats.spikes++;
  g.spikeWatch = { until: Infinity, diverge: false, safeTokens: g.ckptTokens };
  g.event = { kind: 'spike' };
  g.paused = true;
  log('warn', `loss 尖峰！${fmtTok(g.tokens)} token 处 loss 突然跳高。`);
  renderEvent();
}

function resolveSpike(choice) {
  if (choice === 'rollback') {
    g.spikeWatch = null;
    rollback('跳过坏数据');
    g.phase = 'restart'; g.busyLeft = ev.restartSec;
    log('info', '回滚到最近 checkpoint，跳过这段数据重新开始。');
  } else {
    g.spikeWatch.until = g.t + (30 + rnd() * 60) * 60;
    g.spikeWatch.diverge = rnd() < 0.4;
    log('info', '继续观察，看 loss 会不会自己回落。');
  }
  g.event = null; g.paused = false;
  renderEvent();
}

function evict(i) {
  if (!g || g.over) return;
  if (g.nodes[i] !== NODE_SLOW) return;
  if (g.phase !== 'run') { flashTip('等作业恢复运行后再换节点。'); return; }
  if (g.spares <= 0) { flashTip('没有热备节点可以替换。先加一台热备，或者忍着。'); return; }
  g.pendingEvict = i;
  g.phase = 'ckpt'; g.busyLeft = ev.ckptSec;
  log('info', `准备换下慢节点 #${i}：先存一个 checkpoint。`);
}

function endGame(win, reason) {
  const sc = SCENARIOS[g.scIdx];
  g.over = { win, reason };
  const goodput = g.tokens / (ev.rate * g.t);
  let stars = 0;
  if (win) {
    stars = 1;
    if (g.cost <= sc.budget * 0.85) stars++;
    if (goodput >= 0.85) stars++;
  }
  $('endTitle').textContent = win ? `交付了！${reason}` : `失败：${reason}`;
  $('endTitle').className = win ? 'win' : 'lose';
  $('endStars').innerHTML = win ? '★'.repeat(stars) + '<s>' + '★'.repeat(3 - stars) + '</s>' : '';
  $('endKv').innerHTML = `
    <dt>用时</dt><dd>${(g.t / 86400).toFixed(1)} / ${sc.deadlineDays} 天</dd>
    <dt>花费</dt><dd>${fmtMoney(g.cost)} / ${fmtMoney(sc.budget)}</dd>
    <dt>有效 token</dt><dd>${fmtTok(g.tokens)} / ${fmtTok(sc.tokens)}</dd>
    <dt>Goodput</dt><dd>${pct(goodput)}</dd>
    <dt>硬件故障 / 链路抖动</dt><dd>${g.stats.failures} / ${g.stats.flaps}</dd>
    <dt>慢节点 / loss 尖峰</dt><dd>${g.stats.stragglers} / ${g.stats.spikes}</dd>
    <dt>回滚丢掉的 token</dt><dd>${fmtTok(g.stats.lost)}</dd>
    <dt>保存 checkpoint</dt><dd>${g.stats.ckpts} 次</dd>`;
  $('endShade').hidden = false;
  log(win ? 'good' : 'bad', win ? `训练完成，${stars} 星。` : `游戏结束：${reason}。`);
  renderOps();
}

// ---------- 值班台界面 ----------
let dirtyLog = true;
let tipTimer = 0;

function showOps() {
  $('setup').hidden = true; $('ops').hidden = false; $('endShade').hidden = true;
  $('opsCkpt').value = g.cfg.ckptMin;
  ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
  layoutGrid();
  renderEvent();
  dirtyLog = true;
  renderOps();
}

function showSetup(isReconfig) {
  reconfig = isReconfig;
  if (isReconfig) { cfg = { ...g.cfg }; scIdx = g.scIdx; g.paused = true; }
  $('ops').hidden = true; $('setup').hidden = false; $('endShade').hidden = true;
  syncForm();
}

function buildOps() {
  $('speed').innerHTML = SPEEDS.map((s, i) => `<button data-i="${i}" aria-pressed="${i === speedIdx}">${s < 1 ? '½' : s}h/s</button>`).join('');
  $('speed').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    speedIdx = +b.dataset.i;
    $('speed').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  }));
  $('pauseBtn').addEventListener('click', togglePause);
  $('ckptBtn').addEventListener('click', () => {
    if (g.phase === 'run') { g.phase = 'ckpt'; g.busyLeft = ev.ckptSec; log('info', '手动触发 checkpoint。'); }
  });
  $('spareAdd').addEventListener('click', () => {
    g.spares++; g.cfg.spares++; ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
    log('info', `多租一台热备节点，现在每小时 ${fmtMoney(ev.costPerHour)}。`);
  });
  $('spareDel').addEventListener('click', () => {
    if (g.spares <= 0) return;
    g.spares--; g.cfg.spares--; ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
    log('info', `退掉一台热备节点，现在每小时 ${fmtMoney(ev.costPerHour)}。`);
  });
  $('opsCkpt').addEventListener('change', (e) => {
    g.cfg.ckptMin = +e.target.value; ev = evaluate(g.cfg, SCENARIOS[g.scIdx]);
    log('info', `checkpoint 间隔改为 ${g.cfg.ckptMin} 分钟。`);
  });
  $('reconfBtn').addEventListener('click', () => showSetup(true));
  $('againBtn').addEventListener('click', () => { scIdx = g.scIdx; cfg = { ...g.cfg }; ev = evaluate(cfg, SCENARIOS[scIdx]); startGame(); });
  $('menuBtn').addEventListener('click', () => { const keep = g; g = null; scIdx = keep.scIdx; cfg = { ...keep.cfg }; showSetup(false); renderTop(); });

  const grid = $('grid');
  grid.addEventListener('mousemove', (e) => {
    const i = nodeAt(e);
    if (i == null || !g) return;
    const s = g.nodes[i];
    setTip(`节点 #${i} · GPU ${i * 8}–${i * 8 + 7} · ${s === NODE_FAIL ? '故障' : s === NODE_SLOW ? '降频中，点击换下' : '正常'}`);
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
  if (Date.now() < tipTimer) return;
  $('tip').textContent = text || '悬停查看节点。琥珀色的是降频的慢节点，点它可以换下来（需要热备）。';
}
function flashTip(text) { tipTimer = 0; setTip(text); tipTimer = Date.now() + 2500; }

function renderEvent() {
  const box = $('eventBox');
  if (!g || !g.event) { box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  box.innerHTML = `<div class="event">
    <p><b>loss 尖峰</b>：可能是一段脏数据，也可能是优化器状态出了问题。已自动暂停。</p>
    <p style="color:var(--muted);font-size:12px">回滚：回到最近 checkpoint，跳过这批数据，稳妥但要重启。继续：大约六成概率自己回落；不回落的话，尖峰之后存的 checkpoint 都不能用。</p>
    <div class="acts"><button class="btn primary" id="evRollback">回滚并跳过数据</button><button class="btn" id="evContinue">继续观察</button></div>
  </div>`;
  $('evRollback').addEventListener('click', () => resolveSpike('rollback'));
  $('evContinue').addEventListener('click', () => resolveSpike('continue'));
}

const PHASE_LABEL = { run: '训练中', ckpt: '写 checkpoint', restart: '重启中', repair: '等待维修' };

function renderTop() {
  const sc = SCENARIOS[g ? g.scIdx : scIdx];
  if (!g) {
    $('mTime').textContent = `0 / ${sc.deadlineDays} 天`; $('mCost').textContent = `$0 / ${fmtMoney(sc.budget)}`;
    $('mTok').textContent = `0 / ${fmtTok(sc.tokens)}`;
    for (const id of ['bTime', 'bCost', 'bTok']) $(id).style.width = '0';
    return;
  }
  $('mTime').textContent = `${(g.t / 86400).toFixed(1)} / ${sc.deadlineDays} 天`;
  $('mCost').textContent = `${fmtMoney(g.cost)} / ${fmtMoney(sc.budget)}`;
  $('mTok').textContent = `${fmtTok(g.tokens)} / ${fmtTok(sc.tokens)}`;
  $('bTime').style.width = Math.min(100, g.t / (sc.deadlineDays * 86400) * 100) + '%';
  $('bCost').style.width = Math.min(100, g.cost / sc.budget * 100) + '%';
  $('bTok').style.width = Math.min(100, g.tokens / sc.tokens * 100) + '%';
}

function renderOps() {
  if (!g) return;
  renderTop();
  const sc = SCENARIOS[g.scIdx];
  const ph = $('phase');
  const label = g.paused ? (g.event ? '等你决定' : '已暂停') : PHASE_LABEL[g.phase];
  ph.textContent = label + (g.phase !== 'run' && g.busyLeft > 0 && !g.paused ? ` · 剩 ${fmtDur(g.busyLeft)}` : '');
  ph.className = 'phase ' + (g.paused ? 'paused' : g.phase);
  $('wallInfo').textContent = `${fmtClock(g.t)} · ${g.cfg.gpus} × ${g.cfg.gpu} · ${g.nodes.length} 节点`;
  $('pauseBtn').textContent = g.paused ? '继续' : '暂停';
  $('pauseBtn').disabled = !!g.over || !!g.event;
  $('ckptBtn').disabled = g.phase !== 'run' || !!g.over;
  $('spareDel').disabled = g.spares <= 0 || !!g.over;
  $('spareAdd').disabled = !!g.over;
  $('reconfBtn').disabled = !!g.over || !!g.event;

  const chips = (n, cls) => `<span class="chips ${cls}">${'<i></i>'.repeat(Math.min(n, 16))}</span>${n > 16 ? ' +' + (n - 16) : ''}`;
  $('pool').innerHTML = `<span>热备 ${g.spares}${chips(g.spares, '')}</span><span>维修中 ${g.repairs.length}${chips(g.repairs.length, 'rep')}</span>`;

  const slow = stragglerCount();
  const elapsed = Math.max(g.t, 1);
  const goodput = g.tokens / (ev.rate * elapsed);
  const left = sc.tokens - g.tokens;
  const etaLeft = left / (ev.rate * (1 - ev.overhead)) / 86400;
  $('liveKv').innerHTML = `
    <dt>步时 / MFU</dt><dd>${(ev.stepSec / (slow ? STRAGGLER_SLOWDOWN : 1)).toFixed(2)}s · ${pct(ev.mfu * (slow ? STRAGGLER_SLOWDOWN : 1))}</dd>
    <dt>慢节点</dt><dd class="${slow ? 'warn' : 'good'}">${slow ? slow + ' 台，吞吐 ×' + STRAGGLER_SLOWDOWN : '无'}</dd>
    <dt>Goodput</dt><dd class="${goodput > 0.85 ? 'good' : goodput < 0.7 ? 'bad' : ''}">${g.t > 3600 ? pct(goodput) : '—'}</dd>
    <dt>距上次 checkpoint</dt><dd>${fmtDur(g.sinceCkpt)}（存了 ${fmtTok(g.ckptTokens)}）</dd>
    <dt>故障 / 抖动 / 尖峰</dt><dd>${g.stats.failures} / ${g.stats.flaps} / ${g.stats.spikes}</dd>
    <dt>每小时花费</dt><dd>${fmtMoney(ev.costPerHour)}</dd>
    <dt>剩余预计</dt><dd class="${g.t / 86400 + etaLeft > sc.deadlineDays ? 'bad' : ''}">${etaLeft.toFixed(1)} 天</dd>`;

  if (dirtyLog) {
    $('log').innerHTML = g.log.map((l) => `<li class="${l.kind}"><time>${fmtClock(l.t)}</time><span>${l.text}</span></li>`).join('');
    dirtyLog = false;
  }
  drawProgress(); drawLoss();
}

// ---------- GPU 墙 ----------
let lay = null;
function layoutGrid() {
  const c = $('grid');
  const W = Math.max(280, c.parentElement.clientWidth - 34);
  const nodes = g.nodes.length;
  const maxH = Math.max(260, Math.min(620, window.innerHeight - 260));
  let pick = null;
  for (let p = 14; p >= 1; p--) {
    const gap = Math.max(1, Math.round(p * 0.5));
    const nodeW = NODE_GPUS * p + gap;
    const perRow = Math.max(1, Math.floor((W + gap) / nodeW));
    const rows = Math.ceil(nodes / perRow);
    const H = rows * (p + gap);
    pick = { p, gap, nodeW, perRow, rows, W, H };
    if (H <= maxH) break;
  }
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(pick.W * dpr); c.height = Math.round(pick.H * dpr);
  c.style.width = pick.W + 'px'; c.style.height = pick.H + 'px';
  pick.dpr = dpr;
  pick.noise = new Float32Array(g.cfg.gpus).map(() => rnd() * Math.PI * 2);
  lay = pick;
}

function nodeAt(e) {
  if (!lay || !g) return null;
  const r = $('grid').getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const col = Math.floor(x / lay.nodeW), row = Math.floor(y / (lay.p + lay.gap));
  if (col >= lay.perRow) return null;
  const i = row * lay.perRow + col;
  return i >= 0 && i < g.nodes.length ? i : null;
}

const css = getComputedStyle(document.documentElement);
const C = {
  ok: css.getPropertyValue('--ok').trim(), ckpt: css.getPropertyValue('--ckpt').trim(), warn: css.getPropertyValue('--warn').trim(),
  bad: css.getPropertyValue('--bad').trim(), idle: css.getPropertyValue('--idle').trim(), bg: css.getPropertyValue('--bg').trim(),
  line: css.getPropertyValue('--line').trim(), muted: css.getPropertyValue('--muted').trim(), fg: css.getPropertyValue('--fg').trim(),
  accent: css.getPropertyValue('--accent').trim(),
};
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function drawGrid(now) {
  if (!g || !lay) return;
  const ctx = $('grid').getContext('2d');
  const { p, gap, nodeW, perRow, dpr, noise } = lay;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, lay.W, lay.H);
  const tsec = reduceMotion ? 0 : now / 1000;
  const blink = Math.sin(tsec * 6) > 0;
  const busyFrac = g.phase !== 'run' ? 1 - g.busyLeft / (g.phase === 'ckpt' ? ev.ckptSec : g.phase === 'restart' ? ev.restartSec : REPAIR_WAIT_SEC) : 0;
  const totalRows = lay.rows;
  for (let n = 0; n < g.nodes.length; n++) {
    const row = Math.floor(n / perRow), col = n % perRow;
    const x0 = col * nodeW, y = row * (p + gap);
    const s = g.nodes[n];
    for (let k = 0; k < NODE_GPUS; k++) {
      const idx = n * NODE_GPUS + k;
      let color, alpha = 1;
      if (s === NODE_FAIL) { color = C.bad; alpha = blink ? 1 : 0.45; }
      else if (s === NODE_SLOW) { color = C.warn; alpha = 0.6 + 0.4 * Math.sin(tsec * 3 + k); }
      else if (g.phase === 'run') { color = C.ok; alpha = g.paused ? 0.5 : 0.62 + 0.38 * Math.sin(tsec * 2.2 + noise[idx]); }
      else if (g.phase === 'ckpt') { color = row / totalRows <= busyFrac ? C.ckpt : C.ok; alpha = 0.85; }
      else { color = C.idle; alpha = row / totalRows <= busyFrac ? 1 : 0.55; }
      ctx.globalAlpha = alpha; ctx.fillStyle = color;
      ctx.fillRect(x0 + k * p, y, Math.max(1, p - (p > 3 ? 1 : 0)), p);
    }
  }
  ctx.globalAlpha = 1;
}

// ---------- 小图 ----------
function prepChart(id) {
  const c = $(id), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '10px "JetBrains Mono", monospace';
  return { ctx, w, h };
}

function drawProgress() {
  const sc = SCENARIOS[g.scIdx];
  const { ctx, w, h } = prepChart('progChart');
  const L = 34, R = 8, T = 8, B = 16, pw = w - L - R, ph = h - T - B;
  const X = (t) => L + t / (sc.deadlineDays * 86400) * pw, Y = (tok) => T + ph - tok / sc.tokens * ph;
  ctx.strokeStyle = C.line; ctx.lineWidth = 1;
  for (let f = 0; f <= 1; f += 0.5) { ctx.beginPath(); ctx.moveTo(L, Y(sc.tokens * f)); ctx.lineTo(L + pw, Y(sc.tokens * f)); ctx.stroke(); }
  ctx.fillStyle = C.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillText(fmtTok(sc.tokens), L - 4, Y(sc.tokens)); ctx.fillText('0', L - 4, Y(0));
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillText('D1', L, h - 3);
  ctx.textAlign = 'right'; ctx.fillText(`D${sc.deadlineDays}`, L + pw, h - 3);
  ctx.setLineDash([4, 3]); ctx.strokeStyle = C.bad;
  ctx.beginPath(); ctx.moveTo(L + pw, T); ctx.lineTo(L + pw, T + ph); ctx.stroke();
  ctx.strokeStyle = C.muted; ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(sc.tokens / (ev.rate * (1 - ev.overhead))), Y(sc.tokens)); ctx.stroke();
  ctx.setLineDash([]);
  const pts = [...g.hist, [g.t, g.tokens]];
  ctx.beginPath(); pts.forEach(([t, k], i) => (i ? ctx.lineTo(X(t), Y(k)) : ctx.moveTo(X(t), Y(k))));
  ctx.lineTo(X(g.t), Y(0)); ctx.lineTo(X(0), Y(0)); ctx.closePath();
  ctx.globalAlpha = 0.18; ctx.fillStyle = C.ok; ctx.fill(); ctx.globalAlpha = 1;
  ctx.beginPath(); pts.forEach(([t, k], i) => (i ? ctx.lineTo(X(t), Y(k)) : ctx.moveTo(X(t), Y(k))));
  ctx.strokeStyle = C.ok; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.fillStyle = C.ok; ctx.beginPath(); ctx.arc(X(g.t), Y(g.tokens), 3, 0, Math.PI * 2); ctx.fill();
}

function drawLoss() {
  const sc = SCENARIOS[g.scIdx];
  const { ctx, w, h } = prepChart('lossChart');
  const L = 34, R = 8, T = 8, B = 16, pw = w - L - R, ph = h - T - B;
  const active = MODELS[sc.model].active;
  const hi = lossAt(active, sc.tokens / 400) + 0.3, lo = lossAt(active, sc.tokens) - 0.05;
  const X = (tok) => L + tok / sc.tokens * pw, Y = (l) => T + Math.max(0, Math.min(ph, (hi - l) / (hi - lo) * ph));
  ctx.strokeStyle = C.line; ctx.lineWidth = 1;
  [lo, (lo + hi) / 2, hi].forEach((v) => { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(L + pw, Y(v)); ctx.stroke(); });
  ctx.fillStyle = C.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  [lo, hi].forEach((v) => ctx.fillText(v.toFixed(2), L - 4, Y(v)));
  ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left'; ctx.fillText('0', L, h - 3);
  ctx.textAlign = 'right'; ctx.fillText(fmtTok(sc.tokens) + ' tok', L + pw, h - 3);
  if (!g.lossHist.length) return;
  ctx.beginPath(); g.lossHist.forEach(([k, l], i) => (i ? ctx.lineTo(X(k), Y(l)) : ctx.moveTo(X(k), Y(l))));
  ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5; ctx.stroke();
  const [k, l] = g.lossHist[g.lossHist.length - 1];
  ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(X(k), Y(l), 3, 0, Math.PI * 2); ctx.fill();
  ctx.textAlign = 'left'; ctx.fillStyle = C.fg; ctx.fillText(l.toFixed(3), Math.min(X(k) + 6, L + pw - 34), Y(l) - 6);
}

// ---------- 主循环 ----------
let last = performance.now();
let acc = 0;
let uiAcc = 0;
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
    drawGrid(now);
    uiAcc += dt;
    if (uiAcc > 0.15) { uiAcc = 0; renderOps(); }
  }
  requestAnimationFrame(frame);
}

window.addEventListener('resize', () => { if (g && !$('ops').hidden) layoutGrid(); });

// 页面热更新时保留这一局
window.claude?.hot?.snapshot?.(() => ({ g, scIdx, cfg, speedIdx, reconfig }));

function boot(data) {
  buildSetup(); buildOps();
  if (data && data.g) {
    ({ g, scIdx, cfg, speedIdx } = data);
    reconfig = !!data.reconfig;
    $('speed').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(+x.dataset.i === speedIdx)));
    if (reconfig) showSetup(true); else showOps();
  } else {
    if (data && data.cfg) { scIdx = data.scIdx; cfg = data.cfg; }
    syncForm();
    renderTop();
  }
  requestAnimationFrame(frame);
}
window.claude?.hot?.ready ? window.claude.hot.ready(boot) : boot(window.claude?.hot?.data ?? {});
