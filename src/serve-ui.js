// 万卡集群模拟器 · 推理部署：选模型、部署方案、线上值班、复盘
import { CATALOG } from './catalog.js';
import {
  SERVE_GPUS, PRECISION, CARD_OPTIONS, BATCH_OPTIONS, TARGET_OPTIONS, SERVE_SCENARIOS, NODE_GPUS, HALL_GPUS,
  parseHfConfig, keyFields, defaultPrecision, replicaPerf, demandAt, demandSeries, seededRng, hallSlots, defaultScale,
  createDay, cloneDay, stepDay, startReplicas, setShed, summarize, finishDay, grade, advise,
  SLOT_IDLE, SLOT_START, SLOT_ON, SLOT_FAIL,
} from './serve.js';
import { HallView, hallLayout, rackName, NODES_PER_RACK } from './scene.js';
import { $, icon, fillIcons, fmtInt, fmtMoney, pad2, segment, stepper, showOnly, registerMode } from './ui.js';

const HALL = '推理大厅 B';
const HALL_SUB = '3 号楼 4F · 32 柜 · 1,024 GPU';
const SPEEDS = [1, 10, 60];                 // 每真实秒推进的模拟分钟
const CHAT = SERVE_SCENARIOS[0].work;
const STORE_KEY = 'wanka.serve.imported';
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const FAULTS = ['GPU 掉卡（Xid 79）', '显存 OOM，进程崩溃', 'NCCL 超时', 'HBM 坏块（ECC）'];

// ---------- 格式化 ----------
const hm = (t) => `${pad2(Math.floor(t / 60) % 24 + (t >= 1440 ? 24 : 0))}:${pad2(Math.floor(t % 60))}`;
const sizeLabel = (n) => (n >= 1e12 ? (n / 1e12).toFixed(2) + 'T' : n >= 1e10 ? Math.round(n / 1e9) + 'B' : (n / 1e9).toFixed(1) + 'B');
const ctxLabel = (n) => (n >= 1024 ? Math.round(n / 1024) + 'K' : String(n));
const bytesLabel = (b) => (b >= 1e12 ? (b / 1e12).toFixed(2) + ' TB' : Math.round(b / 1e9) + ' GB');
const kbLabel = (b) => (b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB' : Math.round(b / 1024) + ' KB');
const fmtReq = (r) => (r >= 100 ? fmtInt(Math.round(r)) : r >= 10 ? r.toFixed(1) : r.toFixed(2));
const fmtPerM = (x) => (x >= 100 ? '$' + Math.round(x) : '$' + x.toFixed(2));
const fmtWan = (n) => (n >= 1e8 ? fmtInt(Math.round(n / 1e8)) + ' 亿' : n >= 1e4 ? fmtInt(Math.round(n / 1e4)) + ' 万' : fmtInt(Math.round(n)));
const fmtWanShort = (n) => (n >= 1e4 ? (n / 1e4).toFixed(n >= 1e5 ? 0 : 1) + ' 万' : fmtInt(Math.round(n)));
const fmtUsd = (d) => (Math.abs(d) < 1000 ? '$' + fmtInt(Math.round(d)) : fmtMoney(d));
const fmtSec = (s) => (s >= 10 ? Math.round(s) + ' 秒' : s.toFixed(s >= 1 ? 1 : 2) + ' 秒');
const pct2 = (x) => x.toFixed(2) + '%';
const precLabel = (m) => (m.native === 'fp8' ? 'FP8 原生' : m.native === 'int4' ? '4 比特原生' : 'BF16');
const attnKind = (m) => (m.mla ? 'MLA' : m.kvHeads < m.heads ? 'GQA' : 'MHA');
const r1 = (v) => Math.round(v * 10) / 10;

// ---------- 状态 ----------
let lib = CATALOG.map((c) => ({ ...c, source: 'catalog' }));
let cur = 0;                // 模型库里选中的
let model = null;           // 解析后的规格
let fitPick = null;         // “放得下吗”里选中的 { gpu, gpus }
let scId = 'launch';
let plan = null;
let sv = null;              // 进行中的一天
let reconf = false;
let speedIdx = 1;
let selected = null, hover = null;
let hfMsg = null;           // 读取 Hugging Face 的结果提示
let pasting = false;
const view = new HallView($('sScene'));
let L = null;
const sc = () => SERVE_SCENARIOS.find((s) => s.id === scId);

function loadImported() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    for (const e of arr) if (e && e.config && !lib.some((x) => x.repo === e.repo)) lib.unshift({ ...e, source: e.source || 'paste' });
  } catch { /* 浏览器不让存就算了 */ }
}
function saveImported() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(lib.filter((e) => e.source !== 'catalog').slice(0, 12))); } catch { /* 同上 */ }
}

// ---------- 选模型 ----------
function modelSub(m, e) {
  const parts = m.experts ? [`MoE ${m.experts} 选 ${m.topk}`, `激活 ${sizeLabel(m.active)}`] : ['稠密'];
  parts.push(attnKind(m), ctxLabel(m.maxCtx));
  if (e.gated) parts.push('需要先同意协议');
  return parts.join(' · ');
}

function renderLib() {
  const q = $('libSearch').value.trim().toLowerCase();
  const rows = lib.map((e, i) => ({ e, i })).filter(({ e }) => !q || e.name.toLowerCase().includes(q) || e.repo.toLowerCase().includes(q));
  $('libCount').textContent = `${lib.length} 个模型${lib.some((e) => e.source !== 'catalog') ? '（含导入的）' : ''}`;
  $('libList').innerHTML = rows.length ? rows.map(({ e, i }) => {
    let m; try { m = parseHfConfig(e.config); } catch { return ''; }
    const sel = i === cur;
    return `<button type="button" class="mrow" data-i="${i}" aria-pressed="${sel}">
      <span class="mk">${e.mark || e.name[0].toUpperCase()}</span>
      <span style="min-width: 0"><span class="nm"><span>${esc(e.name)}</span>${e.gated ? `<span title="需要授权" style="display: grid; flex: none; color: ${sel ? 'rgba(255,255,255,0.72)' : 'var(--ink-2)'}">${icon('lock', 13, 2.2)}</span>` : ''}${e.source !== 'catalog' ? '<span class="tag">导入</span>' : ''}</span><span class="sb">${modelSub(m, e)}</span></span>
      <span><span class="sz">${sizeLabel(m.total)}</span><span class="pr">${m.native === 'fp8' ? 'FP8 原生' : m.native === 'int4' ? '4 比特' : 'BF16'}</span></span>
    </button>`;
  }).join('') : '<div class="lib-empty">没找到。可以在右边填 Hugging Face 仓库名导入。</div>';
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function selectModel(i, msg = null) {
  cur = i;
  model = parseHfConfig(lib[i].config);
  hfMsg = msg;
  fitPick = cheapestFit();
  $('hfRepo').value = lib[i].repo.startsWith('pasted/') ? '' : lib[i].repo;
  renderModel();
}

const perfCache = new WeakMap();
function perfOf(m, p, work) {
  let c = perfCache.get(m); if (!c) perfCache.set(m, (c = new Map()));
  const key = `${p.gpu}|${p.gpus}|${p.precision}|${p.kvFp8}|${p.maxBatch}|${work.inTok}|${work.outTok}|${work.ttftSlo}|${work.tpotSlo}`;
  if (!c.has(key)) c.set(key, replicaPerf(m, p, work));
  return c.get(key);
}
function fitPerf(gpu, gpus, precision = defaultPrecision(model), work = CHAT, maxBatch = 256) {
  return perfOf(model, { gpu, gpus, precision, kvFp8: true, maxBatch }, work);
}
function fitCols() {
  const prec = defaultPrecision(model);
  const c1 = CARD_OPTIONS.find((c) => Object.keys(SERVE_GPUS).some((gp) => fitPerf(gp, c, prec).ok)) || 16;
  return CARD_OPTIONS.filter((c) => c >= c1).slice(0, 3);
}
function cheapestFit() {
  let best = null;
  for (const c of fitCols()) for (const gp of Object.keys(SERVE_GPUS)) {
    const p = fitPerf(gp, c);
    if (p.ok && (!best || p.costPerMOut < best.cost)) best = { gpu: gp, gpus: c, cost: p.costPerMOut };
  }
  return best ? { gpu: best.gpu, gpus: best.gpus } : { gpu: 'B200', gpus: 16 };
}

function renderModel() {
  renderLib();
  const e = lib[cur], m = model, cfg = e.config.text_config ? { ...e.config, ...e.config.text_config } : e.config;
  // 读取状态
  const q = cfg.quantization_config || {};
  const block = Array.isArray(q.weight_block_size) ? `，${q.weight_block_size.join('×')} 分块` : '';
  const precTxt = m.native === 'fp8' ? `原生 FP8${block}` : m.native === 'int4' ? `原生 4 比特（${q.quant_method || 'int4'}）` : 'BF16 权重';
  let status;
  if (hfMsg && hfMsg.kind === 'busy') status = `<span class="chip" style="background: var(--chip); color: var(--ink-2)">读取中</span><span class="muted">${esc(hfMsg.text)}</span>`;
  else if (hfMsg && hfMsg.kind === 'bad') status = `<span class="chip bad">${icon('alert', 12, 2.4)}读不到</span><span class="t-bad" style="font-weight: 500">${esc(hfMsg.text)}</span>`;
  else {
    const chip = e.source === 'hf' ? '已读取 config.json' : e.source === 'paste' ? '已解析粘贴的 config.json' : '内置';
    status = `<span class="chip ok">${icon('check', 12, 3)}${chip}</span><span class="mono muted">${esc(m.arch)} · ${precTxt}</span>${hfMsg && hfMsg.kind === 'note' ? `<span class="muted">${esc(hfMsg.text)}</span>` : ''}`;
  }
  $('hfStatus').innerHTML = status;
  $('cfgFields').innerHTML = (() => {
    const f = keyFields(e.config), half = Math.ceil(f.length / 2);
    const col = (arr) => `<div class="kvcol">${arr.map(([k, v]) => `<div class="kv"><span>${k}</span><span>${v}</span></div>`).join('')}</div>`;
    return f.length ? col(f.slice(0, half)) + col(f.slice(half)) : '<div class="muted">没有认得出的字段</div>';
  })();
  const kvFp8 = m.kvPerTokenBf16 / 2;
  $('modelTiles').innerHTML = [
    ['总参数', sizeLabel(m.total)],
    [m.experts ? '每个 token 激活' : '稠密，全部激活', sizeLabel(m.active)],
    ['KV cache 每 token', kbLabel(kvFp8)],
  ].map(([k, v]) => `<div class="mtile"><div class="small muted">${k}</div><div class="num">${v}</div></div>`).join('');
  const kb = kbLabel(kvFp8);
  let note = m.mla ? `MLA 把每层的 KV 压成 ${m.mla.kvLora} 维，FP8 存每个 token 只要 ${kb}。同样 ${m.layers} 层的 GQA 模型通常要几百 KB。`
    : m.kvHeads < m.heads ? `GQA：${m.heads} 个 query 头共用 ${m.kvHeads} 组 KV，FP8 存每个 token ${kb}。`
      : `每个头都单独存 KV，FP8 存每个 token 也要 ${kb}，长上下文很吃显存。`;
  if (m.experts) note += ` ${m.experts} 个专家每个 token 只用 ${m.topk} 个${m.sharedExperts ? `（外加 ${m.sharedExperts} 个共享专家）` : ''}：算力按 ${sizeLabel(m.active)} 算，显存要放下全部 ${sizeLabel(m.total)}。`;
  $('modelNote').textContent = note;
  const prec0 = defaultPrecision(m);
  $('weightBars').innerHTML = ['fp8', 'bf16', 'int4'].map((k) => {
    const b = m.total * PRECISION[k].bytes;
    return `<div class="wbar${k === prec0 ? ' on' : ''}"><span>${PRECISION[k].label}</span><span class="bar"><i style="width: ${(PRECISION[k].bytes / 2 * 100).toFixed(0)}%"></i></span><span>${bytesLabel(b)}</span></div>`;
  }).join('');
  renderDrop();
  renderFit();
}

function renderDrop() {
  const z = $('dropZone');
  z.classList.toggle('paste', pasting);
  if (pasting) {
    z.innerHTML = `<label class="small" style="font-weight: 500" for="pasteBox">把 config.json 的内容贴到这里</label>
      <textarea id="pasteBox" spellcheck="false" placeholder='{ "architectures": ["..."], "hidden_size": 4096, ... }'></textarea>
      <div style="display: flex; gap: 8px; justify-content: flex-end"><button type="button" class="btn" id="pasteCancel" style="height: 36px">取消</button><button type="button" class="btn primary" id="pasteGo" style="height: 36px">解析</button></div>`;
    setTimeout(() => $('pasteBox')?.focus(), 0);
    return;
  }
  z.innerHTML = `<span style="color: var(--ink-2); display: grid; flex: none">${icon('upload', 28, 1.8)}</span>
    <div class="txt">私有仓库、要先同意协议的模型（比如 Llama），或者在 Claude 里打开时连不上 Hugging Face：把 config.json 拖到这里，或者直接粘贴。</div>
    <div class="btns"><button type="button" class="btn" id="pasteOpen">粘贴 config.json</button><button type="button" class="btn" id="fileOpen">选文件</button><input type="file" id="fileInput" accept=".json,application/json" hidden></div>`;
}

function renderFit() {
  const m = model, cols = fitCols(), prec = defaultPrecision(m);
  const gpus = Object.keys(SERVE_GPUS);
  const cells = {};
  let best = null;
  const K = (gp, c) => `${gp}|${c}`;
  for (const gp of gpus) for (const c of cols) {
    const p = fitPerf(gp, c, prec);
    cells[K(gp, c)] = p;
    if (p.ok && (!best || p.costPerMOut < cells[K(best.gp, best.c)].costPerMOut)) best = { gp, c };
  }
  $('fitSub').textContent = `一个副本 · ${PRECISION[prec].label} 权重 + FP8 KV`;
  $('fitChips').innerHTML = `<span>权重 ${bytesLabel(m.total * PRECISION[prec].bytes)}</span><span>上下文 ${ctxLabel(m.maxCtx)}</span><span>${precLabel(m)}</span>`;
  const wide = cols.length <= 2;
  const grid = $('fitGrid');
  grid.style.gridTemplateColumns = `56px repeat(${cols.length}, minmax(0, 1fr))`;
  let html = '<span></span>' + cols.map((c) => `<span class="gh">${c === 8 ? '8 卡 · 一台' : c === 16 ? '16 卡 · 两台' : `${c} 卡`}</span>`).join('');
  for (const gp of gpus) {
    html += `<div class="gl"><span style="font-size: 14px; font-weight: 700">${gp}</span><span class="small muted">${SERVE_GPUS[gp].hbm} GB</span></div>`;
    for (const c of cols) {
      const p = cells[K(gp, c)], sel = fitPick && fitPick.gpu === gp && fitPick.gpus === c, isBest = best && best.gp === gp && best.c === c;
      if (!p.ok) {
        const why = p.wPerGpu > SERVE_GPUS[gp].hbm - p.overhead ? `每卡 ${Math.round(p.wPerGpu)} GB 权重` : (p.errors[0] || '').replace(/^.*?：/, '');
        html += `<div class="fcell no"><span class="a">放不下</span><span class="b">${esc(why)}</span></div>`;
      } else {
        html += `<button type="button" class="fcell" data-gpu="${gp}" data-c="${c}" aria-pressed="${sel}" title="KV ${Math.round(p.kvGbPerGpu)} GB/卡，冷启动 ${(p.loadSec / 60).toFixed(1)} 分钟">
          <span class="a"><span>${fmtReq(p.rMax)}<span class="u"> 请求/秒</span><span class="us">/秒</span></span>${isBest && wide ? '<span class="best">最省</span>' : ''}</span>
          <span class="b" style="display: flex; justify-content: space-between; align-items: center; gap: 4px"><span>${wide ? `KV ${Math.round(p.kvGbPerGpu)} GB/卡 · ` : ''}${fmtPerM(p.costPerMOut)}/M</span>${isBest && !wide ? '<span class="best">最省</span>' : ''}</span></button>`;
      }
    }
  }
  grid.innerHTML = html;
  $('fitFoot').textContent = '请求/秒是一个副本在 SLO 内能扛的量（按聊天请求：输入 1,200、输出 350 token；首 token p99 ≤ 1.5 秒，每 token p99 ≤ 50 ms）；$/M 是按 75% 负载跑时，每百万输出 token 的成本。';
  // 说明：最省的那格为什么省，再挑一个放不下的说一句
  let note = '';
  const miss = gpus.map((gp) => [gp, cols[cols.length - 1]]).find(([gp, c]) => !cells[K(gp, c)].ok);
  const anyNo = gpus.flatMap((gp) => cols.map((c) => [gp, c])).reverse().find(([gp, c]) => !cells[K(gp, c)].ok && cells[K(gp, c)].wPerGpu > SERVE_GPUS[gp].hbm - cells[K(gp, c)].overhead);
  if (anyNo) {
    const [gp, c] = anyNo, wb = model.total * PRECISION[prec].bytes;
    note += `${gp} ${c} 卡一共 ${fmtInt(SERVE_GPUS[gp].hbm * c)} GB，放不下 ${bytesLabel(wb)} 的权重${miss ? '' : '，要拼更多卡'}。`;
  }
  if (best) {
    const p = cells[K(best.gp, best.c)];
    note += `${best.gp} × ${best.c} 每卡放 ${Math.round(p.wPerGpu)} GB 权重，还剩 ${Math.round(p.kvGbPerGpu)} GB 给 KV cache，算下来最省。`;
  } else note += '这几种卡都放不下，换个小一点的模型，或者用 4 比特权重。';
  $('fitNote').textContent = note;
  const ok = !!best;
  $('toPlanBtn').disabled = !ok;
  $('toPlanTitle').textContent = `用 ${lib[cur].name} 做部署方案`;
  $('toPlanSub').textContent = ok && fitPick ? `先按 ${fitPick.gpus} × ${fitPick.gpu} · ${PRECISION[prec].label} 来配` : '先换一个放得下的模型';
}

// 读 Hugging Face：成功就加进模型库；连不上的话，仓库在内置里就用内置的
const normRepo = (s) => {
  let r = String(s || '').trim().replace(/^https?:\/\/(www\.)?huggingface\.co\//, '').replace(/^hf\.co\//, '');
  r = r.replace(/\/(tree|blob|resolve)\/.*$/, '').replace(/\/+$/, '');
  return /^[\w.-]+\/[\w.-]+$/.test(r) ? r : null;
};
let lastTried = null;      // 读不到的那个仓库名，粘贴时拿来当名字
async function readHf(raw) {
  const repo = normRepo(raw);
  lastTried = repo;
  if (!repo) { hfMsg = { kind: 'bad', text: '仓库名要写成“组织/模型名”，比如 Qwen/Qwen3-8B' }; renderModel(); return; }
  hfMsg = { kind: 'busy', text: `正在读 huggingface.co/${repo}/config.json` };
  renderModel();
  $('hfRepo').value = repo;
  const known = lib.findIndex((e) => e.repo.toLowerCase() === repo.toLowerCase());
  let why;
  try {
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 9000);
    const res = await fetch(`https://huggingface.co/${repo}/resolve/main/config.json`, { signal: ctl.signal, credentials: 'omit' });
    clearTimeout(timer);
    if (res.status === 401 || res.status === 403) why = '这个模型要先在 Hugging Face 上同意协议才能下载，网页里读不到。把 config.json 粘贴进来吧。';
    else if (res.status === 404) why = `没找到 huggingface.co/${repo}，或者它没有 config.json。`;
    else if (!res.ok) why = `Hugging Face 返回了 ${res.status}。`;
    else {
      const json = await res.json();
      parseHfConfig(json);
      lastTried = null;
      addModel({ repo, name: repo.split('/')[1], config: json, source: 'hf' });
      return;
    }
  } catch (e) {
    why = e instanceof SyntaxError ? '读到的 config.json 不是合法的 JSON。'
      : e.message && /缺|不是/.test(e.message) ? e.message
        : '连不上 Hugging Face（在 Claude 里打开时，网页不能访问外部网站）。把 config.json 粘贴进来，或者用左边的内置模型。';
  }
  if (known >= 0 && lib[known].source === 'catalog' && !/同意协议/.test(why)) { lastTried = null; selectModel(known, { kind: 'note', text: '连不上 Hugging Face，先用内置的这份' }); }
  else { hfMsg = { kind: 'bad', text: why }; renderModel(); }
}

function addModel(entry) {
  const m = parseHfConfig(entry.config);
  const name = entry.name || entry.config._name_or_path?.split('/').pop() || `粘贴的模型 · ${m.arch}`;
  const e = { ...entry, name, mark: name.replace(/[^A-Za-z0-9]/g, '')[0]?.toUpperCase() || 'M' };
  const at = lib.findIndex((x) => x.repo === e.repo);
  if (at >= 0) lib.splice(at, 1);
  lib.unshift(e);
  saveImported();
  pasting = false;
  selectModel(0);
}

function parsePasted(text, repoHint) {
  let json;
  try { json = JSON.parse(text); } catch { hfMsg = { kind: 'bad', text: '这段不是合法的 JSON，检查一下有没有少了括号或逗号。' }; renderModel(); return; }
  try { parseHfConfig(json); } catch (e) { hfMsg = { kind: 'bad', text: e.message }; renderModel(); return; }
  const repo = normRepo(repoHint) || normRepo(json._name_or_path) || `pasted/${(json._name_or_path || (json.architectures || ['model'])[0]).toString().split('/').pop()}`;
  const known = lib.find((x) => x.repo === repo);
  addModel({ repo, name: known && known.source === 'catalog' ? `${known.name}（粘贴）` : repo.startsWith('pasted/') ? null : repo.split('/')[1], config: json, source: 'paste', gated: known?.gated });
}

function buildModel() {
  loadImported();
  $('libSearch').oninput = renderLib;
  $('libList').onclick = (e) => { const b = e.target.closest('.mrow'); if (b) selectModel(+b.dataset.i); };
  $('hfForm').onsubmit = (e) => { e.preventDefault(); readHf($('hfRepo').value); };
  $('fitGrid').onclick = (e) => { const b = e.target.closest('.fcell[data-gpu]'); if (b) { fitPick = { gpu: b.dataset.gpu, gpus: +b.dataset.c }; renderFit(); } };
  $('toPlanBtn').onclick = () => { if (!$('toPlanBtn').disabled) openPlan(true); };
  const zone = $('dropZone'), card = $('hfCard');
  zone.onclick = (e) => {
    if (e.target.closest('#pasteOpen')) { pasting = true; renderDrop(); }
    else if (e.target.closest('#pasteCancel')) { pasting = false; renderDrop(); }
    else if (e.target.closest('#pasteGo')) parsePasted($('pasteBox').value, lastTried);
    else if (e.target.closest('#fileOpen')) $('fileInput').click();
  };
  zone.onchange = (e) => { const f = e.target.files?.[0]; if (f) f.text().then((t) => parsePasted(t, lastTried)); };
  card.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
  card.addEventListener('dragleave', (e) => { if (!card.contains(e.relatedTarget)) zone.classList.remove('over'); });
  card.addEventListener('drop', (e) => {
    e.preventDefault(); zone.classList.remove('over');
    const f = e.dataTransfer?.files?.[0];
    if (f) f.text().then((t) => parsePasted(t, lastTried));
    else { const t = e.dataTransfer?.getData('text'); if (t) parsePasted(t, lastTried); }
  });
}

function openModel() {
  if (sv && !sv.over) { showLive(); return; }
  if (!model) selectModel(cur); else renderModel();
  showOnly('sModel');
}

// ---------- 部署方案 ----------
const MIN_OPTS = [1, 2, 3, 4, 5, 6, 7, 8, ...Array.from({ length: 30 }, (_, i) => 12 + i * 4)];
const PRE_OPTS = [0, ...Array.from({ length: 16 }, (_, i) => 8 + i * 8)];
let planPerf = null, planFc = null, planAdv = null;

function basePlan(gpu, gpus, precision) {
  const s = sc(), p0 = { gpu, gpus, precision, kvFp8: true, maxBatch: 256 };
  const perf = perfOf(model, p0, s.work);
  return { ...p0, ...(perf.ok ? defaultScale(s, perf, gpus) : { min: 4, max: 32, target: 0.75, pre: 0 }) };
}
function rescale() {
  const s = sc(), perf = perfOf(model, plan, s.work);
  if (perf.ok) Object.assign(plan, defaultScale(s, perf, plan.gpus, plan.target));
}

function openPlan(fresh, isReconf = false) {
  reconf = isReconf;
  if (isReconf) { plan = { ...sv.plan }; scId = sv.sc.id; sv.paused = true; }
  else if (fresh || !plan) plan = basePlan(fitPick.gpu, fitPick.gpus, defaultPrecision(model));
  $('planModes').hidden = reconf;
  $('planSteps').hidden = reconf;
  showOnly('sPlan');
  syncPlan();
}

const fits = (gpu, gpus, prec) => perfOf(model, { gpu, gpus, precision: prec, kvFp8: plan.kvFp8, maxBatch: plan.maxBatch }, sc().work).ok;

function syncPlan() {
  const s = sc(), slots = hallSlots(plan.gpus);
  plan.max = Math.min(plan.max, slots); plan.min = Math.min(plan.min, plan.max);
  planPerf = perfOf(model, plan, s.work);
  if (reconf) {
    // 改方案：从现在的状态接着往下跑完今天
    const smooth = smoothDemand(s);
    const d = cloneDay(sv.day, { demand: mixDemand(sv.day, smooth), record: true, keepRows: true });
    Object.assign(d.plan, { min: plan.min, max: plan.max, target: plan.target, pre: plan.pre });
    planFc = { day: d, sum: finishDay(d), perf: planPerf };
    planAdv = { base: planFc, tips: [] };
  } else {
    planAdv = planPerf.ok ? advise(model, plan, s, { perf: planPerf }) : null;
    planFc = planAdv?.base || null;
  }
  $('planSub').textContent = `推理部署：配方案 · ${lib[cur].name}`;

  // 任务
  $('sMissions').innerHTML = SERVE_SCENARIOS.map((x, i) => {
    const sel = x.id === scId, lvl = x.level === '入门' ? 'ok' : x.level === '困难' ? 'hard' : 'bad';
    return `<button type="button" class="mission${sel ? ' sel' : ''}" data-id="${x.id}" aria-pressed="${sel}"${reconf && !sel ? ' disabled' : ''}>
      <span class="row1"><span class="mono small muted">${pad2(i + 1)}</span><span style="display: flex; align-items: center; gap: 8px"><span class="chip ${lvl}">${x.level}</span><span class="check">${icon('check', 13, 3)}</span></span></span>
      <span class="big"><span class="size">${x.big}</span><span class="muted" style="font-size: 13px">${x.bigSub}</span></span>
      <span class="brief">${x.brief}</span>
      <span class="facts"><span>SLO ${Math.round(x.sloTarget * 100)}%</span><span>${fmtMoney(x.budget)}/天</span><span>最多 ${fmtInt(HALL_GPUS)} 卡</span></span>
    </button>`;
  }).join('');
  const w = s.work;
  $('sWork').innerHTML = `<div style="display: flex; justify-content: space-between; align-items: baseline; gap: 8px"><span style="font-size: 13px; font-weight: 700">请求长什么样</span><span class="small muted">${s.id === 'agent' ? '编程 Agent' : '聊天'}</span></div>
    <div>
      <div class="wrow"><span class="muted">平均输入</span><span><b>${fmtInt(w.inTok)}</b> token</span></div>
      <div class="wrow"><span class="muted">平均输出</span><span><b>${fmtInt(w.outTok)}</b> token</span></div>
      <div class="wrow"><span class="muted">首 token p99</span><span>≤ <b>${w.ttftSlo}</b> 秒</span></div>
      <div class="wrow"><span class="muted">每 token p99</span><span>≤ <b>${Math.round(w.tpotSlo * 1000)}</b> ms</span></div>
      <div class="wrow"><span class="muted">达标率</span><span>≥ <b>${Math.round(s.sloTarget * 100)}%</b> · 按天算</span></div>
      <div class="wrow"><span class="muted">预算</span><span>≤ <b>${fmtMoney(s.budget)}</b> / 天</span></div>
    </div>
    <div class="small muted">两条延迟都在线内，这个请求才算达标。</div>`;

  // 副本
  $('sCfgSummary').textContent = `${lib[cur].name} · ${plan.gpu} × ${plan.gpus} · ${PRECISION[plan.precision].label} · ${plan.min}–${plan.max} 个副本`;
  $('sReplicaSec').classList.toggle('locked', reconf);
  $('sReplicaHint').textContent = reconf ? '副本配置要重新部署才能改，今天先不动' : '一个副本放一整份权重，单独接请求';
  $('sGpuTiles').innerHTML = Object.values(SERVE_GPUS).map((G) => {
    const minCards = CARD_OPTIONS.find((c) => fits(G.id, c, plan.precision));
    const line = minCards == null ? '<span class="small t-bad" style="font-weight: 500">放不下</span>'
      : !fits(G.id, plan.gpus, plan.precision) ? `<span class="small t-warn" style="font-weight: 500">要 ${minCards} 卡${minCards > 8 ? '拼一个' : ''}</span>`
        : `<span class="small muted">$${G.price.toFixed(1)}/卡时</span>`;
    return `<button type="button" class="tile" data-v="${G.id}" aria-pressed="${plan.gpu === G.id}"${minCards == null ? ' disabled' : ''}>
      <span style="display: flex; align-items: baseline; gap: 6px"><span style="font-size: 16px; font-weight: 700">${G.id}</span><span class="small muted">${G.hbm} GB</span></span>${line}
      <span class="tick">${icon('check', 11, 3)}</span></button>`;
  }).join('');
  segment($('sSegCards'), CARD_OPTIONS.map((c) => ({ v: c, label: c, disabled: !fits(plan.gpu, c, plan.precision), title: fits(plan.gpu, c, plan.precision) ? `${c} 卡一个副本` : '放不下' })), plan.gpus,
    (v) => { plan.gpus = +v; rescale(); syncPlan(); }, true);
  $('sCardsInfo').textContent = plan.gpus === 8 ? '8 卡正好一台' : plan.gpus > 8 ? '两台拼一个，跨机走 IB' : `一台放 ${NODE_GPUS / plan.gpus} 个副本`;
  const precLbl = { bf16: 'BF16', fp8: model.native === 'fp8' ? 'FP8 原生' : 'FP8', int4: model.native === 'int4' ? 'INT4 原生' : 'INT4' };
  segment($('sSegPrec'), ['bf16', 'fp8', 'int4'].map((k) => ({ v: k, label: precLbl[k], disabled: !fits(plan.gpu, plan.gpus, k) })), plan.precision,
    (v) => { plan.precision = v; rescale(); syncPlan(); });
  const bfOk = fits(plan.gpu, plan.gpus, 'bf16');
  $('sPrecInfo').textContent = !bfOk ? `BF16 要 ${bytesLabel(model.total * 2)}` : plan.precision === 'int4' ? '4 比特省显存，算得不快' : '';
  segment($('sSegBatch'), BATCH_OPTIONS.map((b) => ({ v: b, label: b })), plan.maxBatch, (v) => { plan.maxBatch = +v; rescale(); syncPlan(); }, true);
  $('sBatchInfo').textContent = batchInfo();
  $('sSwKv').setAttribute('aria-checked', String(plan.kvFp8));
  $('sKvTxt').textContent = plan.kvFp8 ? '开' : '关';
  renderHbm();

  // 扩缩容
  const mins = MIN_OPTS.filter((v) => v <= Math.min(slots, plan.max)), maxs = MIN_OPTS.filter((v) => v >= plan.min && v <= slots);
  if (!maxs.includes(slots)) maxs.push(slots);
  stepper($('sStMin'), '最少副本', mins, () => plan.min, (v) => { plan.min = v; syncPlan(); });
  stepper($('sStMax'), '最多副本', maxs, () => plan.max, (v) => { plan.max = v; syncPlan(); });
  $('sMinInfo').textContent = `夜里也开着 · $${fmtInt(planPerf.costPerHour)}/小时`;
  $('sMaxInfo').textContent = `大厅 B 最多 ${slots} 个`;
  segment($('sSegTarget'), TARGET_OPTIONS.map((t) => ({ v: t, label: Math.round(t * 100) + '%' })), plan.target, (v) => { plan.target = +v; syncPlan(); }, true);
  $('sPreField').hidden = !s.pre;
  if (s.pre) {
    const pres = PRE_OPTS.filter((v) => v <= plan.max);
    if (!pres.includes(plan.pre)) { pres.push(plan.pre); pres.sort((a, b) => a - b); }
    stepper($('sStPre'), '预扩', pres, () => plan.pre, (v) => { plan.pre = v; syncPlan(); });
    $('sStPre').sync((v) => (v ? v : '不扩'));
    $('sPreInfo').textContent = reconf && sv.day.t > s.pre.at ? `${hm(s.pre.at)} 已经过了` : `${hm(s.pre.at)} 开始拉起`;
  }

  renderTrafficChart();
  renderPlanForecast();
  renderLatency();
  renderNotes();
  $('sStartBtn').disabled = !planPerf.ok;
  $('sStartTitle').textContent = reconf ? '按新方案继续' : '上线';
  $('sStartSub').textContent = reconf ? '扩缩容从现在起按新设置走' : `${HALL} · 从 00:00 跑满一天`;
  $('sCancelBtn').hidden = !reconf;
}

function batchInfo() {
  if (!planPerf.ok) return '';
  const i = BATCH_OPTIONS.indexOf(plan.maxBatch), out = [];
  const r = (b) => perfOf(model, { ...plan, maxBatch: b }, sc().work);
  if (planPerf.kvConc < plan.maxBatch) return `KV cache 只够 ${fmtInt(planPerf.kvConc)} 个请求同时跑`;
  if (i > 0) out.push(`${BATCH_OPTIONS[i - 1]} 只扛 ${fmtReq(r(BATCH_OPTIONS[i - 1]).rMax)} 请求/秒`);
  if (i < BATCH_OPTIONS.length - 1) {
    const up = r(BATCH_OPTIONS[i + 1]), gain = up.rMax - planPerf.rMax;
    out.push(gain > planPerf.rMax * 0.08 ? `${BATCH_OPTIONS[i + 1]} 能扛 ${fmtReq(up.rMax)}` : gain < 0.05 ? `开到 ${BATCH_OPTIONS[i + 1]} 也一样` : `${BATCH_OPTIONS[i + 1]} 也只多 ${fmtReq(gain)}`);
  }
  return out.join('；');
}

function renderHbm() {
  const p = planPerf, G = SERVE_GPUS[plan.gpu], el = $('sHbm');
  if (!p.wPerGpu) { el.innerHTML = ''; return; }
  const over = p.kvGbPerGpu <= 1, scale = Math.max(G.hbm, p.wPerGpu + p.overhead);
  const parts = over ? [[p.wPerGpu, 'var(--bad)'], [p.overhead, 'var(--gray)']] : [[p.wPerGpu, 'var(--ink)'], [p.kvGbPerGpu, 'var(--yellow)'], [p.overhead, 'var(--gray)']];
  el.innerHTML = `<div class="field-head" style="font-size: 13px"><span style="font-weight: 700">每卡显存</span><span class="small ${over ? 't-bad' : 'muted'}"><b class="num" style="font-size: 18px; color: ${over ? 'var(--bad-ink)' : 'var(--ink)'}">${G.hbm}</b> GB${over ? `，权重就要 ${Math.round(p.wPerGpu)} GB` : ' 全部用上'}</span></div>
    <div class="stack" role="img" aria-label="每卡 ${G.hbm} GB：权重 ${Math.round(p.wPerGpu)} GB，KV cache ${Math.max(0, Math.round(p.kvGbPerGpu))} GB，其他 ${Math.round(p.overhead)} GB" style="background: none">${parts.map(([v, c]) => `<i style="width: ${(v / scale * 100).toFixed(2)}%; background: ${c}"></i>`).join('')}</div>
    <div class="legend2" style="display: flex; flex-wrap: wrap; gap: 4px 16px"><span><b style="background: var(--ink)"></b>权重 ${Math.round(p.wPerGpu)} GB</span><span><b style="background: var(--yellow)"></b>KV cache ${Math.max(0, Math.round(p.kvGbPerGpu))} GB</span><span><b style="background: var(--gray)"></b>其他 ${Math.round(p.overhead)} GB</span>${over ? '' : `<span style="margin-left: auto">一个副本能缓存 ${fmtWan(p.capTokens)} token</span>`}</div>`;
}

// ---------- 图表 ----------
const niceMax = (v) => { const e = 10 ** Math.floor(Math.log10(Math.max(1e-9, v))); for (const k of [1, 1.2, 1.5, 1.8, 2, 2.4, 3, 4, 4.5, 6, 8, 9, 10]) if (k * e >= v) return k * e; return 10 * e; };
const numLabel = (v) => (v >= 1000 ? fmtInt(v) : v >= 10 ? String(Math.round(v)) : String(+v.toFixed(1)));
const pts = (arr) => arr.map(([x, y], i) => `${i ? 'L' : 'M'}${r1(x)} ${r1(y)}`).join(' ');

// 流量、能扛的量、排队区域；rows 是逐分钟的记录
function flowPaths(rows, a, b, W, H, ymax, step = 10, wa = -1, wb = -1) {
  const X = (t) => (t - a) / (b - a) * W, Y = (v) => H - Math.min(v, ymax) / ymax * H;
  const win = rows.filter((r) => r.t >= a && r.t <= b);
  if (!win.length) return { demand: '', area: '', cap: '', over: '', X, Y };
  const last = win[win.length - 1].t;
  const s = win.filter((r) => r.t % step === 0 || (r.t >= wa && r.t <= wb) || r.t === last || r.t === a);
  const demand = pts(s.map((r) => [X(r.t), Y(r.lam)]));
  const area = `${demand} L${r1(X(s[s.length - 1].t))} ${H} L${r1(X(s[0].t))} ${H} Z`;
  const seg = []; let prev = null;
  for (const r of win) {
    if (prev === null) seg.push([X(r.t), Y(r.cap)]);
    else if (Math.abs(r.cap - prev) > 0.01) { seg.push([X(r.t), Y(prev)]); seg.push([X(r.t), Y(r.cap)]); }
    prev = r.cap;
  }
  seg.push([X(last), Y(prev)]);
  const runs = []; let run = null;
  for (let i = 0; i < win.length; i++) {
    const r = win[i], q = win[i - 1];
    if (r.lam > r.cap && !r.shed) {
      if (!run) { run = []; if (q && q.lam < q.cap) { const k = (q.cap - q.lam) / ((r.lam - q.lam) - (r.cap - q.cap) || 1); run.push({ t: q.t + k, lam: q.lam + (r.lam - q.lam) * k, cap: r.cap }); } }
      run.push(r);
    } else if (run) {
      if (q && r.lam < r.cap) { const k = (q.lam - q.cap) / ((q.lam - r.lam) + (r.cap - q.cap) || 1); run.push({ t: q.t + k, lam: q.lam + (r.lam - q.lam) * k, cap: q.cap }); }
      runs.push(run); run = null;
    }
  }
  if (run) runs.push(run);
  const over = runs.filter((x) => x.length > 1).map((x) => {
    const top = x.map((r) => [X(r.t), Y(r.lam)]), bot = [];
    for (let i = x.length - 1; i >= 0; i--) { bot.push([X(x[i].t), Y(x[i].cap)]); if (i > 0 && Math.abs(x[i - 1].cap - x[i].cap) > 0.01) bot.push([X(x[i].t), Y(x[i - 1].cap)]); }
    return `${pts([...top, ...bot])} Z`;
  }).join(' ');
  return { demand, area, cap: pts(seg), over, X, Y };
}
function ttftPath(rows, a, b, W, top, h, step = 10, wa = -1, wb = -1) {
  const lo = Math.log10(0.2), hi = Math.log10(60);
  const Lg = (v) => top + h - (Math.log10(Math.max(0.2, Math.min(60, v))) - lo) / (hi - lo) * h;
  const X = (t) => (t - a) / (b - a) * W;
  const win = rows.filter((r) => r.t >= a && r.t <= b);
  const last = win.length ? win[win.length - 1].t : a;
  const s = win.filter((r) => r.t % step === 0 || (r.t >= wa && r.t <= wb) || r.t === last || r.t === a || r.ttft > 1.5);
  return { path: pts(s.map((r) => [X(r.t), Lg(r.ttft)])), Lg };
}
const txt = (x, y, s, o = {}) => `<text x="${r1(x)}" y="${r1(y)}"${o.anchor ? ` text-anchor="${o.anchor}"` : ''} font-family="${o.mono ? 'IBM Plex Mono, monospace' : o.num ? 'Barlow Condensed, sans-serif' : 'Noto Sans SC, sans-serif'}" font-size="${o.size || 11}"${o.weight ? ` font-weight="${o.weight}"` : ''} fill="${o.fill || '#59636e'}"${o.halo ? ' paint-order="stroke" stroke="#ffffff" stroke-width="4" stroke-linejoin="round"' : ''}>${s}</text>`;
const hline = (y, x2, col = '#eceff2', extra = '') => `<line x1="0" y1="${r1(y)}" x2="${r1(x2)}" y2="${r1(y)}" stroke="${col}"${extra}></line>`;

function renderTrafficChart() {
  const el = $('sTrafficChart');
  if (!planFc) { el.innerHTML = '<div class="small t-bad" style="padding: 40px 0; text-align: center; font-weight: 500">方案不成立，算不出流量能不能扛住</div>'; return; }
  const s = sc(), rows = planFc.day.rows, sum = planFc.sum;
  const W = 536, H = 120;
  let peakCap = 0; for (const r of rows) peakCap = Math.max(peakCap, r.cap);
  const ymax = niceMax(Math.max(sum.peakLam, peakCap) * 1.05);
  const wa = s.traffic.spike ? s.traffic.spike.at - 5 : sum.peakT - 30, wb = s.traffic.spike ? s.traffic.spike.at + 40 : sum.peakT + 30;
  const f = flowPaths(rows, 0, 1439, W, H, ymax, 10, wa, wb);
  let g = '';
  for (let k = 0; k <= 3; k++) g += hline(H - k * H / 3, W, k ? '#eceff2' : '#cfd5db') + txt(-8, H - k * H / 3 + 4, numLabel(ymax * k / 3), { anchor: 'end', mono: true });
  g += `<path d="${f.area}" fill="rgba(20,24,29,0.05)"></path><path d="${f.cap}" fill="none" stroke="#19b26b" stroke-width="1.5" stroke-linejoin="round"></path><path d="${f.demand}" fill="none" stroke="#14181d" stroke-width="1.75" stroke-linejoin="round"></path>`;
  if (f.over) g += `<path d="${f.over}" fill="#e5383b"></path>`;
  const px = f.X(sum.peakT), py = f.Y(sum.peakLam);
  g += `<circle cx="${r1(px)}" cy="${r1(py)}" r="3.5" fill="#14181d" stroke="#ffffff" stroke-width="1.5"></circle>`;
  const right = px < W - 90;
  g += txt(right ? px + 8 : px - 8, py - 3, `峰值 ${fmtInt(sum.peakLam)}`, { anchor: right ? null : 'end', weight: 500, fill: '#14181d', halo: true });
  if (s.traffic.spike) {
    const x = f.X(s.traffic.spike.at);
    g += `<line x1="${r1(x)}" y1="-4" x2="${r1(x)}" y2="${H}" stroke="#9aa4ae" stroke-dasharray="3 3"></line>` + txt(x - 5.7, 10, `${hm(s.traffic.spike.at)} 发布会`, { anchor: 'end', halo: true });
  }
  if (sum.badMin) {
    const x = f.X(sum.longestFrom ?? sum.badFrom);
    g += txt(x - 5.7, H * 0.48, `${hm(sum.longestFrom ?? sum.badFrom)} 起排队 ${sum.longest} 分钟`, { anchor: 'end', weight: 700, fill: '#bf2629', halo: true });
  }
  // 夜里闲着的提醒：凌晨能扛的量远高于流量时才说
  const night = rows[180];
  if (night && night.cap > night.lam * 2.2) g += txt(8, Math.min(f.Y(night.cap) - 6, H - 14), `夜里也留 ${night.active} 个，大半闲着`, { halo: true });
  g += [0, 360, 720, 1080, 1440].map((t, i) => txt(f.X(Math.min(1439, t)) + (i === 4 ? 0.4 : 0), 138, hm(t), { anchor: i === 0 ? null : i === 4 ? 'end' : 'middle', mono: true })).join('');
  el.innerHTML = `<svg viewBox="-44 -8 584 150" role="img" aria-label="预计流量：峰值 ${fmtInt(sum.peakLam)} 请求/秒${sum.badMin ? `，${hm(sum.badFrom)} 起排队` : '，全天扛得住'}">${g}</svg>`;
}

function renderPlanForecast() {
  const s = sc(), box = $('sFc');
  if (!planPerf.ok || !planFc) {
    box.innerHTML = `<div class="fc-head"><h2>上线前预估</h2><span class="chip bad">方案不成立</span></div><div class="metric"><div class="small muted">达标率 / 花费</div><div class="line"><span class="big" style="color: var(--ink-2)">—</span></div><div class="small t-bad" style="font-weight: 500">${esc(planPerf.errors[0] || '先把副本配到能跑起来')}</div></div>`;
    return;
  }
  const sum = planFc.sum, target = s.sloTarget * 100, gr = grade(sum, s);
  const chip = !gr.sloOk ? ['bad', sum.badMin ? '会排队，SLO 守不住' : 'SLO 守不住'] : !gr.costOk ? ['bad', '会超预算'] : !gr.perMOk ? ['warn', '扛得住，成本偏高'] : ['ok', '扛得住，也不贵'];
  const lo = 95, frac = Math.max(0, Math.min(1, (sum.sloPct - lo) / (100 - lo))), mark = (target - lo) / (100 - lo);
  const costFrac = sum.cost / s.budget;
  box.innerHTML = `<div class="fc-head"><h2>${reconf ? '照新方案跑完今天' : '上线前预估'}</h2><span class="chip ${chip[0]}">${chip[1]}</span></div>
    <div class="metric"><div class="small muted">全天达标率</div>
      <div class="line"><span class="big">${sum.sloPct.toFixed(sum.sloPct >= 99.95 ? 2 : sum.sloPct >= 99 ? 2 : 1)}<span class="unit" style="font-size: 15px; font-weight: 500; color: var(--ink)">%</span></span><span class="muted" style="font-size: 13px">目标 ${target}%</span></div>
      <div class="track" style="height: 10px; border-radius: 5px"><i style="height: 10px; border-radius: 5px; width: ${(frac * 100).toFixed(1)}%; background: ${gr.sloOk ? 'var(--ink)' : 'var(--bad)'}"></i><b style="left: calc(${(mark * 100).toFixed(1)}% - 1.5px); background: var(--ink); top: -3px"></b></div>
      <div class="small" style="display: flex; justify-content: space-between; gap: 8px"><span class="${gr.sloOk ? 't-ok' : 't-bad'}" style="font-weight: 500">${gr.sloOk ? (sum.badMin ? `排队 ${sum.badMin} 分钟，还在线内` : '一分钟都不排队') : `差 ${(target - sum.sloPct).toFixed(1)} 个点：${sum.badMin ? `排队 ${sum.badMin} 分钟` : '延迟超线'}`}</span><span class="mono muted">刻度 95–100%</span></div>
    </div>
    <div class="metric"><div class="small muted">每天花费</div>
      <div class="line"><span class="big">${fmtMoney(sum.cost)}</span><span class="muted" style="font-size: 13px">预算 ${fmtMoney(s.budget)}</span></div>
      <div class="bar"><i style="width: ${Math.min(100, costFrac * 100).toFixed(1)}%; background: ${costFrac > 1 ? 'var(--bad)' : 'var(--ink)'}"></i></div>
      <div class="small ${gr.costOk ? (gr.perMOk ? 't-ok' : 't-warn') : 't-bad'}" style="font-weight: 500">${gr.costOk ? `富余 ${fmtMoney(s.budget - sum.cost)}` : `超了 ${fmtMoney(sum.cost - s.budget)}`} · 每百万输出 token ${fmtPerM(sum.perM)}${gr.perMOk ? '' : `，拿 S 要压到 ${fmtPerM(s.perMTarget)}`}</div>
    </div>`;
}

function renderLatency() {
  const box = $('sLat'), p = planPerf, w = sc().work;
  if (!p.ok) { box.innerHTML = '<div class="fc-head"><span style="font-size: 13px; font-weight: 700">一个副本的延迟</span><span class="small muted">配置不成立</span></div>'; return; }
  const W = 318, H = 76, xmax = p.rMax * 1.06, X = (r) => r / xmax * W, Y = (v) => H - Math.min(v, 2) / 2 * H;
  const rs = []; for (let k = 1; k <= 64; k++) rs.push(p.rMax * 0.94 * k / 64); for (let k = 0; k <= 40; k++) rs.push(p.rMax * (0.94 + 0.12 * k / 40));
  const line = (key, slo) => {
    const o = [];
    for (const r of rs) {
      const st = p.at(r);
      if (!st || st[key] / slo > 2) { o.push([X(r), 0]); break; }
      o.push([X(r), Y(st[key] / slo)]);
    }
    return pts(o);
  };
  const op = p.at(0.75 * p.rMax), xo = X(0.75 * p.rMax), xk = X(p.rMax);
  const ticks = [];
  for (let v = 0; v <= xmax + 1e-9; v += niceTick(xmax)) if (X(v) < W - 46) ticks.push(v);
  let g = hline(0, W) + hline(H, W, '#cfd5db') + `<line x1="0" y1="${H / 2}" x2="${W}" y2="${H / 2}" stroke="#e5383b" stroke-dasharray="4 3"></line>`;
  g += txt(-8, 4, '2×', { anchor: 'end', mono: true }) + txt(-8, H / 2 + 4, 'SLO', { anchor: 'end', mono: true, weight: 500, fill: '#bf2629' }) + txt(-8, H + 4, '0', { anchor: 'end', mono: true });
  g += `<line x1="${r1(xo)}" y1="0" x2="${r1(xo)}" y2="${H}" stroke="#9aa4ae" stroke-dasharray="2 3"></line><line x1="${r1(xk)}" y1="0" x2="${r1(xk)}" y2="${H}" stroke="#cfd5db"></line>`;
  g += `<path d="${line('tpotP99', w.tpotSlo)}" fill="none" stroke="#3b7cf3" stroke-width="1.75" stroke-linejoin="round"></path><path d="${line('ttftP99', w.ttftSlo)}" fill="none" stroke="#14181d" stroke-width="1.75" stroke-linejoin="round"></path>`;
  if (op) g += `<circle cx="${r1(xo)}" cy="${r1(Y(op.ttftP99 / w.ttftSlo))}" r="3.5" fill="#14181d" stroke="#ffffff" stroke-width="1.5"></circle><circle cx="${r1(xo)}" cy="${r1(Y(op.tpotP99 / w.tpotSlo))}" r="3.5" fill="#3b7cf3" stroke="#ffffff" stroke-width="1.5"></circle>`;
  g += txt(xo - 5, 12, `按 ${Math.round(plan.target * 100)}% 跑 · ${fmtReq(plan.target * p.rMax)}`, { anchor: 'end', halo: true });
  g += txt(xk - 6, 12, fmtReq(p.rMax), { anchor: 'end', num: true, size: 14, weight: 600, fill: '#14181d', halo: true });
  g += ticks.map((v, i) => txt(X(v), H + 16, numLabel(v), { anchor: i ? 'middle' : null, mono: true })).join('') + txt(W, H + 16, '请求/秒', { anchor: 'end' });
  const opT = p.at(plan.target * p.rMax);
  box.innerHTML = `<div class="fc-head" style="align-items: baseline"><span style="font-size: 13px; font-weight: 700">一个副本最多扛 <span class="num" style="font-size: 22px">${fmtReq(p.rMax)}</span> 请求/秒</span>
      <span class="chart-legend"><span><i style="width: 12px; height: 2px; background: var(--ink)"></i>首 token</span><span><i style="width: 12px; height: 2px; background: var(--info)"></i>每 token</span></span></div>
    <div class="lat"><svg viewBox="-32 -10 364 106" role="img" aria-label="一个副本的 p99 延迟随负载变化，最多扛 ${fmtReq(p.rMax)} 请求/秒">${g}</svg></div>
    <div class="small muted" style="line-height: 1.5">${opT ? `按 ${Math.round(plan.target * 100)}% 跑：首 token ${opT.ttftP99.toFixed(2)} 秒，每 token ${Math.round(opT.tpotP99 * 1000)} ms` : ''}${p.kvConc < plan.maxBatch ? '；KV cache 先满' : ''}。</div>`;
  box.style.gap = '8px';
}
const niceTick = (xmax) => { const raw = xmax / 3.6, e = 10 ** Math.floor(Math.log10(raw)); for (const k of [1, 2, 2.5, 5, 10]) if (k * e >= raw) return k * e; return 10 * e; };

function renderNotes() {
  const s = sc(), out = [];
  if (!planPerf.ok) out.push(`<li class="err">${esc(planPerf.errors[0])}</li>`);
  else if (reconf) {
    const sum = planFc.sum, base = summarize(sv.day);
    out.push(`<li>照新设置跑完今天：达标率 ${pct2(sum.sloPct)}，花 ${fmtMoney(sum.cost)}。</li>`);
    if (sv.day.t < 1440 && base) out.push(`<li>已经过去的 ${hm(sv.day.t)} 不会变，只影响接下来的扩缩容。</li>`);
  } else {
    const a = planAdv, base = a.base.sum;
    if (a.fix) {
      const f = a.fix, d = f.r.sum.cost - base.cost;
      const what = f.kind === 'pre' ? `开场前预扩到 ${f.plan.pre} 个` : f.kind === 'target' ? `目标负载降到 ${Math.round(f.plan.target * 100)}%` : `最多副本放到 ${f.plan.max} 个`;
      out.push(`<li>${what}，达标率就有 ${pct2(f.r.sum.sloPct)}，每天${d >= 0 ? `只多花 ${fmtUsd(d)}` : `还省 ${fmtUsd(-d)}`}。</li>`);
    } else if (base.sloPct < s.sloTarget * 100) out.push('<li class="err">扩缩容怎么调都守不住 SLO，换更快的卡，或者每个副本多给几张卡。</li>');
    if (a.save) {
      const ref = a.fix ? a.fix.r.sum : base, d = ref.cost - a.save.r.sum.cost;
      if (d > 50) out.push(`<li>夜里流量${nightRatio(s)}，最少留 ${a.save.plan.min} 个，每天省 ${fmtUsd(d)}。</li>`);
    }
    const hw = hardwareTip(s, plan, planPerf);
    if (hw) out.push(`<li>${hw}</li>`);
    if (planPerf.loadSec > 480) out.push(`<li>权重 ${bytesLabel(planPerf.wBytes)}，冷启动要 ${(planPerf.loadSec / 60).toFixed(1)} 分钟，扩容得提前。</li>`);
  }
  $('sNotes').innerHTML = out.length ? out.slice(0, 3).join('') : '<li>方案没问题，上线吧。</li>';
}
function nightRatio(s) {
  let lo = Infinity, hi = 0; for (let t = 0; t < 1440; t += 10) { const d = demandAt(s, t); lo = Math.min(lo, d); hi = Math.max(hi, d); }
  const r = lo / hi;
  return r < 0.2 ? '不到高峰的五分之一' : r < 0.34 ? '只有高峰的三分之一' : '比高峰少一半';
}
// 换卡、换并发能不能便宜一截：只比单副本的成本，快
function hardwareTip(s, pl, pf, m = model) {
  let best = null;
  for (const gpu of Object.keys(SERVE_GPUS)) for (const gpus of CARD_OPTIONS) for (const maxBatch of BATCH_OPTIONS) {
    const p = perfOf(m, { ...pl, gpu, gpus, maxBatch }, s.work);
    const better = !best || p.costPerMOut < best.p.costPerMOut * 0.995 || (p.costPerMOut < best.p.costPerMOut * 1.005 && maxBatch === pl.maxBatch && best.maxBatch !== pl.maxBatch);
    if (p.ok && better) best = { gpu, gpus, maxBatch, p };
  }
  if (!best || best.p.costPerMOut > pf.costPerMOut * 0.92) return null;
  const same = best.gpu === pl.gpu && best.gpus === pl.gpus;
  const what = same ? `并发开到 ${best.maxBatch}` : `换 ${best.gpu} × ${best.gpus}${best.maxBatch !== pl.maxBatch ? `、并发 ${best.maxBatch}` : ''}`;
  return `${what}：每个副本扛 ${fmtReq(best.p.rMax)} 请求/秒，便宜 ${Math.round((1 - best.p.costPerMOut / pf.costPerMOut) * 100)}%。`;
}

function buildPlan() {
  $('sMissions').onclick = (e) => {
    const b = e.target.closest('.mission'); if (!b || b.disabled || reconf) return;
    scId = b.dataset.id;
    plan = basePlan(plan.gpu, plan.gpus, plan.precision);
    if (!perfOf(model, plan, sc().work).ok) plan = basePlan(fitPick.gpu, fitPick.gpus, defaultPrecision(model));
    syncPlan();
  };
  $('sGpuTiles').onclick = (e) => {
    const b = e.target.closest('.tile'); if (!b || b.disabled || reconf) return;
    plan.gpu = b.dataset.v;
    if (!fits(plan.gpu, plan.gpus, plan.precision)) plan.gpus = CARD_OPTIONS.find((c) => fits(plan.gpu, c, plan.precision)) ?? plan.gpus;
    rescale(); syncPlan();
  };
  $('sSwKv').onclick = () => { if (reconf) return; plan.kvFp8 = !plan.kvFp8; rescale(); syncPlan(); };
  $('sStartBtn').onclick = () => { if ($('sStartBtn').disabled) return; reconf ? applyReconf() : startDay(); };
  $('sCancelBtn').onclick = () => { reconf = false; sv.paused = false; showLive(); };
  $('backToModel').onclick = () => { reconf = false; showOnly('sModel'); renderModel(); };
}

// 实际流量用到现在，往后用平滑的预测
const smoothCache = {};
function smoothDemand(s) { return smoothCache[s.id] || (smoothCache[s.id] = demandSeries(s)); }
function mixDemand(d, smooth) { const out = smooth.slice(); for (let t = 0; t < d.t && t < 1440; t++) out[t] = d.demand[t]; return out; }

// ---------- 线上值班 ----------
function startDay() {
  const s = sc(), seed = (Date.now() % 100000) + 1;
  const perf = perfOf(model, plan, s.work);
  const day = createDay(model, plan, s, perf, { demand: demandSeries(s, seededRng(seed)), failRng: seededRng(seed * 7 + 3) });
  sv = {
    day, sc: s, perf, model, plan: { ...plan }, name: lib[cur].name, repo: lib[cur].repo, seed,
    log: [], paused: false, over: false, event: null, lastEventT: -999, choices: [],
    flags: { breach: false, q2: false, below: false, slow: false }, evIdx: 0, shown: 0,
  };
  selected = null; hover = null; speedIdx = 1; reconf = false;
  log('info', 'flag', '上线', `${plan.gpu} × ${plan.gpus} 一个副本 · 先起 ${day.active} 个`);
  showLive();
}

function applyReconf() {
  const d = sv.day, old = d.plan;
  Object.assign(d.plan, { min: plan.min, max: plan.max, target: plan.target, pre: plan.pre });
  sv.plan = { ...d.plan };
  const ch = [];
  if (old.min !== plan.min) ch.push(`最少 ${plan.min}`); if (old.max !== plan.max) ch.push(`最多 ${plan.max}`);
  if (old.target !== plan.target) ch.push(`目标负载 ${Math.round(plan.target * 100)}%`); if (old.pre !== plan.pre) ch.push(plan.pre ? `预扩 ${plan.pre}` : '不预扩');
  log('info', 'sliders', '改了扩缩容', ch.join(' · ') || '没变');
  reconf = false; sv.paused = false;
  showLive();
}

function log(kind, ico, title, sub = '') {
  sv.log.unshift({ t: sv.logT ?? sv.day.t, kind, ico, title, sub });
  if (sv.log.length > 120) sv.log.pop();
}
// 连着的扩容、缩容合成一条：30 分钟内的个数累加
function logCount(key, kind, ico, n, title, sub) {
  const t = sv.logT ?? sv.day.t, top = sv.log[0];
  if (top && top.key === key && t - top.t <= 30) { top.n += n; top.t = t; top.title = title(top.n); top.sub = sub; return; }
  log(kind, ico, title(n), sub);
  Object.assign(sv.log[0], { key, n });
}

function showLive() {
  showOnly('sLive');
  L = hallLayout(HALL_GPUS);
  view.setLayout(L, { title: HALL, sub: HALL_SUB });
  $('sLiveSub').textContent = `${sv.name} · ${sv.sc.title} · ${HALL}`;
  layoutLive();
  renderEvent();
  renderLiveHud(true);
}

const narrowQuery = window.matchMedia('(max-width: 860px)');
function layoutLive() {
  if (!sv || $('sLive').hidden) return;
  const stage = $('sStage'), sr = stage.getBoundingClientRect();
  let free;
  if (narrowQuery.matches) free = { x: 6, y: 6, w: sr.width - 12, h: sr.height - 12 };
  else {
    const rail = document.querySelector('#sLive .rail').getBoundingClientRect(), dock = document.querySelector('#sLive .dock').getBoundingClientRect();
    free = { x: 12, y: 112, w: rail.left - sr.left - 32, h: dock.top - sr.top - 126 };
  }
  view.resize(sr.width, sr.height, window.devicePixelRatio || 1, free, narrowQuery.matches ? 2 : 2);
}

function nodesFromSlots() {
  const d = sv.day, g = d.plan.gpus, nodes = new Uint8Array(L.nodes).fill(3);
  const code = (s) => (s === SLOT_ON ? 0 : s === SLOT_START ? 4 : s === SLOT_FAIL ? 1 : 3);
  const rank = { 1: 3, 4: 2, 0: 1, 3: 0 };
  for (let i = 0; i < d.nSlots; i++) {
    const c = code(d.slots[i]);
    if (g >= NODE_GPUS) { const k = g / NODE_GPUS; for (let j = i * k; j < (i + 1) * k && j < nodes.length; j++) nodes[j] = c; }
    else { const j = Math.floor(i * g / NODE_GPUS); if (j < nodes.length && rank[c] > rank[nodes[j]]) nodes[j] = c; }
  }
  return nodes;
}
const slotsOfNode = (j) => { const g = sv.day.plan.gpus; if (g >= NODE_GPUS) return [Math.floor(j / (g / NODE_GPUS))]; const k = NODE_GPUS / g; return Array.from({ length: k }, (_, i) => j * k + i); };
const nodeOfSlot = (i) => { const g = sv.day.plan.gpus; return g >= NODE_GPUS ? i * g / NODE_GPUS : Math.floor(i * g / NODE_GPUS); };
const rid = (i) => 'R' + pad2(i + 1);

// 推进一分钟，并把这一分钟里发生的事写进告警
function tick() {
  const d = sv.day, s = sv.sc, w = s.work;
  if (s.traffic.spike && d.t === s.traffic.spike.at - 5 && speedIdx > 0) { sv.speedBefore = speedIdx; speedIdx = 0; log('warn', 'timer', `${hm(s.traffic.spike.at)} 发布会，速度调慢`, '每秒 1 分钟，盯着开场'); }
  if (sv.speedBefore != null && s.traffic.spike && d.t >= s.traffic.spike.at + 30 && !sv.flags.breach && !d.shed) {
    if (speedIdx === 0) { speedIdx = sv.speedBefore; log('info', 'timer', '开场过去了，速度调回来', `每秒 ${SPEEDS[speedIdx] === 60 ? '1 小时' : SPEEDS[speedIdx] + ' 分钟'}`); }
    sv.speedBefore = null;
  }
  const row = stepDay(d);
  sv.logT = row.t;
  for (; sv.evIdx < d.events.length; sv.evIdx++) {
    const e = d.events[sv.evIdx];
    if (e.kind === 'start') {
      if (e.why === 'pre') log('info', 'server', `开场前预扩 +${e.n}`, `${hm(Math.ceil(e.ready))} 全部上线`);
      else if (e.why === 'manual') log('info', 'server', `手动加 ${e.n} 个副本`, `冷启动 ${(sv.perf.loadSec / 60).toFixed(1)} 分钟，${hm(Math.ceil(e.ready))} 上线`);
      else if (e.why === 'scale') logCount('scale', 'info', 'server', e.n, (n) => `自动扩容 +${n}`, `冷启动 ${(sv.perf.loadSec / 60).toFixed(1)} 分钟，${hm(Math.ceil(e.ready))} 上线`);
      else if (e.why === 'floor') log('info', 'server', `补到最少 ${d.plan.min} 个`, `${hm(Math.ceil(e.ready))} 上线`);
    } else if (e.kind === 'stop') logCount('stop', 'plain', 'down', e.n, (n) => `缩容 −${n}`, `负载低，现在 ${d.active} 个在线`);
    else if (e.kind === 'fail') {
      const node = nodeOfSlot(e.slot);
      log('bad', 'fail', `${rid(e.slot)} ${FAULTS[e.kindIdx % FAULTS.length]}`, `${node < L.nodes ? rackName(L, Math.floor(node / NODES_PER_RACK)) + ' 机柜 · ' : ''}已经自动补一个，冷启动 ${(sv.perf.loadSec / 60).toFixed(1)} 分钟`);
    } else if (e.kind === 'shed') {
      if (e.on && !e.auto) log('warn', 'filter', '开始限流', `只放 ${fmtInt(row.cap)} 请求/秒进来，多的直接回“稍后再试”`);
      else if (!e.on && e.auto) log('ok', 'check', '限流解除', '能扛的量追上来了');
      else if (!e.on) log('info', 'filter', '关掉限流', '请求全部放进来');
    }
  }
  if (s.traffic.spike && d.t - 1 === s.traffic.spike.at) log('warn', 'spike', '发布会开场', '流量开始往上冲');
  // 延迟和排队
  const f = sv.flags;
  if (row.ttft > w.ttftSlo && !f.breach) { f.breach = true; f.breachFrom = row.t; log('bad', 'timer', '首 token 超过 SLO', row.backlog > 0 ? `流量 ${fmtInt(row.lam)}，能扛 ${fmtInt(row.cap)}，开始排队` : `p99 ${fmtSec(row.ttft)}，SLO ${w.ttftSlo} 秒`); }
  else if (row.ttft <= w.ttftSlo && f.breach) { f.breach = false; f.q2 = false; log('ok', 'check', '首 token 回到 SLO 内', `排了 ${row.t - f.breachFrom} 分钟的队`); }
  if (row.backlog > 20000 && !f.q2) { f.q2 = true; log('bad', 'queue', `排队超过 ${fmtWanShort(row.backlog)} 个请求`, `首 token p99 ${fmtSec(row.ttft)}，SLO ${w.ttftSlo} 秒`); }
  const pctNow = d.acc.ok / Math.max(1, d.acc.served) * 100;
  if (pctNow < s.sloTarget * 100 && !f.below) { f.below = true; log('bad', 'alert', `今天达标率跌破 ${Math.round(s.sloTarget * 100)}%`, `现在 ${pct2(pctNow)}`); }
  sv.trend = sv.trend || [];
  sv.trend.push(pctNow); if (sv.trend.length > 6) sv.trend.shift();
  // 过载：自动暂停，二选一
  if (!d.shed && row.ttft > 4 * w.ttftSlo && d.t - sv.lastEventT > 60 && !sv.event && d.t < 1440) openEvent(row);
  sv.logT = null;
  if (d.t >= 1440) endDay();
}

function predict(shed) {
  const d = sv.day, c = cloneDay(d, { demand: mixDemand(d, smoothDemand(sv.sc)), record: false });
  if (shed) setShed(c, true, true);
  return finishDay(c).sloPct;
}

function openEvent(row) {
  const d = sv.day;
  const starting = []; for (let i = 0; i < d.nSlots; i++) if (d.slots[i] === SLOT_START) starting.push(d.until[i]);
  starting.sort((a, b) => a - b);
  sv.event = { t: d.t, lam: row.lam, cap: row.cap, backlog: row.backlog, ttft: row.ttft, starting: starting.length, first: starting[0], pShed: predict(true), pWait: predict(false), rejectNow: row.backlog };
  sv.paused = true;
  sv.logT = d.t;
  log('warn', 'alert', '扛不住了，等你决定', `流量 ${fmtInt(row.lam)}，能扛 ${fmtInt(row.cap)}`);
  renderEvent();
}

function resolveEvent(choice) {
  if (!sv || !sv.event) return;
  const e = sv.event;
  sv.choices.push({ t: e.t, choice, pShed: e.pShed, pWait: e.pWait });
  if (choice === 'shed') setShed(sv.day, true, true);
  else log('info', 'eye', '先扛着，一个都不拒', '等新副本上线');
  sv.lastEventT = sv.day.t; sv.event = null; sv.paused = false;
  // 把刚加的限流事件写进告警
  const d = sv.day;
  for (; sv.evIdx < d.events.length; sv.evIdx++) { const ev = d.events[sv.evIdx]; if (ev.kind === 'shed' && ev.on) log('warn', 'filter', '开始限流', `只放 ${fmtInt(e.cap)} 请求/秒进来，多的直接回“稍后再试”`); }
  renderEvent();
  renderLiveHud(true);
}

function renderEvent() {
  const box = $('sEvent');
  if (!sv || !sv.event) { box.hidden = true; return; }
  const e = sv.event, s = sv.sc, target = s.sloTarget * 100;
  const into = s.traffic.spike && e.t >= s.traffic.spike.at && e.t < s.traffic.spike.at + 90 ? `开场 ${e.t - s.traffic.spike.at} 分钟，扛不住了` : '流量超了能扛的量';
  const chip = (p) => `<span class="chip ${p >= target ? 'ok' : 'bad'}">今天 ${p.toFixed(1)}%</span>`;
  box.hidden = false;
  box.innerHTML = `<div style="display: flex; align-items: center; gap: 12px">
      <div class="ins-ico ico bad">${icon('alert', 18, 2)}</div>
      <div style="flex: 1; min-width: 0"><div style="font-size: 16px; font-weight: 700; line-height: 1.3" id="sEventTitle">${into}</div><div class="small muted">${hm(e.t)} · 自动暂停</div></div>
    </div>
    <p style="font-size: 13px; line-height: 1.6">流量 ${fmtInt(e.lam)}，${sv.day.active} 个副本只能扛 ${fmtInt(e.cap)}。${e.starting ? `冷启动的 ${e.starting} 个 ${hm(Math.ceil(e.first))} 起才陆续上线，` : '没有在冷启动的副本，'}排着的 ${fmtWanShort(e.backlog)} 个请求已经超时。</p>
    <div style="display: flex; flex-direction: column; gap: 8px">
      <button type="button" class="sev-choice" data-choice="shed"><span class="hd"><kbd style="color: var(--ink); font-size: 12px">1</kbd><span class="ttl">限流到 ${fmtInt(e.cap)} 请求/秒</span>${chip(e.pShed)}</span><span class="sub">多出来的直接回“稍后再试”，排着的也退回去。拒掉的算不达标，但后面的请求不用排队</span></button>
      <button type="button" class="sev-choice" data-choice="wait"><span class="hd"><kbd style="color: var(--ink); font-size: 12px">2</kbd><span class="ttl">先扛着</span>${chip(e.pWait)}</span><span class="sub">一个都不拒，等新副本上线把队排完</span></button>
    </div>
    <div class="small muted">“今天”是照预测的流量跑完今天的达标率。按 1 或 2 选。</div>`;
  box.querySelector('[data-choice="shed"]').focus({ preventScroll: true });
}

function renderLiveHud(force = false) {
  if (!sv) return;
  const d = sv.day, s = sv.sc, w = s.work, row = d.rows[d.rows.length - 1] || { lam: d.demand[0], cap: d.active * sv.perf.rMax, ttft: 0.4, tpot: 0.03, backlog: 0, active: d.active, starting: d.starting };
  // 时钟
  $('sClock').textContent = hm(d.t);
  const spike = s.traffic.spike, inSpike = spike && d.t >= spike.at && d.t < spike.at + 60;
  $('sClockSub').textContent = sv.event ? '已暂停 · 等你决定' : sv.paused ? '已暂停 · 空格继续' : `${inSpike ? `发布会第 ${d.t - spike.at} 分钟` : '今天'} · 每秒 ${SPEEDS[speedIdx] === 60 ? '1 小时' : SPEEDS[speedIdx] + ' 分钟'}`;
  $('sClockSub').style.color = sv.event ? 'var(--warn-ink)' : '';
  $('sSpeed').querySelectorAll('button').forEach((b) => { const i = +b.dataset.i; b.setAttribute('aria-pressed', String(i < 0 ? sv.paused : !sv.paused && i === speedIdx)); });

  // 今天达标率
  const pctNow = d.acc.served ? d.acc.ok / d.acc.served * 100 : 99.7, target = s.sloTarget * 100;
  const tr = sv.trend || [], slope = tr.length > 1 ? (tr[tr.length - 1] - tr[0]) / (tr.length - 1) : 0;
  const chip = slope < -0.01 ? ['bad', `每分钟掉 ${(-slope).toFixed(2)} 个点`] : pctNow < target ? ['bad', `差 ${(target - pctNow).toFixed(2)} 个点`] : ['ok', '达标'];
  const lo = 95, frac = Math.max(0, Math.min(1, (pctNow - lo) / (100 - lo))), mark = (target - lo) / (100 - lo);
  const perHour = (d.active + d.starting) * sv.perf.costPerHour;
  $('sSlo').innerHTML = `<div style="display: flex; justify-content: space-between; align-items: center; gap: 8px"><span class="small muted" style="font-weight: 500">今天达标率</span><span class="chip ${chip[0]}">${chip[1]}</span></div>
    <div style="display: flex; align-items: baseline; gap: 10px"><span class="big">${pct2(pctNow)}</span><span class="muted" style="font-size: 13px">目标 ${target}%</span></div>
    <div class="track"><i style="width: ${(frac * 100).toFixed(1)}%; background: ${pctNow < target ? 'var(--bad)' : pctNow - target < 0.3 ? 'var(--warn)' : 'var(--ink)'}"></i><b style="left: calc(${(mark * 100).toFixed(1)}% - 1.5px); background: var(--ink)"></b></div>
    <div class="small muted" style="display: flex; justify-content: space-between; gap: 8px"><span class="${pctNow < target ? 't-bad' : pctNow - target < 0.3 ? 't-warn' : ''}" style="font-weight: 500">${pctNow >= target ? `离 ${target}% 还有 ${(pctNow - target).toFixed(2)} 个点` : `已经差 ${(target - pctNow).toFixed(2)} 个点`}</span><span class="mono">刻度 95–100%</span></div>
    <div class="minis">
      <div><div class="small muted">今天已花</div><div class="num">${fmtMoney(d.acc.cost)}</div><div class="small muted">预算 ${fmtMoney(s.budget)}</div></div>
      <div><div class="small muted">现在每小时</div><div class="num">$${fmtInt(perHour)}</div><div class="small muted">${d.active + d.starting} 个副本在计费</div></div>
    </div>`;

  // 告警
  const key = `${sv.log.length}|${sv.log[0]?.t}|${sv.log[0]?.n}`;
  if (force || key !== sv.alertsKey) {
    sv.alertsKey = key;
    const cls = { bad: 'bad', warn: 'warn', ok: 'ok', info: 'info', plain: 'plain' };
    $('sAlerts').innerHTML = sv.log.map((x, i) => `<li class="${i === 0 && (x.kind === 'bad' || x.kind === 'warn') && d.t - x.t < 30 ? 'hot' : ''}"><span class="ico ${cls[x.kind] || 'plain'}">${icon(x.ico || 'info', 17, 2)}</span><div style="min-width: 0"><div class="ttl"><span>${x.title}</span><time>${hm(x.t)}</time></div>${x.sub ? `<div class="sub">${x.sub}</div>` : ''}</div></li>`).join('');
    $('sAlertsCount').textContent = `今天 ${sv.log.length} 条`;
  }

  // 流量
  const over = row.lam > row.cap, load = row.cap ? row.lam / row.cap : 9;
  $('sTraffic').innerHTML = `<div style="display: flex; justify-content: space-between; align-items: flex-end; gap: 8px">
      <div><div class="small muted" style="font-weight: 500">流量 · 每秒请求</div><div class="num" style="font-size: 36px">${fmtReq(row.lam)}</div></div>
      <div class="small muted" style="text-align: right">能扛 ${fmtReq(row.cap)}<br>${over ? `<b class="t-bad">${d.shed ? '限流中，' : ''}超出 ${Math.round((load - 1) * 100)}%</b>` : `余量 ${Math.round((1 - load) * 100)}%`}</div></div>
    ${liveChart()}
    <div class="stats3">
      <div><div class="small muted">首 token p99</div><div class="num" style="color: ${row.ttft > w.ttftSlo ? 'var(--bad-ink)' : ''}">${row.ttft >= 10 ? Math.round(row.ttft) : row.ttft.toFixed(row.ttft >= 1 ? 1 : 2)}<span class="unit">s</span></div><div class="small muted">SLO ${w.ttftSlo}</div></div>
      <div><div class="small muted">每 token p99</div><div class="num" style="color: ${row.tpot > w.tpotSlo * 0.97 ? 'var(--warn-ink)' : ''}">${Math.round(row.tpot * 1000)}<span class="unit">ms</span></div><div class="small muted">SLO ${Math.round(w.tpotSlo * 1000)}${row.tpot > w.tpotSlo * 0.97 ? '，顶格' : ''}</div></div>
      <div><div class="small muted">排队</div><div class="num" style="color: ${row.backlog > 0 ? 'var(--bad-ink)' : ''}">${row.backlog >= 1e4 ? (row.backlog / 1e4).toFixed(1) + '<span class="unit">万</span>' : fmtInt(row.backlog)}</div><div class="small muted">个请求</div></div>
    </div>`;

  // 底栏
  const status = sv.event ? ['等你决定', 'var(--warn)', '流量超了能扛的量'] : sv.paused ? ['已暂停', 'var(--gray)', '空格继续']
    : d.shed ? ['限流中', 'var(--warn)', `只放 ${fmtInt(row.cap)} 请求/秒进来`] : over || row.ttft > w.ttftSlo ? ['过载', 'var(--bad)', over ? `流量比能扛的多 ${Math.round((load - 1) * 100)}%` : `首 token p99 ${fmtSec(row.ttft)}`]
      : ['正常', 'var(--ok)', `${d.active} 个副本在线 · 负载 ${Math.round(load * 100)}%`];
  $('sStatusDot').style.background = status[1];
  $('sStatusDot').style.boxShadow = `0 0 0 4px color-mix(in srgb, ${status[1]} 18%, transparent)`;
  $('sStatusT').textContent = status[0];
  $('sStatusSub').textContent = status[2];
  let idle = 0; for (let i = 0; i < d.nSlots; i++) if (d.slots[i] === SLOT_IDLE) idle++;
  $('sCOn').innerHTML = `${d.active}<span class="unit">个</span>`;
  $('sCStart').innerHTML = `${d.starting}<span class="unit">个</span>`;
  $('sCStart').style.color = d.starting ? 'var(--info-ink)' : '';
  $('sCFail').innerHTML = `${d.failed}<span class="unit">个</span>`;
  $('sCFail').style.color = d.failed ? 'var(--bad-ink)' : '';
  $('sCIdle').innerHTML = `${fmtInt(idle * d.plan.gpus / NODE_GPUS)}<span class="unit">台</span>`;
  $('sCRoom').innerHTML = `${Math.max(0, d.plan.max - d.active - d.starting)}<span class="unit">个 · 上限 ${d.plan.max}</span>`;
  const busy = !!sv.event || sv.over;
  $('sAddBtn').disabled = busy || idle === 0;
  $('sShedBtn').disabled = busy;
  $('sShedBtn').setAttribute('aria-pressed', String(d.shed));
  $('sShedTxt').textContent = d.shed ? '限流中' : '限流';
  $('sReconfBtn').disabled = busy;
  renderPins();
  renderInspector(false);
}

function liveChart() {
  const d = sv.day, rows = d.rows, now = d.t, a = Math.max(0, now - 30), b = Math.max(36, now + 6);
  const W = 296, H = 56;
  let ymax = 0; for (const r of rows.slice(-31)) ymax = Math.max(ymax, r.lam, r.cap);
  // 冷启动中的副本上线后能扛多少：虚线
  const fut = [];
  for (let i = 0; i < d.nSlots; i++) if (d.slots[i] === SLOT_START) fut.push(d.until[i]);
  fut.sort((x, y) => x - y);
  let capNow = d.active * sv.perf.rMax;
  const future = [[now, capNow]];
  for (const u of fut) { if (u > b) break; future.push([Math.max(now, u), capNow]); capNow += sv.perf.rMax; future.push([Math.max(now, u), capNow]); }
  future.push([b, capNow]);
  ymax = niceMax(Math.max(ymax, capNow) * 1.08);
  const f = flowPaths(rows, a, b, W, H, ymax, 1, a, b);
  const X = f.X, Y = f.Y;
  const fpath = pts(future.map(([t, v]) => [X(t), Y(v)]));
  const last = rows[rows.length - 1];
  const ticks = []; for (let t = Math.ceil(a / 10) * 10; t <= b; t += 10) if (Math.abs(t - now) > 4) ticks.push(t);
  return `<svg viewBox="0 -4 296 90" role="img" aria-label="最近半小时的流量和能扛的量" style="height: auto">
    <path d="${f.area}" fill="rgba(20,24,29,0.06)"></path>${f.over ? `<path d="${f.over}" fill="#e5383b" fill-opacity="0.85"></path>` : ''}
    <path d="${f.cap}" fill="none" stroke="#19b26b" stroke-width="1.75" stroke-linejoin="round"></path>
    ${fut.length ? `<path d="${fpath}" fill="none" stroke="#19b26b" stroke-width="1.75" stroke-dasharray="2 3" stroke-linejoin="round"></path>` : ''}
    <path d="${f.demand}" fill="none" stroke="#14181d" stroke-width="1.75" stroke-linejoin="round"></path>
    <line x1="${r1(X(now))}" y1="0" x2="${r1(X(now))}" y2="${H}" stroke="#9aa4ae" stroke-dasharray="2 2"></line>
    ${last ? `<circle cx="${r1(X(last.t))}" cy="${r1(Y(last.lam))}" r="3" fill="#14181d"></circle>` : ''}
    ${ticks.map((t) => txt(X(t), 70, hm(t), { anchor: 'middle', mono: true, size: 10 })).join('')}
    ${txt(X(now), 70, '现在', { anchor: 'middle', size: 10, weight: 500, fill: '#14181d' })}
  </svg>`;
}

let pinsKey = '';
function renderPins() {
  const d = sv.day, items = [];
  for (let i = 0; i < d.nSlots; i++) if (d.slots[i] === SLOT_FAIL) { const r = Math.floor(nodeOfSlot(i) / NODES_PER_RACK); if (r < L.racks && !items.includes(r)) items.push(r); }
  let first = null; for (let i = 0; i < d.nSlots; i++) if (d.slots[i] === SLOT_START && (first == null || d.until[i] < d.until[first])) first = i;
  const nFirst = first == null ? 0 : (() => { let n = 0; for (let i = 0; i < d.nSlots; i++) if (d.slots[i] === SLOT_START && Math.abs(d.until[i] - d.until[first]) < 0.01) n++; return n; })();
  const key = `${items.join(',')}|${selected}|${first}|${d.starting}|${nFirst}|${view.s}|${view.offX}|${view.offY}`;
  if (key === pinsKey) return;
  pinsKey = key;
  let html = '';
  for (const r of items) {
    const sel = r === selected, stem = sel ? 30 : 20, a = view.anchor(r), size = sel ? 36 : 28;
    html += `<button type="button" class="pin fail${sel ? ' sel' : ' small'}" data-r="${r}" aria-label="${rackName(L, r)} 有副本故障，查看详情" style="left: ${a.x.toFixed(1)}px; top: ${(a.y - stem - size / 2).toFixed(1)}px; --stem: ${stem}px">${icon('x', sel ? 17 : 14, 2.4)}</button>`;
  }
  if (first != null && !narrowQuery.matches) {
    const r = Math.min(L.racks - 1, Math.floor(nodeOfSlot(first) / NODES_PER_RACK)), a = view.anchor(r);
    html += `<div class="chip-scene" style="left: ${a.x.toFixed(1)}px; top: ${(a.y - 10).toFixed(1)}px"><span class="dot" style="background: var(--info); box-shadow: 0 0 0 3px rgba(59,124,243,0.2)"></span><span>${d.starting} 个副本冷启动中 · ${hm(Math.ceil(d.until[first]))} 先上 ${nFirst} 个</span></div>`;
  }
  $('sPins').innerHTML = html;
}

let insKey = '';
function renderInspector(force) {
  const box = $('sInspector');
  if (selected == null || !sv || !L) { box.hidden = true; insKey = ''; renderHover(); return; }
  const d = sv.day, r = selected, name = rackName(L, r), rk = L.rackList[r];
  const nodes = [0, 1, 2, 3].map((j) => r * NODES_PER_RACK + j);
  const row = d.rows[d.rows.length - 1];
  const per = row && d.active ? Math.min(row.lam, row.cap) / d.active : 0;
  const key = `${r}|${nodes.map((j) => slotsOfNode(j).map((i) => d.slots[i]).join('')).join('/')}|${Math.round(per * 10)}|${Math.floor(d.t / 5)}`;
  if (!force && key === insKey) { positionInspector(); return; }
  insKey = key;
  let anyFail = false;
  const rowsHtml = nodes.map((j) => {
    const ids = slotsOfNode(j).filter((i) => i < d.nSlots);
    const st = ids.map((i) => d.slots[i]);
    const host = `${name}-${j % NODES_PER_RACK + 1}`;
    let cls = '', dot = 'var(--gray)', txtS = '空闲', right = '—';
    if (st.includes(SLOT_FAIL)) { const i = ids[st.indexOf(SLOT_FAIL)]; anyFail = true; cls = 'fail'; dot = 'var(--bad)'; txtS = `${rid(i)} 故障`; right = `修 ${Math.max(1, Math.ceil(d.until[i] - d.t))} 分钟`; }
    else if (st.includes(SLOT_START)) { const i = ids[st.indexOf(SLOT_START)]; dot = 'var(--info)'; txtS = `${rid(i)} 冷启动`; right = `${hm(Math.ceil(d.until[i]))} 上线`; }
    else if (st.includes(SLOT_ON)) { const on = ids.filter((i) => d.slots[i] === SLOT_ON); dot = 'var(--ok)'; txtS = on.length > 1 ? `${on.length} 个副本在线` : `${rid(on[0])} 在线`; right = `${fmtReq(per * on.length)} 请求/秒`; }
    if (!ids.length) { txtS = '没用上'; }
    return `<div class="${cls}"><span class="dot" style="background: ${dot}"></span><span class="mono">${host}</span><span class="${cls ? '' : 'muted'}">${txtS}</span><span class="mono small ${cls ? '' : 'muted'}">${right}</span></div>`;
  }).join('');
  const g = d.plan.gpus;
  const span = g > NODE_GPUS ? `一个副本占 ${g / NODE_GPUS} 台机器` : g === NODE_GPUS ? '一台机器一个副本' : `一台机器放 ${NODE_GPUS / g} 个副本`;
  const msg = anyFail ? '故障的副本已经下线送修，自动补的那个在别的机器上冷启动。' : `${span}。在线的副本平均每个接 ${fmtReq(per)} 请求/秒，最多能扛 ${fmtReq(sv.perf.rMax)}。`;
  box.innerHTML = `<span class="tip-arrow"></span>
    <div class="ins-head"><div class="ins-ico ico ${anyFail ? 'bad' : 'plain'}">${icon(anyFail ? 'fail' : 'server', 18, 2)}</div>
      <div style="flex: 1; min-width: 0"><div style="font-size: 16px; font-weight: 700; line-height: 1.3">${name} 机柜</div><div class="small muted">${L.rows[rk.row].letter} 行 ${rk.n} 号 · 4 台 · 32 卡</div></div>
      <button type="button" class="icon-btn" id="sInsClose" aria-label="关闭">${icon('x', 16, 2)}</button></div>
    <div class="nodes">${rowsHtml}</div>
    <p style="font-size: 13px; line-height: 1.6">${msg}</p>`;
  box.hidden = false;
  positionInspector();
  renderHover();
}
function positionInspector() {
  const box = $('sInspector');
  if (box.hidden || narrowQuery.matches || selected == null) { box.style.left = ''; box.style.top = ''; return; }
  const a = view.anchor(selected), stage = $('sStage').getBoundingClientRect(), rail = document.querySelector('#sLive .rail').getBoundingClientRect();
  const w = 300, h = box.offsetHeight || 300, pinY = a.y - 20;
  let left = a.x + 38, side = 'left';
  if (left + w > rail.left - stage.left - 12) { left = a.x - 38 - w; side = 'right'; }
  left = Math.max(12, left);
  const top = Math.max(100, Math.min(pinY - 76, stage.height - 116 - h));
  box.style.left = left + 'px'; box.style.top = top + 'px';
  const arrow = box.querySelector('.tip-arrow');
  if (arrow) { arrow.style.left = side === 'left' ? '-6px' : ''; arrow.style.right = side === 'right' ? '-6px' : ''; arrow.style.top = Math.max(16, Math.min(h - 28, pinY - top - 6)) + 'px'; }
}
function renderHover() {
  const tag = $('sHover');
  if (hover == null || !L || hover === selected) { tag.hidden = true; return; }
  const a = view.anchor(hover);
  tag.hidden = false; tag.textContent = rackName(L, hover);
  tag.style.left = a.x + 'px'; tag.style.top = (a.y - 10) + 'px';
}

function addEight() {
  if (!sv || sv.over || sv.event) return;
  startReplicas(sv.day, 8, 'manual');
  renderLiveHud();
}
function toggleShed() {
  if (!sv || sv.over || sv.event) return;
  setShed(sv.day, !sv.day.shed);
  const d = sv.day, row = d.rows[d.rows.length - 1];
  for (; sv.evIdx < d.events.length; sv.evIdx++) {
    const e = d.events[sv.evIdx];
    if (e.kind === 'shed') log(e.on ? 'warn' : 'info', 'filter', e.on ? '开始限流' : '关掉限流', e.on ? `只放 ${fmtInt(row ? row.cap : 0)} 请求/秒进来，多的直接回“稍后再试”` : '请求全部放进来');
  }
  renderLiveHud();
}

function buildLive() {
  fillIcons($('sLive'));
  $('sSpeed').innerHTML = [['pause', '暂停'], ['play1', '每秒 1 分钟'], ['play2', '每秒 10 分钟'], ['play3', '每秒 1 小时']]
    .map(([ic, label], i) => `<button type="button" data-i="${i - 1}" aria-label="${label}">${icon(ic, ic === 'play3' ? 20 : ic === 'play2' ? 18 : 16)}</button>`).join('');
  $('sSpeed').onclick = (e) => {
    const b = e.target.closest('button'); if (!b || !sv || sv.over || sv.event) return;
    const i = +b.dataset.i;
    if (i < 0) sv.paused = !sv.paused; else { speedIdx = i; sv.paused = false; }
    renderLiveHud();
  };
  $('sAddBtn').onclick = addEight;
  $('sShedBtn').onclick = toggleShed;
  $('sReconfBtn').onclick = () => { if (sv && !sv.over && !sv.event) openPlan(false, true); };
  $('sEvent').onclick = (e) => { const b = e.target.closest('[data-choice]'); if (b) resolveEvent(b.dataset.choice); };
  const canvas = $('sScene');
  canvas.addEventListener('mousemove', (e) => {
    if (!sv) return;
    const r = canvas.getBoundingClientRect(), h = view.pick(e.clientX - r.left, e.clientY - r.top);
    if (h !== hover) { hover = h; canvas.style.cursor = h == null ? 'default' : 'pointer'; renderHover(); }
  });
  canvas.addEventListener('mouseleave', () => { hover = null; renderHover(); });
  canvas.addEventListener('click', (e) => {
    if (!sv) return;
    const r = canvas.getBoundingClientRect(), h = view.pick(e.clientX - r.left, e.clientY - r.top);
    selected = h == null || h === selected ? null : h;
    renderInspector(true);
  });
  $('sPins').onclick = (e) => { const b = e.target.closest('.pin'); if (!b) return; const r = +b.dataset.r; selected = selected === r ? null : r; renderInspector(true); };
  $('sInspector').onclick = (e) => { if (e.target.closest('#sInsClose')) { selected = null; renderInspector(true); } };
  document.addEventListener('keydown', (e) => {
    if (!sv || $('sLive').hidden || e.target.closest?.('input, select, textarea')) return;
    if (sv.event) { if (e.key === '1') resolveEvent('shed'); else if (e.key === '2') resolveEvent('wait'); return; }
    if (sv.over) return;
    if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); sv.paused = !sv.paused; }
    else if (['1', '2', '3'].includes(e.key)) { speedIdx = +e.key - 1; sv.paused = false; }
    else if (e.key === 'a' || e.key === 'A') addEight();
    else if (e.key === 'l' || e.key === 'L') toggleShed();
    else if (e.key === 'c' || e.key === 'C') openPlan(false, true);
    else if (e.key === 'Escape') { selected = null; renderInspector(true); }
    else return;
    renderLiveHud();
  });
}

function endDay() {
  sv.over = true; sv.paused = true; selected = null;
  log('ok', 'flag', '一天结束', '看复盘');
  showReview();
}

// ---------- 复盘 ----------
function showReview() {
  showOnly('sReview');
  const d = sv.day, s = sv.sc, sum = summarize(d), gr = grade(sum, s), target = s.sloTarget * 100;
  $('sReviewSub').textContent = `${sv.name} · ${s.title} · 复盘`;
  // 最长的一段排队：吃掉了多少个点
  let lostReq = 0;
  if (sum.longest) for (const r of d.rows) if (r.t >= sum.longestFrom && r.t < sum.longestFrom + sum.longest) lostReq += r.lam * 60 * (1 - r.good);
  const lostPts = lostReq / Math.max(1, sum.served) * 100;
  const title = gr.sloOk ? 'SLO 守住了' : 'SLO 没守住';
  const lead = `全天 ${pct2(sum.sloPct)} 达标，目标 ${target}%。` + (sum.longest
    ? `${hm(sum.longestFrom)} 起排了 ${sum.longest} 分钟的队，${fmtWan(lostReq)}个请求超时${lostPts >= 0.1 ? `，吃掉 ${lostPts.toFixed(1)} 个点` : ''}。`
    : sum.rejected > 0 ? `限流退回了 ${fmtWan(sum.rejected)}个请求，没让队排起来。` : '一分钟都没排队。');
  // 照今天的流量换个方案重跑
  const replayOpts = { demand: d.demand, failAt: d.failTimes };
  const adv = advise(sv.model, sv.plan, s, { ...replayOpts, perf: sv.perf });
  const best = adv.combined, bestGr = best ? grade(best.sum, s) : null;
  const changes = [];
  if (adv.fix) changes.push(adv.fix.kind === 'pre' ? `开场前预扩到 ${adv.fix.plan.pre} 个` : adv.fix.kind === 'target' ? `目标负载降到 ${Math.round(adv.fix.plan.target * 100)}%` : `最多副本放到 ${adv.fix.plan.max} 个`);
  if (adv.save) changes.push(`最少副本降到 ${adv.save.plan.min} 个`);
  const lines = [];
  if (gr.letter === 'S') lines.push('三条全中。');
  else if (changes.length && bestGr && rankOf(bestGr.letter) > rankOf(gr.letter)) lines.push(`要拿 ${bestGr.letter}：${changes.join('，')}。照今天的流量重跑，达标率 ${pct2(best.sum.sloPct)}，百万 token ${fmtPerM(best.sum.perM)}。`);
  else {
    const miss = [];
    if (!gr.sloOk) miss.push('先把达标率守住：开场前多预扩一些，或者把目标负载调低');
    if (!gr.costOk) miss.push(`每天得压到 ${fmtMoney(s.budget)} 以内`);
    if (!gr.perMOk) miss.push(`每百万 token 得压到 ${fmtPerM(s.perMTarget)}，试试换卡或者开大并发`);
    lines.push(`要拿 S：${miss.join('；')}。`);
  }
  const wait = sv.choices.find((c) => c.choice === 'wait');
  if (wait) {
    const shedRun = finishDay(createDay(sv.model, sv.plan, s, sv.perf, { ...replayOpts, shedAt: wait.t }));
    lines.push(`${hm(wait.t)} 那次要是选限流，达标率是 ${pct2(shedRun.sloPct)}${shedRun.sloPct >= target ? '，能过线' : ''}。`);
  }
  const hw = hardwareTip(s, sv.plan, sv.perf, sv.model);
  if (hw && gr.letter !== 'S' && lines.length < 2) lines.push(hw);
  const tick = (ok, t) => `<span class="${ok ? '' : 'muted'}">${ok ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0e7a47" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>` : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#bf2629" stroke-width="2.6" stroke-linecap="round" aria-hidden="true"><path d="M7 7l10 10M17 7L7 17"/></svg>`}${t}</span>`;
  const parts = [['在算', sum.busyPct, 'var(--ink)'], ['白天留的余量', sum.idleDayPct, '#a4adb7'], ['夜里闲着', sum.idleNightPct, '#d5dae0'], ['冷启动', sum.startPct, 'var(--info)']];
  $('sVerdict').innerHTML = `
    <span class="small muted" style="font-weight: 500">任务 ${pad2(SERVE_SCENARIOS.indexOf(s) + 1)} · ${HALL} · 24 小时</span>
    <div style="display: flex; align-items: center; gap: 14px; margin-top: 14px">
      <span class="vmark" style="background: ${gr.sloOk ? 'var(--ok-bg)' : 'var(--bad-bg)'}; color: ${gr.sloOk ? 'var(--ok-ink)' : 'var(--bad-ink)'}">${gr.sloOk ? icon('check', 26, 2.6) : icon('x', 24, 2.6)}</span>
      <h1>${title}</h1>
    </div>
    <p style="margin-top: 12px; font-size: 15px; line-height: 1.65; color: var(--ink-2)">${lead}</p>
    <div style="display: flex; align-items: center; gap: 20px; margin-top: 20px; flex-wrap: wrap">
      <div class="grade" style="width: 108px; height: 108px; font-size: 90px; border-radius: 22px" role="img" aria-label="评级 ${gr.letter}">${gr.letter}</div>
      <div class="crit"><span class="small muted">评级</span>${tick(gr.sloOk, `达标率 ${pct2(sum.sloPct)}，${gr.sloOk ? '过了' : '没到'} ${target}%`)}${tick(gr.costOk, `每天 ${fmtMoney(sum.cost)}，${gr.costOk ? '没超' : '超了'} ${fmtMoney(s.budget)}`)}${tick(gr.perMOk, `百万 token ${fmtPerM(sum.perM)}，${gr.perMOk ? '压在' : '超了'} ${fmtPerM(s.perMTarget)}${gr.perMOk ? ' 以内' : ''}`)}</div>
    </div>
    <div class="small muted" style="margin-top: 10px">达标率过线才评 S/A/B；没过线最多 C。</div>
    <div class="hintbox" style="margin-top: 14px">${lines.map((x, i) => `<span${i ? ' class="muted"' : ''}>${x}</span>`).join('')}</div>
    <div style="margin-top: 18px; display: flex; flex-direction: column; gap: 8px">
      <div style="display: flex; justify-content: space-between; align-items: baseline; gap: 8px"><span style="font-size: 13px; font-weight: 700">GPU 时间花在哪了</span><span class="small muted">一共 ${fmtInt(sum.gpuHours)} 卡时</span></div>
      <div class="stack" role="img" aria-label="${parts.map((x) => `${x[0]} ${x[1].toFixed(1)}%`).join('，')}" style="background: none">${parts.map((x) => `<i style="width: ${x[1].toFixed(2)}%; min-width: ${x[1] > 0.05 ? 3 : 0}px; background: ${x[2]}"></i>`).join('')}</div>
      <div class="legend2">${parts.map((x) => `<span><b style="background: ${x[2]}"></b>${x[0]} ${x[1].toFixed(1)}%</span>`).join('')}</div>
    </div>
    <div class="mono small muted" style="margin-top: 16px; line-height: 1.7">${sv.plan.gpu} × ${sv.plan.gpus} · ${PRECISION[sv.plan.precision].label} · 最大并发 ${sv.plan.maxBatch}<br>副本 ${sv.plan.min}–${sv.plan.max} · 目标负载 ${Math.round(sv.plan.target * 100)}%${s.pre && sv.plan.pre ? ` · ${hm(s.pre.at)} 预扩到 ${sv.plan.pre}` : ''}</div>
    <div style="flex: 1; min-height: 20px"></div>
    <div style="display: flex; flex-direction: column; gap: 10px">
      <button type="button" class="btn primary" style="height: 52px; justify-content: center; font-size: 16px; border-radius: 14px" id="sAgainBtn">改方案，再来一天</button>
      <button type="button" class="btn" style="height: 52px; justify-content: center; font-size: 16px; border-radius: 14px" id="sModelBtn">换个模型</button>
    </div>`;
  const tile = (label, big, sub) => `<div class="card tile4"><span class="small muted" style="font-weight: 500">${label}</span><span class="num">${big}</span><span class="small muted">${sub}</span></div>`;
  const unit = (s2) => s2.replace(/ (万|亿)$/, '<span class="unit" style="font-size: 14px; font-weight: 500; color: var(--ink)">$1</span>');
  $('sTiles').innerHTML = tile('请求', unit(fmtWan(sum.served)), `达标 ${pct2(sum.sloPct)}，目标 ${target}%`)
    + tile('输出 token', unit(fmtWan(sum.outTok)), `平均每个请求 ${fmtInt(s.work.outTok)}`)
    + tile('每百万输出 token', fmtPerM(sum.perM), gr.perMOk ? `压在 ${fmtPerM(s.perMTarget)} 以内` : `要拿 S 得压到 ${fmtPerM(s.perMTarget)}`)
    + tile('当天花费', fmtMoney(sum.cost), `预算 ${fmtMoney(s.budget)} 的 ${Math.round(sum.cost / s.budget * 100)}%`);
  $('sDayTitle').textContent = s.traffic.spike ? '发布会这一天' : '这一天';
  $('sDayCharts').innerHTML = reviewCharts(d, sum);
  const inc = (n, label, color) => `<div><div class="num">${n}</div><div class="small"><span class="dot" style="width: 7px; height: 7px; background: ${color}"></span>${label}</div></div>`;
  $('sIncidents').innerHTML = inc(`${sum.badMin}<span class="unit">分钟</span>`, '排队或超时', 'var(--bad)')
    + inc(`${fmtSec(sum.maxTtft).replace(' 秒', '<span class="unit">秒</span>')}`, '最长首 token', 'var(--bad)')
    + (sum.rejected > 0 ? inc(fmtWanShort(sum.rejected).replace(' 万', '<span class="unit">万</span>'), '限流退回的请求', 'var(--warn)') : inc(fmtWanShort(sum.maxBacklog).replace(' 万', '<span class="unit">万</span>'), '最多排队请求', 'var(--bad)'))
    + inc(`${sum.starts}<span class="unit">次</span>`, '冷启动', 'var(--info)')
    + inc(`${sum.peakActive}<span class="unit">个</span>`, sum.peakActive >= sv.plan.max ? '最多在线，到顶了' : '最多同时在线', 'var(--warn)')
    + inc(`${sum.failures}<span class="unit">次</span>`, '副本故障', 'var(--bad)');
  $('sAgainBtn').onclick = () => { plan = { ...sv.plan }; scId = s.id; sv = null; openPlan(false); };
  $('sModelBtn').onclick = () => { sv = null; openModel(); };
}
const rankOf = (l) => ({ C: 0, B: 1, A: 2, S: 3 }[l]);

function reviewCharts(d, sum) {
  const rows = d.rows, s = d.sc, W = 548, H = 236;
  let peakCap = 0; for (const r of rows) peakCap = Math.max(peakCap, r.cap);
  const ymax = niceMax(Math.max(sum.peakLam, peakCap) * 1.05);
  // 放大窗口：最长的一段排队，没有就围着峰值
  let za = Math.floor(((sum.longest ? sum.longestFrom - 20 : sum.peakT - 35)) / 10) * 10;
  za = Math.max(0, Math.min(1440 - 70, za)); const zb = za + 70;
  const big = flowPaths(rows, 0, 1439, W, H, ymax, 10, za, zb);
  const T = ttftPath(rows, 0, 1439, W, 300, 80, 10, za, zb);
  const grid = (w) => [0, 1, 2, 3].map((k) => hline(H - k * H / 3, w, k ? '#eceff2' : '#cfd5db')).join('');
  const ylab = (x) => [1, 2, 3].map((k) => txt(x, H - k * H / 3 + 4, numLabel(ymax * k / 3), { anchor: 'end', mono: true })).join('') + txt(x, H + 4, '0', { anchor: 'end', mono: true });
  const sloY = T.Lg(s.work.ttftSlo);
  const ttftAxis = (w) => hline(sloY, w, '#e5383b', ' stroke-dasharray="4 3"') + hline(380, w, '#cfd5db') + txt(-8, 304, '60 s', { anchor: 'end', mono: true }) + txt(-8, sloY + 4, `${s.work.ttftSlo} s`, { anchor: 'end', mono: true, weight: 500, fill: '#bf2629' }) + txt(-8, 384, '0.2 s', { anchor: 'end', mono: true });
  let g = grid(W) + ylab(-8);
  g += `<path d="${big.area}" fill="rgba(20,24,29,0.05)"></path><path d="${big.cap}" fill="none" stroke="#19b26b" stroke-width="1.5" stroke-linejoin="round"></path><path d="${big.demand}" fill="none" stroke="#14181d" stroke-width="1.75" stroke-linejoin="round"></path>`;
  if (big.over) g += `<path d="${big.over}" fill="#e5383b"></path>`;
  if (s.traffic.spike) { const x = big.X(s.traffic.spike.at); g += `<line x1="${r1(x)}" y1="-8" x2="${r1(x)}" y2="380" stroke="#9aa4ae" stroke-dasharray="3 3"></line>` + txt(x - 5.7, -4, `${hm(s.traffic.spike.at)} 发布会`, { anchor: 'end', halo: true }); }
  const px = big.X(sum.peakT), py = big.Y(sum.peakLam), pr = px < W - 90;
  g += `<circle cx="${r1(px)}" cy="${r1(py)}" r="3.5" fill="#14181d" stroke="#ffffff" stroke-width="1.5"></circle>` + txt(pr ? px + 7 : px - 7, py - 6, `峰值 ${fmtInt(sum.peakLam)}`, { anchor: pr ? null : 'end', weight: 500, fill: '#14181d', halo: true });
  if (sum.longest) { const x = big.X(sum.longestFrom); g += txt(x - 5.7, H * 0.47, `${hm(sum.longestFrom)}–${hm(sum.longestFrom + sum.longest - 1)} 排队`, { anchor: 'end', weight: 700, fill: '#bf2629', halo: true }); }
  for (const ft of d.failTimes) { const r = rows[Math.min(rows.length - 1, ft)]; if (!r) continue; const x = big.X(ft), y = big.Y(r.cap); g += `<circle cx="${r1(x)}" cy="${r1(y)}" r="3.5" fill="#e5383b" stroke="#ffffff" stroke-width="1.5"></circle>`; }
  if (d.failTimes.length) { const ft = d.failTimes[d.failTimes.length - 1], x = big.X(ft), y = big.Y(rows[Math.min(rows.length - 1, ft)].cap); g += txt(Math.min(W - 30, Math.max(30, x)), y + 18 < H - 4 ? y + 18 : y - 8, `${hm(ft)} 副本故障`, { anchor: 'middle', fill: '#bf2629', halo: true }); }
  const night = rows[180];
  if (night && night.cap > night.lam * 2.2) g += txt(8, Math.min(big.Y(night.cap) - 7, H - 12), `夜里也开着 ${night.active} 个，只用了${Math.round(night.lam / night.cap * 10) <= 3 ? '三成' : Math.round(night.lam / night.cap * 10) + '成'}`, { halo: true });
  g += txt(0, 288, '首 token p99 · 对数刻度', { weight: 500 }) + ttftAxis(W);
  g += `<path d="${T.path}" fill="none" stroke="#14181d" stroke-width="1.5" stroke-linejoin="round"></path>`;
  if (sum.maxTtft > s.work.ttftSlo) { const x = T.Lg(sum.maxTtft); g += txt(Math.min(W - 70, big.X(sum.maxTtftT) + 8), x + 11, `最长 ${fmtSec(sum.maxTtft)}`, { weight: 500, fill: '#bf2629', halo: true }); }
  g += [0, 360, 720, 1080, 1440].map((t, i) => txt(i === 4 ? W : big.X(t), 400, hm(t), { anchor: i === 0 ? null : i === 4 ? 'end' : 'middle', mono: true })).join('');
  const left = `<svg viewBox="-46 -16 600 430" role="img" aria-label="24 小时的流量、能扛的量和首 token 延迟：峰值 ${fmtInt(sum.peakLam)} 请求/秒${sum.longest ? `，${hm(sum.longestFrom)} 起排队 ${sum.longest} 分钟` : '，没有排队'}">${g}</svg>`;

  // 放大
  const ZW = 264, z = flowPaths(rows, za, zb, ZW, H, ymax, 1, za, zb), ZT = ttftPath(rows, za, zb, ZW, 300, 80, 1, za, zb);
  let h = txt(0, -4, `${hm(za)}–${hm(zb)} 放大`, { weight: 500, fill: '#14181d' }) + grid(ZW) + [1, 2].map((k) => txt(-8, H - k * H / 3 + 4, numLabel(ymax * k / 3), { anchor: 'end', mono: true })).join('') + txt(-8, H + 4, '0', { anchor: 'end', mono: true });
  h += `<path d="${z.area}" fill="rgba(20,24,29,0.05)"></path>${z.over ? `<path d="${z.over}" fill="#e5383b" fill-opacity="0.85"></path>` : ''}<path d="${z.cap}" fill="none" stroke="#19b26b" stroke-width="1.75" stroke-linejoin="round"></path><path d="${z.demand}" fill="none" stroke="#14181d" stroke-width="1.75" stroke-linejoin="round"></path>`;
  if (s.traffic.spike && s.traffic.spike.at > za && s.traffic.spike.at < zb) { const x = z.X(s.traffic.spike.at); h += `<line x1="${r1(x)}" y1="0" x2="${r1(x)}" y2="380" stroke="#9aa4ae" stroke-dasharray="3 3"></line>`; }
  const r0 = rows[za]; if (r0) h += txt(4, z.Y(r0.cap) - 7, `${r0.active} 个能扛 ${fmtInt(r0.cap)}`, { weight: 500, fill: '#0e7a47', halo: true });
  if (sum.peakT >= za && sum.peakT <= zb) { const x = z.X(sum.peakT), y = z.Y(sum.peakLam), rr = x < ZW - 70; h += `<circle cx="${r1(x)}" cy="${r1(y)}" r="3.5" fill="#14181d" stroke="#ffffff" stroke-width="1.5"></circle>` + txt(rr ? x + 7 : x - 7, y - 6, `峰值 ${fmtInt(sum.peakLam)}`, { anchor: rr ? null : 'end', weight: 500, fill: '#14181d', halo: true }); }
  h += ttftAxis(ZW);
  h += `<path d="${ZT.path}" fill="none" stroke="#14181d" stroke-width="1.5" stroke-linejoin="round"></path>`;
  if (sum.longest && sum.longestFrom >= za && sum.longestFrom <= zb) {
    const x = Math.min(ZW - 80, z.X(sum.longestFrom + sum.longest) + 6);
    h += txt(x, 314, `排队 ${sum.longest} 分钟`, { weight: 700, fill: '#bf2629', halo: true });
    if (sum.maxTtftT >= za && sum.maxTtftT <= zb) h += txt(x, 330, `最长 ${fmtSec(sum.maxTtft)}`, { fill: '#bf2629', halo: true });
  }
  h += [0, 20, 40, 70].map((k, i) => txt(i === 3 ? ZW : z.X(za + k), 400, hm(za + k), { anchor: i === 0 ? null : i === 3 ? 'end' : 'middle', mono: true })).join('');
  const right = `<svg viewBox="-40 -16 312 430" role="img" aria-label="${hm(za)} 到 ${hm(zb)} 放大">${h}</svg>`;
  return left + right;
}

// ---------- 主循环 ----------
let last = performance.now(), acc = 0, uiAcc = 0;
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (sv && !sv.over && !$('sLive').hidden) {
    if (!sv.paused) {
      acc += dt * SPEEDS[speedIdx];
      let n = Math.floor(acc); acc -= n;
      while (n-- > 0 && !sv.paused && !sv.over) tick();
    }
    if (sv && !sv.over && !$('sLive').hidden) {
      view.draw({ nodes: nodesFromSlots(), phase: 'run', progress: 0, paused: sv.paused, selected, hover, techs: [], now: reduceMotion ? 0 : now });
      uiAcc += dt;
      if (uiAcc > 0.15) { uiAcc = 0; renderLiveHud(); }
    }
  }
  requestAnimationFrame(frame);
}

window.addEventListener('resize', () => {
  requestAnimationFrame(() => { if (sv && !$('sLive').hidden) { layoutLive(); pinsKey = ''; renderLiveHud(); } });
});

// ---------- 对外 ----------
export function serveSnapshot() {
  const screen = ['sModel', 'sPlan', 'sLive', 'sReview'].find((id) => !$(id).hidden) || null;
  let snap = null;
  if (sv) {
    const { day, perf, model: _m, sc: s, ...rest } = sv;
    const { model: _dm, perf: _dp, sc: _ds, slots, until, demand, failRng, ...d } = day;
    snap = { ...rest, scId: s.id, day: { ...d, slots: Array.from(slots), until: Array.from(until), demand: Array.from(demand) } };
  }
  return { screen, cur, scId, plan, fitPick, reconf, speedIdx, imported: lib.filter((e) => e.source !== 'catalog'), sv: snap };
}

export function serveBoot(data) {
  fillIcons($('sModel')); fillIcons($('sPlan')); fillIcons($('sReview'));
  buildModel(); buildPlan(); buildLive();
  registerMode('serve', openModel);
  requestAnimationFrame(frame);
  if (!data || !data.screen) return false;
  for (const e of data.imported || []) if (!lib.some((x) => x.repo === e.repo)) lib.unshift(e);
  cur = Math.min(lib.length - 1, data.cur || 0);
  model = parseHfConfig(lib[cur].config);
  fitPick = data.fitPick || cheapestFit();
  scId = data.scId || scId; plan = data.plan; speedIdx = data.speedIdx ?? 1;
  if (data.sv && data.sv.day) {
    const s = SERVE_SCENARIOS.find((x) => x.id === data.sv.scId) || sc();
    const perf = perfOf(model, data.sv.plan, s.work), dd = data.sv.day;
    sv = { ...data.sv, sc: s, perf, model };
    sv.day = { ...dd, model, perf, sc: s, slots: Uint8Array.from(dd.slots), until: Float64Array.from(dd.until), demand: Float64Array.from(dd.demand), failRng: seededRng((sv.seed || 1) * 7 + 3 + dd.t) };
  }
  if (data.screen === 'sLive' && sv && !sv.over) showLive();
  else if (data.screen === 'sReview' && sv) showReview();
  else if (data.screen === 'sPlan' && plan) openPlan(false, !!(data.reconf && sv && !sv.over));
  else openModel();
  return true;
}
