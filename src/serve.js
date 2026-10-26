// 万卡集群模拟器 · 推理部署模型
// 从 Hugging Face 的 config.json 估参数和 KV cache，再估一个副本能扛多少流量，
// 最后按分钟推进一整天：流量、自动扩缩容、冷启动、故障、排队、限流。
// 纯函数加一个可变的“这一天”状态，不依赖 DOM，可以在 Node 里测试。量级要对，不是逐 kernel 仿真。

export const SERVE_GPUS = {
  L40S: { id: 'L40S', hbm: 48, bw: 0.864, tf16: 362, tf8: 733, link: 32, price: 1.0 },
  H100: { id: 'H100', hbm: 80, bw: 3.35, tf16: 989, tf8: 1979, link: 450, price: 2.2 },
  H200: { id: 'H200', hbm: 141, bw: 4.8, tf16: 989, tf8: 1979, link: 450, price: 3.0 },
  B200: { id: 'B200', hbm: 192, bw: 8.0, tf16: 2250, tf8: 4500, link: 900, price: 5.5 },
};
export const PRECISION = {
  bf16: { id: 'bf16', label: 'BF16', bytes: 2, fast: false },
  fp8: { id: 'fp8', label: 'FP8', bytes: 1, fast: true },
  int4: { id: 'int4', label: 'INT4', bytes: 0.5, fast: false },
};
export const CARD_OPTIONS = [1, 2, 4, 8, 16];
export const BATCH_OPTIONS = [64, 128, 256, 512];
export const TARGET_OPTIONS = [0.6, 0.75, 0.9];
export const NODE_GPUS = 8;
export const HALL_GPUS = 1024;            // 推理大厅 B：32 柜
const SLOT_FAIL_MTBF = 8000;              // 每卡平均多少小时出一次让副本挂掉的问题（掉卡、OOM、进程崩溃）
const BASE_TTFT = 0.18;                   // 分词、调度、网络往返

// ---------- config.json → 模型规格 ----------
export function parseHfConfig(c) {
  if (!c || typeof c !== 'object') throw new Error('不是一个 JSON 对象');
  const cfg = c.text_config ? { ...c, ...c.text_config } : c;
  const H = cfg.hidden_size || cfg.d_model || cfg.n_embd, N = cfg.num_hidden_layers || cfg.n_layer || cfg.num_layers;
  const A = cfg.num_attention_heads || cfg.n_head;
  if (!H || !N || !A) throw new Error('config.json 里缺 hidden_size、num_hidden_layers 或 num_attention_heads');
  const K = cfg.num_key_value_heads || cfg.num_kv_heads || A;
  const d = cfg.head_dim || H / A;
  const V = cfg.vocab_size || 32000;
  const I = cfg.intermediate_size || cfg.ffn_hidden_size || 4 * H;
  const E = cfg.n_routed_experts || cfg.num_local_experts || cfg.num_experts || 0;
  const topk = E ? (cfg.num_experts_per_tok || cfg.moe_topk || cfg.top_k || 2) : 0;
  const moeI = cfg.moe_intermediate_size || I;
  const shared = cfg.n_shared_experts || (cfg.shared_expert_intermediate_size ? 1 : 0);
  const sharedI = (cfg.n_shared_experts || 0) * moeI + (cfg.shared_expert_intermediate_size || 0);
  const denseLayers = E ? Math.min(N, cfg.first_k_dense_replace || 0) : N;
  const mla = cfg.kv_lora_rank ? {
    kvLora: cfg.kv_lora_rank, qLora: cfg.q_lora_rank || 0, rope: cfg.qk_rope_head_dim || 64,
    nope: cfg.qk_nope_head_dim || 128, v: cfg.v_head_dim || 128,
  } : null;

  const attn = mla
    ? (mla.qLora ? H * mla.qLora + mla.qLora * A * (mla.nope + mla.rope) : H * A * (mla.nope + mla.rope))
      + H * (mla.kvLora + mla.rope) + mla.kvLora * A * (mla.nope + mla.v) + A * mla.v * H
    : H * A * d + 2 * H * K * d + A * d * H;
  const expertP = 3 * H * moeI, sharedP = 3 * H * sharedI;
  const sparse = N - denseLayers;
  const embed = V * H * (cfg.tie_word_embeddings ? 1 : 2);
  const total = embed + N * attn + denseLayers * 3 * H * I + sparse * (E * expertP + sharedP + H * E);
  const active = V * H + N * attn + denseLayers * 3 * H * I + sparse * (topk * expertP + sharedP);
  const routed = sparse * E * expertP;

  const q = cfg.quantization_config || {};
  const method = String(q.quant_method || '').toLowerCase();
  const native = method === 'fp8' ? 'fp8' : /mxfp4|awq|gptq|int4/.test(method) ? 'int4' : 'bf16';
  return {
    arch: (cfg.architectures || [cfg.model_type || '未知'])[0],
    layers: N, hidden: H, heads: A, kvHeads: K, headDim: d, vocab: V, intermediate: I,
    experts: E, topk, moeI, sharedExperts: shared, denseLayers, mla,
    maxCtx: cfg.max_position_embeddings || cfg.max_sequence_length || 4096, native,
    total, active, routed,
    kvPerTokenBf16: mla ? N * (mla.kvLora + mla.rope) * 2 : 2 * N * K * d * 2,
  };
}

// config.json 里值得给人看的字段，按这个顺序挑
export const KEY_FIELDS = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'kv_lora_rank',
  'qk_rope_head_dim', 'n_routed_experts', 'num_experts', 'num_local_experts', 'num_experts_per_tok', 'n_shared_experts',
  'first_k_dense_replace', 'moe_intermediate_size', 'intermediate_size', 'vocab_size', 'max_position_embeddings'];
export function keyFields(c, n = 12) {
  const cfg = c.text_config ? { ...c, ...c.text_config } : c;
  const out = [];
  for (const k of KEY_FIELDS) if (cfg[k] != null && typeof cfg[k] !== 'object' && out.length < n) out.push([k, cfg[k]]);
  return out;
}

// 默认用哪种精度：原生 FP8 / 4 比特的就用原生的，否则在线量化成 FP8
export const defaultPrecision = (m) => (m.native === 'int4' ? 'int4' : 'fp8');

// ---------- 一个副本的性能 ----------
// plan: { gpu, gpus, precision, kvFp8, maxBatch }；work: { inTok, outTok, ttftSlo, tpotSlo }
export function replicaPerf(m, plan, work) {
  const G = SERVE_GPUS[plan.gpu], g = plan.gpus, P = PRECISION[plan.precision];
  const tp = Math.min(g, NODE_GPUS), nodes = Math.ceil(g / NODE_GPUS);
  const tf = (P.fast ? G.tf8 : G.tf16) * 1e12;
  const wBytes = m.total * P.bytes;
  const wPerGpu = wBytes / g / 1e9;
  const overhead = 4 + 0.06 * G.hbm;
  const kvGbPerGpu = G.hbm - wPerGpu - overhead;
  const errors = [];
  if (kvGbPerGpu <= 1) errors.push(`放不下：每卡要放 ${wPerGpu.toFixed(0)} GB 权重，${G.id} 只有 ${G.hbm} GB`);
  // MLA 用数据并行注意力，KV 不重复存；GQA 的 KV 头比张量并行的份数少时，每张卡都要存一份
  const rep = m.mla ? 1 : Math.max(1, tp / m.kvHeads);
  const kvTok = m.kvPerTokenBf16 * (plan.kvFp8 ? 0.5 : 1) * rep;
  const capTokens = Math.max(0, g * kvGbPerGpu * 1e9 / kvTok);
  const ctx = work.inTok + work.outTok / 2;
  const kvConc = Math.floor(capTokens / ctx);
  const maxConc = Math.max(0, Math.min(plan.maxBatch, kvConc));
  if (!errors.length && work.inTok + work.outTok > m.maxCtx) errors.push(`上下文不够：请求要 ${fmtK(work.inTok + work.outTok)} token，模型最多 ${fmtK(m.maxCtx)}`);
  if (!errors.length && kvConc < 1) errors.push('KV cache 连一个请求都放不下');

  const commLayer = (tokens) => (g > 1 ? 2 * (8e-6 + tokens * m.hidden * 2 / (G.link * 1e9)) + (nodes > 1 ? 2 * 25e-6 : 0) : 0);
  const step = (B) => {
    const touched = m.experts ? 1 - Math.pow(1 - m.topk / m.experts, B) : 1;
    const wRead = (m.total - m.routed + m.routed * touched) * P.bytes / g;
    const kvRead = B * ctx * kvTok / g;
    const tMem = (wRead + kvRead) / (G.bw * 1e12 * 0.82);
    const tCmp = 2 * m.active * B / (g * tf * 0.5);
    return Math.max(tMem, tCmp) + m.layers * commLayer(B) + 0.002;
  };
  const attnDim = m.mla ? m.heads * (m.mla.nope + m.mla.rope + m.mla.v) : m.heads * m.headDim * 2;
  const prefill = (L) => (2 * m.active * L + 2 * m.layers * L * L * attnDim) / (g * tf * 0.4) + m.layers * commLayer(L);

  // 每副本每秒 r 个请求时的稳定状态
  const cache = new Map();
  const at = (r) => {
    const key = Math.round(r * 1000);
    if (cache.has(key)) return cache.get(key);
    const tp1 = prefill(work.inTok), share = r * tp1;
    let res = null;
    if (share < 0.92 && maxConc > 0) {
      let B = Math.max(1, r * work.outTok * step(1));
      for (let k = 0; k < 40; k++) B = Math.max(1, r * work.outTok * step(B) / (1 - share));
      const tpot = step(B) / (1 - share);
      const wait = share / (2 * (1 - share)) * tp1;
      // 解码槽位当成 M/M/c 排队（Sakasegawa 近似）：槽位快满时排队时间陡增
      const c = Math.max(1, maxConc), svc = work.outTok * tpot, rho = Math.min(0.999, r * svc / c);
      const admit = Math.pow(rho, Math.sqrt(2 * (c + 1))) / (1 - rho) * svc / c;
      res = {
        r, B, share, rho, tpot, tpotP99: tpot * 1.25,
        ttft: BASE_TTFT + tp1 + wait + admit + step(B),
        ttftP99: BASE_TTFT * 1.6 + tp1 * 1.5 + (wait + admit) * 4.6 + step(B),
      };
    }
    if (cache.size < 50000) cache.set(key, res);
    return res;
  };
  const ok = (s) => s && s.rho < 0.98 && s.ttftP99 <= work.ttftSlo && s.tpotP99 <= work.tpotSlo;
  let lo = 0, hi = 4000;
  if (!errors.length) for (let k = 0; k < 60; k++) { const mid = (lo + hi) / 2; if (ok(at(mid))) lo = mid; else hi = mid; }
  if (!errors.length && lo < 1e-3) errors.push(at(0.001) && at(0.001).ttftP99 > work.ttftSlo ? `一个请求的首 token 就要 ${at(0.001).ttftP99.toFixed(1)} 秒，过不了 SLO` : '每 token 延迟过不了 SLO');
  const rMax = errors.length ? 0 : lo;
  const loadSec = 120 + wBytes / 1e9 / (3 * nodes);   // 每台机器 3 GB/s 读权重，外加 2 分钟初始化
  return {
    ok: !errors.length, errors, rMax, at, step, prefill,
    wBytes, wPerGpu, kvGbPerGpu, overhead, kvTok, capTokens, kvConc, maxConc, loadSec, nodes,
    hbm: G.hbm, costPerHour: g * G.price,
    costPerMOut: rMax ? g * G.price / 3600 / (0.75 * rMax * work.outTok) * 1e6 : Infinity,
  };
}
const fmtK = (n) => (n >= 1024 ? Math.round(n / 1024) + 'K' : String(n));

// ---------- 任务 ----------
const CHAT_HOURLY = [0.45, 0.32, 0.24, 0.2, 0.19, 0.22, 0.3, 0.45, 0.62, 0.75, 0.82, 0.86, 0.8, 0.78, 0.84, 0.88, 0.86, 0.84, 0.86, 0.92, 0.98, 1.0, 0.92, 0.7];
const WORK_HOURLY = [0.12, 0.08, 0.06, 0.05, 0.05, 0.06, 0.1, 0.25, 0.55, 0.85, 0.97, 1.0, 0.82, 0.9, 1.0, 0.98, 0.94, 0.85, 0.66, 0.48, 0.38, 0.32, 0.24, 0.17];
const CHAT = { inTok: 1200, outTok: 350, ttftSlo: 1.5, tpotSlo: 0.05 };

export const SERVE_SCENARIOS = [
  {
    id: 'daily', title: '日常流量', level: '入门', big: '600', bigSub: '请求/秒 · 日常晚高峰',
    brief: '一天的流量跟着作息走：凌晨最少，晚上九点最多。夜里少开几个副本，白天留够余量。',
    work: CHAT, sloTarget: 0.99, budget: 22000, perMTarget: 1.65,
    traffic: { peak: 600, hourly: CHAT_HOURLY, noise: 0.06 },
  },
  {
    id: 'launch', title: '发布会', level: '困难', big: '20:00', bigSub: '新模型发布会',
    brief: '平时晚高峰 600 请求/秒，开场 10 分钟翻一倍。冷启动要好几分钟，扩容慢了就排队。',
    work: CHAT, sloTarget: 0.99, budget: 25000, perMTarget: 1.7,
    traffic: { peak: 600, hourly: CHAT_HOURLY, noise: 0.06, spike: { at: 1200, add: 640, rise: 7, hold: 25, decay: 85 } },
    pre: { at: 1180, until: 1380 },
  },
  {
    id: 'agent', title: '长上下文', level: '专家', big: '32K', bigSub: 'Agent · 平均输入 token',
    brief: '编程 Agent 每次都把整个仓库塞进来：预填充又长又贵，KV cache 一个请求就是几个 GB，白天上班时间最忙。',
    work: { inTok: 32768, outTok: 800, ttftSlo: 6, tpotSlo: 0.06 }, sloTarget: 0.99, budget: 30000, perMTarget: 20.5,
    traffic: { peak: 32, hourly: WORK_HOURLY, noise: 0.08 },
  },
];

export function demandAt(sc, t) {
  const tr = sc.traffic, h = t / 60, i = Math.floor(h) % 24, f = h - Math.floor(h);
  let d = tr.peak * (tr.hourly[i] * (1 - f) + tr.hourly[(i + 1) % 24] * f);
  const s = tr.spike;
  if (s && t >= s.at) d += s.add * (1 - Math.exp(-(t - s.at) / s.rise)) * Math.exp(-Math.max(0, t - s.at - s.hold) / s.decay);
  return d;
}
// 一天 1,440 分钟的流量；rng 给了就加噪声
export function demandSeries(sc, rng = null) {
  const out = new Float64Array(1440);
  for (let t = 0; t < 1440; t++) out[t] = demandAt(sc, t) * (rng ? 1 + (rng() - 0.5) * sc.traffic.noise : 1);
  return out;
}

export function seededRng(seed) {
  let s = (seed >>> 0) % 2147483647 || 7;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}

export const hallSlots = (gpus) => Math.floor(HALL_GPUS / gpus);
const ceilTo = (x, k) => Math.max(k, Math.ceil(x / k) * k);

// 按模型和任务给一套“能跑但不一定好”的默认扩缩容：最少按晚高峰六成算，最多按全天峰值；默认不预扩，等玩家自己发现
export function defaultScale(sc, perf, gpus, target = 0.75) {
  const per = Math.max(1e-6, target * perf.rMax), slots = hallSlots(gpus);
  let peak = 0; for (let t = 0; t < 1440; t++) peak = Math.max(peak, demandAt(sc, t));
  const step = per * 4 >= sc.traffic.peak ? 1 : 4;
  const min = Math.min(slots, ceilTo(0.6 * sc.traffic.peak / per, step));
  const max = Math.min(slots, Math.max(min, ceilTo(peak / per, step === 1 ? 2 : 8)));
  return { min, max, target, pre: 0 };
}

// ---------- 一天 ----------
// 副本槽位：0 空闲，1 冷启动，2 在线，3 故障维修中
export const SLOT_IDLE = 0, SLOT_START = 1, SLOT_ON = 2, SLOT_FAIL = 3;

// opts: { demand: Float64Array(1440), failRng, failAt: [分钟], shedAt, record }
export function createDay(model, plan, sc, perf, opts = {}) {
  const nSlots = hallSlots(plan.gpus);
  const day = {
    sc, plan: { ...plan }, perf, model, t: 0, nSlots,
    slots: new Uint8Array(nSlots), until: new Float64Array(nSlots),
    active: 0, starting: 0, failed: 0,
    backlog: 0, shed: false, shedAuto: false, shedCalm: 0, lastUp: -99, lastDown: -99,
    demand: opts.demand || demandSeries(sc), failRng: opts.failRng || null, failAt: opts.failAt ? [...opts.failAt] : null,
    shedAt: opts.shedAt ?? Infinity, record: opts.record !== false,
    acc: { served: 0, ok: 0, gpuMin: 0, busyMin: 0, startMin: 0, idleNight: 0, idleDay: 0, cost: 0, starts: 0, failures: 0, rejected: 0 },
    rows: [], events: [], failTimes: [],
  };
  const want0 = Math.ceil(day.demand[0] / (plan.target * Math.max(1e-6, perf.rMax)));
  const n0 = Math.min(plan.max, Math.max(plan.min, want0));
  for (let i = 0; i < n0 && i < nSlots; i++) { day.slots[i] = SLOT_ON; day.active++; }
  return day;
}

export function cloneDay(d, opts = {}) {
  return {
    ...d, plan: { ...d.plan }, slots: d.slots.slice(), until: d.until.slice(), acc: { ...d.acc },
    rows: opts.keepRows ? d.rows.slice() : [], events: [], failTimes: d.failTimes.slice(),
    demand: opts.demand || d.demand, failRng: opts.failRng ?? null, failAt: opts.failAt ?? null,
    shedAt: opts.shedAt ?? d.shedAt, record: opts.record ?? d.record,
  };
}

const minFloor = (d, t) => (d.sc.pre && d.plan.pre && t >= d.sc.pre.at && t < d.sc.pre.until ? Math.max(d.plan.min, d.plan.pre) : d.plan.min);
function emit(d, e) { if (d.record) d.events.push({ t: d.t, ...e }); }

export function startReplicas(d, n, why = 'scale') {
  let k = 0;
  for (let i = 0; i < d.nSlots && k < n; i++) {
    if (d.slots[i] !== SLOT_IDLE) continue;
    d.slots[i] = SLOT_START; d.until[i] = d.t + d.perf.loadSec / 60; d.starting++; d.acc.starts++; k++;
  }
  if (k) emit(d, { kind: 'start', n: k, why, ready: d.t + d.perf.loadSec / 60 });
  return k;
}
function stopReplicas(d, n) {
  let k = 0;
  for (let i = d.nSlots - 1; i >= 0 && k < n; i--) if (d.slots[i] === SLOT_ON) { d.slots[i] = SLOT_IDLE; d.active--; k++; }
  if (k) emit(d, { kind: 'stop', n: k });
  return k;
}
function failReplica(d, rnd) {
  const on = []; for (let i = 0; i < d.nSlots; i++) if (d.slots[i] === SLOT_ON) on.push(i);
  if (!on.length) return;
  const i = on[Math.floor((rnd ? rnd() : 0.5) * on.length) % on.length];
  d.slots[i] = SLOT_FAIL; d.until[i] = d.t + 30 + (rnd ? rnd() : 0.5) * 60; d.active--; d.failed++;
  d.acc.failures++; d.failTimes.push(d.t);
  emit(d, { kind: 'fail', slot: i, kindIdx: rnd ? Math.floor(rnd() * 4) : 0 });
  startReplicas(d, 1, 'replace');
}

// 推进一分钟，返回这一分钟的记录
export function stepDay(d) {
  const t = d.t, plan = d.plan, perf = d.perf, work = d.sc.work;
  const lam = d.demand[Math.min(1439, t)];
  // 冷启动完的上线，修好的回到空闲
  for (let i = 0; i < d.nSlots; i++) {
    const s = d.slots[i];
    if (s === SLOT_START && d.until[i] <= t) { d.slots[i] = SLOT_ON; d.starting--; d.active++; }
    else if (s === SLOT_FAIL && d.until[i] <= t) { d.slots[i] = SLOT_IDLE; d.failed--; }
  }
  // 故障
  if (d.failAt) { while (d.failAt.length && d.failAt[0] <= t) { d.failAt.shift(); failReplica(d, null); } }
  else if (d.failRng) {
    const p = d.active * plan.gpus / SLOT_FAIL_MTBF / 60;
    if (d.failRng() < p) failReplica(d, d.failRng);
  }
  // 扩缩容：先保底，再按负载每 2 分钟加最多 8 个；缩容要等 10 分钟没扩过
  const floor = minFloor(d, t);
  const want = Math.ceil(lam / (plan.target * Math.max(1e-6, perf.rMax)));
  if (d.active + d.starting < floor) startReplicas(d, floor - d.active - d.starting, t === d.sc.pre?.at ? 'pre' : 'floor');
  if (t % 2 === 0) {
    const goal = Math.min(plan.max, Math.max(want, floor));
    if (d.active + d.starting < goal) { if (startReplicas(d, Math.min(8, goal - d.active - d.starting))) d.lastUp = t; }
  }
  if (t % 5 === 0 && !d.starting && t - d.lastUp >= 10 && t - d.lastDown >= 5) {
    const keep = Math.max(floor, Math.ceil(want * 1.2));
    if (d.active > keep) { stopReplicas(d, Math.min(2, d.active - keep)); d.lastDown = t; }
  }
  // 限流
  if (t === d.shedAt && !d.shed) { d.shed = true; d.shedAuto = true; emit(d, { kind: 'shed', on: true, auto: true }); }
  const cap = d.active * perf.rMax;
  if (d.shed) {
    if (d.backlog > 0) { d.acc.rejected += d.backlog; d.backlog = 0; }
    if (lam > cap) d.acc.rejected += (lam - cap) * 60;
    // 为了扛过一波而开的限流，能扛的量追上来 5 分钟就自动解除；手动开的要手动关
    d.shedCalm = cap >= lam * 1.05 ? d.shedCalm + 1 : 0;
    if (d.shedAuto && d.shedCalm >= 5) { d.shed = false; d.shedAuto = false; d.shedCalm = 0; emit(d, { kind: 'shed', on: false, auto: true }); }
  } else d.backlog = Math.max(0, d.backlog + (lam - cap) * 60);
  // 延迟和达标
  let ttft, tpot, good, B = 0;
  if (d.active > 0) {
    const r = Math.min(lam, cap) / d.active;
    const st = perf.at(Math.min(r, perf.rMax)) || { ttftP99: 60, tpotP99: work.tpotSlo, B: perf.maxConc };
    B = st.B; tpot = st.tpotP99;
    ttft = st.ttftP99 + d.backlog / Math.max(1, cap);
    good = ttft <= work.ttftSlo && tpot <= work.tpotSlo + 1e-9 ? 0.997 : Math.max(0.2, work.ttftSlo / ttft);
    if (d.shed && lam > cap) good *= cap / lam;
  } else { ttft = 60 + d.backlog / Math.max(1, lam); tpot = 0; good = 0; }
  const a = d.acc, gpus = plan.gpus, busy = Math.min(lam, cap) / Math.max(1e-6, perf.rMax) * gpus;
  a.served += lam * 60; a.ok += lam * 60 * good;
  a.gpuMin += (d.active + d.starting) * gpus; a.busyMin += busy; a.startMin += d.starting * gpus;
  const idle = Math.max(0, d.active * gpus - busy);
  if (t < 420 || t >= 1380) a.idleNight += idle; else a.idleDay += idle;
  a.cost += (d.active + d.starting) * perf.costPerHour / 60;
  const row = { t, lam, cap, active: d.active, starting: d.starting, failed: d.failed, ttft, tpot, good, backlog: d.backlog, shed: d.shed, batch: B };
  if (d.record) d.rows.push(row);
  d.t = t + 1;
  return row;
}

// autoRelease：扛过这一波就自动解除（过载时选“限流”就是这种）
export function setShed(d, on, autoRelease = false) {
  if (d.shed === on) return;
  d.shed = on; d.shedAuto = on && autoRelease; d.shedCalm = 0;
  emit(d, { kind: 'shed', on, auto: false });
}

export function finishDay(d) { while (d.t < 1440) stepDay(d); return summarize(d); }

export function summarize(d) {
  const a = d.acc, rows = d.rows, work = d.sc.work;
  const outTok = a.served * work.outTok;
  let peakLam = 0, peakT = 0, peakActive = 0, minActive = Infinity, maxTtft = 0, maxBacklog = 0, maxTtftT = 0;
  let badMin = 0, badFrom = null, badTo = null, run = 0, longest = 0, longestFrom = null, curFrom = null;
  for (const r of rows) {
    if (r.lam > peakLam) { peakLam = r.lam; peakT = r.t; }
    peakActive = Math.max(peakActive, r.active); minActive = Math.min(minActive, r.active);
    if (r.ttft > maxTtft) { maxTtft = r.ttft; maxTtftT = r.t; }
    maxBacklog = Math.max(maxBacklog, r.backlog);
    if (r.good < 0.99) {
      badMin++; badFrom ??= r.t; badTo = r.t;
      if (!run) curFrom = r.t;
      run++; if (run > longest) { longest = run; longestFrom = curFrom; }
    } else run = 0;
  }
  const gpuMin = Math.max(1, a.gpuMin);
  return {
    sloPct: a.served ? a.ok / a.served * 100 : 0, served: a.served, outTok, cost: a.cost,
    perM: outTok ? a.cost / (outTok / 1e6) : Infinity, gpuHours: a.gpuMin / 60,
    busyPct: a.busyMin / gpuMin * 100, startPct: a.startMin / gpuMin * 100,
    idleDayPct: a.idleDay / gpuMin * 100, idleNightPct: a.idleNight / gpuMin * 100,
    starts: a.starts, failures: a.failures, rejected: a.rejected,
    peakLam, peakT, peakActive, minActive: minActive === Infinity ? 0 : minActive, maxTtft, maxTtftT, maxBacklog,
    badMin, badFrom, badTo, longest, longestFrom,
  };
}

export function grade(s, sc) {
  const sloOk = s.sloPct >= sc.sloTarget * 100 - 1e-9, costOk = s.cost <= sc.budget, perMOk = s.perM <= sc.perMTarget;
  const n = (costOk ? 1 : 0) + (perMOk ? 1 : 0);
  return { sloOk, costOk, perMOk, letter: !sloOk ? 'C' : n === 2 ? 'S' : n === 1 ? 'A' : 'B' };
}

// 整天预估：平滑的流量，不出故障
export function forecast(model, plan, sc, perf = null, opts = {}) {
  const p = perf || replicaPerf(model, plan, sc.work);
  if (!p.ok) return null;
  const d = createDay(model, plan, sc, p, { demand: opts.demand || demandSeries(sc), failAt: opts.failAt || [], shedAt: opts.shedAt, record: opts.record !== false });
  return { day: d, sum: finishDay(d), perf: p };
}

// 老周：在默认方案附近试几种改法，挑最管用的说
// opts: { perf, demand, failAt }——复盘时用当天的实际流量和故障时间重跑
export function advise(model, plan, sc, opts = {}) {
  const perf = opts.perf || replicaPerf(model, plan, sc.work);
  if (!perf.ok) return { base: null, tips: [] };
  const run = (p, record = false) => forecast(model, p, sc, perf, { ...opts, record });
  const base = run(plan, true);
  const tips = [];
  const target = sc.sloTarget * 100, slots = hallSlots(plan.gpus);
  // 1. 达标率不够：发布会先试预扩，再试降低目标负载、提高最少副本
  let fix = null;
  if (base.sum.sloPct < target) {
    if (sc.pre) for (let pre = (plan.pre || plan.min) + 8; pre <= Math.min(plan.max, slots); pre += 8) {
      const r = run({ ...plan, pre }); if (r.sum.sloPct >= target) { fix = { kind: 'pre', plan: { ...plan, pre }, r }; break; }
    }
    if (!fix) for (const tg of TARGET_OPTIONS.filter((x) => x < plan.target).reverse()) {
      const r = run({ ...plan, target: tg }); if (r.sum.sloPct >= target) { fix = { kind: 'target', plan: { ...plan, target: tg }, r }; break; }
    }
    if (!fix && plan.max < slots) {
      const r = run({ ...plan, max: slots }); if (r.sum.sloPct >= target) fix = { kind: 'max', plan: { ...plan, max: slots }, r };
    }
  }
  // 2. 夜里最少留几个就够：在不掉达标率的前提下往下压
  const ref = fix ? fix.plan : plan, refSum = fix ? fix.r.sum : base.sum;
  let save = null;
  for (let min = ref.min - (ref.min > 8 ? 4 : 1); min >= 1; min -= ref.min > 8 ? 4 : 1) {
    const r = run({ ...ref, min });
    if (r.sum.sloPct < Math.min(refSum.sloPct, target) - 0.02 || r.sum.sloPct < target && refSum.sloPct >= target) break;
    save = { plan: { ...ref, min }, r };
  }
  return { base, perf, fix, save, combined: save ? save.r : fix ? fix.r : base };
}
