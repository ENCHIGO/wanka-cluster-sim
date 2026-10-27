// 万卡集群模拟器 · 推理部署模型
// 从 Hugging Face 的 config.json 估参数和 KV cache，再估一个副本能扛多少流量，
// 最后按分钟推进一整天：流量、自动扩缩容、冷启动、故障、排队、限流。
// 纯函数加一个可变的“这一天”状态，不依赖 DOM，可以在 Node 里测试。量级要对，不是逐 kernel 仿真。

// 加速卡的规格表在 gpus.js，训练和推理共用
import { GPUS as SERVE_GPUS, VENDORS, GPU_IDS, gpuName } from './gpus.js';
export { SERVE_GPUS, VENDORS, GPU_IDS, gpuName };
export const PRECISION = {
  bf16: { id: 'bf16', label: 'BF16', bytes: 2 },
  fp8: { id: 'fp8', label: 'FP8', bytes: 1 },
  // 4 比特加分组缩放因子。只压专家和 FFN（开源的 MXFP4 / NVFP4 / INT4 权重都是这么做的），注意力、词表按 8 比特
  fp4: { id: 'fp4', label: 'FP4', bytes: 0.53 },
};
export const PREC_KEYS = ['bf16', 'fp8', 'fp4'];
export const CARD_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 72];
export const BATCH_OPTIONS = [32, 64, 128, 256, 512, 1024, 2048, 4096];
export const TARGET_OPTIONS = [0.6, 0.75, 0.9];
export const NODE_GPUS = 8;
export const HALL_GPUS = 1024;            // 推理大厅 B：32 柜
const SLOT_FAIL_MTBF = 8000;              // 每卡平均多少小时出一次让副本挂掉的问题（掉卡、OOM、进程崩溃）
const BASE_TTFT = 0.18;                   // 分词、调度、网络往返
const MTP_ACCEPT = [0.75, 0.56, 0.42];    // MTP 第 1、2、3 个草稿 token 平均被接受的概率

export const gpuStock = (id) => SERVE_GPUS[id]?.stock || HALL_GPUS;
// 一个副本可以用几张卡：72 只有整柜 72 卡一个域的才行
export const cardOptions = (id) => CARD_OPTIONS.filter((c) => (c !== 72 || SERVE_GPUS[id].domain === 72) && c <= gpuStock(id));
// 默认最大并发：每卡 32 个左右
export const defaultBatch = (gpus) => BATCH_OPTIONS.reduce((a, b) => (Math.abs(Math.log2(b / (32 * gpus))) < Math.abs(Math.log2(a / (32 * gpus))) ? b : a), 64);

// ---------- config.json → 模型规格 ----------
// 每层注意力是哪一种：full 全注意力；sparse 稀疏注意力（DSA 这类：KV 全存，每个 token 只挑 index_topk 个看）；
// swa 滑窗；linear 线性注意力（Gated DeltaNet、KDA：每个请求只存一份固定大小的状态）；
// csa / hca：DeepSeek V4 的压缩注意力，每 r 个 token 压成一条（csa 再用索引器挑 top-k，hca 全看），两种都另带一个滑窗
function layerKinds(cfg, N, hasIdx) {
  const cr = Array.isArray(cfg.compress_ratios) ? cfg.compress_ratios : null;
  const lt = Array.isArray(cfg.layer_types) ? cfg.layer_types : null;
  const lin = cfg.linear_attn_config && (cfg.linear_attn_config.kda_layers || cfg.linear_attn_config.full_attn_layers) ? cfg.linear_attn_config : null;
  const hp = Array.isArray(cfg.hybrid_layer_pattern) ? cfg.hybrid_layer_pattern : null;
  const sp = Array.isArray(cfg.sparse_attention_freq) ? cfg.sparse_attention_freq : null;
  const swp = cfg.sliding_window_pattern;
  const out = [];
  for (let i = 0; i < N; i++) {
    let k = 'full', r = 0;
    if (cr) { r = cr[i] || 0; k = !r ? 'swa' : r <= 16 ? 'csa' : 'hca'; }
    else if (lt) { const t = String(lt[i] || ''); k = /linear|delta|mamba|kda/.test(t) ? 'linear' : /sliding|chunk|local/.test(t) ? 'swa' : 'full'; }
    else if (cfg.full_attention_interval > 1) k = (i + 1) % cfg.full_attention_interval ? 'linear' : 'full';
    else if (lin) k = (lin.kda_layers ? lin.kda_layers.includes(i + 1) : !lin.full_attn_layers.includes(i + 1)) ? 'linear' : 'full';   // 这两个列表从 1 数
    else if (hp) k = hp[i] ? 'swa' : 'full';
    else if (swp > 1 && cfg.sliding_window) k = (i + 1) % swp ? 'swa' : 'full';
    if (k === 'full' && hasIdx && (!sp || sp[i])) k = 'sparse';
    out.push({ k, r });
  }
  const groups = [];
  for (const x of out) { const g = groups.find((y) => y.k === x.k && y.r === x.r); if (g) g.n++; else groups.push({ ...x, n: 1 }); }
  return groups;
}

export function parseHfConfig(c) {
  if (!c || typeof c !== 'object' || Array.isArray(c)) throw new Error('不是一个 JSON 对象');
  const cfg = c.text_config ? { ...c, ...c.text_config } : c;
  const H = cfg.hidden_size || cfg.d_model || cfg.n_embd, N = cfg.num_hidden_layers || cfg.n_layer || cfg.num_layers;
  const A = cfg.num_attention_heads || cfg.n_head;
  if (!H || !N || !A) throw new Error('config.json 里缺 hidden_size、num_hidden_layers 或 num_attention_heads');
  const arch = (cfg.architectures || [cfg.model_type || '未知'])[0];
  const K = cfg.num_key_value_heads || cfg.num_kv_heads || A;
  const d = cfg.head_dim || H / A;
  const V = cfg.vocab_size || 32000;
  const I = cfg.intermediate_size || cfg.ffn_hidden_size || 4 * H;
  const E = cfg.n_routed_experts || cfg.num_local_experts || cfg.num_experts || 0;
  const topk = E ? (cfg.num_experts_per_tok || cfg.num_experts_per_token || cfg.moe_topk || cfg.top_k || 2) : 0;
  const moeI = cfg.moe_intermediate_size || I;
  const eH = E ? cfg.routed_expert_hidden_size || cfg.moe_latent_size || H : H;   // 有的模型先把 token 投影到小一点的维度再进专家
  const nShared = cfg.n_shared_experts ?? cfg.num_shared_experts ?? 0;
  const sharedI = nShared * moeI + (cfg.shared_expert_intermediate_size || 0);
  const denseLayers = !E ? N : Array.isArray(cfg.moe_layer_freq) ? cfg.moe_layer_freq.slice(0, N).filter((x) => !x).length
    : Array.isArray(cfg.mlp_only_layers) ? cfg.mlp_only_layers.length : Math.min(N, cfg.first_k_dense_replace || 0);
  const mla = cfg.kv_lora_rank ? {
    kvLora: cfg.kv_lora_rank, qLora: cfg.q_lora_rank || 0, rope: cfg.qk_rope_head_dim || 64,
    nope: cfg.qk_nope_head_dim || 128, v: cfg.v_head_dim || 128,
  } : null;
  // DeepSeek V4：一个 512 维的 KV 头，K 和 V 共用；输出投影按组做低秩
  const v4 = !mla && (Array.isArray(cfg.compress_ratios) || cfg.o_lora_rank) ? { qLora: cfg.q_lora_rank || 0, oLora: cfg.o_lora_rank || 0, oGroups: cfg.o_groups || 1 } : null;
  const dv = mla ? mla.v : v4 ? d : cfg.v_head_dim || d;
  const gate = cfg.attn_output_gate ?? /qwen3_?5|qwen3next/i.test(arch);     // Qwen3.5 系列的注意力输出带门控
  const idx = cfg.index_topk ? { topk: cfg.index_topk, heads: cfg.index_n_heads || 64, dim: cfg.index_head_dim || 128 } : null;
  const groups = layerKinds(cfg, N, !!idx);
  const count = (k) => groups.filter((g) => g.k === k).reduce((s, g) => s + g.n, 0);
  const lin = cfg.linear_attn_config || {};
  const la = !count('linear') ? null : cfg.linear_attn_config
    ? { hv: lin.num_heads || A, hk: lin.num_heads || A, dk: lin.head_dim || 128, dv: lin.head_dim || 128, conv: lin.short_conv_kernel_size || 4 }
    : { hv: cfg.linear_num_value_heads || A, hk: cfg.linear_num_key_heads || cfg.linear_num_value_heads || A, dk: cfg.linear_key_head_dim || 128, dv: cfg.linear_value_head_dim || 128, conv: cfg.linear_conv_kernel_dim || 4 };

  // 参数量
  const qLora = mla ? mla.qLora : v4 ? v4.qLora : 0;
  const qP = (w) => (qLora ? H * qLora + qLora * w : H * w);
  const softP = mla
    ? qP(A * (mla.nope + mla.rope)) + H * (mla.kvLora + mla.rope) + mla.kvLora * A * (mla.nope + mla.v) + A * mla.v * H
    : v4 ? qP(A * d) + H * d + (v4.oLora ? A * d * v4.oLora + v4.oGroups * v4.oLora * H : A * d * H)
      : H * A * d * (gate ? 2 : 1) + H * K * (d + dv) + A * dv * H;
  const idxP = idx ? (qLora || H) * idx.heads * idx.dim + H * idx.dim + H * idx.heads : 0;
  const linP = la ? H * (2 * la.hk * la.dk + 2 * la.hv * la.dv) + 2 * H * la.hv + la.hv * la.dv * H + (2 * la.hk * la.dk + la.hv * la.dv) * la.conv : 0;
  let attnP = 0;
  for (const g of groups) attnP += g.n * (g.k === 'linear' ? linP : softP + (g.k === 'csa' || g.k === 'sparse' ? idxP : 0));
  const expertP = 3 * eH * moeI, sharedP = 3 * H * sharedI, latentP = eH !== H ? 2 * H * eH : 0;
  const sparseL = N - denseLayers;
  const denseFfn = denseLayers * 3 * H * I;
  const embed = V * H * (cfg.tie_word_embeddings ? 1 : 2);
  const routed = sparseL * E * expertP;
  const total = embed + attnP + denseFfn + routed + sparseL * (sharedP + latentP + H * E);
  const ffnActive = denseFfn + sparseL * (topk * expertP + sharedP + latentP);
  const active = V * H + attnP + ffnActive;

  const q = cfg.quantization_config || {};
  const method = String(q.quant_method || '').toLowerCase(), algo = String(q.quant_algo || '').toLowerCase();
  const bits = Object.values(q.config_groups || {}).map((x) => x && x.weights && x.weights.num_bits).find(Boolean);
  const native = /fp4|int4|awq|gptq|w4a/.test(method + algo) || bits === 4 ? 'fp4' : method === 'fp8' || /fp8/.test(algo) || bits === 8 ? 'fp8' : 'bf16';

  // KV：每层每 token 存的字节（BF16）、索引器的 key、滑窗长度、线性注意力每个请求的状态
  const e = mla ? (mla.kvLora + mla.rope) * 2 : v4 ? d * 2 : K * (d + dv) * 2;
  const att = {
    e, groups, idxE: idx ? idx.dim + 4 : 0, topk: idx ? idx.topk : 0,
    win: cfg.sliding_window || cfg.window_size || cfg.attention_chunk_size || 4096,
    state: la ? (la.hv * la.dk * la.dv + (2 * la.hk * la.dk + la.hv * la.dv) * (la.conv - 1)) * 2 : 0,
    // 每看一个 token 的乘加：解码时 MLA 在压缩后的 512 维上算，预填充时展开算
    attended: mla ? 2 * A * (2 * mla.kvLora + mla.rope) : 2 * A * (d + dv),
    attendedPre: mla ? 2 * A * (mla.nope + mla.rope + mla.v) : 2 * A * (d + dv),
    idxFl: idx ? 2 * idx.heads * idx.dim : 0, linFl: la ? 4 * la.hv * la.dk * la.dv : 0,
  };
  return {
    arch, layers: N, hidden: H, heads: A, kvHeads: v4 ? 1 : K, headDim: d, vocab: V, intermediate: I,
    experts: E, topk, moeI, sharedExperts: nShared || (cfg.shared_expert_intermediate_size ? 1 : 0), denseLayers, mla, v4: !!v4, idx, la,
    gqa: !mla && !v4, linName: cfg.linear_attn_config ? 'KDA' : 'Gated DeltaNet',
    counts: { full: count('full'), sparse: count('sparse'), swa: count('swa'), linear: count('linear'), csa: count('csa'), hca: count('hca') },
    maxCtx: cfg.max_position_embeddings || cfg.max_sequence_length || cfg.seq_length || 4096, native, quant: method,
    total, active, routed, q4: routed + denseFfn, ffnShare: ffnActive / active,
    mtp: cfg.num_nextn_predict_layers || cfg.num_mtp_modules || cfg.mtp_num_hidden_layers || 0,
    att,
  };
}

// config.json 里值得给人看的字段，按这个顺序挑
export const KEY_FIELDS = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'kv_lora_rank',
  'compress_ratios', 'index_topk', 'layer_types', 'full_attention_interval', 'linear_attn_config', 'hybrid_layer_pattern', 'sliding_window',
  'n_routed_experts', 'num_experts', 'num_local_experts', 'num_experts_per_tok', 'num_experts_per_token', 'n_shared_experts', 'num_shared_experts',
  'moe_intermediate_size', 'num_nextn_predict_layers', 'num_mtp_modules', 'vocab_size', 'max_position_embeddings'];
// 数组字段（每层是什么注意力）按种类数一数
const tally = (arr, name) => {
  const m = new Map(); for (const x of arr) m.set(name(x), (m.get(name(x)) || 0) + 1);
  return [...m].map(([k, n]) => `${k}×${n}`).join(' ');
};
export function keyFields(c, n = 14) {
  const cfg = c.text_config ? { ...c, ...c.text_config } : c;
  const out = [];
  for (const k of KEY_FIELDS) {
    const v = cfg[k];
    if (v == null || out.length >= n) continue;
    if (k === 'compress_ratios' && Array.isArray(v)) out.push([k, tally(v, (x) => x || 'swa')]);
    else if (k === 'layer_types' && Array.isArray(v)) out.push([k, tally(v, (x) => String(x).replace('_attention', ''))]);
    else if (k === 'hybrid_layer_pattern' && Array.isArray(v)) out.push([k, tally(v, (x) => (x ? 'swa' : 'full'))]);
    else if (k === 'linear_attn_config' && v.kda_layers) out.push(['kda_layers', `${v.kda_layers.length} 层`]);
    else if (typeof v !== 'object') out.push([k, v]);
  }
  return out;
}

// 默认用哪种精度：原生 FP8 / 4 比特的就用原生的，否则在线量化成 FP8
export const defaultPrecision = (m) => (m.native === 'fp4' ? 'fp4' : 'fp8');
export const weightBytes = (m, prec) => (prec === 'fp4' ? m.q4 * PRECISION.fp4.bytes + (m.total - m.q4) : m.total * PRECISION[prec].bytes);

// ---------- KV cache 和注意力 ----------
// 上下文长 n 的一个请求，KV cache（加上线性注意力的状态）占多少字节；rep：GQA 的 KV 头在张量并行里复制了几份
export function kvSeqBytes(m, n, fp8 = true, rep = 1) {
  const a = m.att, e = a.e * (fp8 ? 0.5 : 1) * rep, w = Math.min(n, a.win);
  let b = 0;
  for (const g of a.groups) {
    const c = g.r ? Math.ceil(n / g.r) : 0;
    b += g.n * (g.k === 'full' ? n * e : g.k === 'sparse' ? n * (e + a.idxE) : g.k === 'swa' ? w * e
      : g.k === 'csa' ? c * (e + a.idxE) + w * e : g.k === 'hca' ? c * e + w * e : a.state);
  }
  return b;
}
// 解码一步，每个请求要从显存里读多少字节的 KV
export function kvReadBytes(m, n, fp8 = true, rep = 1) {
  const a = m.att, e = a.e * (fp8 ? 0.5 : 1) * rep, w = Math.min(n, a.win);
  let b = 0;
  for (const g of a.groups) {
    const c = g.r ? Math.ceil(n / g.r) : 0;
    b += g.n * (g.k === 'full' ? n * e : g.k === 'sparse' ? Math.min(n, a.topk) * e + n * a.idxE : g.k === 'swa' ? w * e
      : g.k === 'csa' ? Math.min(c, a.topk) * e + c * a.idxE + w * e : g.k === 'hca' ? c * e + w * e : 2 * a.state);
  }
  return b;
}
// 注意力本身的浮点运算（不含 QKV、输出投影）：解码一个 token
export function attnDecodeFlops(m, n) {
  const a = m.att, w = Math.min(n, a.win);
  let f = 0;
  for (const g of a.groups) {
    const c = g.r ? Math.ceil(n / g.r) : 0;
    f += g.n * (g.k === 'full' ? n * a.attended : g.k === 'sparse' ? Math.min(n, a.topk) * a.attended + n * a.idxFl
      : g.k === 'swa' ? w * a.attended : g.k === 'csa' ? (Math.min(c, a.topk) + w) * a.attended + c * a.idxFl
        : g.k === 'hca' ? (c + w) * a.attended : a.linFl);
  }
  return f;
}
// 预填充一段 L 长的提示词
export function attnPrefillFlops(m, L) {
  const a = m.att, tri = (cap) => (L <= cap ? L * L / 2 : cap * L - cap * cap / 2);   // Σ min(t, cap)
  let f = 0;
  for (const g of a.groups) {
    f += g.n * (g.k === 'full' ? tri(L) * a.attendedPre
      : g.k === 'sparse' ? tri(a.topk) * a.attendedPre + L * L / 2 * a.idxFl
        : g.k === 'swa' ? tri(a.win) * a.attendedPre
          : g.k === 'csa' ? (tri(a.topk * g.r) / g.r + tri(a.win)) * a.attendedPre + L * L / 2 / g.r * a.idxFl
            : g.k === 'hca' ? (L * L / 2 / g.r + tri(a.win)) * a.attendedPre : 1.5 * L * a.linFl);
  }
  return f;
}

// ---------- 一个副本的性能 ----------
// plan: { gpu, gpus, precision, kvFp8, maxBatch, mtp }；work: { inTok, outTok, ttftSlo, tpotSlo, hit }
export function replicaPerf(m, plan, work) {
  const G = SERVE_GPUS[plan.gpu] || SERVE_GPUS.H200, g = plan.gpus;
  const prec = PRECISION[plan.precision] ? plan.precision : 'fp8';
  const tp = Math.min(g, NODE_GPUS), nodes = Math.ceil(g / NODE_GPUS);
  // 矩阵乘按权重精度的速度算；4 比特只有带 FP4 张量核的卡算得更快，别的卡解开成 8 比特算
  const tf16 = G.tf16 * 1e12, tf8 = G.tf8 * 1e12;
  const tfW = prec === 'bf16' ? tf16 : prec === 'fp8' ? tf8 : 1 / (m.ffnShare / ((G.tf4 || G.tf8 * 0.85) * 1e12) + (1 - m.ffnShare) / tf8);
  // MTP 投机解码：每步每个请求多猜 k 个 token 一起验，平均出 tps 个
  const k = plan.mtp && m.mtp ? Math.min(3, m.mtp) : 0;
  const tps = 1 + MTP_ACCEPT.slice(0, k).reduce((s, x) => s + x, 0);
  const wBytes = weightBytes(m, prec) * (1 + k / m.layers);   // 一个 MTP 模块差不多是一层
  const routedBytes = m.routed * (prec === 'fp4' ? PRECISION.fp4.bytes : PRECISION[prec].bytes) * (1 + k / m.layers);
  const wPerGpu = wBytes / g / 1e9;
  const overhead = 4 + 0.06 * G.hbm;
  const kvGbPerGpu = G.hbm - wPerGpu - overhead;
  const errors = [];
  if (kvGbPerGpu <= 1) errors.push(`放不下：每卡要放 ${wPerGpu.toFixed(0)} GB 权重，${G.name} 只有 ${G.hbm} GB`);
  // MLA、V4 用数据并行注意力，KV 不重复存；GQA 的 KV 头比张量并行的份数少时，每张卡都要存一份
  const rep = m.gqa ? Math.max(1, tp / m.kvHeads) : 1;
  const ctx = work.inTok + work.outTok / 2;
  const kvSeq = kvSeqBytes(m, ctx, plan.kvFp8, rep), kvRead = kvReadBytes(m, ctx, plan.kvFp8, rep);
  const capSeqs = g * Math.max(0, kvGbPerGpu) * 1e9 / kvSeq;
  const kvConc = Math.floor(capSeqs);
  const maxConc = Math.max(0, Math.min(plan.maxBatch, kvConc));
  if (!errors.length && work.inTok + work.outTok > m.maxCtx) errors.push(`上下文不够：请求要 ${fmtK(work.inTok + work.outTok)} token，模型最多 ${fmtK(m.maxCtx)}`);
  if (!errors.length && kvConc < 1) errors.push('KV cache 连一个请求都放不下');

  // 通信。张量并行：每层两次 all-reduce；副本超过 8 卡的 MoE 走大规模专家并行：注意力按数据并行切，每层一次 all-to-all 发出去再收回来。
  // 超出 scale-up 域的那部分走网卡
  const out = g > G.domain ? 1 - G.domain / g : 0;
  const bwComm = 1 / ((1 - out) / (G.link * 1e9) + out / (G.nic * 1e9));
  const lat = out ? 25e-6 : 8e-6, wideEp = m.experts && g > NODE_GPUS;
  const commLayer = (tokens) => (g <= 1 ? 0 : wideEp ? 2 * lat + tokens / g * m.topk * m.hidden * 3 / bwComm : 2 * (lat + tokens * m.hidden * 2 / bwComm));
  const attnTok = attnDecodeFlops(m, ctx);
  const step = (B) => {
    const T = B * (1 + k);                    // 这一步要验算的位置
    const touched = m.experts ? 1 - Math.pow(1 - m.topk / m.experts, T) : 1;
    const wRead = (wBytes - routedBytes * (1 - touched)) / g;
    const tMem = (wRead + B * kvRead / g) / (G.bw * 1e12 * 0.82);
    const tCmp = T * (2 * m.active / tfW + attnTok / tf16) / (g * 0.5);
    return Math.max(tMem, tCmp) + m.layers * commLayer(T) * (wideEp ? 0.5 : 1) + 0.002;   // 专家并行两批交替，通信藏一半
  };
  // 预填充：前缀缓存命中的部分不用重算，新的 token 照样要看完整的上下文
  const hit = work.hit || 0;
  const prefill = (L) => {
    const fresh = L * (1 - hit);
    return (2 * m.active * fresh / tfW + (attnPrefillFlops(m, L) - attnPrefillFlops(m, L - fresh)) / tf16) / (g * 0.4) + m.layers * commLayer(fresh) * 0.5;
  };

  // 每副本每秒 r 个请求时的稳定状态
  const cache = new Map();
  const tp1 = prefill(work.inTok);
  const at = (r) => {
    const key = Math.round(r * 1000);
    if (cache.has(key)) return cache.get(key);
    const share = r * tp1;
    let res = null;
    if (share < 0.92 && maxConc > 0) {
      let B = Math.max(1, r * work.outTok * step(1) / tps);
      for (let i = 0; i < 40; i++) B = Math.max(1, r * work.outTok * step(B) / tps / (1 - share));
      const tpot = step(B) / tps / (1 - share);
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
  if (!errors.length) for (let i = 0; i < 40; i++) { const mid = (lo + hi) / 2; if (ok(at(mid))) lo = mid; else hi = mid; }
  if (!errors.length && lo < 1e-3) {
    const one = at(0.001);
    errors.push(one && one.ttftP99 > work.ttftSlo ? `一个请求的首 token 就要 ${one.ttftP99.toFixed(1)} 秒，过不了 SLO` : `每 token 要 ${Math.round(step(1) / tps * 1250)} ms，过不了 SLO`);
  }
  const rMax = errors.length ? 0 : lo;
  const loadSec = 120 + wBytes / 1e9 / (3 * nodes);   // 每台机器 3 GB/s 读权重，外加 2 分钟初始化
  return {
    ok: !errors.length, errors, rMax, at, step, prefill, tps, mtpK: k, rep,
    wBytes, wPerGpu, kvGbPerGpu, overhead, kvSeq, capTokens: capSeqs * ctx, kvConc, maxConc, loadSec, nodes,
    hbm: G.hbm, costPerHour: g * G.price,
    costPerMOut: rMax ? g * G.price / 3600 / (0.75 * rMax * work.outTok) * 1e6 : Infinity,
  };
}
const fmtK = (n) => (n >= 1048576 ? Math.round(n / 1048576) + 'M' : n >= 1024 ? Math.round(n / 1024) + 'K' : String(n));

// 每百万输出 token 最便宜的副本：所有卡型、卡数、并发都试一遍（精度用默认的，KV 存 FP8，有 MTP 就开）
export function bestReplica(m, work, perfFn = replicaPerf, prec = defaultPrecision(m), gpuIds = GPU_IDS) {
  let best = null;
  for (const gpu of gpuIds) for (const gpus of cardOptions(gpu)) for (const maxBatch of BATCH_OPTIONS) {
    const plan = { gpu, gpus, precision: prec, kvFp8: true, maxBatch, mtp: !!m.mtp };
    const p = perfFn(m, plan, work);
    if (p.ok && (!best || p.costPerMOut < best.perf.costPerMOut)) best = { plan, perf: p };
  }
  return best;
}

// ---------- 任务 ----------
const CHAT_HOURLY = [0.45, 0.32, 0.24, 0.2, 0.19, 0.22, 0.3, 0.45, 0.62, 0.75, 0.82, 0.86, 0.8, 0.78, 0.84, 0.88, 0.86, 0.84, 0.86, 0.92, 0.98, 1.0, 0.92, 0.7];
const WORK_HOURLY = [0.12, 0.08, 0.06, 0.05, 0.05, 0.06, 0.1, 0.25, 0.55, 0.85, 0.97, 1.0, 0.82, 0.9, 1.0, 0.98, 0.94, 0.85, 0.66, 0.48, 0.38, 0.32, 0.24, 0.17];
const CHAT = { inTok: 2000, outTok: 600, ttftSlo: 1.5, tpotSlo: 0.05 };
// 编程 Agent：每次带着大半个仓库来，八成上下文命中前缀缓存，只有新加的部分要预填充
const AGENT = { inTok: 65536, outTok: 1200, ttftSlo: 8, tpotSlo: 0.06, hit: 0.8 };

export const SERVE_SCENARIOS = [
  {
    id: 'daily', title: '日常流量', level: '入门', big: '1,000', bigSub: '请求/秒 · 日常晚高峰',
    brief: '一天的流量跟着作息走：凌晨最少，晚上九点最多。夜里少开几个副本，白天留够余量。',
    work: CHAT, sloTarget: 0.99,
    traffic: { peak: 1000, hourly: CHAT_HOURLY, noise: 0.06 },
  },
  {
    id: 'launch', title: '发布会', level: '困难', big: '20:00', bigSub: '新模型发布会',
    brief: '平时晚高峰 1,000 请求/秒，开场 10 分钟翻一倍。冷启动要好几分钟，扩容慢了就排队。',
    work: CHAT, sloTarget: 0.99,
    traffic: { peak: 1000, hourly: CHAT_HOURLY, noise: 0.06, spike: { at: 1200, add: 1100, rise: 7, hold: 25, decay: 85 } },
    pre: { at: 1180, until: 1380 },
  },
  {
    id: 'agent', title: '长上下文', level: '专家', big: '64K', bigSub: 'Agent · 平均上下文 token',
    brief: '编程 Agent 每次带着大半个仓库来：八成能命中前缀缓存，剩下的预填充也不便宜；KV cache 一个请求就要好几个 GB，白天上班时间最忙。',
    work: AGENT, sloTarget: 0.99,
    traffic: { peak: 48, hourly: WORK_HOURLY, noise: 0.08 },
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

export const hallSlots = (gpus, gpu) => Math.floor(gpuStock(gpu) / gpus);
const ceilTo = (x, k) => Math.max(k, Math.ceil(x / k) * k);

// 按模型和任务给一套“能跑但不一定好”的默认扩缩容：最少按晚高峰六成算，最多按全天峰值；默认不预扩，等玩家自己发现
export function defaultScale(sc, perf, plan, target = plan.target || 0.75) {
  const per = Math.max(1e-6, target * perf.rMax), slots = hallSlots(plan.gpus, plan.gpu);
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
  const nSlots = hallSlots(plan.gpus, plan.gpu);
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
  if (d.starting || d.failed) for (let i = 0; i < d.nSlots; i++) {
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

// tg：这个模型在这个任务上的预算和单价目标（scenarioTargets）
export function grade(s, sc, tg) {
  const sloOk = s.sloPct >= sc.sloTarget * 100 - 1e-9, costOk = s.cost <= tg.budget, perMOk = s.perM <= tg.perM;
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
  const target = sc.sloTarget * 100, slots = hallSlots(plan.gpus, plan.gpu);
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

// 预算和每百万输出 token 的目标按模型定：每种卡挑一个最省的副本配置，单价最低的几种照默认规则配扩缩容、
// 让老周调一遍跑完一天（新卡数量少，扛不住高峰的就换下一种），取一天花得最少的那个当参考，
// 预算给它的 1.2 倍，单价给 1.08 倍。拿 S 得接近最优；随手配一般只能拿 A
export function scenarioTargets(m, sc, perfFn = replicaPerf) {
  const cands = GPU_IDS.map((id) => bestReplica(m, sc.work, perfFn, defaultPrecision(m), [id])).filter(Boolean)
    .sort((a, b) => a.perf.costPerMOut - b.perf.costPerMOut);
  let ref = null;
  for (const c of cands.slice(0, 5)) {
    const plan = { ...c.plan, ...defaultScale(sc, c.perf, c.plan) };
    const a = advise(m, plan, sc, { perf: c.perf });
    const r = { plan: (a.save || a.fix || { plan }).plan, sum: a.combined.sum };
    r.ok = r.sum.sloPct >= sc.sloTarget * 100;
    if (!ref || (r.ok && (!ref.ok || r.sum.cost < ref.sum.cost)) || (!ref.ok && !r.ok && r.sum.sloPct > ref.sum.sloPct)) ref = r;
  }
  if (!ref) return { budget: Infinity, perM: Infinity, ref: null };
  return { budget: niceCeil(ref.sum.cost * 1.2), perM: niceCeil(ref.sum.perM * 1.08), ref };
}
const niceCeil = (x) => { const e = 10 ** (Math.floor(Math.log10(x)) - 1); return Math.ceil(x / e - 1e-9) * e; };   // 两位有效数字，往上取
