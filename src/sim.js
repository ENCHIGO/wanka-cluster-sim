// 万卡集群模拟器 · 性能与可靠性模型
// 纯函数，不依赖 DOM，可以在 Node 里直接测试。
// 模型刻意做了简化：数量级要对，取舍要有意思，而不是逐包仿真。

import { GPUS, gpuName } from './gpus.js';

export { GPUS };
export const SEQ_LEN = 4096;

// bw: 每卡有效带宽 GB/s（还要受卡上网卡的限制）；price: 每卡每小时 $；flapMtbf: 每卡平均多少小时出一次链路抖动
export const NETWORKS = {
  roce200: { id: 'roce200', name: 'RoCE 200G', short: 'RoCE 200G', bw: 25 * 0.7, price: 0.12, flapMtbf: 300000 },
  ib400: { id: 'ib400', name: 'InfiniBand NDR 400G', short: 'IB 400G', bw: 50 * 0.85, price: 0.3, flapMtbf: 900000 },
  ib800: { id: 'ib800', name: 'InfiniBand XDR 800G', short: 'IB 800G', bw: 100 * 0.85, price: 0.6, flapMtbf: 900000 },
  x1600: { id: 'x1600', name: 'ConnectX-9 1.6T', short: '1.6T', bw: 200 * 0.85, price: 1.0, flapMtbf: 700000 },
};

// 训练精度。eff：矩阵乘能跑到这个精度峰值的几成（量化开销、留在 BF16 的算子都算进去）；act：激活显存是 BF16 的几成；
// spike：loss 尖峰是 BF16 的几倍
export const TRAIN_PRECISION = {
  bf16: { id: 'bf16', label: 'BF16', eff: 1, act: 1, spike: 1 },
  fp8: { id: 'fp8', label: 'FP8', eff: 0.78, act: 0.7, spike: 1.3 },
  fp4: { id: 'fp4', label: 'FP4', eff: 0.55, act: 0.55, spike: 2 },
};
export const precPeak = (gpu, p) => (p === 'fp4' ? gpu.tf4 : p === 'fp8' ? gpu.tf8 : gpu.tf16);
// 8 比特是 INT8 的卡不能用 FP8 训练；没有 4 比特张量核的不能用 FP4
export const precOk = (gpu, p) => p === 'bf16' || (p === 'fp8' && !gpu.f8) || (p === 'fp4' && gpu.tf4 > 0);
export const SPIKE_PER_HOUR = 1 / 70;     // BF16 训练平均每 70 小时一次 loss 尖峰

// bw: 聚合写带宽 GB/s；price: 每小时 $
export const STORAGE = {
  std: { id: 'std', name: '并行文件系统 · 标准', bw: 100, price: 400 },
  fast: { id: 'fast', name: '全闪存储 · 高速', bw: 600, price: 1800 },
};

// dense：MoE 里不是路由专家的那部分参数（注意力、共享专家、词表），每张卡都要放一份切片
export const MODELS = {
  dense27b: { id: 'dense27b', name: 'Dense 27B', params: 27e9, active: 27e9, hidden: 5120, layers: 64, moe: false, critBatch: 8 * 2 ** 20 },
  moe35b: {
    id: 'moe35b', name: 'MoE 35B-A3B', params: 35e9, active: 3e9, dense: 2.5e9, hidden: 2048, layers: 40,
    moe: true, experts: 256, topk: 8, critBatch: 8 * 2 ** 20,
  },
  moe744b: {
    id: 'moe744b', name: 'MoE 744B-A40B', params: 744e9, active: 40e9, dense: 20e9, hidden: 6144, layers: 78,
    moe: true, experts: 256, topk: 8, critBatch: 32 * 2 ** 20,
  },
  moe1600b: {
    id: 'moe1600b', name: 'MoE 1.6T-A49B', params: 1.6e12, active: 49e9, dense: 26e9, hidden: 7168, layers: 61,
    moe: true, experts: 384, topk: 6, critBatch: 64 * 2 ** 20,
  },
  moe2800b: {
    id: 'moe2800b', name: 'MoE 2.8T-A104B', params: 2.8e12, active: 104e9, dense: 57e9, hidden: 7168, layers: 93,
    moe: true, experts: 896, topk: 16, critBatch: 64 * 2 ** 20,
  },
};

// gpus：这单只能用哪几种卡（没写就随便挑）
export const SCENARIOS = [
  {
    id: 'dense', title: '27B 稠密', model: 'dense27b', tokens: 10e12, deadlineDays: 14, budget: 4.0e6, maxGpus: 4096,
    requester: '预训练组', hall: '1 号楼 3F', size: '27B', kind: 'Dense', level: '入门',
    brief: '新人练手的单子：一个 27B 的稠密基座。卡不多，一天坏一两次，主要看并行切分、训练精度和存盘间隔定得对不对。',
    defaults: { gpu: 'H200', gpus: 4096, net: 'ib400', storage: 'std', precision: 'bf16', tp: 4, pp: 2, ep: 1, gbs: 2048, recompute: false, ckptMin: 30, spares: 6 },
  },
  {
    id: 'small-moe', title: '35B-A3B 小 MoE', model: 'moe35b', tokens: 24e12, deadlineDays: 12, budget: 2.5e6, maxGpus: 4096,
    requester: '端侧模型组', hall: '1 号楼 3F', size: '35B', kind: 'MoE · 激活 3B', level: '进阶',
    brief: '每个 token 只算 3B，256 个小专家。算得快，通信反倒成了大头：EP 切多大、all-to-all 出不出 NVLink 域，工期能差出一倍。',
    defaults: { gpu: 'H200', gpus: 2048, net: 'ib400', storage: 'std', precision: 'bf16', tp: 1, pp: 1, ep: 8, gbs: 2048, recompute: false, ckptMin: 30, spares: 4 },
  },
  {
    id: 'domestic', title: '国产算力', model: 'moe744b', tokens: 15e12, deadlineDays: 30, budget: 22e6, maxGpus: 16384, gpus: ['A910C', 'A950PR'],
    requester: '国产化项目组', hall: '5 号楼 2F', size: '744B', kind: 'MoE · 激活 40B', level: '困难',
    brief: '这单只能用昇腾。910C 没有 FP8，只能 BF16 硬算，但 384 张卡一个超节点，专家并行不出灵衢；950PR 能跑 FP8 和 FP4，可一台只有 8 卡，EP 一大就得走网卡。',
    defaults: { gpu: 'A910C', gpus: 16384, net: 'ib400', storage: 'std', precision: 'bf16', tp: 4, pp: 2, ep: 32, gbs: 8192, recompute: true, ckptMin: 20, spares: 24 },
  },
  {
    id: 'wanka', title: '1.6T MoE 万卡', model: 'moe1600b', tokens: 33e12, deadlineDays: 12, budget: 20e6, maxGpus: 16384,
    requester: '基座模型部', hall: '3 号楼 2F', size: '1.6T', kind: 'MoE · 激活 49B', level: '困难',
    brief: '一万多张卡，平均一两个小时坏一台。备机留几台、多久存一次盘、用 FP8 还是 FP4、上不上 NVL72，决定了能不能按期交。',
    defaults: { gpu: 'B200', gpus: 16384, net: 'ib400', storage: 'fast', precision: 'fp8', tp: 1, pp: 32, ep: 8, gbs: 16384, recompute: true, ckptMin: 20, spares: 20 },
  },
  {
    id: 'giant', title: '2.8T MoE', model: 'moe2800b', tokens: 30e12, deadlineDays: 24, budget: 55e6, maxGpus: 16384,
    requester: '旗舰模型组', hall: '3 号楼 4F', size: '2.8T', kind: 'MoE · 激活 104B', level: '专家',
    brief: '896 个专家，每个 token 挑 16 个。HGX 一台 8 卡，EP 一大 all-to-all 就走网卡；NVL72 一柜 72 卡全走 NVLink，可首批的 Rubin 只有 4,608 张。',
    defaults: { gpu: 'GB200', gpus: 16384, net: 'ib400', storage: 'fast', precision: 'fp8', tp: 2, pp: 16, ep: 32, gbs: 16384, recompute: true, ckptMin: 20, spares: 24 },
  },
];

export const GPU_COUNTS = [512, 1024, 2048, 4096, 8192, 10240, 12288, 16384];
export const TP_OPTIONS = [1, 2, 4, 8, 16];
export const PP_OPTIONS = [1, 2, 4, 8, 16, 32];
export const EP_OPTIONS = [1, 8, 16, 32, 64, 128, 256];
export const GBS_OPTIONS = [256, 512, 1024, 2048, 4096, 8192, 16384];
export const CKPT_OPTIONS = [10, 20, 30, 60, 120, 240, 480];

export const RESTART_BASE_SEC = 600;      // 重新调度 + NCCL 初始化
export const REPAIR_WAIT_SEC = 3 * 3600;  // 没有备机时原地等维修
export const NODE_GPUS = 8;
// 送修：坏机 12–48 小时回来，换下的慢机器 12–36 小时回来
export const REPAIR_HOURS = [12, 48];
export const SWAP_REPAIR_HOURS = [12, 36];
// 过热降频：同步训练整体按 0.62 倍速跑；七成会在 2–8 小时内自己降下来
export const STRAGGLER_SLOWDOWN = 0.62;
export const STRAGGLER_RECOVER = { chance: 0.7, hours: [2, 8] };
// 预估里假设玩家发现慢机器后大约 1.5 小时内处理掉
const STRAGGLER_HANDLED_HOURS = 1.5;

// 泊松分布 P(X >= k)
function poissonTail(lambda, k) {
  if (k <= 0) return 1;
  let term = Math.exp(-lambda), cdf = term;
  for (let i = 1; i < k; i++) { term *= lambda / i; cdf += term; }
  return Math.max(0, 1 - cdf);
}

// Chinchilla 拟合 (Hoffmann et al. 2022)
export function lossAt(activeParams, tokens) {
  const d = Math.max(tokens, 1e8);
  return 1.69 + 406.4 / activeParams ** 0.34 + 410.7 / d ** 0.28;
}

export function evaluate(cfg, sc) {
  const model = MODELS[sc.model];
  const gpu = GPUS[cfg.gpu] || GPUS.H100;
  const net = NETWORKS[cfg.net] || NETWORKS.ib400;
  const sto = STORAGE[cfg.storage] || STORAGE.std;
  const prec = TRAIN_PRECISION[cfg.precision] ? cfg.precision : 'bf16', P = TRAIN_PRECISION[prec];
  const N = cfg.gpus, tp = cfg.tp, pp = cfg.pp;
  const ep = model.moe ? cfg.ep : 1;
  const errors = [], warnings = [];

  const mp = tp * pp;
  const dp = N / mp;
  if (sc.gpus && !sc.gpus.includes(gpu.id)) errors.push(`这单只能用${sc.gpus.map(gpuName).join('、')}`);
  if (N > sc.maxGpus) errors.push(`这单最多批 ${sc.maxGpus} 卡`);
  if (gpu.trainStock && N > gpu.trainStock) errors.push(`${gpu.name} 训练大厅里只有 ${gpu.trainStock} 张`);
  if (!precOk(gpu, prec)) errors.push(prec === 'fp8' ? `${gpu.name} 的 8 比特是 INT8，不能用 FP8 训练` : `${gpu.name} 没有 FP4 张量核，不能用 FP4 训练`);
  if (!Number.isInteger(dp)) errors.push(`${N} 卡除不尽 TP×PP=${mp}`);
  if (pp > model.layers) errors.push(`PP${pp} 比层数 ${model.layers} 还多`);
  if (model.moe) {
    if (Number.isInteger(dp) && dp % ep) errors.push(`EP${ep} 除不尽 DP${dp}`);
    if (model.experts % ep) errors.push(`EP${ep} 除不尽 ${model.experts} 个专家`);
  }
  const m = cfg.gbs / dp;
  if (Number.isInteger(dp) && m < 1) errors.push(`DP${dp} 比全局批次 ${cfg.gbs} 条还大，有的副本分不到样本`);
  if (errors.length) return { ok: false, errors, warnings };

  const tokensPerStep = cfg.gbs * SEQ_LEN;
  const peak = precPeak(gpu, prec) * 1e12;

  // 计算：TP 越大单卡 GEMM 越小，效率越低；MoE 每卡放的专家越多，每个专家分到的 token 越少，GEMM 越碎；
  // FP8 / FP4 只有矩阵乘能跑到低精度的峰值，量化和留在 BF16 的算子吃掉一部分；软件栈不成熟的卡再打个折
  const moeEff = model.moe ? 0.9 - 0.07 * Math.log2(model.experts / ep) : 1;
  const kEff = 0.66 * (1 - 0.035 * Math.log2(tp)) * moeEff * P.eff * gpu.sw;
  const flopsStep = 6 * model.active * tokensPerStep * (cfg.recompute ? 4 / 3 : 1);
  const tComp = flopsStep / (N * peak * kEff);

  const layersPerStage = model.layers / pp;
  const tokPerReplica = tokensPerStep / dp;

  // 网卡：交换机再快，也受卡上网卡的限制
  const netBw = Math.min(net.bw, gpu.nic * 0.85);
  // TP：每层 4 次激活集合通信，在一个 scale-up 域（HGX 一台 8 卡，NVL72 一柜 72 卡）里走 NVLink 这类互联，出了域走网卡；约 40% 能和计算重叠
  const tpBw = (tp <= gpu.domain ? gpu.link : netBw) * 1e9;
  const tTP = tp > 1 ? tokPerReplica * layersPerStage * model.hidden * 2 * 4 * (tp - 1) / tp / tpBw * 0.6 : 0;

  // EP：dispatch + combine，前向反向各一次，top-k 份拷贝；一半能被重叠
  let tEP = 0;
  if (model.moe && ep > 1) {
    const epBw = (ep * tp <= gpu.domain ? gpu.link : netBw) * 1e9;
    tEP = (tokPerReplica / tp) * layersPerStage * model.hidden * 2 * model.topk * 4 * (ep - 1) / ep / epBw * 0.5;
  }

  const work = tComp + tTP + tEP;
  const tBubble = pp > 1 ? work * (pp - 1) / m : 0;

  // 每卡参数分片
  const expert = model.moe ? model.params - model.dense : 0;
  const denseP = model.moe ? model.dense : model.params;
  const shard = denseP / mp + expert / (mp * ep);

  // DP：梯度 reduce-scatter + all-gather，约 70% 藏在反向里
  const tAR = dp > 1 ? 2 * (dp - 1) / dp * shard * 2 / (netBw * 1e9) : 0;
  const tDP = tAR * 0.3;
  const tOpt = shard * 16 / (gpu.bw * 1e12 * 0.8);   // 优化器读写一遍参数、梯度和 Adam 状态，看显存带宽
  const stepSec = work + tBubble + tDP + tOpt;
  const mfu = 6 * model.active * tokensPerStep / (N * peak * stepSec);   // 按所选精度的峰值算

  // 显存：bf16 权重 + fp32 梯度，Adam 状态按 DP 切分 (ZeRO-1)
  const memWeights = shard * 6;
  const memOpt = (denseP / mp) * 12 / dp + (expert / (mp * ep)) * 12 / Math.max(1, dp / ep);
  const perLayer = SEQ_LEN * model.hidden * (cfg.recompute ? 2 : 34) / tp * (model.moe ? 1.5 : 1) * P.act;
  const inflight = Math.min(pp, Math.ceil(m));
  const memAct = layersPerStage * perLayer * inflight;
  const memOther = 5e9;
  const memGB = (memWeights + memOpt + memAct + memOther) / 1e9;
  const mem = { weights: memWeights / 1e9, opt: memOpt / 1e9, act: memAct / 1e9, other: memOther / 1e9, total: memGB, cap: gpu.hbm };

  // 全局批次超过临界批次后，多出来的 token 收益递减
  const tokenEff = tokensPerStep <= model.critBatch ? 1 : 1 / (1 + 0.5 * (tokensPerStep / model.critBatch - 1));
  const rate = tokensPerStep / stepSec * tokenEff;

  const ckptSec = model.params * 14 / (sto.bw * 1e9) + 15;
  const restartSec = RESTART_BASE_SEC + ckptSec;
  const hwPerHour = N / gpu.mtbf;
  const failPerHour = hwPerHour + N / net.flapMtbf;
  const mtbfHours = 1 / failPerHour;
  const mtbfSec = mtbfHours * 3600;
  const youngMin = Math.sqrt(2 * ckptSec * mtbfSec) / 60;
  const intervalSec = cfg.ckptMin * 60;
  // 备机池：送修的机器平均有多少台在路上，坏机时没备机的概率
  const slowPerHour = N / (gpu.mtbf * 4);
  const inRepair = hwPerHour * (REPAIR_HOURS[0] + REPAIR_HOURS[1]) / 2 + slowPerHour * (1 - STRAGGLER_RECOVER.chance) * (SWAP_REPAIR_HOURS[0] + SWAP_REPAIR_HOURS[1]) / 2;
  const noSpare = poissonTail(inRepair, cfg.spares);
  let sparesNeeded = 0;
  while (poissonTail(inRepair, sparesNeeded) > 0.05) sparesNeeded++;
  const slowLoss = Math.min(0.6, slowPerHour * STRAGGLER_HANDLED_HOURS) * (1 - STRAGGLER_SLOWDOWN);
  // loss 尖峰按回滚处理：丢半个存盘间隔，再重启一次
  const spikePerHour = SPIKE_PER_HOUR * P.spike;
  const overhead = Math.min(0.95, ckptSec / intervalSec
    + (intervalSec / 2 + restartSec) / mtbfSec
    + hwPerHour * noSpare * REPAIR_WAIT_SEC / 3600
    + slowLoss
    + spikePerHour * (intervalSec / 2 + restartSec) / 3600);
  const etaDays = sc.tokens / (rate * (1 - overhead)) / 86400;
  const costPerHour = (N + cfg.spares * NODE_GPUS) * (gpu.price + net.price) + sto.price;
  const estCost = costPerHour * etaDays * 24;

  if (memGB > gpu.hbm) errors.push(`显存不够：每卡要 ${memGB.toFixed(1)} GB，${gpu.name} 只有 ${gpu.hbm} GB。`);
  else if (memGB > gpu.hbm * 0.92) warnings.push(`显存只剩 ${(gpu.hbm - memGB).toFixed(1)} GB 余量，碎片一多就 OOM。`);
  const dom = gpu.domain > NODE_GPUS ? `${gpu.domain} 卡的 ${gpu.fabric} 域` : '单机';
  if (tp > gpu.domain) warnings.push(`TP${tp} 出了${dom}，张量并行要走网卡，很慢。`);
  if (model.moe && ep * tp > gpu.domain) warnings.push(`EP×TP=${ep * tp} 出了${dom}，all-to-all 要走网卡。`);
  if (net.bw > gpu.nic * 0.85 + 1e-9) warnings.push(`${gpu.name} 的网卡只到 ${gpu.nic * 8}G，${net.short} 用不满，白花钱。`);
  if (prec === 'fp4' && gpu.tf4 <= gpu.tf8) warnings.push(`${gpu.name} 的 FP4 不比 FP8 快，用 FP4 只会多出 loss 尖峰。`);
  else if (prec === 'fp4') warnings.push('FP4 训练每步快，但 loss 尖峰大约是 BF16 的两倍，存盘间隔别拉太长。');
  if (pp > 1 && tBubble / stepSec > 0.15) warnings.push(`气泡占了 ${(tBubble / stepSec * 100).toFixed(0)}%，PP${pp} 只分到 ${m.toFixed(1)} 个微批。`);
  if (tokenEff < 1) warnings.push(`批次 ${(tokensPerStep / 2 ** 20).toFixed(0)}M 超过临界批次，每个 token 只顶 ${(tokenEff * 100).toFixed(0)}% 用。`);
  if (cfg.ckptMin > youngMin * 2) warnings.push(`存盘间隔太长，坏一次丢得多。${Math.round(youngMin)} 分钟左右合适。`);
  if (cfg.ckptMin < youngMin / 2) warnings.push(`存盘太勤，时间都花在写盘上。${Math.round(youngMin)} 分钟左右合适。`);
  if (cfg.spares === 0) warnings.push('不留备机的话，坏一台就要停 3 小时等现场修。');
  else if (noSpare > 0.2) warnings.push(`备机不够：平均有 ${inRepair.toFixed(1)} 台在送修，${cfg.spares} 台备机经常用光，用光了坏机就要停 3 小时。至少留 ${sparesNeeded} 台。`);
  if (etaDays > sc.deadlineDays) warnings.push(`要 ${etaDays.toFixed(1)} 天，超期了。`);
  if (estCost > sc.budget) warnings.push(`要 $${(estCost / 1e6).toFixed(2)}M，超预算了。`);
  else if (estCost > sc.budget * 0.95) warnings.push(`预算只剩 $${((sc.budget - estCost) / 1e6).toFixed(2)}M 余量，坏得多一点就会超支。`);
  if (etaDays <= sc.deadlineDays && etaDays > sc.deadlineDays * 0.92) warnings.push(`只富余 ${(sc.deadlineDays - etaDays).toFixed(1)} 天，出点状况就会超期。`);

  return {
    ok: errors.length === 0, errors, warnings,
    dp, m, tokensPerStep, stepSec, mfu, mem, tokenEff, rate,
    breakdown: { comp: tComp, tp: tTP, ep: tEP, bubble: tBubble, dp: tDP + tOpt },
    ckptSec, restartSec, mtbfHours, youngMin, overhead, etaDays, costPerHour, estCost,
    failPerHour, inRepair, noSpare, sparesNeeded, spikePerHour, model, gpu, net, sto, prec, netBw,
  };
}

// 老周的建议：先换卡、精度、卡数（PP 跟着调），挑出最有希望的几种，再调 TP、EP、存储和网络；存盘间隔和备机按建议值。
// 能按期、按预算交的里面挑最省的，都交不了就挑最快的。比现在好不到一成就不说
const nearestCkpt = (v) => CKPT_OPTIONS.reduce((a, b) => (Math.abs(Math.log(b / v)) < Math.abs(Math.log(a / v)) ? b : a));
export function suggestHardware(cfg, sc, cur = evaluate(cfg, sc)) {
  const model = MODELS[sc.model];
  const tune = (c0) => {
    const e0 = evaluate(c0, sc);
    if (!e0.ok) return null;
    const c = { ...c0, ckptMin: nearestCkpt(e0.youngMin), spares: Math.min(64, e0.sparesNeeded) };
    const e = evaluate(c, sc);
    return e.ok ? { cfg: c, ev: e, onTime: e.etaDays <= sc.deadlineDays * 0.9 && e.estCost <= sc.budget * 0.9 } : null;
  };
  const better = (x, y) => !y || (x.onTime && (!y.onTime || x.ev.estCost < y.ev.estCost)) || (!x.onTime && !y.onTime && x.ev.etaDays < y.ev.etaDays);
  const firsts = [];
  for (const gpu of sc.gpus || Object.keys(GPUS)) for (const precision of ['bf16', 'fp8', 'fp4']) {
    if (!precOk(GPUS[gpu], precision)) continue;
    let b = null;
    for (const gpus of GPU_COUNTS) for (const pp of PP_OPTIONS) {
      if (gpus > sc.maxGpus || (GPUS[gpu].trainStock && gpus > GPUS[gpu].trainStock)) continue;
      const r = tune({ ...cfg, gpu, precision, gpus, pp });
      if (r && better(r, b)) b = r;
    }
    if (b) firsts.push(b);
  }
  firsts.sort((x, y) => (better(x, y) ? -1 : better(y, x) ? 1 : 0));
  let best = firsts[0] || null;
  for (const f of firsts.slice(0, 5)) {
    const i = GPU_COUNTS.indexOf(f.cfg.gpus), counts = GPU_COUNTS.slice(Math.max(0, i - 1), i + 2);
    for (const gpus of counts) for (const tp of [1, 2, 4, 8]) for (const pp of PP_OPTIONS) for (const ep of model.moe ? EP_OPTIONS : [1]) for (const storage of ['std', 'fast']) for (const net of ['ib400', 'ib800', 'x1600']) {
      if (gpus > sc.maxGpus || (GPUS[f.cfg.gpu].trainStock && gpus > GPUS[f.cfg.gpu].trainStock)) continue;
      const r = tune({ ...f.cfg, gpus, tp, pp, ep, storage, net });
      if (r && better(r, best)) best = r;
    }
  }
  if (!best) return null;
  const curOk = cur.ok && cur.etaDays <= sc.deadlineDays && cur.estCost <= sc.budget;
  if (curOk && best.ev.estCost > cur.estCost * 0.9) return null;
  if (!curOk && cur.ok && !best.onTime && best.ev.etaDays > cur.etaDays * 0.9) return null;
  return best;
}
