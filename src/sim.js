// 万卡集群模拟器 · 性能与可靠性模型
// 纯函数，不依赖 DOM，可以在 Node 里直接测试。
// 模型刻意做了简化：数量级要对，取舍要有意思，而不是逐包仿真。

export const SEQ_LEN = 4096;

export const GPUS = {
  H100: { id: 'H100', name: 'H100 SXM 80GB', tflops: 989, hbm: 80, nvlink: 450, price: 2.2, mtbf: 50000 },
  B200: { id: 'B200', name: 'B200 192GB', tflops: 2250, hbm: 192, nvlink: 900, price: 5.5, mtbf: 40000 },
};

// bw: 每卡有效带宽 GB/s；price: 每卡每小时 $；flapMtbf: 每卡平均多少小时出一次链路抖动
export const NETWORKS = {
  roce200: { id: 'roce200', name: 'RoCE 200G', bw: 25 * 0.7, price: 0.12, flapMtbf: 300000 },
  ib400: { id: 'ib400', name: 'InfiniBand NDR 400G', bw: 50 * 0.85, price: 0.3, flapMtbf: 900000 },
  ib800: { id: 'ib800', name: 'InfiniBand XDR 800G', bw: 100 * 0.85, price: 0.6, flapMtbf: 900000 },
};

// bw: 聚合写带宽 GB/s；price: 每小时 $
export const STORAGE = {
  std: { id: 'std', name: '并行文件系统 · 标准', bw: 100, price: 400 },
  fast: { id: 'fast', name: '全闪存储 · 高速', bw: 600, price: 1800 },
};

export const MODELS = {
  dense70b: { id: 'dense70b', name: 'Dense 70B', params: 70e9, active: 70e9, hidden: 8192, layers: 80, moe: false, critBatch: 4 * 2 ** 20 },
  dense405b: { id: 'dense405b', name: 'Dense 405B', params: 405e9, active: 405e9, hidden: 16384, layers: 126, moe: false, critBatch: 16 * 2 ** 20 },
  moe671b: {
    id: 'moe671b', name: 'MoE 671B-A37B', params: 671e9, active: 37e9, dense: 17e9, hidden: 7168, layers: 61,
    moe: true, experts: 256, topk: 8, critBatch: 16 * 2 ** 20,
  },
};

export const SCENARIOS = [
  {
    id: 'warmup', title: '入职第一周', model: 'dense70b', tokens: 2e12, deadlineDays: 16, budget: 1.7e6, maxGpus: 4096,
    brief: '老板要一个 70B 基座，16 天交付。集群不大，先把并行策略和 checkpoint 节奏摸清楚。',
    defaults: { gpu: 'H100', gpus: 2048, net: 'ib400', storage: 'std', tp: 8, pp: 4, ep: 1, gbs: 1024, recompute: false, ckptMin: 60, spares: 2 },
  },
  {
    id: 'wanka', title: '万卡 405B', model: 'dense405b', tokens: 8e12, deadlineDays: 36, budget: 40e6, maxGpus: 16384,
    brief: '一万张卡，平均几个小时就坏一张。算力不是瓶颈，能不能一直跑下去才是。',
    defaults: { gpu: 'H100', gpus: 10240, net: 'ib400', storage: 'std', tp: 8, pp: 16, ep: 1, gbs: 2048, recompute: false, ckptMin: 120, spares: 4 },
  },
  {
    id: 'moe', title: 'MoE 冲刺', model: 'moe671b', tokens: 15e12, deadlineDays: 20, budget: 11e6, maxGpus: 16384,
    brief: '671B 稀疏模型，每个 token 只激活 37B，但专家之间的 all-to-all 会把网络打满。',
    defaults: { gpu: 'H100', gpus: 10240, net: 'ib400', storage: 'std', tp: 1, pp: 16, ep: 64, gbs: 4096, recompute: false, ckptMin: 60, spares: 4 },
  },
];

export const GPU_COUNTS = [512, 1024, 2048, 4096, 8192, 10240, 12288, 16384];
export const TP_OPTIONS = [1, 2, 4, 8, 16];
export const PP_OPTIONS = [1, 2, 4, 8, 16, 32];
export const EP_OPTIONS = [1, 8, 16, 32, 64, 128, 256];
export const GBS_OPTIONS = [256, 512, 1024, 2048, 4096, 8192];
export const CKPT_OPTIONS = [10, 20, 30, 60, 120, 240, 480];

export const RESTART_BASE_SEC = 600;      // 重新调度 + NCCL 初始化
export const REPAIR_WAIT_SEC = 3 * 3600;  // 没有热备时原地等维修
export const NODE_GPUS = 8;

// Chinchilla 拟合 (Hoffmann et al. 2022)
export function lossAt(activeParams, tokens) {
  const d = Math.max(tokens, 1e8);
  return 1.69 + 406.4 / activeParams ** 0.34 + 410.7 / d ** 0.28;
}

export function evaluate(cfg, sc) {
  const model = MODELS[sc.model];
  const gpu = GPUS[cfg.gpu];
  const net = NETWORKS[cfg.net];
  const sto = STORAGE[cfg.storage];
  const N = cfg.gpus, tp = cfg.tp, pp = cfg.pp;
  const ep = model.moe ? cfg.ep : 1;
  const errors = [], warnings = [];

  const mp = tp * pp;
  const dp = N / mp;
  if (N > sc.maxGpus) errors.push(`本关最多只能申请 ${sc.maxGpus} 张卡`);
  if (!Number.isInteger(dp)) errors.push(`${N} 张卡不能被 TP×PP=${mp} 整除`);
  if (pp > model.layers) errors.push(`PP=${pp} 超过模型层数 ${model.layers}`);
  if (model.moe) {
    if (Number.isInteger(dp) && dp % ep) errors.push(`EP=${ep} 必须整除 DP=${dp}`);
    if (model.experts % ep) errors.push(`EP=${ep} 必须整除专家数 ${model.experts}`);
  }
  const m = cfg.gbs / dp;
  if (Number.isInteger(dp) && m < 1) errors.push(`DP=${dp} 大于全局批次 ${cfg.gbs}，每个副本分不到一条样本`);
  if (errors.length) return { ok: false, errors, warnings };

  const tokensPerStep = cfg.gbs * SEQ_LEN;
  const peak = gpu.tflops * 1e12;

  // 计算：TP 越大单卡 GEMM 越小，效率越低；MoE 每卡放的专家越多，每个专家分到的 token 越少，GEMM 越碎
  const moeEff = model.moe ? 0.9 - 0.07 * Math.log2(model.experts / ep) : 1;
  const kEff = 0.66 * (1 - 0.035 * Math.log2(tp)) * moeEff;
  const flopsStep = 6 * model.active * tokensPerStep * (cfg.recompute ? 4 / 3 : 1);
  const tComp = flopsStep / (N * peak * kEff);

  const layersPerStage = model.layers / pp;
  const tokPerReplica = tokensPerStep / dp;

  // TP：每层 4 次激活集合通信，节点内走 NVLink，跨节点走网卡；约 40% 能和计算重叠
  const tpBw = (tp <= NODE_GPUS ? gpu.nvlink : net.bw) * 1e9;
  const tTP = tp > 1 ? tokPerReplica * layersPerStage * model.hidden * 2 * 4 * (tp - 1) / tp / tpBw * 0.6 : 0;

  // EP：dispatch + combine，前向反向各一次，top-k 份拷贝；一半能被重叠
  let tEP = 0;
  if (model.moe && ep > 1) {
    const epBw = (ep * tp <= NODE_GPUS ? gpu.nvlink : net.bw) * 1e9;
    tEP = (tokPerReplica / tp) * layersPerStage * model.hidden * 2 * model.topk * 4 * (ep - 1) / ep / epBw * 0.5;
  }

  const work = tComp + tTP + tEP;
  const tBubble = pp > 1 ? work * (pp - 1) / m : 0;

  // 每卡参数分片
  const expert = model.moe ? model.params - model.dense : 0;
  const denseP = model.moe ? model.dense : model.params;
  const shard = denseP / mp + expert / (mp * ep);

  // DP：梯度 reduce-scatter + all-gather，约 70% 藏在反向里
  const tAR = dp > 1 ? 2 * (dp - 1) / dp * shard * 2 / (net.bw * 1e9) : 0;
  const tDP = tAR * 0.3;
  const tOpt = shard * 16 / 3e12;
  const stepSec = work + tBubble + tDP + tOpt;
  const mfu = 6 * model.active * tokensPerStep / (N * peak * stepSec);

  // 显存：bf16 权重 + fp32 梯度，Adam 状态按 DP 切分 (ZeRO-1)
  const memWeights = shard * 6;
  const memOpt = (denseP / mp) * 12 / dp + (expert / (mp * ep)) * 12 / Math.max(1, dp / ep);
  const perLayer = SEQ_LEN * model.hidden * (cfg.recompute ? 2 : 34) / tp * (model.moe ? 1.5 : 1);
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
  const failPerHour = N / gpu.mtbf + N / net.flapMtbf;
  const mtbfHours = 1 / failPerHour;
  const mtbfSec = mtbfHours * 3600;
  const youngMin = Math.sqrt(2 * ckptSec * mtbfSec) / 60;
  const intervalSec = cfg.ckptMin * 60;
  const overhead = Math.min(0.95, ckptSec / intervalSec + (intervalSec / 2 + restartSec + (cfg.spares > 0 ? 0 : REPAIR_WAIT_SEC)) / mtbfSec);
  const etaDays = sc.tokens / (rate * (1 - overhead)) / 86400;
  const costPerHour = (N + cfg.spares * NODE_GPUS) * (gpu.price + net.price) + sto.price;
  const estCost = costPerHour * etaDays * 24;

  if (memGB > gpu.hbm) errors.push(`显存溢出：每卡需要 ${memGB.toFixed(1)} GB，${gpu.id} 只有 ${gpu.hbm} GB`);
  else if (memGB > gpu.hbm * 0.92) warnings.push(`显存只剩 ${(gpu.hbm - memGB).toFixed(1)} GB 余量，碎片一多就会 OOM`);
  if (tp > NODE_GPUS) warnings.push(`TP=${tp} 跨节点了，张量并行的通信走网卡，非常慢`);
  if (model.moe && ep * tp > NODE_GPUS) warnings.push(`EP×TP=${ep * tp} 超出单节点，all-to-all 走网卡`);
  if (pp > 1 && tBubble / stepSec > 0.15) warnings.push(`流水线气泡占步时 ${(tBubble / stepSec * 100).toFixed(0)}%：微批次数 ${m.toFixed(1)} 相对 PP=${pp} 太少`);
  if (tokenEff < 1) warnings.push(`全局批次 ${(tokensPerStep / 2 ** 20).toFixed(0)}M token 超过临界批次，token 利用率降到 ${(tokenEff * 100).toFixed(0)}%`);
  if (cfg.ckptMin > youngMin * 2) warnings.push(`checkpoint 间隔偏长，故障时丢的进度多（建议约 ${Math.round(youngMin)} 分钟）`);
  if (cfg.ckptMin < youngMin / 2) warnings.push(`checkpoint 太频繁，写盘时间浪费多（建议约 ${Math.round(youngMin)} 分钟）`);
  if (cfg.spares === 0) warnings.push('没有热备节点：坏一台就得原地等维修，约 3 小时');
  if (etaDays > sc.deadlineDays) warnings.push(`预计 ${etaDays.toFixed(1)} 天完成，超过 ${sc.deadlineDays} 天的期限`);
  if (estCost > sc.budget) warnings.push(`预计花费 $${(estCost / 1e6).toFixed(2)}M，超出预算`);

  return {
    ok: errors.length === 0, errors, warnings,
    dp, m, tokensPerStep, stepSec, mfu, mem, tokenEff, rate,
    breakdown: { comp: tComp, tp: tTP, ep: tEP, bubble: tBubble, dp: tDP + tOpt },
    ckptSec, restartSec, mtbfHours, youngMin, overhead, etaDays, costPerHour, estCost,
    failPerHour, model, gpu, net, sto,
  };
}
