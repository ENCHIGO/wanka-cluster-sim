import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CATALOG, specOf } from '../src/catalog.js';
import {
  parseHfConfig, replicaPerf, SERVE_SCENARIOS, SERVE_GPUS, defaultScale, defaultPrecision, forecast, advise, grade, scenarioTargets,
  createDay, finishDay, demandSeries, seededRng, setShed, stepDay, cloneDay, hallSlots, cardOptions, kvSeqBytes, kvReadBytes, weightBytes,
} from '../src/serve.js';

const byName = (n) => specOf(CATALOG.find((c) => c.name === n));
const v4pro = byName('DeepSeek-V4-Pro'), k3 = byName('Kimi-K3'), k25 = byName('Kimi-K2.5'), glm = byName('GLM-5.3');
const qwen35 = byName('Qwen3.5-397B-A17B'), oss = byName('gpt-oss-120b');
const [daily, launch, agent] = SERVE_SCENARIOS;
const chat = daily.work;
const plan8 = (gpu, m, extra = {}) => ({ gpu, gpus: 8, precision: defaultPrecision(m), kvFp8: true, maxBatch: 1024, mtp: !!m.mtp, ...extra });

test('从 config.json 估出来的参数量和模型卡对得上', () => {
  for (const c of CATALOG) {
    const m = specOf(c);
    const [total, active] = c.known;
    assert.ok(Math.abs(m.total / 1e9 / total - 1) < 0.08, `${c.repo} 总参数 ${(m.total / 1e9).toFixed(1)}B，模型卡 ${total}B`);
    // 模型卡的“激活参数”把输入 embedding 也算进去了，算力上它只是查表，所以这里放宽一点
    assert.ok(Math.abs(m.active / 1e9 / active - 1) < 0.12, `${c.repo} 激活 ${(m.active / 1e9).toFixed(1)}B，模型卡 ${active}B`);
  }
});

test('KV cache：MLA 只存压缩后的 576 维，GQA 存 KV 头', () => {
  assert.equal(kvSeqBytes(k25, 1000, false), 61 * (512 + 64) * 2 * 1000);
  assert.equal(kvSeqBytes(k25, 1000, true), 61 * (512 + 64) * 1000);
  const plain = parseHfConfig({ hidden_size: 4096, num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, vocab_size: 128256 });
  assert.equal(kvSeqBytes(plain, 100, false), 2 * 32 * 8 * 128 * 2 * 100);
});

test('KV cache：V4 的压缩注意力、线性注意力、滑窗都比全注意力省', () => {
  const n = 65536;
  // V4：64K 的请求不到同样层数 MLA 的三分之一
  assert.ok(kvSeqBytes(v4pro, n) < v4pro.layers * 576 * n / 3, `V4 ${(kvSeqBytes(v4pro, n) / 1e9).toFixed(2)} GB`);
  // Qwen3.5：只有 15 层全注意力随上下文涨，45 层 Gated DeltaNet 是固定的一份状态
  assert.equal(qwen35.counts.linear, 45);
  const grow = kvSeqBytes(qwen35, 2 * n) - kvSeqBytes(qwen35, n);
  assert.equal(grow, 15 * 2 * 256 * 2 * 2 * 0.5 * n);
  assert.ok(kvSeqBytes(qwen35, 10) > 45 * 64 * 128 * 128 * 2, '线性注意力的状态要算进去');
  // gpt-oss：一半的层是 128 token 的滑窗
  assert.equal(oss.counts.swa, 18);
  assert.equal(kvSeqBytes(oss, 2 * n) - kvSeqBytes(oss, n), 18 * 8 * 128 * 2 * 0.5 * n);
  // DSA：KV 全存，但解码每步只读 index_topk 个加上索引器的 key
  assert.ok(kvReadBytes(glm, n) < kvSeqBytes(glm, n) * 0.3);
});

test('缺关键字段时给出看得懂的错误', () => {
  assert.throws(() => parseHfConfig({ architectures: ['X'] }), /hidden_size/);
  assert.throws(() => parseHfConfig([1, 2]), /JSON 对象/);
});

test('原生精度：V4 的 FP4 专家、Kimi 的 INT4、gpt-oss 的 MXFP4 都按 4 比特', () => {
  assert.equal(defaultPrecision(v4pro), 'fp4');
  assert.equal(defaultPrecision(k25), 'fp4');
  assert.equal(defaultPrecision(oss), 'fp4');
  assert.equal(defaultPrecision(glm), 'fp8');
  assert.equal(defaultPrecision(byName('MiniMax-M3')), 'fp8');
  // 4 比特只压专家和 FFN，注意力、词表留 8 比特
  const w = weightBytes(v4pro, 'fp4');
  assert.ok(w > v4pro.total * 0.53 && w < v4pro.total * 0.6, `${(w / 1e9).toFixed(0)} GB`);
});

test('放不放得下：DeepSeek-V4-Pro 一台 H200 够，H100 不够；Kimi-K3 要 GB300 这一档', () => {
  assert.ok(replicaPerf(v4pro, plan8('H200', v4pro), chat).ok);
  assert.equal(replicaPerf(v4pro, plan8('H100', v4pro), chat).ok, false);
  assert.ok(replicaPerf(v4pro, plan8('H100', v4pro, { gpus: 16 }), chat).ok);
  assert.ok(replicaPerf(k3, plan8('GB300', k3), chat).ok);
  assert.equal(replicaPerf(k3, plan8('H200', k3), chat).ok, false);
  const p = replicaPerf(v4pro, plan8('B300', v4pro), chat);
  assert.ok(p.rMax > 15 && p.rMax < 120, `一个副本 ${p.rMax.toFixed(1)} 请求/秒`);
  assert.ok(p.loadSec > 120 && p.loadSec < 600, `冷启动 ${p.loadSec} 秒`);
});

test('卡数：72 卡只有整柜一个 NVLink 域的才行，新卡只到了一部分', () => {
  assert.ok(cardOptions('GB300').includes(72));
  assert.ok(!cardOptions('B300').includes(72));
  assert.equal(hallSlots(16, 'VR200'), Math.floor(SERVE_GPUS.VR200.stock / 16));
  assert.equal(hallSlots(8, 'H200'), 128);
});

test('16 卡一个副本：NVL72 柜内走 NVLink，比两台 B200 走网卡快', () => {
  const nvl = replicaPerf(k25, { ...plan8('GB200', k25), gpus: 16 }, chat), ib = replicaPerf(k25, { ...plan8('B200', k25), gpus: 16 }, chat);
  assert.ok(nvl.ok && ib.ok);
  assert.ok(nvl.step(1024) < ib.step(1024), `NVL72 ${(nvl.step(1024) * 1000).toFixed(1)} ms，IB ${(ib.step(1024) * 1000).toFixed(1)} ms`);
});

test('FP4 只有带 FP4 张量核的卡才算得更快', () => {
  const pre = (gpu, precision) => replicaPerf(glm, plan8(gpu, glm, { precision }), chat).prefill(chat.inTok);
  assert.ok(pre('B200', 'fp4') < pre('B200', 'fp8') * 0.8);
  assert.ok(pre('H200', 'fp4') >= pre('H200', 'fp8'));
});

test('MTP：每步多出 token，每 token 延迟下来（MoE 多验几个位置要多读专家，省得没那么多）', () => {
  const on = replicaPerf(glm, plan8('B300', glm), chat), off = replicaPerf(glm, plan8('B300', glm, { mtp: false }), chat);
  assert.ok(on.tps > 1.5 && off.tps === 1);
  assert.ok(on.step(16) / on.tps < off.step(16) * 0.85, `${(on.step(16) / on.tps * 1000).toFixed(1)} ms vs ${(off.step(16) * 1000).toFixed(1)} ms`);
});

test('并发开得越小，一个副本扛得越少', () => {
  const lo = replicaPerf(v4pro, plan8('B300', v4pro, { maxBatch: 64 }), chat).rMax, hi = replicaPerf(v4pro, plan8('B300', v4pro), chat).rMax;
  assert.ok(lo < hi * 0.6, `${lo.toFixed(1)} vs ${hi.toFixed(1)}`);
});

test('发布会：默认方案开场会排队，预扩以后就扛得住', () => {
  const p0 = plan8('B300', glm);
  const perf = replicaPerf(glm, p0, launch.work);
  const plan = { ...p0, ...defaultScale(launch, perf, p0) };
  const base = forecast(glm, plan, launch, perf).sum;
  assert.ok(base.sloPct < 99, `默认 ${base.sloPct.toFixed(2)}%`);
  assert.ok(base.badMin >= 5 && base.badFrom >= 1200 && base.badFrom < 1215, `排队 ${base.badMin} 分钟，从 ${base.badFrom}`);
  const a = advise(glm, plan, launch, { perf });
  assert.equal(a.fix.kind, 'pre');
  assert.ok(a.fix.r.sum.sloPct >= 99);
  assert.ok(a.save && a.save.plan.min < plan.min, '还能建议夜里少留几个');
});

test('每个内置模型在每个任务上都有一套能拿 S 的参考方案', () => {
  for (const c of CATALOG) {
    const m = specOf(c);
    for (const sc of SERVE_SCENARIOS) {
      const tg = scenarioTargets(m, sc);
      assert.ok(tg.ref && tg.ref.ok, `${c.name} ${sc.id}：参考方案 ${tg.ref?.sum.sloPct.toFixed(2)}%`);
      assert.ok(tg.ref.plan.max <= hallSlots(tg.ref.plan.gpus, tg.ref.plan.gpu));
      assert.equal(grade(tg.ref.sum, sc, tg).letter, 'S', `${c.name} ${sc.id}`);
    }
  }
});

test('入门任务：照模型页挑的卡和默认扩缩容，至少能达标', () => {
  for (const name of ['DeepSeek-V4-Pro', 'GLM-5.3', 'Qwen3.5-397B-A17B']) {
    const m = byName(name), tg = scenarioTargets(m, daily);
    const p0 = { ...tg.ref.plan };
    const perf = replicaPerf(m, p0, daily.work);
    const s = forecast(m, { ...p0, ...defaultScale(daily, perf, p0) }, daily, perf).sum;
    assert.ok(s.sloPct >= 99, `${name} ${s.sloPct}`);
  }
});

test('排队时限流比硬扛达标率高', () => {
  const p0 = plan8('B300', glm);
  const perf = replicaPerf(glm, p0, launch.work);
  const plan = { ...p0, ...defaultScale(launch, perf, p0) };
  const demand = demandSeries(launch, seededRng(3));
  const a = createDay(glm, plan, launch, perf, { demand, failAt: [] });
  while (a.t < 1207) stepDay(a);
  const b = cloneDay(a, { record: false });
  setShed(b, true);
  const wait = finishDay(cloneDay(a, { record: false })).sloPct, shed = finishDay(b).sloPct;
  assert.ok(shed > wait + 0.3, `限流 ${shed.toFixed(2)}%，硬扛 ${wait.toFixed(2)}%`);
});

test('同样的种子跑出同样的一天', () => {
  const p0 = plan8('B300', glm);
  const perf = replicaPerf(glm, p0, launch.work);
  const plan = { ...p0, ...defaultScale(launch, perf, p0) };
  const run = () => finishDay(createDay(glm, plan, launch, perf, { demand: demandSeries(launch, seededRng(11)), failRng: seededRng(5) }));
  const x = run(), y = run();
  assert.equal(x.sloPct, y.sloPct);
  assert.equal(x.cost, y.cost);
  assert.equal(x.failures, y.failures);
});
