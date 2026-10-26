import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CATALOG } from '../src/catalog.js';
import {
  parseHfConfig, replicaPerf, SERVE_SCENARIOS, defaultScale, defaultPrecision, forecast, advise, grade,
  createDay, finishDay, demandSeries, seededRng, setShed, stepDay, cloneDay, hallSlots,
} from '../src/serve.js';

const byRepo = (r) => CATALOG.find((c) => c.repo === r);
const ds3 = parseHfConfig(byRepo('deepseek-ai/DeepSeek-V3').config);
const launch = SERVE_SCENARIOS.find((s) => s.id === 'launch');
const chat = launch.work;
const h200 = { gpu: 'H200', gpus: 8, precision: 'fp8', kvFp8: true, maxBatch: 256 };

test('从 config.json 估出来的参数量和模型卡对得上', () => {
  for (const c of CATALOG) {
    const m = parseHfConfig(c.config);
    const [total, active] = c.known;
    assert.ok(Math.abs(m.total / 1e9 / total - 1) < 0.08, `${c.repo} 总参数 ${(m.total / 1e9).toFixed(1)}B，模型卡 ${total}B`);
    // 模型卡的“激活参数”把输入 embedding 也算进去了，算力上它只是查表，所以这里放宽一点
    assert.ok(Math.abs(m.active / 1e9 / active - 1) < 0.12, `${c.repo} 激活 ${(m.active / 1e9).toFixed(1)}B，模型卡 ${active}B`);
  }
});

test('MLA 每个 token 的 KV 只存压缩后的 576 维', () => {
  assert.equal(ds3.kvPerTokenBf16, 61 * (512 + 64) * 2);
  assert.equal(ds3.native, 'fp8');
  const llama = parseHfConfig(byRepo('meta-llama/Llama-3.1-70B-Instruct').config);
  assert.equal(llama.kvPerTokenBf16, 2 * 80 * 8 * 128 * 2);
});

test('缺关键字段时给出看得懂的错误', () => {
  assert.throws(() => parseHfConfig({ architectures: ['X'] }), /hidden_size/);
});

test('DeepSeek-V3：一台 H200 放得下，一台 H100 和 L40S 放不下', () => {
  assert.ok(replicaPerf(ds3, h200, chat).ok);
  assert.equal(replicaPerf(ds3, { ...h200, gpu: 'H100' }, chat).ok, false);
  assert.equal(replicaPerf(ds3, { ...h200, gpu: 'L40S', gpus: 16 }, chat).ok, false);
  assert.ok(replicaPerf(ds3, { ...h200, gpu: 'H100', gpus: 16 }, chat).ok);
  const p = replicaPerf(ds3, h200, chat);
  assert.ok(p.rMax > 10 && p.rMax < 30, `一个副本 ${p.rMax.toFixed(1)} 请求/秒`);
  assert.ok(p.loadSec > 240 && p.loadSec < 480, `冷启动 ${p.loadSec} 秒`);
});

test('并发开得越小，一个副本扛得越少', () => {
  const lo = replicaPerf(ds3, { ...h200, maxBatch: 64 }, chat).rMax, hi = replicaPerf(ds3, h200, chat).rMax;
  assert.ok(lo < hi * 0.6);
});

test('发布会：默认方案开场会排队，预扩以后就扛得住', () => {
  const perf = replicaPerf(ds3, h200, chat);
  const plan = { ...h200, ...defaultScale(launch, perf, 8) };
  const base = forecast(ds3, plan, launch, perf).sum;
  assert.ok(base.sloPct < 99, `默认 ${base.sloPct.toFixed(2)}%`);
  assert.ok(base.badMin >= 5 && base.badFrom >= 1200 && base.badFrom < 1215, `排队 ${base.badMin} 分钟，从 ${base.badFrom}`);
  const fixed = forecast(ds3, { ...plan, pre: 72 }, launch, perf).sum;
  assert.ok(fixed.sloPct >= 99, `预扩以后 ${fixed.sloPct.toFixed(2)}%`);
  const a = advise(ds3, plan, launch, { perf });
  assert.equal(a.fix.kind, 'pre');
  assert.ok(a.save && a.save.plan.min < plan.min, '还能建议夜里少留几个');
  assert.equal(grade(a.save.r.sum, launch).letter, 'S');
});

test('每个任务的默认方案都能跑完一天，入门任务默认就能达标', () => {
  for (const sc of SERVE_SCENARIOS) {
    const perf = replicaPerf(ds3, h200, sc.work);
    assert.ok(perf.ok, sc.id);
    const plan = { ...h200, ...defaultScale(sc, perf, 8) };
    assert.ok(plan.max <= hallSlots(8));
    const s = forecast(ds3, plan, sc, perf).sum;
    assert.ok(s.cost > 0 && s.served > 0);
    if (sc.level === '入门') assert.ok(s.sloPct >= 99 && s.cost <= sc.budget, `${sc.id} ${s.sloPct}`);
  }
});

test('排队时限流比硬扛达标率高', () => {
  const perf = replicaPerf(ds3, h200, chat);
  const plan = { ...h200, ...defaultScale(launch, perf, 8) };
  const demand = demandSeries(launch, seededRng(3));
  const a = createDay(ds3, plan, launch, perf, { demand, failAt: [] });
  while (a.t < 1207) stepDay(a);
  const b = cloneDay(a, { record: false });
  setShed(b, true);
  const wait = finishDay(cloneDay(a, { record: false })).sloPct, shed = finishDay(b).sloPct;
  assert.ok(shed > wait + 0.5, `限流 ${shed.toFixed(2)}%，硬扛 ${wait.toFixed(2)}%`);
});

test('同样的种子跑出同样的一天', () => {
  const perf = replicaPerf(ds3, h200, chat);
  const plan = { ...h200, ...defaultScale(launch, perf, 8) };
  const run = () => finishDay(createDay(ds3, plan, launch, perf, { demand: demandSeries(launch, seededRng(11)), failRng: seededRng(5) }));
  const x = run(), y = run();
  assert.equal(x.sloPct, y.sloPct);
  assert.equal(x.cost, y.cost);
  assert.equal(x.failures, y.failures);
});

test('默认精度：原生 FP8 / 4 比特就用原生的', () => {
  assert.equal(defaultPrecision(ds3), 'fp8');
  assert.equal(defaultPrecision(parseHfConfig(byRepo('openai/gpt-oss-120b').config)), 'int4');
  assert.equal(defaultPrecision(parseHfConfig(byRepo('Qwen/Qwen3-8B').config)), 'fp8');
});
