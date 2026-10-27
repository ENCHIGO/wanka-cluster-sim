import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate, SCENARIOS, lossAt, suggestHardware, TRAIN_PRECISION } from '../src/sim.js';

const byId = (id) => SCENARIOS.find((s) => s.id === id);
const dense = byId('dense'), wanka = byId('wanka'), domestic = byId('domestic');

test('每个关卡的默认配置都能启动', () => {
  for (const sc of SCENARIOS) {
    const ev = evaluate(sc.defaults, sc);
    assert.ok(ev.ok, `${sc.id}: ${ev.errors.join('; ')}`);
    assert.ok(ev.mfu > 0 && ev.mfu < 0.7, `${sc.id} MFU ${ev.mfu}`);
  }
});

test('第一关默认配置可以按期、按预算完成', () => {
  const ev = evaluate(dense.defaults, dense);
  assert.ok(ev.etaDays < dense.deadlineDays, `${ev.etaDays} 天`);
  assert.ok(ev.estCost < dense.budget);
});

test('每一关老周都给得出一个能按期、按预算交的配置', () => {
  for (const sc of SCENARIOS) {
    const tip = suggestHardware(sc.defaults, sc);
    const ok = (e) => e.etaDays <= sc.deadlineDays && e.estCost <= sc.budget;
    const cur = evaluate(sc.defaults, sc);
    assert.ok(ok(cur) || (tip && ok(tip.ev)), `${sc.id}: ${tip ? `${tip.ev.etaDays.toFixed(1)} 天 $${(tip.ev.estCost / 1e6).toFixed(1)}M` : '没有建议'}`);
  }
});

test('DP 大于全局批次时报错', () => {
  const ev = evaluate({ ...dense.defaults, tp: 1, pp: 1, gbs: 2048 }, dense);
  assert.equal(ev.ok, false);
  assert.match(ev.errors.join(), /分不到/);
});

test('1.6T 不切分会显存溢出', () => {
  const ev = evaluate({ ...wanka.defaults, gpus: 512, tp: 1, pp: 1, ep: 8, gbs: 512 }, wanka);
  assert.equal(ev.ok, false);
  assert.match(ev.errors.join(), /显存不够/);
});

test('TP 出了 scale-up 域就慢；NVL72 里 TP16 还走 NVLink', () => {
  const a = evaluate({ ...dense.defaults, tp: 8, pp: 1 }, dense);
  const b = evaluate({ ...dense.defaults, tp: 16, pp: 1 }, dense);
  assert.ok(b.breakdown.tp > a.breakdown.tp * 3);
  assert.match(b.warnings.join(), /TP16 出了单机/);
  const nvl = evaluate({ ...dense.defaults, gpu: 'GB200', tp: 16, pp: 1 }, dense);
  assert.ok(nvl.breakdown.tp < b.breakdown.tp / 3);
  assert.doesNotMatch(nvl.warnings.join(), /TP16 出了/);
});

test('MoE 的 all-to-all 放进 NVL72 柜里比走网卡快', () => {
  const sc = byId('small-moe');
  const hgx = evaluate({ ...sc.defaults, gpu: 'B200', ep: 64, precision: 'fp8' }, sc);
  const nvl = evaluate({ ...sc.defaults, gpu: 'GB200', ep: 64, precision: 'fp8' }, sc);
  assert.ok(nvl.breakdown.ep < hgx.breakdown.ep / 5, `NVL72 ${nvl.breakdown.ep.toFixed(3)} 秒，HGX ${hgx.breakdown.ep.toFixed(3)} 秒`);
});

test('训练精度：FP8 比 BF16 快，FP4 只有带 FP4 张量核的卡能用，尖峰更多', () => {
  const bf = evaluate(dense.defaults, dense), f8 = evaluate({ ...dense.defaults, precision: 'fp8' }, dense);
  assert.ok(f8.etaDays < bf.etaDays * 0.8);
  assert.equal(evaluate({ ...dense.defaults, precision: 'fp4' }, dense).ok, false);   // H200 没有 FP4
  const b300 = evaluate({ ...dense.defaults, gpu: 'B300', precision: 'fp4' }, dense);
  assert.ok(b300.ok && b300.spikePerHour === bf.spikePerHour * TRAIN_PRECISION.fp4.spike);
  assert.equal(evaluate({ ...domestic.defaults, precision: 'fp8' }, domestic).ok, false);   // 910C 的 8 比特是 INT8
});

test('国产算力只能用昇腾；新卡训练大厅里只有一部分', () => {
  assert.match(evaluate({ ...domestic.defaults, gpu: 'H200' }, domestic).errors.join(), /只能用/);
  assert.match(evaluate({ ...wanka.defaults, gpu: 'VR200', gpus: 16384 }, wanka).errors.join(), /只有/);
});

test('网卡快不过卡上的网卡', () => {
  const ev = evaluate({ ...dense.defaults, gpu: 'H100', net: 'ib800' }, dense);
  assert.match(ev.warnings.join(), /网卡只到 400G/);
  assert.ok(Math.abs(ev.netBw - 50 * 0.85) < 1e-9);
});

test('卡越多，集群故障间隔越短', () => {
  const small = evaluate({ ...wanka.defaults, gpus: 2048, gbs: 2048 }, wanka);
  const big = evaluate(wanka.defaults, wanka);
  assert.ok(big.mtbfHours < small.mtbfHours / 4);
});

test('loss 随 token 单调下降', () => {
  assert.ok(lossAt(27e9, 1e12) < lossAt(27e9, 1e11));
});

const WANKA = { gpu: 'GB300', gpus: 12288, net: 'ib400', storage: 'fast', precision: 'fp4', tp: 2, pp: 32, ep: 32, gbs: 16384, recompute: true, ckptMin: 20, spares: 19 };

test('万卡：GB300 加 FP4 能按期、按预算交', () => {
  const ev = evaluate(WANKA, wanka);
  assert.ok(ev.ok, ev.errors.join());
  assert.ok(ev.etaDays < wanka.deadlineDays && ev.estCost < wanka.budget, `${ev.etaDays.toFixed(1)} 天 $${(ev.estCost / 1e6).toFixed(1)}M`);
});

test('万卡规模下备机留少了，预估会把等维修算进去', () => {
  const few = evaluate({ ...WANKA, spares: 8 }, wanka), enough = evaluate(WANKA, wanka);
  assert.ok(few.noSpare > 0.3 && enough.noSpare < 0.1, `${few.noSpare} / ${enough.noSpare}`);
  assert.ok(few.etaDays > enough.etaDays * 1.3);
  assert.match(few.warnings.join(), /备机不够/);
});
