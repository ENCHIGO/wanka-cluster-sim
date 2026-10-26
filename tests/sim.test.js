import { test } from 'node:test';
import assert from 'node:assert/strict';
import { evaluate, SCENARIOS, lossAt } from '../src/sim.js';

test('每个关卡的默认配置都能启动', () => {
  for (const sc of SCENARIOS) {
    const ev = evaluate(sc.defaults, sc);
    assert.ok(ev.ok, `${sc.id}: ${ev.errors.join('; ')}`);
    assert.ok(ev.mfu > 0 && ev.mfu < 0.7, `${sc.id} MFU ${ev.mfu}`);
  }
});

test('第一关默认配置可以按期、按预算完成', () => {
  const sc = SCENARIOS[0];
  const ev = evaluate(sc.defaults, sc);
  assert.ok(ev.etaDays < sc.deadlineDays);
  assert.ok(ev.estCost < sc.budget);
});

test('DP 大于全局批次时报错', () => {
  const sc = SCENARIOS[1];
  const ev = evaluate({ ...sc.defaults, tp: 1, pp: 1, gbs: 2048 }, sc);
  assert.equal(ev.ok, false);
  assert.match(ev.errors.join(), /分不到/);
});

test('405B 不切分会显存溢出', () => {
  const sc = SCENARIOS[1];
  const ev = evaluate({ ...sc.defaults, gpus: 512, tp: 1, pp: 1, gbs: 512 }, sc);
  assert.equal(ev.ok, false);
  assert.match(ev.errors.join(), /显存不够/);
});

test('TP 跨节点比节点内慢', () => {
  const sc = SCENARIOS[1];
  const a = evaluate({ ...sc.defaults, tp: 8, pp: 16 }, sc);
  const b = evaluate({ ...sc.defaults, tp: 16, pp: 8 }, sc);
  assert.ok(b.breakdown.tp > a.breakdown.tp * 3);
});

test('卡越多，集群故障间隔越短', () => {
  const sc = SCENARIOS[1];
  const small = evaluate({ ...sc.defaults, gpus: 2048, pp: 4, gbs: 2048 }, sc);
  const big = evaluate(sc.defaults, sc);
  assert.ok(big.mtbfHours < small.mtbfHours / 4);
});

test('loss 随 token 单调下降', () => {
  assert.ok(lossAt(70e9, 1e12) < lossAt(70e9, 1e11));
});

const WANKA = { gpu: 'B200', gpus: 10240, net: 'ib800', storage: 'fast', tp: 8, pp: 8, ep: 1, gbs: 4096, recompute: false, ckptMin: 10, spares: 16 };

test('万卡的推荐配置能交，但预算吃紧', () => {
  const sc = SCENARIOS[1];
  const ev = evaluate(WANKA, sc);
  assert.ok(ev.ok);
  assert.ok(ev.etaDays < sc.deadlineDays && ev.estCost < sc.budget);
  assert.match(ev.warnings.join(), /预算只剩/);
});

test('万卡规模下备机留少了，预估会把等维修算进去', () => {
  const sc = SCENARIOS[1];
  const few = evaluate({ ...WANKA, spares: 8 }, sc), enough = evaluate(WANKA, sc);
  assert.ok(few.noSpare > 0.3 && enough.noSpare < 0.05);
  assert.ok(few.etaDays > enough.etaDays * 1.5);
  assert.match(few.warnings.join(), /备机不够/);
});
