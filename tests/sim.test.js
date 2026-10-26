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
  assert.match(ev.errors.join(), /显存溢出/);
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
