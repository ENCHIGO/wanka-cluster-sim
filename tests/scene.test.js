import { test } from 'node:test';
import assert from 'node:assert/strict';
import { hallLayout, nodeHost, rackName } from '../src/scene.js';
import { GPU_COUNTS } from '../src/sim.js';

test('每种卡数都能排满，机柜数正好是卡数 / 32', () => {
  for (const n of GPU_COUNTS) {
    const L = hallLayout(n);
    assert.equal(L.racks, n / 32, `${n} 卡`);
    assert.equal(L.nodes, n / 8);
    const seen = new Set(L.rackList.map((r) => `${r.x},${r.y}`));
    assert.equal(seen.size, L.racks, '机柜位置不能重叠');
    assert.ok(L.rows.length <= 16, '行号只到 P');
  }
});

test('万卡大厅：8 排，每排 40 柜，中间一条主通道', () => {
  const L = hallLayout(10240);
  assert.equal(L.rows.length, 8);
  assert.equal(L.perRow, 40);
  assert.equal(L.blocks.length, 2);
  assert.equal(rackName(L, 0), 'A01');
  assert.equal(rackName(L, 2 * 40 + 26), 'C27');
  assert.equal(nodeHost(L, (2 * 40 + 26) * 4 + 2), 'C27-3');
});
