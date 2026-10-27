// 万卡集群模拟器 · 换卡弹层：按厂商分组，可以搜索，训练和推理共用
import { GPUS, VENDORS } from './gpus.js';
import { icon } from './ui.js';

const ALIAS = {
  NVIDIA: '英伟达 nvidia', AMD: 'amd 超威', Google: 'google 谷歌 tpu', AWS: 'aws 亚马逊 amazon',
  华为: 'huawei 华为 昇腾 ascend 国产', Intel: 'intel 英特尔',
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// 按卡名、厂商（中英文都行）、互联、显存、备注搜，空格分开的几个词都要对上
export function gpuMatches(G, q) {
  const words = String(q || '').toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const hay = [G.id, G.name, G.short, G.vendor, ALIAS[G.vendor], G.fabric, G.note, `${G.hbm}gb ${G.hbm} gb`,
    G.domain === 72 ? 'nvl72 整柜' : G.domain > 72 ? '超节点' : '', G.tf4 ? 'fp4' : '', G.stock || G.trainStock ? '新卡' : ''].join(' ').toLowerCase();
  return words.every((w) => hay.includes(w));
}

// o: { wrap, btn, pop, items(): [{ id, sub, right, disabled, badge }], value(): id, onPick(id), locked(): bool }；badge 是名字后面的小黄标
// items 每次打开时算一遍（推理要按当前任务算每种卡最省多少钱，训练要按当前切分算工期）
export function gpuPicker(o) {
  let open = false, q = '', items = [];
  o.pop.innerHTML = `<label class="gp-search"><span>${icon('search', 16, 2)}</span><input type="search" placeholder="搜卡名、厂商、显存，比如 NVL72、HBM4、昇腾" aria-label="搜索加速卡" autocomplete="off" spellcheck="false"></label><div class="gp-list"></div>`;
  const input = o.pop.querySelector('input'), list = o.pop.querySelector('.gp-list');
  const item = (x) => {
    const G = GPUS[x.id];
    return `<button type="button" class="gp-item" role="option" data-v="${x.id}" aria-selected="${x.id === o.value()}"${x.disabled ? ' disabled' : ''} title="${esc(G.note || G.name)}">
      <span style="min-width: 0"><span class="nm">${G.name}${x.badge ? `<span class="best">${x.badge}</span>` : ''}</span><span class="a">${x.sub}</span></span>
      <span class="r">${x.right}</span></button>`;
  };
  function renderList() {
    const shown = items.filter((x) => gpuMatches(GPUS[x.id], q));
    if (!shown.length) { list.className = 'gp-list'; list.innerHTML = `<div class="gp-empty">没有对得上“${esc(q)}”的卡</div>`; return; }
    if (q) {
      // 搜索时不分组，直接按原来的顺序列出来
      list.className = 'gp-list flat';
      list.innerHTML = `<div class="label">${shown.length} 种卡</div>${shown.map(item).join('')}`;
      return;
    }
    const group = (vs) => `<div class="gp-col">${vs.map((v) => { const xs = shown.filter((x) => GPUS[x.id].vendor === v); return xs.length ? `<div class="label">${v}</div>${xs.map(item).join('')}` : ''; }).join('')}</div>`;
    list.className = 'gp-list';
    list.innerHTML = group(['NVIDIA']) + group(VENDORS.filter((v) => v !== 'NVIDIA'));
  }
  function setOpen(v, focusBtn = false) {
    open = v;
    o.btn.setAttribute('aria-expanded', String(open));
    o.pop.hidden = !open;
    if (open) {
      q = ''; input.value = '';   // 每次打开都从头搜
      items = o.items();
      renderList();
      input.focus({ preventScroll: true });
    } else if (focusBtn) o.btn.focus({ preventScroll: true });
  }
  o.btn.onclick = () => { if (!o.locked || !o.locked()) setOpen(!open); };
  input.oninput = () => { q = input.value.trim(); renderList(); };
  list.onclick = (e) => { const b = e.target.closest('.gp-item'); if (b && !b.disabled) { setOpen(false, true); o.onPick(b.dataset.v); } };
  o.pop.addEventListener('keydown', (e) => {
    const btns = [...list.querySelectorAll('.gp-item:not(:disabled)')];
    const i = btns.indexOf(document.activeElement);
    if (e.key === 'ArrowDown') { e.preventDefault(); btns[Math.min(btns.length - 1, i + 1)]?.focus(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); if (i <= 0) input.focus(); else btns[i - 1].focus(); }
    else if (e.key === 'Enter' && document.activeElement === input && btns.length) { e.preventDefault(); btns[0].click(); }
  });
  document.addEventListener('mousedown', (e) => { if (open && !o.wrap.contains(e.target)) setOpen(false); });
  document.addEventListener('keydown', (e) => { if (open && e.key === 'Escape') { e.stopPropagation(); setOpen(false, true); } }, true);
  return {
    get open() { return open; },
    close: () => { if (open) setOpen(false); },
    refresh: () => { if (open) { items = o.items(); renderList(); } },
  };
}
