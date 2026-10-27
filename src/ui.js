// 万卡集群模拟器 · 训练和推理两种模式共用的界面小件：图标、格式化、分段按钮、步进器、切页面
export const $ = (id) => document.getElementById(id);

// ---------- 图标 ----------
export const STROKE = {
  cube: '<path d="M12 2.8l8 4.6v9.2l-8 4.6-8-4.6V7.4z"/><path d="M4 7.4l8 4.6 8-4.6M12 12v9.2"/><path d="M7 11.4v3.4M9.4 12.8v3.4"/>',
  save: '<path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
  server: '<rect x="3" y="4" width="13" height="6" rx="1.5"/><rect x="3" y="14" width="13" height="6" rx="1.5"/><path d="M6.5 7h.01M6.5 17h.01M20 9v6M17 12h6"/>',
  minus: '<path d="M6 12h12"/>',
  plus: '<path d="M12 6v12M6 12h12"/>',
  sliders: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  thermo: '<path d="M14 14.8V5a2 2 0 1 0-4 0v9.8a4 4 0 1 0 4 0z"/>',
  refresh: '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.5 4.5v4h-4"/>',
  plug: '<path d="M8 3v5M16 3v5M6 8h12v3a6 6 0 0 1-12 0zM12 17v4"/>',
  wrench: '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  rollback: '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.5 4.5v4h4"/>',
  eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
  fail: '<path d="M8.5 3h7L21 8.5v7L15.5 21h-7L3 15.5v-7z"/><path d="M9.5 9.5l5 5M14.5 9.5l-5 5"/>',
  spike: '<path d="M3 17l5-5 4 3 8-9"/><path d="M15 6h5v5"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8h.01"/>',
  flag: '<path d="M5 21V4M5 4h11l-2 4 2 4H5"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>',
  upload: '<path d="M12 15V4M7.5 8.5L12 4l4.5 4.5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
  lock: '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
  alert: '<path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v4.5M12 17.5h.01"/>',
  queue: '<path d="M4 6h16M4 12h16M4 18h10"/>',
  timer: '<circle cx="12" cy="13" r="7.5"/><path d="M12 9.5V13l2.5 2M9.5 2.5h5"/>',
  filter: '<path d="M4 5h16l-6 7.5V19l-4 1.5v-8z"/>',
  down: '<path d="M3.5 6.5l5 6 4-3.5 7.5 9"/><path d="M15 18h5v-5"/>',
  chev: '<path d="M6 9.5l6 6 6-6"/>',
};
export const FILL = {
  pause: '<rect x="6" y="5" width="4" height="14" rx="1.2"/><rect x="14" y="5" width="4" height="14" rx="1.2"/>',
  play1: '<path d="M8 5.5v13l10.5-6.5z"/>',
  play2: '<path d="M3.5 6v12l8-6zM12.5 6v12l8-6z"/>',
  play3: '<path d="M1.5 7v10l6-5zM9 7v10l6-5zM16.5 7v10l6-5z"/>',
};
export const icon = (name, size = 18, sw = 1.9) => FILL[name]
  ? `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">${FILL[name]}</svg>`
  : `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${STROKE[name]}</svg>`;
export function fillIcons(root = document) {
  root.querySelectorAll('[data-icon]').forEach((el) => {
    const n = el.dataset.icon;
    el.innerHTML = icon(n, n === 'cube' ? 24 : n === 'arrow' ? 20 : 18, n === 'cube' ? 1.8 : n === 'arrow' ? 2.2 : 1.9);
    el.style.display = el.style.display || 'grid';
  });
}

// ---------- 格式化 ----------
export const fmtInt = (n) => Math.floor(n).toLocaleString('en-US');
export const fmtMoney = (d) => Math.abs(d) >= 1e6 ? '$' + (d / 1e6).toFixed(2) + 'M' : '$' + (d / 1e3).toFixed(1) + 'K';
export const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';
export const pad2 = (n) => String(n).padStart(2, '0');

// ---------- 控件 ----------
export function segment(el, items, value, onPick, numeric = false) {
  el.innerHTML = items.map((it) => `<button type="button" class="${numeric ? 'n' : ''}" data-v="${it.v}" aria-pressed="${String(it.v) === String(value)}"${it.disabled ? ' disabled' : ''}${it.title ? ` title="${it.title}"` : ''}>${it.label}</button>`).join('');
  el.onclick = (e) => { const b = e.target.closest('button'); if (b && !b.disabled) onPick(b.dataset.v); };
}

export function stepper(el, name, options, get, set) {
  el.innerHTML = `<button type="button" aria-label="${name}减一档">${icon('minus', 16, 2)}</button><output></output><button type="button" aria-label="${name}加一档">${icon('plus', 16, 2)}</button>`;
  const [dec, out, inc] = el.children;
  const at = () => { const v = get(), i = options.indexOf(v); return i >= 0 ? i : options.findIndex((o) => o > v); };
  dec.onclick = () => { const v = get(), i = options.indexOf(v), j = i >= 0 ? i - 1 : options.findLastIndex((o) => o < v); if (j >= 0) set(options[j]); };
  inc.onclick = () => { const i = at(), v = get(), j = options.indexOf(v) >= 0 ? i + 1 : i; if (j >= 0 && j < options.length) set(options[j]); };
  el.sync = (fmt = (v) => v) => {
    const v = get();
    out.textContent = fmt(v);
    dec.disabled = !options.some((o) => o < v);
    inc.disabled = !options.some((o) => o > v);
  };
  el.sync();
}

// ---------- 页面和模式 ----------
// 同一时间只显示 #app 下的一个页面
export function showOnly(id) {
  for (const s of document.querySelectorAll('#app > section')) s.hidden = s.id !== id;
  window.scrollTo(0, 0);
}

const modes = {};
export function registerMode(name, open) { modes[name] = open; }
// 训练 / 推理部署 切换按钮：页面上带 data-mode 的按钮都走这里
document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-mode]');
  if (!b || b.getAttribute('aria-pressed') === 'true' || !modes[b.dataset.mode]) return;
  modes[b.dataset.mode]();
});
