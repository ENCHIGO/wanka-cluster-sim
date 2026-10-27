// 万卡集群模拟器 · 加速卡规格，训练和推理两种玩法共用
// hbm：每卡显存 GB；bw：显存带宽 TB/s；tf16 / tf8 / tf4：稠密峰值 TFLOPS，tf4 为 0 表示没有 4 比特张量核
// （4 比特权重只省显存，算的时候解开按 8 比特）；f8：8 比特是 INT8 而不是 FP8 时写上（这种卡不能用 FP8 训练）；
// link：scale-up 互联每卡单向 GB/s；fabric / domain：scale-up 互联叫什么、一个域几张卡，超出要走网卡；
// nic：网卡每卡 GB/s（插多快的网卡都受它限制）；price：每卡每小时美元，按 2026 年包年的行情取整；
// mtbf：每卡平均多少小时出一次让训练作业挂掉的硬件故障（游戏值，新卡刚量产时故障多）；
// sw：训练软件栈有多成熟，乘在内核效率上（CUDA 最成熟；ROCm、Neuron、CANN 和刚出的新卡要打折，游戏值）；
// stock：推理大厅里有几张，trainStock：训练大厅里有几张（新卡只到了一部分，没写就是要多少有多少）。
// 规格取厂商公开的数字，没公布的（Rubin、MI455X 的 BF16 等）按同代 FP8 的一半估。
export const GPUS = {
  L40S: { id: 'L40S', name: 'L40S', vendor: 'NVIDIA', hbm: 48, bw: 0.864, tf16: 362, tf8: 733, tf4: 0, link: 32, fabric: 'PCIe', domain: 8, nic: 25, price: 0.9, mtbf: 60000, sw: 1.0, note: 'PCIe 卡，卡间没有 NVLink' },
  RTX6000: { id: 'RTX6000', name: 'RTX PRO 6000', vendor: 'NVIDIA', hbm: 96, bw: 1.6, tf16: 500, tf8: 1000, tf4: 2000, link: 64, fabric: 'PCIe', domain: 8, nic: 50, price: 1.5, mtbf: 60000, sw: 0.95, note: 'GDDR7 的 PCIe 卡，有 FP4' },
  H20: { id: 'H20', name: 'H20', vendor: 'NVIDIA', hbm: 96, bw: 4.0, tf16: 148, tf8: 296, tf4: 0, link: 450, fabric: 'NVLink', domain: 8, nic: 50, price: 1.1, mtbf: 50000, sw: 1.0, note: '带宽不差，算力只有 H100 的 15%' },
  H100: { id: 'H100', name: 'H100', vendor: 'NVIDIA', hbm: 80, bw: 3.35, tf16: 989, tf8: 1979, tf4: 0, link: 450, fabric: 'NVLink', domain: 8, nic: 50, price: 2.0, mtbf: 50000, sw: 1.0 },
  H200: { id: 'H200', name: 'H200', vendor: 'NVIDIA', hbm: 141, bw: 4.8, tf16: 989, tf8: 1979, tf4: 0, link: 450, fabric: 'NVLink', domain: 8, nic: 50, price: 2.6, mtbf: 50000, sw: 1.0 },
  B200: { id: 'B200', name: 'B200', vendor: 'NVIDIA', hbm: 180, bw: 8.0, tf16: 2250, tf8: 4500, tf4: 9000, link: 900, fabric: 'NVLink', domain: 8, nic: 50, price: 4.0, mtbf: 40000, sw: 1.0 },
  B300: { id: 'B300', name: 'B300', vendor: 'NVIDIA', hbm: 288, bw: 8.0, tf16: 2250, tf8: 4500, tf4: 13500, link: 900, fabric: 'NVLink', domain: 8, nic: 100, price: 5.0, mtbf: 40000, sw: 0.95, note: 'Blackwell Ultra，FP4 比 B200 快一半' },
  GB200: { id: 'GB200', name: 'GB200 NVL72', short: 'GB200', vendor: 'NVIDIA', hbm: 186, bw: 8.0, tf16: 2500, tf8: 5000, tf4: 10000, link: 900, fabric: 'NVLink', domain: 72, nic: 50, price: 4.8, mtbf: 30000, sw: 0.95, note: '72 卡一个 NVLink 域' },
  GB300: { id: 'GB300', name: 'GB300 NVL72', short: 'GB300', vendor: 'NVIDIA', hbm: 288, bw: 8.0, tf16: 2500, tf8: 5000, tf4: 15000, link: 900, fabric: 'NVLink', domain: 72, nic: 100, price: 6.0, mtbf: 32000, sw: 0.9, note: '72 卡一个 NVLink 域' },
  VR200: { id: 'VR200', name: 'Rubin NVL72', short: 'Rubin', vendor: 'NVIDIA', hbm: 288, bw: 22, tf16: 8000, tf8: 16000, tf4: 35000, link: 1800, fabric: 'NVLink', domain: 72, nic: 200, price: 14, mtbf: 25000, sw: 0.8, trainStock: 2304, stock: 144, note: 'HBM4，首批只到了两个柜' },
  MI300X: { id: 'MI300X', name: 'MI300X', vendor: 'AMD', hbm: 192, bw: 5.3, tf16: 1307, tf8: 2615, tf4: 0, link: 448, fabric: 'Infinity Fabric', domain: 8, nic: 50, price: 1.8, mtbf: 40000, sw: 0.85 },
  MI355X: { id: 'MI355X', name: 'MI355X', vendor: 'AMD', hbm: 288, bw: 8.0, tf16: 2500, tf8: 5000, tf4: 10000, link: 504, fabric: 'Infinity Fabric', domain: 8, nic: 50, price: 3.4, mtbf: 38000, sw: 0.8 },
  MI455X: { id: 'MI455X', name: 'MI455X Helios', short: 'MI455X', vendor: 'AMD', hbm: 432, bw: 19.6, tf16: 10000, tf8: 20000, tf4: 40000, link: 1800, fabric: 'UALink', domain: 72, nic: 100, price: 12, mtbf: 25000, sw: 0.7, trainStock: 2304, stock: 144, note: 'HBM4，首批只到了两个柜' },
  TPUv6e: { id: 'TPUv6e', name: 'TPU v6e', vendor: 'Google', hbm: 32, bw: 1.64, tf16: 918, tf8: 1836, f8: 'INT8', tf4: 0, link: 224, fabric: 'ICI', domain: 256, nic: 25, price: 1.4, mtbf: 70000, sw: 0.95, note: 'Trillium，每片只有 32 GB' },
  TPUv7: { id: 'TPUv7', name: 'TPU v7', vendor: 'Google', hbm: 192, bw: 7.37, tf16: 2307, tf8: 4614, tf4: 0, link: 600, fabric: 'ICI', domain: 256, nic: 50, price: 4.2, mtbf: 60000, sw: 0.9, note: 'Ironwood，ICI 连成一大片' },
  Trn2: { id: 'Trn2', name: 'Trainium2', vendor: 'AWS', hbm: 96, bw: 2.9, tf16: 667, tf8: 1299, tf4: 0, link: 512, fabric: 'NeuronLink', domain: 64, nic: 50, price: 1.5, mtbf: 45000, sw: 0.85, note: 'UltraServer 64 片一组' },
  Trn3: { id: 'Trn3', name: 'Trainium3', vendor: 'AWS', hbm: 144, bw: 4.9, tf16: 1260, tf8: 2520, tf4: 2520, link: 1000, fabric: 'NeuronLink', domain: 144, nic: 100, price: 3.0, mtbf: 40000, sw: 0.8, note: 'MXFP4 和 FP8 一样快，只省显存' },
  A910C: { id: 'A910C', name: '昇腾 910C', vendor: '华为', hbm: 128, bw: 3.2, tf16: 780, tf8: 1560, f8: 'INT8', tf4: 0, link: 196, fabric: '灵衢', domain: 384, nic: 50, price: 1.4, mtbf: 30000, sw: 0.75, note: 'CloudMatrix 384 超节点' },
  A950PR: { id: 'A950PR', name: '昇腾 950PR', vendor: '华为', hbm: 128, bw: 1.6, tf16: 500, tf8: 1000, tf4: 2000, link: 1000, fabric: '灵衢', domain: 8, nic: 50, price: 1.5, mtbf: 30000, sw: 0.7, note: '给预填充做的卡，带宽低' },
  Gaudi3: { id: 'Gaudi3', name: 'Gaudi 3', vendor: 'Intel', hbm: 128, bw: 3.7, tf16: 1835, tf8: 1835, tf4: 0, link: 525, fabric: 'RoCE', domain: 8, nic: 75, price: 1.4, mtbf: 40000, sw: 0.75, note: 'FP8 和 BF16 一样快' },
};
export const VENDORS = ['NVIDIA', 'AMD', 'Google', 'AWS', '华为', 'Intel'];
export const GPU_IDS = Object.keys(GPUS);
export const gpuName = (id) => GPUS[id]?.name || id;
