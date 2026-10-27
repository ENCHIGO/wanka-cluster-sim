import { parseHfConfig } from './serve.js';

// 内置模型：2026 年 10 月前后的前沿开源模型，config.json 的关键字段按各仓库公开的 config.json、模型卡和技术报告整理。
// 在连不上 Hugging Face 的地方（比如在 Claude 里打开）就用这份。
// known：模型卡上写的总参数 / 激活参数（十亿），只用来在测试里核对估算。
// inferred：公开资料里没找到原文、按参数量和同系列模型反推的字段。
// native：权重原生是什么精度，config.json 说不清的时候写在这里（DeepSeek V4 的专家是 FP4，其余 FP8）。
const range = (n, f) => Array.from({ length: n }, (_, i) => f(i));
// DeepSeek V4：前两层和最后一层只有滑窗，中间 CSA（4 个 token 压一条）和 HCA（128 个压一条）交替
const v4Ratios = (n) => [0, 0, ...range(n - 3, (i) => (i % 2 ? 128 : 4)), 0];
// Kimi K3：每 4 层一层 Gated MLA，最后一层也是，其余 69 层是 KDA（从 1 数）
const k3Full = [...range(23, (i) => 4 * (i + 1)), 93];
const FP8_BLOCK = { quant_method: 'fp8', fmt: 'e4m3', weight_block_size: [128, 128] };

export const CATALOG = [
  {
    repo: 'deepseek-ai/DeepSeek-V4-Pro', name: 'DeepSeek-V4-Pro', mark: 'D', known: [1600, 49], native: 'fp4', inferred: ['compress_ratios'],
    config: {
      architectures: ['DeepseekV4ForCausalLM'], hidden_size: 7168, num_hidden_layers: 61, num_attention_heads: 128, num_key_value_heads: 1,
      head_dim: 512, qk_rope_head_dim: 64, q_lora_rank: 1536, o_lora_rank: 1024, o_groups: 16, compress_ratios: v4Ratios(61),
      index_topk: 1024, index_n_heads: 64, index_head_dim: 128, sliding_window: 128,
      n_routed_experts: 384, n_shared_experts: 1, num_experts_per_tok: 6, moe_intermediate_size: 3072, num_nextn_predict_layers: 1,
      vocab_size: 129280, max_position_embeddings: 1048576, tie_word_embeddings: false, torch_dtype: 'bfloat16', quantization_config: FP8_BLOCK,
    },
  },
  {
    repo: 'moonshotai/Kimi-K3', name: 'Kimi-K3', mark: 'K', known: [2800, 104], inferred: ['intermediate_size', 'vocab_size', 'quantization_config'],
    config: {
      architectures: ['KimiK3ForConditionalGeneration'], hidden_size: 7168, intermediate_size: 18432, moe_intermediate_size: 3072, routed_expert_hidden_size: 3584,
      num_hidden_layers: 93, num_attention_heads: 96, num_key_value_heads: 96, first_k_dense_replace: 1,
      kv_lora_rank: 512, q_lora_rank: 1536, qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128,
      linear_attn_config: { kda_layers: range(93, (i) => i + 1).filter((i) => !k3Full.includes(i)), full_attn_layers: k3Full, num_heads: 96, head_dim: 128, short_conv_kernel_size: 4 },
      num_experts: 896, num_experts_per_token: 16, num_shared_experts: 2,
      vocab_size: 163840, max_position_embeddings: 1048576, tie_word_embeddings: false, torch_dtype: 'bfloat16', quantization_config: { quant_method: 'mxfp4' },
    },
  },
  {
    repo: 'Qwen/Qwen3.8-2.4T-A95B', name: 'Qwen3.8-2.4T-A95B', mark: 'Q', known: [2400, 95], inferred: ['moe_intermediate_size', 'shared_expert_intermediate_size', 'vocab_size', 'mtp_num_hidden_layers'],
    config: {
      architectures: ['Qwen3_5MoeForCausalLM'], hidden_size: 8192, num_hidden_layers: 92, num_attention_heads: 64, num_key_value_heads: 4, head_dim: 256,
      full_attention_interval: 4, linear_num_value_heads: 128, linear_num_key_heads: 16, linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4,
      num_experts: 512, num_experts_per_tok: 10, moe_intermediate_size: 2048, shared_expert_intermediate_size: 2048, mtp_num_hidden_layers: 1,
      vocab_size: 248320, max_position_embeddings: 262144, tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
  {
    repo: 'XiaomiMiMo/MiMo-V2.6-Pro', name: 'MiMo-V2.6-Pro', mark: 'Mi', known: [1020, 42], inferred: ['moe_intermediate_size', 'intermediate_size', 'vocab_size', 'hybrid_layer_pattern', 'quantization_config'],
    config: {
      architectures: ['MiMoV2ForCausalLM'], hidden_size: 6144, intermediate_size: 16384, num_hidden_layers: 70, num_attention_heads: 128, num_key_value_heads: 8,
      head_dim: 192, v_head_dim: 128, hybrid_layer_pattern: range(70, (i) => (i % 7 ? 1 : 0)), sliding_window: 128,
      n_routed_experts: 384, num_experts_per_tok: 8, moe_intermediate_size: 2048, first_k_dense_replace: 1,
      vocab_size: 152576, max_position_embeddings: 1048576, tie_word_embeddings: false, torch_dtype: 'bfloat16', quantization_config: FP8_BLOCK,
    },
  },
  {
    repo: 'zai-org/GLM-5.3', name: 'GLM-5.3', mark: 'Z', known: [744, 40], inferred: ['vocab_size'],
    config: {
      architectures: ['GlmMoeDsaForCausalLM'], hidden_size: 6144, intermediate_size: 12288, moe_intermediate_size: 2048,
      num_hidden_layers: 78, num_attention_heads: 64, num_key_value_heads: 64, first_k_dense_replace: 3,
      kv_lora_rank: 512, q_lora_rank: 2048, qk_nope_head_dim: 192, qk_rope_head_dim: 64, v_head_dim: 256,
      index_topk: 2048, index_n_heads: 32, index_head_dim: 128,
      n_routed_experts: 256, n_shared_experts: 1, num_experts_per_tok: 8, num_nextn_predict_layers: 1,
      vocab_size: 154880, max_position_embeddings: 202752, tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
  {
    repo: 'moonshotai/Kimi-K2.5', name: 'Kimi-K2.5', mark: 'K', known: [1040, 32],
    config: {
      architectures: ['KimiK25ForConditionalGeneration'],
      text_config: {
        model_type: 'kimi_k2', hidden_size: 7168, intermediate_size: 18432, moe_intermediate_size: 2048,
        num_hidden_layers: 61, num_attention_heads: 64, num_key_value_heads: 64, n_routed_experts: 384, n_shared_experts: 1,
        num_experts_per_tok: 8, first_k_dense_replace: 1, kv_lora_rank: 512, q_lora_rank: 1536, qk_nope_head_dim: 128,
        qk_rope_head_dim: 64, v_head_dim: 128, vocab_size: 163840, max_position_embeddings: 262144, tie_word_embeddings: false,
      },
      quantization_config: { quant_method: 'compressed-tensors', format: 'pack-quantized', config_groups: { group_0: { weights: { num_bits: 4, type: 'int', group_size: 32, symmetric: true } } } },
    },
  },
  {
    repo: 'MiniMaxAI/MiniMax-M3', name: 'MiniMax-M3', mark: 'M', known: [428, 23], inferred: ['vocab_size', 'intermediate_size'],
    config: {
      architectures: ['MiniMaxM3ForCausalLM'], hidden_size: 6144, intermediate_size: 12288, num_hidden_layers: 60, num_attention_heads: 64,
      num_key_value_heads: 4, head_dim: 128, first_k_dense_replace: 3, sparse_attention_freq: range(60, (i) => (i < 3 ? 0 : 1)),
      index_topk: 2048, index_n_heads: 4, index_head_dim: 128,
      num_local_experts: 128, num_experts_per_tok: 4, n_shared_experts: 1, moe_intermediate_size: 3072,
      vocab_size: 200064, max_position_embeddings: 1048576, tie_word_embeddings: false, torch_dtype: 'bfloat16', quantization_config: FP8_BLOCK,
    },
  },
  {
    repo: 'Qwen/Qwen3.5-397B-A17B', name: 'Qwen3.5-397B-A17B', mark: 'Q', known: [397, 17],
    config: {
      architectures: ['Qwen3_5MoeForConditionalGeneration'], hidden_size: 4096, num_hidden_layers: 60, num_attention_heads: 32, num_key_value_heads: 2, head_dim: 256,
      full_attention_interval: 4, linear_num_value_heads: 64, linear_num_key_heads: 16, linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4,
      num_experts: 512, num_experts_per_tok: 10, moe_intermediate_size: 1024, shared_expert_intermediate_size: 1024, mtp_num_hidden_layers: 1,
      vocab_size: 248320, max_position_embeddings: 262144, tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
  {
    repo: 'deepseek-ai/DeepSeek-V4-Flash', name: 'DeepSeek-V4-Flash', mark: 'D', known: [284, 13], native: 'fp4', inferred: ['compress_ratios'],
    config: {
      architectures: ['DeepseekV4ForCausalLM'], hidden_size: 4096, num_hidden_layers: 43, num_attention_heads: 64, num_key_value_heads: 1,
      head_dim: 512, qk_rope_head_dim: 64, q_lora_rank: 1024, o_lora_rank: 1024, o_groups: 8, compress_ratios: v4Ratios(43),
      index_topk: 512, index_n_heads: 64, index_head_dim: 128, sliding_window: 128,
      n_routed_experts: 256, n_shared_experts: 1, num_experts_per_tok: 6, moe_intermediate_size: 2048, num_nextn_predict_layers: 1,
      vocab_size: 129280, max_position_embeddings: 1048576, tie_word_embeddings: false, torch_dtype: 'bfloat16', quantization_config: FP8_BLOCK,
    },
  },
  {
    repo: 'openai/gpt-oss-120b', name: 'gpt-oss-120b', mark: 'O', known: [117, 5.1],
    config: {
      architectures: ['GptOssForCausalLM'], hidden_size: 2880, intermediate_size: 2880, num_hidden_layers: 36,
      num_attention_heads: 64, num_key_value_heads: 8, head_dim: 64, layer_types: range(36, (i) => (i % 2 ? 'full_attention' : 'sliding_attention')), sliding_window: 128,
      num_local_experts: 128, num_experts_per_tok: 4, vocab_size: 201088, max_position_embeddings: 131072, tie_word_embeddings: false,
      torch_dtype: 'bfloat16', quantization_config: { quant_method: 'mxfp4' },
    },
  },
  {
    repo: 'Qwen/Qwen3.6-35B-A3B', name: 'Qwen3.6-35B-A3B', mark: 'Q', known: [35, 3],
    config: {
      architectures: ['Qwen3_5MoeForConditionalGeneration'], hidden_size: 2048, num_hidden_layers: 40, num_attention_heads: 16, num_key_value_heads: 2, head_dim: 256,
      full_attention_interval: 4, linear_num_value_heads: 32, linear_num_key_heads: 16, linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4,
      num_experts: 256, num_experts_per_tok: 8, moe_intermediate_size: 512, shared_expert_intermediate_size: 512, mtp_num_hidden_layers: 1,
      vocab_size: 248320, max_position_embeddings: 262144, tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
  {
    repo: 'Qwen/Qwen3.6-27B', name: 'Qwen3.6-27B', mark: 'Q', known: [27, 27], inferred: ['intermediate_size'],
    config: {
      architectures: ['Qwen3_5ForConditionalGeneration'], hidden_size: 5120, intermediate_size: 17408, num_hidden_layers: 64, num_attention_heads: 24,
      num_key_value_heads: 4, head_dim: 256, full_attention_interval: 4, linear_num_value_heads: 48, linear_num_key_heads: 16,
      linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4, mtp_num_hidden_layers: 1,
      vocab_size: 248320, max_position_embeddings: 262144, tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
];

// 解析内置或导入的条目：config.json 说不清的原生精度用条目上写的
export function specOf(entry) {
  const m = parseHfConfig(entry.config);
  if (entry.native) m.native = entry.native;
  return m;
}
