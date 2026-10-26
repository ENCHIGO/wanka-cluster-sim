// 常用开源模型的 config.json 关键字段，按各仓库公开的 config.json 整理。
// 在连不上 Hugging Face 的地方（比如在 Claude 里打开）就用这份。
// known：模型卡上写的总参数 / 激活参数（十亿），只用来在测试里核对估算。
export const CATALOG = [
  {
    repo: 'deepseek-ai/DeepSeek-V3', name: 'DeepSeek-V3', mark: 'D', known: [671, 37],
    config: {
      architectures: ['DeepseekV3ForCausalLM'], hidden_size: 7168, intermediate_size: 18432, moe_intermediate_size: 2048,
      num_hidden_layers: 61, num_attention_heads: 128, num_key_value_heads: 128, n_routed_experts: 256, n_shared_experts: 1,
      num_experts_per_tok: 8, first_k_dense_replace: 3, kv_lora_rank: 512, q_lora_rank: 1536, qk_nope_head_dim: 128,
      qk_rope_head_dim: 64, v_head_dim: 128, vocab_size: 129280, max_position_embeddings: 163840, tie_word_embeddings: false,
      torch_dtype: 'bfloat16', quantization_config: { quant_method: 'fp8', fmt: 'e4m3', weight_block_size: [128, 128] },
    },
  },
  {
    repo: 'moonshotai/Kimi-K2-Instruct', name: 'Kimi-K2-Instruct', mark: 'K', known: [1040, 32],
    config: {
      architectures: ['DeepseekV3ForCausalLM'], hidden_size: 7168, intermediate_size: 18432, moe_intermediate_size: 2048,
      num_hidden_layers: 61, num_attention_heads: 64, num_key_value_heads: 64, n_routed_experts: 384, n_shared_experts: 1,
      num_experts_per_tok: 8, first_k_dense_replace: 1, kv_lora_rank: 512, q_lora_rank: 1536, qk_nope_head_dim: 128,
      qk_rope_head_dim: 64, v_head_dim: 128, vocab_size: 163840, max_position_embeddings: 131072, tie_word_embeddings: false,
      torch_dtype: 'bfloat16', quantization_config: { quant_method: 'fp8', fmt: 'e4m3', weight_block_size: [128, 128] },
    },
  },
  {
    repo: 'zai-org/GLM-4.5', name: 'GLM-4.5', mark: 'Z', known: [355, 32],
    config: {
      architectures: ['Glm4MoeForCausalLM'], hidden_size: 5120, intermediate_size: 12288, moe_intermediate_size: 1536,
      num_hidden_layers: 92, num_attention_heads: 96, num_key_value_heads: 8, head_dim: 128, n_routed_experts: 160,
      n_shared_experts: 1, num_experts_per_tok: 8, first_k_dense_replace: 3, vocab_size: 151552,
      max_position_embeddings: 131072, tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
  {
    repo: 'Qwen/Qwen3-235B-A22B', name: 'Qwen3-235B-A22B', mark: 'Q', known: [235, 22],
    config: {
      architectures: ['Qwen3MoeForCausalLM'], hidden_size: 4096, intermediate_size: 12288, moe_intermediate_size: 1536,
      num_hidden_layers: 94, num_attention_heads: 64, num_key_value_heads: 4, head_dim: 128, num_experts: 128,
      num_experts_per_tok: 8, vocab_size: 151936, max_position_embeddings: 40960, tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
  {
    repo: 'openai/gpt-oss-120b', name: 'gpt-oss-120b', mark: 'O', known: [117, 5.1],
    config: {
      architectures: ['GptOssForCausalLM'], hidden_size: 2880, intermediate_size: 2880, num_hidden_layers: 36,
      num_attention_heads: 64, num_key_value_heads: 8, head_dim: 64, num_local_experts: 128, num_experts_per_tok: 4,
      vocab_size: 201088, max_position_embeddings: 131072, tie_word_embeddings: false, torch_dtype: 'bfloat16',
      quantization_config: { quant_method: 'mxfp4' },
    },
  },
  {
    repo: 'Qwen/Qwen2.5-72B-Instruct', name: 'Qwen2.5-72B-Instruct', mark: 'Q', known: [72.7, 72.7],
    config: {
      architectures: ['Qwen2ForCausalLM'], hidden_size: 8192, intermediate_size: 29568, num_hidden_layers: 80,
      num_attention_heads: 64, num_key_value_heads: 8, vocab_size: 152064, max_position_embeddings: 32768,
      tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
  {
    repo: 'meta-llama/Llama-3.1-70B-Instruct', name: 'Llama-3.1-70B-Instruct', mark: 'M', gated: true, known: [70.6, 70.6],
    config: {
      architectures: ['LlamaForCausalLM'], hidden_size: 8192, intermediate_size: 28672, num_hidden_layers: 80,
      num_attention_heads: 64, num_key_value_heads: 8, vocab_size: 128256, max_position_embeddings: 131072,
      tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
  {
    repo: 'Qwen/Qwen3-30B-A3B', name: 'Qwen3-30B-A3B', mark: 'Q', known: [30.5, 3.3],
    config: {
      architectures: ['Qwen3MoeForCausalLM'], hidden_size: 2048, intermediate_size: 6144, moe_intermediate_size: 768,
      num_hidden_layers: 48, num_attention_heads: 32, num_key_value_heads: 4, head_dim: 128, num_experts: 128,
      num_experts_per_tok: 8, vocab_size: 151936, max_position_embeddings: 40960, tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
  {
    repo: 'Qwen/Qwen3-8B', name: 'Qwen3-8B', mark: 'Q', known: [8.2, 8.2],
    config: {
      architectures: ['Qwen3ForCausalLM'], hidden_size: 4096, intermediate_size: 12288, num_hidden_layers: 36,
      num_attention_heads: 32, num_key_value_heads: 8, head_dim: 128, vocab_size: 151936, max_position_embeddings: 40960,
      tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
  {
    repo: 'meta-llama/Llama-3.1-8B-Instruct', name: 'Llama-3.1-8B-Instruct', mark: 'M', gated: true, known: [8.03, 8.03],
    config: {
      architectures: ['LlamaForCausalLM'], hidden_size: 4096, intermediate_size: 14336, num_hidden_layers: 32,
      num_attention_heads: 32, num_key_value_heads: 8, vocab_size: 128256, max_position_embeddings: 131072,
      tie_word_embeddings: false, torch_dtype: 'bfloat16',
    },
  },
];
