
#include "ilu_ops_api.h"
#include "ixformer.h"
#include "ixinfer.h"
#include "utils.h"

using namespace ixformer;

namespace xllm::kernel::ilu {

void reshape_paged_cache(const at::Tensor& key,
                         const at::Tensor& value,
                         at::Tensor& key_cache,
                         at::Tensor& value_cache,
                         at::Tensor& slot_mapping) {
  int64_t key_token_stride = key.stride(0);
  int64_t value_token_stride = value.stride(0);
  slot_mapping = slot_mapping.to(at::kLong);

  auto scalar_type = key.scalar_type();
  TORCH_CHECK(scalar_type == at::ScalarType::Half ||
              scalar_type == at::ScalarType::BFloat16);

  check_tensor_contiguous(key_cache, scalar_type);
  check_tensor_contiguous(value_cache, scalar_type);
  check_tensor_contiguous(slot_mapping, at::ScalarType::Long);

  TORCH_CHECK(key.scalar_type() == scalar_type);
  TORCH_CHECK(key.is_cuda());
  TORCH_CHECK(value.scalar_type() == scalar_type);
  TORCH_CHECK(value.is_cuda());

  // num_tokens, num_heads, head_size
  TORCH_CHECK(key.dim() == 3);
  TORCH_CHECK(value.dim() == 3);

  int num_tokens = key.size(0);
  size_t num_heads = key.size(1);
  size_t head_size = key.size(2);
  size_t value_head_size = value.size(2);
  size_t value_head_stride = value.stride(1);
  // this kernel is only used in vllm, key is always contiguous on dim [1,2],
  // and so should v. But v is not contiguous on model deepseek, wihch is split
  // form tensor: [b, num_heads, k + v], k is ready for used by using mla
  // kernels. so add a value head stride here to store k..

  TORCH_CHECK(key.stride(1) == head_size);  // key must contiguous on dim [1, 2]
  TORCH_CHECK(value.size(0) == num_tokens);
  TORCH_CHECK(value.size(1) == num_heads);
  // TORCH_CHECK(value.size(2) == head_size); # support value head_dim != key
  // head_dim for deepseek(128 and 192)

  // num_blocks, num_heads, block_size, head_size
  TORCH_CHECK(key_cache.dim() == 4);
  TORCH_CHECK(value_cache.dim() == 4);

  // translate kvcache shape from [n_blocks, block_size, n_heads, head_dim] to
  // (num_blocks, num_heads, block_size, head_size)
  key_cache = key_cache.permute({0, 2, 1, 3}).contiguous();
  if (value_cache.defined()) {
    value_cache = value_cache.permute({0, 2, 1, 3}).contiguous();
  }

  size_t num_blocks = key_cache.size(0);
  size_t block_size = key_cache.size(2);

  TORCH_CHECK(key_cache.numel() ==
              num_blocks * num_heads * block_size * head_size);
  TORCH_CHECK(value_cache.numel() ==
              num_blocks * num_heads * block_size *
                  head_size);  // still use head_size, but we only fill
                               // value_head_stride

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (scalar_type == at::ScalarType::Half) {
    kernels::infer::vllm_reshape_and_cache(
        (const half*)key.data_ptr(),
        (const half*)value.data_ptr(),
        (half*)key_cache.data_ptr(),
        (half*)value_cache.data_ptr(),
        (const int64_t*)slot_mapping.data_ptr(),
        key_token_stride,
        value_token_stride,
        value_head_stride,
        num_heads,
        head_size,
        value_head_size,
        block_size,
        num_tokens,
        stream);
  } else {
    kernels::infer::vllm_reshape_and_cache(
        (const __nv_bfloat16*)key.data_ptr(),
        (const __nv_bfloat16*)value.data_ptr(),
        (__nv_bfloat16*)key_cache.data_ptr(),
        (__nv_bfloat16*)value_cache.data_ptr(),
        (const int64_t*)slot_mapping.data_ptr(),
        key_token_stride,
        value_token_stride,
        value_head_stride,
        num_heads,
        head_size,
        value_head_size,
        block_size,
        num_tokens,
        stream);
  }
}

void batch_prefill(torch::Tensor& query,
                   torch::Tensor& key,
                   torch::Tensor& value,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& q_cu_seq_lens,
                   const std::optional<torch::Tensor>& kv_cu_seq_lens,
                   const std::optional<torch::Tensor>& alibi_slope,
                   const std::optional<torch::Tensor>& attn_bias,
                   const std::optional<torch::Tensor>& q_quant_scale,
                   const std::optional<torch::Tensor>& k_quant_scale,
                   const std::optional<torch::Tensor>& v_quant_scale,
                   int64_t max_query_len,
                   int64_t max_seq_len,
                   float scale,
                   bool is_causal,
                   int64_t window_size_left,
                   int64_t window_size_right,
                   const std::string& compute_dtype,
                   bool return_lse) {
  double softcap = 0.0;
  bool sqrt_alibi = false;
  auto q_cu_seq_lens_ = q_cu_seq_lens.value_or(torch::Tensor());
  auto kv_cu_seq_lens_ = kv_cu_seq_lens.value_or(torch::Tensor());
  auto q_quant_scale_ = q_quant_scale.value_or(torch::Tensor());
  auto k_quant_scale_ = k_quant_scale.value_or(torch::Tensor());
  auto v_quant_scale_ = v_quant_scale.value_or(torch::Tensor());

  infer::ixinfer_flash_attn_unpad(query,
                                  key,
                                  value,
                                  output,
                                  q_cu_seq_lens_,
                                  kv_cu_seq_lens_,
                                  max_query_len,
                                  max_seq_len,
                                  is_causal,
                                  window_size_left,
                                  window_size_right,
                                  static_cast<double>(scale),
                                  softcap,
                                  sqrt_alibi,
                                  alibi_slope,
                                  c10::nullopt,
                                  output_lse);
}

size_t get_paged_attention_workspace(int num_seqs,
                                     int num_heads,
                                     int num_kv_heads,
                                     int head_size,
                                     int block_size,
                                     int max_context_len) {
  size_t workspace_size = 0;
  std::cout << num_seqs << "," << num_heads << "," << num_kv_heads << ","
            << head_size << "," << block_size << "," << max_context_len
            << std::endl;
  CUINFER_CHECK(cuInferPageAttentionGetWorkspaceV7(num_seqs,
                                                   num_heads,
                                                   num_kv_heads,
                                                   head_size,
                                                   block_size,
                                                   max_context_len,
                                                   &workspace_size));
  return workspace_size;
}

at::Tensor vllm_paged_attention(at::Tensor& out,
                                at::Tensor& query,
                                at::Tensor& key_cache,
                                at::Tensor& value_cache,
                                int64_t num_kv_heads,
                                double scale,
                                at::Tensor& block_tables,
                                at::Tensor& context_lens,
                                int64_t block_size,
                                int64_t max_context_len,
                                const c10::optional<at::Tensor>& alibi_slopes,
                                bool causal,
                                int window_left,
                                int window_right,
                                double softcap,
                                bool enable_cuda_graph,
                                bool use_sqrt_alibi,
                                const c10::optional<at::Tensor>& sinks) {
  kernels::TensorDesc out_desc = to_tensor(out);
  kernels::TensorDesc query_desc = to_tensor(query);
  kernels::TensorDesc key_cache_desc = to_tensor(key_cache);
  kernels::TensorDesc value_cache_desc = to_tensor(value_cache);
  kernels::TensorDesc block_tables_desc = to_tensor(block_tables);
  kernels::TensorDesc context_lens_desc = to_tensor(context_lens);
  std::optional<kernels::TensorDesc> alibi_slopes_desc;
  std::optional<kernels::TensorDesc> sinks_desc;

  if (alibi_slopes) {
    kernels::TensorDesc alibi_slopes_desc_ = to_tensor(alibi_slopes.value());
    alibi_slopes_desc = alibi_slopes_desc_;
  }

  if (sinks) {
    kernels::TensorDesc sinks_desc_ = to_tensor(sinks.value());
    sinks_desc = sinks_desc_;
  }

  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  size_t workspace_size = get_paged_attention_workspace(num_seqs,
                                                        num_heads,
                                                        num_kv_heads,
                                                        head_size,
                                                        block_size,
                                                        max_context_len);
  char* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    auto options = torch::TensorOptions()
                       .dtype(torch::kChar)
                       .layout(query.layout())
                       .device(query.device())
                       .requires_grad(false);
    at::Tensor workspace = at::empty({(int64_t)workspace_size}, options);
    workspace_ptr = reinterpret_cast<char*>(workspace.data_ptr());
  }
  kernels::desc::paged_attention(out_desc,
                                 query_desc,
                                 key_cache_desc,
                                 value_cache_desc,
                                 num_kv_heads,
                                 scale,
                                 block_tables_desc,
                                 context_lens_desc,
                                 block_size,
                                 max_context_len,
                                 alibi_slopes_desc,
                                 causal,
                                 window_left,
                                 window_right,
                                 softcap,
                                 enable_cuda_graph,
                                 use_sqrt_alibi,
                                 sinks_desc,
                                 stream,
                                 workspace_ptr);
  return out;
}

void batch_decode(torch::Tensor& query,
                  torch::Tensor& k_cache,
                  torch::Tensor& output,
                  torch::Tensor& block_table,
                  torch::Tensor& seq_lens,
                  const std::optional<torch::Tensor>& v_cache,
                  std::optional<torch::Tensor>& output_lse,
                  const std::optional<torch::Tensor>& q_quant_scale,
                  const std::optional<torch::Tensor>& k_cache_quant_scale,
                  const std::optional<torch::Tensor>& v_cache_quant_scale,
                  const std::optional<torch::Tensor>& out_quant_scale,
                  const std::optional<torch::Tensor>& alibi_slope,
                  const std::optional<torch::Tensor>& mask,
                  const std::string& compute_dtype,
                  int64_t max_seq_len,
                  int64_t window_size_left,
                  int64_t window_size_right,
                  float scale,
                  bool return_lse,
                  bool is_causal,
                  int64_t kv_cache_quant_bit_size) {
  if (query.dim() == 4) {
    query =
        query
            .view({query.size(0) * query.size(1), query.size(2), query.size(3)})
            .contiguous();
  }
  if (output.dim() == 4) {
    output = output
                 .view({output.size(0) * output.size(1),
                        output.size(2),
                        output.size(3)})
                 .contiguous();
  }
  auto v_cache_ = v_cache.value_or(torch::Tensor());
  k_cache = k_cache.permute({0, 2, 1, 3}).contiguous();
  v_cache_ = v_cache_.permute({0, 2, 1, 3}).contiguous();
  auto num_kv_heads = k_cache.size(1);
  auto page_block_size = k_cache.size(2);
  double softcap = 0.0;
  bool enable_cuda_graph = false;
  bool use_sqrt_alibi = false;
  // check_tensor_contiguous(k_cache, query.dtype());

  infer::vllm_paged_attention(output,
                              query,
                              k_cache,
                              v_cache_,
                              static_cast<int64_t>(num_kv_heads),
                              scale,
                              block_table,
                              seq_lens,
                              page_block_size,
                              max_seq_len,
                              alibi_slope,
                              is_causal,
                              window_size_left,
                              window_size_right,
                              softcap,
                              enable_cuda_graph,
                              use_sqrt_alibi,
                              c10::nullopt);
}

}  // namespace xllm::kernel::ilu