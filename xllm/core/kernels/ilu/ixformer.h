#include <torch/all.h>
#include "ATen/Tensor.h"
#include "utils.h"

namespace ixformer::infer
{
        at::Tensor ixinfer_flash_attn_unpad(
        at::Tensor &query, at::Tensor &key, at::Tensor &value,
        at::Tensor &out, at::Tensor &cu_seq_q, at::Tensor &cu_seq_k,
        int64_t max_seq_q, int64_t max_seq_k, bool is_causal,
        int64_t window_left, int64_t window_right, double scale, double softcap,
        bool sqrt_alibi, const c10::optional<at::Tensor> &alibi_slopes,
        const c10::optional<at::Tensor> &sinks, c10::optional<at::Tensor> &lse);

        void silu_and_mul(at::Tensor &input, at::Tensor &output);

        at::Tensor vllm_paged_attention(
        at::Tensor &out, at::Tensor &query, at::Tensor &key_cache,
        at::Tensor &value_cache, int64_t num_kv_heads, double scale,
        at::Tensor &block_tables, at::Tensor &context_lens, int64_t block_size,
        int64_t max_context_len, const c10::optional<at::Tensor> &alibi_slopes,
        bool causal, int window_left, int window_right, double softcap, bool enable_cuda_graph, bool use_sqrt_alibi, 
        const c10::optional<at::Tensor> &sinks);
}

namespace ixformer {

kernels::TensorDesc to_tensor(const at::Tensor &src);
}// namespace ixformer

namespace ixformer::kernels::desc {
size_t get_paged_attention_workspace(int num_seqs, int num_heads, int num_kv_heads,
                                     int head_size, int block_size, int max_context_len);

void paged_attention(
        TensorDesc &out, TensorDesc &query, TensorDesc &key_cache,
        TensorDesc &value_cache, int64_t num_kv_heads, double scale,
        TensorDesc &block_tables, TensorDesc &context_lens, int64_t block_size,
        int64_t max_context_len, const std::optional<TensorDesc> &alibi_slopes,
        bool causal, int window_left, int window_right,
        double softcap, bool enable_cuda_graph, bool use_sqrt_alibi, const std::optional<TensorDesc> &sinks, cudaStream_t stream,
        void *workspace_ptr// size_t of char
);

}

