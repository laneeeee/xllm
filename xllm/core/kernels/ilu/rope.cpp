/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ilu_ops_api.h"
#include "utils.h"

namespace xllm::kernel::ilu {

void apply_rope_pos_ids_cos_sin_cache(at::Tensor& query,
                                      at::Tensor& key,
                                      at::Tensor& cos_sin_cache,
                                      at::Tensor& positions,
                                      bool interleave) {
  const int64_t head_size = cos_sin_cache.size(-1) / 2;
  infer::vllm_rotary_embedding(
      positions, query, key, head_size, cos_sin_cache, !interleave);
}
/*
void apply_rope_pos_ids_cos_sin_cache(at::Tensor &query,
                                        at::Tensor &key,
                                        at::Tensor &cos_sin_cache,
                                        at::Tensor &positions,
                                        bool interleave) {
    check_tensor_half_bf_float(query);
    check_tensor_half_bf_float(key);
    check_tensor_half_bf_float(cos_sin_cache);

    positions = positions.to(at::kLong);
    TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long);
    TORCH_CHECK(positions.is_cuda());
    TORCH_CHECK(positions.is_contiguous());

    TORCH_CHECK(positions.dim() <= 2);
    TORCH_CHECK(query.dim() == 3 || query.dim() == 2);
    TORCH_CHECK(key.dim() == 3 || key.dim() == 2);
    TORCH_CHECK(cos_sin_cache.dim() == 2);

    const int64_t head_size = cos_sin_cache.size(-1) / 2;
    query = query.view({query.size(0), -1, head_size});
    key = key.view({key.size(0), -1, head_size});
    int64_t num_tokens = 0;
    int rot_dim = cos_sin_cache.size(1);
    int num_heads = 0;
    int num_kv_heads = 0;
    int64_t query_head_stride = head_size;
    int64_t query_token_stride = 0;
    int64_t key_head_stride = head_size;
    int64_t key_token_stride = 0;

    TORCH_CHECK(query.stride(-1) == 1, "query must contiguous on stride(-1)");
    TORCH_CHECK(key.stride(-1) == 1, "key must contiguous on stride(-1)");

    key_head_stride = key.stride(-2);
    key_token_stride = key.stride(-3);
    num_kv_heads = key.size(-2);
    query_head_stride = query.stride(-2);
    query_token_stride = query.stride(-3);
    num_heads = query.size(-2);
    num_tokens = query.numel() / (num_heads * head_size);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define CALL_VLLM_ROTARY_EMBEDDING(SCALAR_T, CACHE_T, IS_NEOX)         \
    kernels::infer::vllm_rotary_embedding<SCALAR_T, CACHE_T, IS_NEOX>( \
            (const int64_t *) positions.data_ptr(),                    \
            (SCALAR_T *) query.data_ptr(),                             \
            (SCALAR_T *) key.data_ptr(),                               \
            (const CACHE_T *) cos_sin_cache.data_ptr(),                \
            rot_dim, query_head_stride, query_token_stride,            \
            key_head_stride, key_token_stride,                         \
            num_heads, num_kv_heads, head_size, num_tokens, stream)

#define CHECK_CACHE_AND_CALL(SCALAR_T, IS_NEOX)                          \
    if (cos_sin_cache.dtype() == at::ScalarType::Float) {                \
        CALL_VLLM_ROTARY_EMBEDDING(SCALAR_T, float, IS_NEOX);            \
    } else if (cos_sin_cache.dtype() == at::ScalarType::Half) {          \
        CALL_VLLM_ROTARY_EMBEDDING(SCALAR_T, half, IS_NEOX);             \
    } else if (cos_sin_cache.dtype() == at::ScalarType::BFloat16) {      \
        CALL_VLLM_ROTARY_EMBEDDING(SCALAR_T, __nv_bfloat16, IS_NEOX);    \
    } else {                                                             \
        throw std::runtime_error("Unsupported cos_sin_cache data type"); \
    }

    if (!interleave) {
        if (query.dtype() == at::ScalarType::Float) {
            CHECK_CACHE_AND_CALL(float, true);
        } else if (query.dtype() == at::ScalarType::Half) {
            CHECK_CACHE_AND_CALL(half, true);
        } else if (query.dtype() == at::ScalarType::BFloat16) {
            CHECK_CACHE_AND_CALL(__nv_bfloat16, true);
        } else {
            throw std::runtime_error("Unsupported query/key data type");
        }
    } else {
        if (query.dtype() == at::ScalarType::Float) {
            CHECK_CACHE_AND_CALL(float, false);
        } else if (query.dtype() == at::ScalarType::Half) {
            CHECK_CACHE_AND_CALL(half, false);
        } else if (query.dtype() == at::ScalarType::BFloat16) {
            CHECK_CACHE_AND_CALL(__nv_bfloat16, false);
        } else {
            throw std::runtime_error("Unsupported query/key data type");
        }
    }

#undef CHECK_CACHE_AND_CALL
#undef CALL_VLLM_ROTARY_EMBEDDING
}
*/
}  // namespace xllm::kernel::ilu
