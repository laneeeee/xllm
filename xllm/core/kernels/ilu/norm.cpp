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

using namespace ixformer;


namespace xllm::kernel::ilu {

void layer_norm(at::Tensor &input,
                at::Tensor &weight,
                c10::optional<at::Tensor>& bias_tensor,
                const c10::optional<at::Tensor> &fused_bias,
                at::Tensor &output,
                double eps) {
    auto scalar_type = input.scalar_type();
    int hidden_size = weight.numel();
    at::Tensor bias = bias_tensor.value_or(
    at::zeros({hidden_size}, 
        at::TensorOptions()
            .dtype(input.scalar_type())
            .device(input.device())
        )
    );


    check_tensor_contiguous(weight, scalar_type);
    check_tensor_contiguous(bias, scalar_type);
    check_tensor_contiguous(output, scalar_type);

    // int hidden_size = weight.numel();
    TORCH_CHECK(bias.numel() == hidden_size);

    void *fused_bias_ptr = nullptr;
    if (fused_bias) {
        check_tensor_contiguous(fused_bias.value(), scalar_type);
        TORCH_CHECK(fused_bias.value().numel() == hidden_size);
        fused_bias_ptr = fused_bias.value().data_ptr();
    }

    int batch_tokens = 1;
    for (int i = 0; i < input.dim() - 1; ++i) {
        batch_tokens *= input.size(i);
    }

    TORCH_CHECK(input.size(input.dim() - 1) == hidden_size,
                "layernorm expects input.size(-1)==hidden_size");
    TORCH_CHECK(output.numel() == hidden_size * batch_tokens,
                "layernorm expects output.numel()==input.numel()");
    int in_stride = input.stride(-2);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::ScalarType::Float) {
        kernels::infer::layernorm<float>(
                reinterpret_cast<float *>(input.data_ptr()),
                reinterpret_cast<float *>(weight.data_ptr()),
                reinterpret_cast<float *>(bias.data_ptr()),
                reinterpret_cast<float *>(fused_bias_ptr),
                reinterpret_cast<float *>(output.data_ptr()), batch_tokens, hidden_size,
                in_stride, eps, stream);
    } else if (input.scalar_type() == at::ScalarType::Half) {
        kernels::infer::layernorm<half>(
                reinterpret_cast<half *>(input.data_ptr()),
                reinterpret_cast<half *>(weight.data_ptr()),
                reinterpret_cast<half *>(bias.data_ptr()),
                reinterpret_cast<half *>(fused_bias_ptr),
                reinterpret_cast<half *>(output.data_ptr()), batch_tokens, hidden_size,
                in_stride, eps, stream);
    } else if (input.scalar_type() == at::ScalarType::BFloat16) {
        kernels::infer::layernorm<__nv_bfloat16>(
                reinterpret_cast<__nv_bfloat16 *>(input.data_ptr()),
                reinterpret_cast<__nv_bfloat16 *>(weight.data_ptr()),
                reinterpret_cast<__nv_bfloat16 *>(bias.data_ptr()),
                reinterpret_cast<__nv_bfloat16 *>(fused_bias_ptr),
                reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()), batch_tokens, hidden_size,
                in_stride, eps, stream);
    } else {
        throw std::runtime_error("layernorm support scalar type half bf16 or float32");
    }
}


}  // namespace xllm::kernel::cuda