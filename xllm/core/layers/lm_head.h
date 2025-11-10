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

#pragma once

#if defined(USE_NPU)
#include "npu/npu_lm_head_impl.h"
#elif defined(USE_ILU)
#include <torch/nn/modules/linear.h>
#endif
#include "core/framework/model_context.h"

namespace xllm {
namespace layer {

#if defined(USE_NPU)
class LmHead : public torch::nn::ModuleHolder<NpuLmHeadImpl> {
 public:
  using torch::nn::ModuleHolder<NpuLmHeadImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuLmHeadImpl;

  LmHead(const ModelContext& context)
      : ModuleHolder(std::make_shared<NpuLmHeadImpl>(context)) {}
};
#elif defined(USE_ILU)
class LmHead : public torch::nn::Module {
  //   public:
  //         const std::shared_ptr<ModelContext> place_holder;
  //     LmHead(const ModelContext&
  //     context):place_holder(std::make_shared<ModelContext>(context)){}
  //     LmHead(std::nullptr_t) : place_holder(nullptr) {}
  //     LmHead() : place_holder(nullptr) {}
  //     torch::Tensor forward(const torch::Tensor& hidden_states,
  //                          const torch::Tensor& selected_idxes,
  //                          int dummy) {

  //         return torch::Tensor();
  //     }
 public:
  LmHead(int64_t in_features, int64_t out_features, bool bias = false) {
    linear_ = register_module(
        "linear",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(bias)));
  }
  LmHead(std::nullptr_t) : linear_(nullptr) {}

  torch::Tensor forward(const torch::Tensor& x) { return linear_->forward(x); }

 private:
  torch::nn::Linear linear_{nullptr};
};
#endif

}  // namespace layer
}  // namespace xllm