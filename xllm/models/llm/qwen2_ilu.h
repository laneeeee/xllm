/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

// QWen2 model compatible with huggingface weights
// ref to:
// https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/qwen2/modeling_qwen2.py
namespace xllm {

class QWen2ModelImpl : public torch::nn::Module {
 public:
  QWen2ModelImpl(const ModelContext& context) {}
};
TORCH_MODULE(QWen2Model);

template <typename QWenModelType>
class QWen2ForCausalLMImpl : public torch::nn::Module {
 public:
  QWen2ForCausalLMImpl(const ModelContext& context) {
    tie_word_embeddings = context.get_model_args().tie_word_embeddings();
    // register submodules
    model_ = register_module("model", QWenModelType(context));
    lm_head_ = register_module("lm_head", layer::LmHead(context));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  virtual torch::Tensor forward(
      const std::vector<torch::Tensor>& tokens,
      const std::vector<torch::Tensor>& positions,
      std::vector<KVCache>& kv_caches,
      const std::vector<ModelInputParams>& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) {
    // select tokens if provided
    auto h = hidden_states;
    // test
    return lm_head_(hidden_states, seleted_idxes, 0);
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "" /*llm model weight prefix*/) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(
          state_dict->get_dict_with_prefix(prefix + "model."));
      if (tie_word_embeddings) {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "model.embed_tokens."));
      } else {
        lm_head_->load_state_dict(
            state_dict->get_dict_with_prefix(prefix + "lm_head."));
      }
    }
    // verify
    model_->verify_loaded_weights(prefix + "model.");
    lm_head_->verify_loaded_weights(prefix + "lm_head.");

    model_->merge_loaded_weights();
    // test
    lm_head_->merge_loaded_weights();
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  virtual layer::LmHead get_lm_head() { return lm_head_; }

  virtual void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  virtual std::vector<layer::WordEmbedding> get_word_embedding() {
    return model_->get_word_embedding();
  }

  virtual void set_word_embedding(
      std::vector<layer::WordEmbedding>& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 protected:
  // parameter members, must be registered
  QWenModelType model_{nullptr};
  int device_id = 0;
  bool tie_word_embeddings{false};
  // test
  layer::LmHead lm_head_{nullptr};
};

TORCH_MODULE(QWen2ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen2, QWen2ForCausalLM);

// register the model args
// example config:
// https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/config.json
REGISTER_MODEL_ARGS(qwen2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen2");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  // LOAD_ARG_OR(no_bias, "no_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // For Qwen2/2.5 model < 7B,  tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
