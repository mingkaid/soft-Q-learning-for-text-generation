# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import json
import torch
import torch.nn as nn
import texar.torch as tx
from functools import partial
from typing import List, Tuple, Union, Dict, Optional, Callable, Any, cast

from configs.models import (
    config_model_transformers_small)

from sql.utils import ForwardMode
from sql.types import (
    BatchType,
    FloatTensor,
    LongTensor)

from transformers import pipeline, AutoTokenizer

def _build_gpt2_vocab_mlp(out_dim, in_dim=768, device=0): 
#     W1 = nn.Linear(in_dim, 1024)
#     A = nn.ReLU()
#     W2 = nn.Linear(1024, 64)
#     O = nn.Linear(64, out_dim)
#     return nn.Sequential(W1, A, W2, O)

#     W1 = nn.Linear(in_dim, 1024)
#     A1 = nn.ReLU()
#     W2 = nn.Linear(1024, 512)
#     A2 = nn.ReLU()
#     W3 = nn.Linear(512, 256)
#     A3 = nn.ReLU()
#     W4 = nn.Linear(256, 64)
#     O = nn.Linear(64, out_dim)
    
    W1 = nn.Linear(in_dim, 2048)
    A1 = nn.ReLU()
    # D1 = nn.Dropout(p=0.1)
    W2 = nn.Linear(2048, 768)
    # N2 = nn.LayerNorm(768)
    # O = nn.Linear(64, out_dim)
    
    # return nn.Sequential(W1, A1, W2, N2)
    return nn.Sequential(W1, A1, W2)

def _build_self_vocab_mlp(out_dim, in_dim=768, device=0): 
    W1 = nn.Linear(in_dim, 2048)
    A1 = nn.ReLU()
    O = nn.Linear(2048, out_dim)
    return nn.Sequential(W1, A1, O)


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    r"""Adapted from
    https://github.com/openai/gpt-2/blob/master/src/sample.py#L63-L77
    """
    if k == 0:
        # no truncation
        return logits

    values, _ = torch.topk(logits, k=k)
    min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_values,
        torch.full_like(logits, float('-inf')), logits)

def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    r"""Adapted from
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317#file-top-k-top-p-py-L16-L27"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the
    # threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    for idx in range(logits.size(0)):
        batch_indices = sorted_indices[idx, sorted_indices_to_remove[idx]]
        logits[idx, batch_indices] = float("-inf")
    return logits

class GPT2ConditionedMLP(nn.Module): 
    input_template = 'Sentence 1: "{sentence_1}"'
    output_template = 'Sentence 2: "'
    sentence_1 = 'thank you for a five star service .'
    sentence_2 = "thank you for the meal ."
    temp_input_2 = "the mojitos are deliciously made with fresh fruit ."
    temp_input_0 = "they are all really friendly and the vets are knowledgable and patient ."
    temp_input_3 = "someone should buy this place and turn it into what it should be."
    temp_input_6 = "manager actually has respect for the customer instead of ignoring them ."
    
    sentence_1 = 'thank you for a five star service .'
#     sentence_1 = 'classification'
#     sentence_1 = 'Prompt: '

    def __init__(self, 
                 train_data: tx.data.PairedTextData,
                 max_source_length: int,
                 max_decoding_length: int,
                 config_name: str,
                 device=0) -> None: 
        super().__init__()
        
        if config_name not in ['gpt2_conditioned_mlp']: 
            raise ValueError
#         if config_name == 'gpt2_conditioned_mlp': 
#             config_model = config_model_gpt2_conditioned_mlp
            
        # self.config_model = config_model
        self.device = device
        self.max_source_length = max_source_length
        self.max_decoding_length = max_decoding_length
        
        self.source_vocab = train_data.source_vocab
        self.target_vocab = train_data.target_vocab
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id
        
        model = 'distilgpt2'
        self.tokenizer = AutoTokenizer.from_pretrained(model, pad_token='<|endoftext|>')
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model=model,
                                  device=0)
        for param in self.generator.model.parameters():
            param.requires_grad = False
        
        mode = 'gpt2_vocab'
        # mode = 'self_vocab'
        # mode = 'prep_vocab'
        if mode == 'gpt2_vocab': 
            self.mlp = _build_gpt2_vocab_mlp(self.target_vocab_size).to(0)
            self._mlp_forward = self._gpt2_vocab_mlp_forward
            self.valid_token_ids = None
        elif mode == 'self_vocab': 
            self.mlp = _build_self_vocab_mlp(self.target_vocab_size - 4).to(0)
            # self._mlp_forward = self.mlp
            self._mlp_forward = self._self_vocab_mlp_forward
        elif mode == 'prep_vocab': 
            self.mlp = _build_gpt2_vocab_mlp(self.target_vocab_size).to(0)
            self._mlp_forward = self._gpt2_vocab_mlp_forward
            self.valid_token_ids = json.load(open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_negative_prep'))
            self.valid_token_ids = json.load(open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_3_tokens'))
            self.valid_token_ids = json.load(open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_3_tokens_rep_100'))
            self.valid_token_ids = json.load(open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_1k_random_tokens'))
            self.valid_token_ids = json.load(open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_3_tokens_1'))

            
#         self.dataset_inputs = ['the carts are in excellent shape, all electric and all equipped with gps. the',
#                                'challenging but fun course! the',
#                                'beautiful views and lots of variety of length and layout of holes. the',
#                                "i'll definitely be back! the",
#                                'the service and prices were great. the',
#                                'i had the buffalo chicken sandwich and it was delicious. the',
#                                'a cool bar off the beaten path that is a worth a trip. the',
#                                'awesome drink specials during happy hour. the',
#                                'fantastic wings that are crispy and delicious, wing night on tuesday and thursday! the',
#                                'the sandwiches are always amazing just as i remember. the',
#                                'the staff is amazing and friendly. the',
#                                'great place for lunch as well. the',
#                                'friendly staff, good food, great beer selection, and relaxing atmosphere. the',
#                                "great wings and the buffalo chicken pizza is the best i've had. the",
#                                'the sandwiches are all on thick cut italian bread and fresh. the',
#                                'if we ever get to the pittsburgh area again, we will go back! the']
        
        self.dataset_inputs = ['the carts are in excellent shape, all electric and all equipped with gps.',
                               'challenging but fun course!',
                               'beautiful views and lots of variety of length and layout of holes.',
                               "i'll definitely be back!",
                               'the service and prices were great.',
                               'i had the buffalo chicken sandwich and it was delicious.',
                               'a cool bar off the beaten path that is a worth a trip.',
                               'awesome drink specials during happy hour.',
                               'fantastic wings that are crispy and delicious, wing night on tuesday and thursday!',
                               'the sandwiches are always amazing just as i remember.',
                               'the staff is amazing and friendly.',
                               'great place for lunch as well.',
                               'friendly staff, good food, great beer selection, and relaxing atmosphere.',
                               "great wings and the buffalo chicken pizza is the best i've had.",
                               'the sandwiches are all on thick cut italian bread and fresh.',
                               'if we ever get to the pittsburgh area again, we will go back!']
        
        self.dataset_inputs = ['the carts are in excellent shape, all electric and all equipped with gps.',
                               'challenging but fun course!',
                               'beautiful views and lots of variety of length and layout of holes.',
                               "i'll definitely be back!",
                               'the service and prices were great.',
                               'i had the buffalo chicken sandwich and it was delicious.',
                               'a cool bar off the beaten path that is a worth a trip.',
                               'awesome drink specials during happy hour.',]
        import itertools
        n_repeats = 4
        self.dataset_inputs = list(itertools.chain(*[[s for _ in range(n_repeats)] for s in self.dataset_inputs]))
        print(self.dataset_inputs)
        
#         self.dataset_inputs = ['challenging but fun course!',
#                                "i'll definitely be back!",
#                                'the service and prices were great.',
#                                'i had the buffalo chicken sandwich and it was delicious.',
#                                'thank you for a five star service.',
#                                # 'a cool bar off the beaten path that is a worth a trip.',
#                                'awesome drink specials during happy hour.',
#                                'fantastic wings that are crispy and delicious, wing night on tuesday and thursday!',
#                                'the sandwiches are always amazing just as i remember.']

        # self.dataset_inputs = ['this is good.']
        
#         self.dataset_inputs = ['thank you for a five star service.',
#                                'this is good.']
        
        # self.dataset_inputs = ['thank you for a five star service.']
        
        # self.dataset_inputs = ['the service and prices were great.']
        
#         self.dataset_inputs = ['thank you for a five star service.',
#                                'the service and prices were great.']
        
        # self.dataset_inputs = ["i'll definitely be back!"]
        
#         self.dataset_inputs = ["i'll definitely be back!",
#                                'this is good.']
        
#         self.dataset_inputs = ["i'll definitely be back!",
#                                'this is good.',
#                                'the prices were great.',
#                                'great place for lunch as well.']
        
#         self.dataset_inputs = ["i'll definitely be back!",
#                                'this is good.',
#                                'the service and prices were great.',
#                                'great place for lunch as well.']

        # self.dataset_inputs = ['the carts are in excellent shape, all electric and all equipped with gps.']
        
        # self.dataset_inputs = ['i had the buffalo chicken sandwich and it was delicious.']
        
        # self.dataset_inputs = ['a cool bar off the beaten path that is a worth a trip.']
        
        # self.dataset_inputs = ['challenging but fun course!']
        
        # self.dataset_inputs = ['awesome drink specials during happy hour.']
        
        # self.dataset_inputs = ['the service and prices were great.']
        
        # self.dataset_inputs = ['beautiful views and lots of variety of length and layout of holes.']
        
#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'beautiful views and lots of variety of length and layout of holes.']
        
#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'the service and prices were great.']
        
#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'the service and prices were great.',
#                                'beautiful views and lots of variety of length and layout of holes.']

#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'the service and prices were great.',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                'challenging but fun course!']
#         self.dataset_inputs = list(itertools.chain(*[[s, s, s, s] for s in self.dataset_inputs]))
#         print(self.dataset_inputs)
    
#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'awesome drink specials during happy hour.',
#                                'awesome drink specials during happy hour.',
#                                'awesome drink specials during happy hour.',
#                                'the service and prices were great.',
#                                'the service and prices were great.',
#                                'the service and prices were great.',
#                                'the service and prices were great.',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                'challenging but fun course!',
#                                'challenging but fun course!',
#                                'challenging but fun course!',
#                                'challenging but fun course!']

#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'thank you for a five star service.']
    
#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'the service and prices were great.',
#                                'thank you for a five star service.']
        
        # self.dataset_inputs = ['the prices were great.']
        
        # self.dataset_inputs = ['the service and prices were good.']
        
        # self.dataset_inputs = ['great place for lunch as well.']
        
#         self.dataset_inputs = ['great place for lunch as well.',
#                                'this is good.']
        
#         self.dataset_inputs = ['thank you for a five star service.',
#                                'this is good.',
#                                'the service and prices were great.',
#                                'great place for lunch as well.']
        
        self.temp_input = 'this is good.'
        # self.temp_input = ' '
        # self.temp_input = 'Prompt:'
        # self.temp_input = 'classification'
        # self.temp_input = self.sentence_1
#         self.temp_input = self.sentence_2
#         self.temp_input = self.temp_input_2
#         self.temp_input = self.temp_input_0
        # self.temp_input = self.temp_input_6
        self._tst_inputs = self._load_tst_inputs()
        self._tst_inputs_idx = {('train', 'LABEL_0'): 0, 
                                ('train', 'LABEL_1'): 0,
                                ('infer', 'LABEL_0'): 0,
                                ('infer', 'LABEL_1'): 0}
        self.fluent = False
        self.logit_bias = -10
        # self.logit_bias = nn.Parameter(torch.tensor(-10.))
        self.normalize_mlp_output = False
        

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.mlp.apply(init_weights)
    
    
    def _gpt2_vocab_mlp_forward(self, state): 
        mlp_output = self.mlp(state)
        
        if self.normalize_mlp_output:
            mlp_output = mlp_output / torch.linalg.norm(mlp_output, dim=-1).unsqueeze(-1)
            
        # print('Norm:', torch.linalg.norm(mlp_output, dim=-1).mean().item())
        logits = self.generator.model.lm_head(mlp_output)
        if self.valid_token_ids is not None: 
            logits = logits[:, self.valid_token_ids]
            
        if self.fluent: 
            plm_logits = self.generator.model.lm_head(state)
            values, _ = torch.topk(plm_logits, k=20)
            min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                plm_logits < min_values,
                torch.full_like(logits, float('-inf')), logits)
        # print(logits.shape)
        zeros = torch.ones_like(logits)[:, :4] * float('-inf')
        # print(zeros)
        modified_logits = torch.cat([zeros, logits], dim=-1)
        # print(modified_logits.shape)
        return modified_logits
    
    def _self_vocab_mlp_forward(self, state): 
        logits = self.mlp(state)
        zeros = torch.ones_like(logits)[:, :4] * float('-inf')
        # print(zeros)
        modified_logits = torch.cat([zeros, logits], dim=-1)
        # print(modified_logits.shape)
        return modified_logits
    
    def _load_tst_inputs(self) -> Dict[Tuple[Any], List[str]]: 
        tst_inputs = {}
        # tokenizer = self._generator.tokenizer
        filepath_train_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.0"
        filepath_train_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.1"
        filepath_dev_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.0"
        filepath_dev_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.1"
        
        with open(filepath_train_0) as f: 
            sentences_train_0 = [line.strip() for line in f]
        with open(filepath_train_1) as f: 
            sentences_train_1 = [line.strip() for line in f]
        with open(filepath_dev_0) as f: 
            sentences_dev_0 = [line.strip() for line in f]
        with open(filepath_dev_1) as f: 
            sentences_dev_1 = [line.strip() for line in f]
            
        idx = 43
        size = len(self.dataset_inputs)
        tst_inputs[('train', 'LABEL_0')] = sentences_train_1[idx:(idx+size)]
        tst_inputs[('train', 'LABEL_1')] = sentences_train_0[idx:(idx+size)]
        # tst_inputs[('train', 'LABEL_0')] = sentences_train_1[idx:]
        # tst_inputs[('train', 'LABEL_1')] = sentences_train_0[idx:]
        tst_inputs[('infer', 'LABEL_0')] = sentences_train_1[idx:(idx+size)]
        tst_inputs[('infer', 'LABEL_1')] = sentences_train_0[idx:(idx+size)]
        
        return tst_inputs
        
    def decode_teacher_forcing(self,
                               batch: BatchType,
                               last_token_hidden_state,
                               past_key_values) \
    -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        
        state = last_token_hidden_state
        sample_ids, sample_logits = batch['target_text_ids'][:, 1:], []
        #print(sample_ids)
        for i in range(self.max_decoding_length): 
            # logits = self.mlp(state)
            logits = self._mlp_forward(state)
            logits = logits + self.logit_bias
            
            actions = sample_ids[:, i]
            tokens = self.target_vocab.map_ids_to_tokens_py(actions.tolist()).tolist()
#             tokens = self.generator.tokenizer.convert_ids_to_tokens(actions.tolist())
            # if i == 0: print(tokens)
            
            # sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))
            
            tokens = [self.generator.tokenizer.convert_tokens_to_string([t]) \
                      for t in tokens]
            # tokens = [' ' + t for t in tokens]
            token_encoding = (self.generator
                               .tokenizer(tokens, 
                                          padding=True,
                                          return_tensors='pt')
                               .to(0))
            input_ids = token_encoding['input_ids']
            input_lengths = token_encoding['attention_mask'].sum(dim=1)

            next_outputs = (self.generator.model
                            .transformer(input_ids, 
                                         past_key_values=past_key_values, 
                                         use_cache=True))
            state = next_outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                   (input_lengths - 1)]
            past_key_values = next_outputs.past_key_values
            
        # sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
            
        decoder_output = tx.modules.TransformerDecoderOutput(
            logits=sample_logits,
            sample_id=sample_ids
        )
        return decoder_output, None
    
    def decode_sampling(self,
                        batch: BatchType,
                        last_token_hidden_state,
                        past_key_values,
                        top_k: Optional[int] = None,
                        top_p: Optional[float] = None) \
    -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        if top_k is not None and top_p is not None:
            raise ValueError
            
        print(self.logit_bias)
            
        state = last_token_hidden_state
        prompt_tokens, sample_ids, sample_logits = [], [], []
        for i in range(self.max_decoding_length): 
            # logits = self.mlp(state)
            logits = self._mlp_forward(state)
            logits = logits + self.logit_bias
            # print(state.min().item(), state.max().item())
            print(logits[:, 4:].min().item(), logits.max().item())
            # print(logits.min().item(), logits.max().item())
            
            if top_k is not None: sampling_logits = _top_k_logits(logits, k=top_k)
            elif top_p is not None: sampling_logits = _top_p_logits(logits, p=top_p)
            else: sampling_logits = logits
            
            # print(sampling_logits)
            actions = (torch.distributions.categorical
                       .Categorical(logits=sampling_logits)
                       .sample())
            # print(actions)
            tokens = self.target_vocab.map_ids_to_tokens_py(actions.tolist()).tolist()
#             tokens = self.generator.tokenizer.convert_ids_to_tokens(actions.tolist())
            # if i == 0: print(tokens)
            
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))
            
            tokens = [self.generator.tokenizer.convert_tokens_to_string([t]) \
                      for t in tokens]
            # print(tokens)
            
            token_encoding = (self.generator
                               .tokenizer(tokens, 
                                          padding=True,
                                          return_tensors='pt')
                               .to(0))
            input_ids = token_encoding['input_ids']
            input_lengths = token_encoding['attention_mask'].sum(dim=1)

            next_outputs = (self.generator.model
                            .transformer(input_ids, 
                                         past_key_values=past_key_values, 
                                         use_cache=True))
            state = next_outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                   (input_lengths - 1)]
            past_key_values = next_outputs.past_key_values
            
        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
        # print(sample_logits.min().item(), sample_logits.max().item())
            
        decoder_output = tx.modules.TransformerDecoderOutput(
            logits=sample_logits,
            sample_id=sample_ids
        )
        return (decoder_output, 
                torch.tensor([self.max_decoding_length \
                              for _ in range(sample_ids.shape[0])]).to(0)
               )

    def decode_greedy(
            self,
            batch: BatchType,
            last_token_hidden_state,
            past_key_values,
            corruption_p: Optional[float] = None,
            input_mode='train',
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        state = last_token_hidden_state
        prompt_tokens, sample_ids, sample_logits = [], [], []
        for i in range(self.max_decoding_length): 
            # logits = self.mlp(state) # [batch_size, vocab_size]
            logits = self._mlp_forward(state)
            if input_mode == 'infer': 
                print('State:', state)
                print('Logits:', logits)
            
            # actions = torch.distributions.categorical.Categorical(logits).sample() # [batch_size]
            actions = logits.argmax(dim=-1) # [batch_size]
            tokens = self.target_vocab.map_ids_to_tokens_py(actions.tolist()).tolist()
            # tokens = self.generator.tokenizer.convert_ids_to_tokens(actions.tolist())
            
            sample_ids.append(actions.unsqueeze(dim=1)) # [batch_size, 1]
            sample_logits.append(logits.unsqueeze(dim=1)) # [batch_size, 1, vocab_size]
            
            tokens = [self.generator.tokenizer.convert_tokens_to_string([t]) \
                      for t in tokens]
            # tokens = [' ' + t for t in tokens]
#             input_ids = (self.generator
#                          .tokenizer(tokens, 
#                                     return_tensors='pt')
#                          ['input_ids']
#                          .to(0))
            token_encoding = (self.generator
                               .tokenizer(tokens, 
                                          padding=True,
                                          return_tensors='pt')
                               .to(0))
            input_ids = token_encoding['input_ids']
            input_lengths = token_encoding['attention_mask'].sum(dim=1)

            next_outputs = (self.generator.model
                            .transformer(input_ids, 
                                         past_key_values=past_key_values, 
                                         use_cache=True))
            # state = next_outputs.last_hidden_state[:, -1, :]
            state = next_outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                   (input_lengths - 1)]
            past_key_values = next_outputs.past_key_values
            
        sample_ids = torch.cat(sample_ids, dim=1) # [batch_size, prompt_length]
        sample_logits = torch.cat(sample_logits, dim=1) # [batch_size, prompt_length, vocab_size]
            
        decoder_output = tx.modules.TransformerDecoderOutput(
            logits=sample_logits,
            sample_id=sample_ids
        )
        return {
                "sample_id": (
                    decoder_output
                    .sample_id
                    .unsqueeze(dim=-1)
                )
            }
    
    def _get_inputs(self, mode: str, target_labels: List[str]): 
        # data_0 = self._tst_inputs[(mode, 'LABEL_0')]
        # data_1 = self._tst_inputs[(mode, 'LABEL_1')]
        
        # idx_0 = self._tst_inputs_idx[(mode, 'LABEL_0')]
        # idx_1 = self._tst_inputs_idx[(mode, 'LABEL_1')]
        
        inputs = []
        for i, label in enumerate(target_labels): 
            idx = self._tst_inputs_idx[(mode, label)]
            data = self._tst_inputs[(mode, label)]
            
            # inputs.append(self.temp_input)
            if mode == 'train': 
                # inputs.append('thank you for a five star service.')
                # inputs.append(self.temp_input)
                inputs.append(self.dataset_inputs[idx])
            else: 
                inputs.append(self.dataset_inputs[idx])
                
            # inputs.append('Prompt:')
            # inputs.append(self.dataset_inputs[idx])
            # inputs.append(data[idx])
            idx += 1
            idx %= len(data)
            self._tst_inputs_idx[(mode, label)] = idx
        
        return inputs
    
    def forward(self,
                batch: BatchType,
                mode: ForwardMode,
                **kwargs) -> Union[Tuple[tx.modules.TransformerDecoderOutput, 
                                         LongTensor], 
                                   Dict]:        
        # print(batch['source_text'])
        target_labels = [t[0] for t in batch['source_text']]
        if mode in [ForwardMode.INFER]: 
            input_mode = 'infer'
        else: 
            input_mode = 'train'
        input_texts = self._get_inputs(input_mode, target_labels)
        
        if input_mode == 'infer': 
            print('Infer Inputs:', input_texts)
        
        # print('Model:', input_texts)
        
        # input_texts = [self.temp_input for _ in range(len(batch['source_text']))]
        
        token_encoding = (self.generator
                          .tokenizer(input_texts, 
                                     padding=True,
                                     return_tensors='pt')
                          .to(0))
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        outputs = (self.generator.model
                   .transformer(input_ids, use_cache=True))
        last_token_hidden_state = outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                            (input_lengths - 1)]
        past_key_values = outputs.past_key_values
        
#         input_ids = (self.generator
#                      .tokenizer(input_texts, 
#                                 return_tensors='pt')['input_ids']
#                      .to(0))
#         outputs = (self.generator.model
#                    .transformer(input_ids, use_cache=True))
#         last_token_hidden_state = outputs.last_hidden_state[:, -1, :]
#         past_key_values = outputs.past_key_values
        
        if mode in [ForwardMode.MLE, ForwardMode.SQL_OFF_GT]:
            return self.decode_teacher_forcing(
                batch=batch,
                last_token_hidden_state=last_token_hidden_state,
                past_key_values=past_key_values,
                **kwargs)

        if mode in [ForwardMode.PG, ForwardMode.SQL_ON]:
            return self.decode_sampling(
                batch=batch,
                last_token_hidden_state=last_token_hidden_state,
                past_key_values=past_key_values,
                **kwargs)

        if mode in [ForwardMode.INFER]:
            return self.decode_greedy(
                batch=batch,
                last_token_hidden_state=last_token_hidden_state,
                past_key_values=past_key_values,
                input_mode=input_mode,
                **kwargs)

        raise ValueError(f"Unknown mode {mode}")


class Transformer(nn.Module):
    r"""A standalone sequence-to-sequence Transformer model, from "Attention
    Is All You Need". The Transformer model consists of the word embedding
    layer, position embedding layer, an encoder and a decoder. Both encoder
    and decoder are stacks of self-attention layers followed by feed-forward
    layers. See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
    for the full description of the model.
    """

    def __init__(
            self,
            train_data: tx.data.PairedTextData,
            max_source_length: int,
            max_decoding_length: int,
            config_name: str,
    ) -> None:
        super().__init__()

        if config_name not in ["transformer_small"]:
            raise ValueError

        if config_name == "transformer_small":
            config_model = config_model_transformers_small

        self.config_model = config_model
        self.max_source_length = max_source_length
        self.max_decoding_length = max_decoding_length

        self.source_vocab = train_data.source_vocab
        self.target_vocab = train_data.target_vocab
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id

        self.source_embedder = tx.modules.WordEmbedder(
            vocab_size=self.source_vocab_size,
            hparams=self.config_model.emb)

        self.target_embedder = tx.modules.WordEmbedder(
            vocab_size=self.target_vocab_size,
            hparams=self.config_model.emb)

        self.source_pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.max_source_length,
            hparams=self.config_model.position_embedder_hparams)

        self.target_pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.max_decoding_length,
            hparams=self.config_model.position_embedder_hparams)

        self.encoder = tx.modules.TransformerEncoder(
            hparams=self.config_model.encoder)
        self.decoder = tx.modules.TransformerDecoder(
            token_pos_embedder=partial(
                self._embedding_fn,
                source_or_target="target"),
            vocab_size=self.target_vocab_size,
            output_layer=self.target_embedder.embedding,
            hparams=self.config_model.decoder)

    def _embedding_fn(
            self,
            tokens: LongTensor,
            positions: LongTensor,
            source_or_target: str,
    ) -> FloatTensor:
        if source_or_target not in ["source", "target"]:
            raise ValueError

        if source_or_target == "source":
            word_embed = self.source_embedder(tokens)
            pos_embed = self.source_pos_embedder(positions)
        if source_or_target == "target":
            word_embed = self.target_embedder(tokens)
            pos_embed = self.target_pos_embedder(positions)

        scale = self.config_model.hidden_dim ** 0.5
        return word_embed * scale + pos_embed

    def decode_teacher_forcing(
            self,
            batch: BatchType,
            memory: FloatTensor
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        decoder_outputs = self.decoder(
            memory=memory,
            memory_sequence_length=batch["source_length"],
            inputs=batch["target_text_ids"][:, :-1],
            sequence_length=batch["target_length"] - 1,
            decoding_strategy="train_greedy")

        # label_lengths = (labels != 0).long().sum(dim=1)
        # We don't really need `sequence_lengths` here
        return decoder_outputs, None

    def decode_greedy(
            self,
            batch: BatchType,
            memory: FloatTensor,
            corruption_p: Optional[float] = None,
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:

        start_tokens = memory.new_full(
            batch["target_length"].size(),
            self.bos_token_id,
            dtype=torch.int64)

        helper = None
        if corruption_p is not None:
            raise NotImplementedError("Deprecated")

        return self.decoder(
            start_tokens=start_tokens,
            end_token=self.eos_token_id,
            helper=helper,
            memory=memory,
            memory_sequence_length=batch["source_length"],
            decoding_strategy="infer_greedy",
            # Probably will hurt the longest sequence,
            # but probably better learning
            max_decoding_length=min(
                self.max_decoding_length,
                batch["target_length"].max().item() - 1))

    def decode_sampling(
            self,
            batch: BatchType,
            memory: FloatTensor,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        if top_k is not None and top_p is not None:
            raise ValueError

        start_tokens = memory.new_full(
            batch["target_length"].size(),
            self.bos_token_id,
            dtype=torch.int64)

        helper = None
        if top_k is not None:
            helper = tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                top_k=top_k)

        if top_p is not None:
            helper = tx.modules.TopPSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                p=top_p)

        decoder_output = self.decoder(
            start_tokens=start_tokens,
            end_token=self.eos_token_id,
            helper=helper,
            memory=memory,
            memory_sequence_length=batch["source_length"],
            decoding_strategy="infer_sample",
            # Probably will hurt the longest sequence,
            # but probably better learning
            max_decoding_length=min(
                self.max_decoding_length,
                batch["target_length"].max().item() - 1))
        
        sample_logits = decoder_output[0].logits
        print(sample_logits.min().item(), sample_logits.max().item())
        return decoder_output

    def decode_beam_search(
            self,
            batch: BatchType,
            memory: FloatTensor,
            beam_width: int,
            corruption_p: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:

        # Only greedy decoding is support for this as of now.
        if corruption_p is not None:
            if beam_width != 1:
                raise NotImplementedError

        # when `beam_width in [None, 1]`, `self.decoder`
        # will switch to default decoding mode, which is
        # not necessarily what we want. Instead, let's
        # explicitly call greedy-decoding.
        # https://sourcegraph.com/github.com/asyml/texar-pytorch/-/blob/texar/torch/modules/decoders/rnn_decoders.py#L717:9
        if beam_width > 1:

            start_tokens = memory.new_full(
                batch["target_length"].size(),
                self.bos_token_id,
                dtype=torch.int64)

            return self.decoder(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                memory=memory,
                memory_sequence_length=batch["source_length"],
                beam_width=beam_width,
                max_decoding_length=self.max_decoding_length)

        else:
            infer_outputs, _ = self.decode_greedy(
                batch=batch,
                memory=memory,
                corruption_p=corruption_p)

            return {
                "sample_id": (
                    infer_outputs
                    .sample_id
                    .unsqueeze(dim=-1)
                )
            }

    def forward(
            self,
            batch: BatchType,
            mode: ForwardMode,
            **kwargs,
    ) -> Union[Tuple[tx.modules.TransformerDecoderOutput, LongTensor], Dict]:

        # Text sequence length excluding padding
        if not (batch["source_length"] == (batch["source_text_ids"] != 0).int().sum(dim=1)).all():
            raise ValueError

        positions: LongTensor = (
            torch.arange(
                batch["source_length"].max(),  # type: ignore
                dtype=torch.long,
                device=batch["source_text_ids"].device)
            .unsqueeze(0)
            .expand(batch["source_text_ids"].size(0), -1)
        )

        encoder_output = self.encoder(
            inputs=self._embedding_fn(
                tokens=batch["source_text_ids"],
                positions=positions,
                source_or_target="source"),
            sequence_length=batch["source_length"])

        if mode in [ForwardMode.MLE, ForwardMode.SQL_OFF_GT]:
            return self.decode_teacher_forcing(
                batch=batch,
                memory=encoder_output)

        if mode in [ForwardMode.PG, ForwardMode.SQL_ON]:
            return self.decode_sampling(
                batch=batch,
                memory=encoder_output,
                **kwargs)

        if mode in [ForwardMode.INFER]:
            return self.decode_beam_search(
                batch=batch,
                memory=encoder_output,
                **kwargs)

        raise ValueError(f"Unknown mode {mode}")
