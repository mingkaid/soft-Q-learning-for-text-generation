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
from typing import Union, Tuple, Dict, Any, Optional

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
    # O = nn.Linear(64, out_dim)
    
    return nn.Sequential(W1, A1, W2)

def _build_self_vocab_mlp(out_dim, in_dim=768, device=0): 
    W1 = nn.Linear(in_dim, 64)
    A1 = nn.ReLU()
    O = nn.Linear(64, out_dim)
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
    
    # sentence_1 = 'thank you for a five star service .'
    sentence_1 = 'classification'
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
        
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>')
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model="gpt2",
                                  device=0)
        for param in self.generator.model.parameters():
            param.requires_grad = False
        
        mode = 'gpt2_vocab'
        if mode == 'gpt2_vocab': 
            self.mlp = _build_gpt2_vocab_mlp(self.target_vocab_size).to(0)
            self._mlp_forward = self._gpt2_vocab_mlp_forward
            self.valid_token_ids = json.load(open('/home/c2hsieh/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_negative_prep'))
            self.valid_token_ids = None
        elif mode == 'self_vocab': 
            self.mlp = _build_self_vocab_mlp(self.target_vocab_size).to(0)
            self._mlp_forward = self.mlp

        self.temp_input = self.sentence_1
#         self.temp_input = self.sentence_2
#         self.temp_input = self.temp_input_2
#         self.temp_input = self.temp_input_0
        # self.temp_input = self.temp_input_6
        

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
                m.bias.data.fill_(0.0001)
        self.mlp.apply(init_weights)
    
    
    def _gpt2_vocab_mlp_forward(self, state): 
        mlp_output = self.mlp(state)
        logits = self.generator.model.lm_head(mlp_output)
        if self.valid_token_ids is not None: 
            logits = logits[:, self.valid_token_ids]
        # print(logits.shape)
        zeros = torch.ones_like(logits)[:, :4] * float('-inf')
        # print(zeros)
        modified_logits = torch.cat([zeros, logits], dim=-1)
        # print(modified_logits.shape)
        return modified_logits
        
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
            
        state = last_token_hidden_state
        prompt_tokens, sample_ids, sample_logits = [], [], []
        for i in range(self.max_decoding_length): 
            # logits = self.mlp(state)
            logits = self._mlp_forward(state)
#             print(state.min().item(), state.max().item())
#             print(logits.min().item(), logits.max().item())
            normalized_logits = torch.softmax(logits, dim=1)
            print(i, torch.topk(normalized_logits, 10).indices[0].cpu(), torch.topk(normalized_logits, 10).values[0].cpu())
    
            if top_k is not None: sampling_logits = _top_k_logits(logits, k=top_k)
            elif top_p is not None: sampling_logits = _top_p_logits(logits, p=top_p)
            else: sampling_logits = logits
            
            actions = (torch.distributions.categorical
                       .Categorical(logits=sampling_logits)
                       .sample())
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
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        state = last_token_hidden_state
        prompt_tokens, sample_ids, sample_logits = [], [], []
        for i in range(self.max_decoding_length): 
            # logits = self.mlp(state) # [batch_size, vocab_size]
            logits = self._mlp_forward(state)
            
            # actions = torch.distributions.categorical.Categorical(logits).sample() # [batch_size]
            actions = logits.argmax(dim=-1) # [batch_size]
            tokens = self.target_vocab.map_ids_to_tokens_py(actions.tolist()).tolist()
            # tokens = self.generator.tokenizer.convert_ids_to_tokens(actions.tolist())
            
            sample_ids.append(actions.unsqueeze(dim=1)) # [batch_size, 1]
            sample_logits.append(logits.unsqueeze(dim=1)) # [batch_size, 1, vocab_size]
            
            tokens = [self.generator.tokenizer.convert_tokens_to_string([t]) \
                      for t in tokens]
            # tokens = [' ' + t for t in tokens]
            input_ids = (self.generator
                         .tokenizer(tokens, 
                                    return_tensors='pt')
                         ['input_ids']
                         .to(0))
            next_outputs = (self.generator.model
                            .transformer(input_ids, 
                                         past_key_values=past_key_values, 
                                         use_cache=True))
            state = next_outputs.last_hidden_state[:, -1, :]
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
    
    def forward(self,
                batch: BatchType,
                mode: ForwardMode,
                **kwargs) -> Union[Tuple[tx.modules.TransformerDecoderOutput, 
                                         LongTensor], 
                                   Dict]:        
        formatted_input_template = self.input_template.format(sentence_1=self.sentence_1)
        input_ids = (self.generator
                     .tokenizer([self.temp_input \
                                 for _ in range(len(batch['source_text']))], 
                                return_tensors='pt')['input_ids']
                     .to(0))
        outputs = (self.generator.model
                   .transformer(input_ids, use_cache=True))
        last_token_hidden_state = outputs.last_hidden_state[:, -1, :]
        past_key_values = outputs.past_key_values
        
        if mode in [ForwardMode.MLE, ForwardMode.SQL_OFF_GT]:
            return self.decode_teacher_forcing(
                batch=batch,
                last_token_hidden_state=last_token_hidden_state,
                past_key_values=past_key_values)

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
