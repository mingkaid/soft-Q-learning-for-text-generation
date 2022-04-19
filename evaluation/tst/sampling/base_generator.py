from transformers import AutoTokenizer, pipeline, GPT2LMHeadModel
import numpy as np
import sacrebleu as scb
from collections import defaultdict 
import torch
import scipy.stats as stats
from tqdm import tqdm
import pandas as pd
from bert_score import BERTScorer
from torch.utils.data import Dataset
import re

class BaseGenerator(): 
    def __init__(self, device, reward_device=None): 
        self.device = device
        self.reward_device = reward_device if reward_device is not None else self.device
        self.classifier = pipeline("sentiment-analysis",
                                   model="/jupyter/prompt-generation/soft-Q-learning-for-text-generation/"
                                         "experiments/yelp_sentiment_classifier/results-bert-base/checkpoint-10410",
                                   tokenizer='bert-base-uncased',
                                   device=self.reward_device)
        
        self.bert_scorer = BERTScorer('roberta-large', 
                                      device=self.reward_device, 
                                      rescale_with_baseline=True, 
                                      lang='en')
        
        self._load_data()
        
    def _load_data(self): 
        raise NotImplementedError
        
    def _model_generate(self, task_name, sample_size, top_k=None, top_p=None, **kwargs): 
        raise NotImplementedError
    
    def sample_generate(self, task_name, sample_size, top_k=None, top_p=None, **kwargs): 
        assert task_name in ['pos2neg', 'neg2pos']
        
        outputs = []
        for sample_id, gen, ref, target_label in self._model_generate(task_name, 
                                                              sample_size, 
                                                              top_k, 
                                                              top_p, 
                                                              **kwargs): 
            output = self._select_output(gen, ref, target_label, sample_id=sample_id)
            outputs.append(output)
        return outputs
    
    def _recon_score(self, **kwargs): 
        return kwargs['bertscore']
        # return (3 * kwargs['bertscore'] + 1 * kwargs['bleu']) / 4
        
    def _select_output(self,
                       generated_texts, 
                       reference_texts, 
                       target_label, 
                       sample_id=None): 
        reference_texts = [r for i, r in enumerate(reference_texts) if len(generated_texts[i]) > 0]
        generated_texts = [g for i, g in enumerate(generated_texts) if len(generated_texts[i]) > 0]
        if len(generated_texts) == 0: 
            output = {'top_reward': 0}
            if sample_id is not None: output['sample_id'] = sample_id
            return output
        
        output = {}
        bleus = [
            scb.sentence_bleu(
                hypothesis=x,
                references=[y])
            for x, y in zip(
                generated_texts,
                reference_texts)
        ]
        eps = 1e-3
        bleu_rewards = [b.score + eps for b in bleus]

        bertscore_f1 = self.bert_scorer.score(generated_texts, reference_texts)[2]
        bertscore_rewards = [max(b, 0) for b in (bertscore_f1 * 100).tolist()]

        recon_rewards = [self._recon_score(bleu=bleu, bertscore=bertscore) \
                         for bleu, bertscore in zip(bleu_rewards, bertscore_rewards)]

        classes = self.classifier(generated_texts, truncation=True)
        label = target_label
        correct = [(c['label'] == label) for c in classes]
        probs = [(c['label'] == label) * c['score'] + (c['label'] != label) * (1 - c['score']) for c in classes]

        recon_weight = 1
        style_weight = 1
        sum_rewards = [(recon_weight * r + style_weight * 100 * p) / (recon_weight + style_weight) \
                       for r, c, p in zip(recon_rewards, correct, probs)]
        max_sum_reward = torch.tensor(sum_rewards).float().max()

        top_index = sum_rewards.index(max_sum_reward)
        
        output['top_reward'] = float(max_sum_reward)
        output['source_sentence'] = reference_texts[top_index]
        output['top_sentence'] = generated_texts[top_index]
        output['top_recon'] = float(recon_rewards[top_index])
        output['top_selfbleu'] = float(bleu_rewards[top_index])
        output['top_style'] = float(probs[top_index])
        # output['top_perplexity'] = self.perplexity(generated_texts[top_index])
        if sample_id is not None: output['sample_id'] = sample_id
            
        return output