import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM
)
from typing import List, Tuple, Union, Dict, Optional, Callable, Any, cast
from sql.types import FloatTensor
from tqdm import tqdm
from tasks.dataset import FewShotDataset
from tasks.processors import compute_metrics_mapping
import pdb
import warnings
warnings.filterwarnings("ignore")



class RoBERTaGLUEReward(object):

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda')
        self._tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self._generator = AutoModelForMaskedLM.from_pretrained('roberta-large').to(self.device)
        self._max_length = self._tokenizer.model_max_length
        
        self.dataset = {
            'train'  : FewShotDataset(self.config, tokenizer=self._tokenizer, mode="train"),
            'infer'  : FewShotDataset(self.config, tokenizer=self._tokenizer, mode="dev"),
            'test'   : FewShotDataset(self.config, tokenizer=self._tokenizer, mode="test"),
        }
        
        self.labels_tok = self.dataset['train'].get_labels_tok()
        self.metrics_fn = compute_metrics_mapping[config.task_name]
        self.best_prompts = []
        self.best_prompts_metrics = []
        self.top_k = 3
        
        self.m_rewards = None
        self.m_metrics = None
        self.manual_metrics = None

    def load_manual_features(self, mode):
        dataset = self.dataset[mode]
        dataset.use_learned_prompt = False
        dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True, batch_size=len(dataset), shuffle=False)
        batch = next(iter(dataloader))
        return batch
        
    def load_learned_features(self, prompt_string, mode):
        dataset = self.dataset[mode]
        dataset.use_learned_prompt = True
        index_prompt = dataset.set_learned_prompt(prompt_string)
        dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True, batch_size=len(dataset), shuffle=False)
        batch = next(iter(dataloader))
        return batch, index_prompt
    
    
    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._tokenizer.convert_tokens_to_string(s.split()) for s in tokens]

    
    def _get_rewards_metrics(self, batch, index_prompt=None):
                
        with torch.no_grad():
            logits = self._generator(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
            ).logits.cpu()
        
        
        logits = logits[range(logits.shape[0]), batch['mask_pos'].squeeze()]
        
        probs = torch.softmax(logits, -1)
        label_probs = probs[:, self.labels_tok]
        label_probs = label_probs / label_probs.sum(-1).unsqueeze(1)
        true_probs  = label_probs[range(logits.shape[0]), batch['labels'].squeeze()].squeeze()
        true_probs_per_prompt = torch.tensor([[true_probs[index].item() for index in i] for i in index_prompt]).float()
        false_probs_per_prompt = 1 - true_probs_per_prompt
        rewards = (true_probs_per_prompt - false_probs_per_prompt) * 100
        rewards = rewards.mean(1).squeeze()
        metrics = self.metrics_fn(self.config.task_name, label_probs.argmax(1).numpy(), batch['labels'].squeeze().numpy())
        
        return rewards, metrics
    
    def evaluate_manual(self):
        dataset = self.dataset['test']
        dataset.use_learned_prompt = False
        dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True, batch_size=16, shuffle=False)
        pred_labels, true_labels = [], []
        for batch in dataloader:
            
            with torch.no_grad():
                logits = self._generator(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                ).logits.cpu()
                
            logits = logits[range(logits.shape[0]), batch['mask_pos'].squeeze()]
            pred_labels += logits[:, self.labels_tok].argmax(1).tolist()
            true_labels += batch['labels'].squeeze().tolist()
            
        metrics = self.metrics_fn(self.config.task_name, np.array(pred_labels), np.array(true_labels))
        return metrics
        
    def evaluate_learned(self, prompt_string):
        dataset = self.dataset['test']
        dataset.use_learned_prompt = True
        dataset.set_learned_prompt(prompt_string)
        dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True, batch_size=16, shuffle=False)
        pred_labels, true_labels = [], []
        for batch in dataloader:
            with torch.no_grad():
                logits = self._generator(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                ).logits.cpu()
                
            logits = logits[range(logits.shape[0]), batch['mask_pos'].squeeze()]
            pred_labels += logits[:, self.labels_tok].argmax(1).tolist()
            true_labels += batch['labels'].squeeze().tolist()

        metrics = self.metrics_fn(self.config.task_name, np.array(pred_labels), np.array(true_labels))
        return metrics
        
        
    def forward(self, target_labels: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        
        if mode not in ["train", "infer"]:
            raise ValueError
        
        
        rewards_log = {}
        prompt_strings = self._convert_tokens_to_string(prompts)
        
        # Learned
        learned_prompts, index_prompt = self.load_learned_features(prompt_strings, mode)
        l_rewards, l_metrics = self._get_rewards_metrics(learned_prompts, index_prompt)        
        rewards_tensor = l_rewards
        rewards_log["reward"] = torch.mean(l_rewards)
        for k, v in l_metrics.items():
            rewards_log[k] = torch.tensor(v)
            
        # Manual
        if self.m_rewards == None and self.m_metrics == None:
            manual_prompts  = self.load_manual_features(mode)
            self.m_rewards, self.m_metrics = self._get_rewards_metrics(manual_prompts, index_prompt)
            
        rewards_log["reward_manual"] = torch.mean(self.m_rewards)   
        for k, v in self.m_metrics.items():
            rewards_log[k+'_manual'] = torch.tensor(v)
            
        # example = self._tokenizer.decode(learned_prompts["input_ids"][0][:torch.sum(learned_prompts["attention_mask"][0]).item()])
        # print(f'Train: {example} - {rewards_log["reward"].item()}', end='\r')
        
        if mode == 'infer':
            metric_key = max(l_metrics.keys(), key=lambda x: len(x))
            
            if len(self.best_prompts_metrics) < self.top_k:
                self.best_prompts.append(prompt_strings[0])
                self.best_prompts_metrics.append(rewards_log[metric_key].item())
            elif rewards_log[metric_key].item() > min(self.best_prompts_metrics):
                idx = np.argmin(self.best_prompts_metrics)
                self.best_prompts.pop(idx)
                self.best_prompts_metrics.pop(idx)
                self.best_prompts.append(prompt_strings[0])
                self.best_prompts_metrics.append(rewards_log[metric_key].item())
            
            if self.manual_metrics == None:
                self.manual_metrics = self.evaluate_manual()
                
            learned_metrics = self.evaluate_learned([self.best_prompts[np.argmax(self.best_prompts_metrics)]])
            rewards_log[metric_key+'_manual_test'] = self.manual_metrics[metric_key]
            rewards_log[metric_key+'_test'] = learned_metrics[metric_key]
            
            print('=' * 20)
            print('Prompts:', self.best_prompts)
            print(f'Dev  {metric_key}:', self.best_prompts_metrics)
            print('Best prompt:', self.best_prompts[np.argmax(self.best_prompts_metrics)])
            print(f'Learned Test {metric_key}:', learned_metrics[metric_key])
            print(f'Manual  Test {metric_key}:',  self.manual_metrics[metric_key])
            
        
        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log
        

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            target_labels=sources,
            prompts=predictions,
            to_tensor=to_tensor,
            mode=mode)
    
    
class RoBERTaGLUEReward_piece(object):

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda')
        self._tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        self._generator = AutoModelForMaskedLM.from_pretrained('roberta-large').to(self.device)
        self._max_length = self._tokenizer.model_max_length
        
        self.dataset = {
            'train'  : FewShotDataset(self.config, tokenizer=self._tokenizer, mode="train"),
            'infer'  : FewShotDataset(self.config, tokenizer=self._tokenizer, mode="dev"),
            'test'   : FewShotDataset(self.config, tokenizer=self._tokenizer, mode="test"),
        }
        
        self.labels_tok = self.dataset['train'].get_labels_tok()
        self.metrics_fn = compute_metrics_mapping[config.task_name]
        self.best_prompts = []
        self.best_prompts_metrics = []
        self.top_k = 3
        
        self.m_rewards = None
        self.m_metrics = None
        self.manual_metrics = None

    def load_manual_features(self, mode):
        dataset = self.dataset[mode]
        dataset.use_learned_prompt = False
        dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True, batch_size=len(dataset), shuffle=False)
        batch = next(iter(dataloader))
        return batch
        
    def load_learned_features(self, prompt_string, mode):
        dataset = self.dataset[mode]
        dataset.use_learned_prompt = True
        index_prompt = dataset.set_learned_prompt(prompt_string)
        dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True, batch_size=len(dataset), shuffle=False)
        batch = next(iter(dataloader))
        return batch, index_prompt
    
    
    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._tokenizer.convert_tokens_to_string(s.split()) for s in tokens]

    
    def _get_rewards_metrics(self, batch, index_prompt=None):
                
        with torch.no_grad():
            logits = self._generator(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
            ).logits.cpu()
        
        
        logits = logits[range(logits.shape[0]), batch['mask_pos'].squeeze()]
        label_logits = logits[:, self.labels_tok]
        label_probs = torch.softmax(label_logits, -1)
        true_probs  = label_probs[range(logits.shape[0]), batch['labels'].squeeze()].squeeze()
        rewards = []
        true_probs_per_prompt = torch.tensor([[true_probs[index].item() for index in i] for i in index_prompt]).float()

        regions_1 = torch.all(true_probs_per_prompt >= (1.0 / len(self.labels_tok)), dim=1).tolist()
        regions_3 = torch.all(true_probs_per_prompt < (1.0 / len(self.labels_tok)), dim=1).tolist()
        for i, (r1, r3) in enumerate(zip(regions_1, regions_3)):
            if r1:
                # [-0.7, inf] + 2
                rewards.append(2 + torch.log(torch.mean(true_probs_per_prompt[i] / (1 - true_probs_per_prompt[i]))))
            elif r3:
                # 2c Prod [-5, -1.38] * k
                # 2c Mean [-5, -0.69] * k
                # 3c Prod [-5, -3.32] * 3
                # 3c Mean [-5, -1.1] * 3
                # [-15, -3.3]
                rewards.append(9 * torch.clip(torch.log(torch.mean(true_probs_per_prompt[i])), min=-5))
            else:
                # Stuck in here
                # [-2.7, -1]
                # [-8.1, -3]
                rel_scores = torch.softmax(true_probs_per_prompt[i], dim=-1)
                rewards.append(3 * -torch.exp(max(rel_scores) - min(rel_scores)))
            
        rewards = torch.FloatTensor(rewards)
        metrics = self.metrics_fn(self.config.task_name, label_probs.argmax(1).numpy(), batch['labels'].squeeze().numpy())
        
        return rewards, metrics
    
    def evaluate_manual(self):
        dataset = self.dataset['test']
        dataset.use_learned_prompt = False
        dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True, batch_size=16, shuffle=False)
        pred_labels, true_labels = [], []
        for batch in dataloader:
            with torch.no_grad():
                logits = self._generator(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                ).logits.cpu()
                
            logits = logits[range(logits.shape[0]), batch['mask_pos'].squeeze()]
            pred_labels += logits[:, self.labels_tok].argmax(1).tolist()
            true_labels += batch['labels'].squeeze().tolist()
            
        metrics = self.metrics_fn(self.config.task_name, np.array(pred_labels), np.array(true_labels))
        return metrics
        
    def evaluate_learned(self, prompt_string):
        dataset = self.dataset['test']
        dataset.use_learned_prompt = True
        dataset.set_learned_prompt(prompt_string)
        dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True, batch_size=16, shuffle=False)
        pred_labels, true_labels = [], []
        for batch in dataloader:
            with torch.no_grad():
                logits = self._generator(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                ).logits.cpu()
                
            logits = logits[range(logits.shape[0]), batch['mask_pos'].squeeze()]
            pred_labels += logits[:, self.labels_tok].argmax(1).tolist()
            true_labels += batch['labels'].squeeze().tolist()

        metrics = self.metrics_fn(self.config.task_name, np.array(pred_labels), np.array(true_labels))
        return metrics
        
        
    def forward(self, target_labels: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        
        if mode not in ["train", "infer"]:
            raise ValueError
        
        
        rewards_log = {}
        prompt_strings = self._convert_tokens_to_string(prompts)
        
        # Learned
        learned_prompts, index_prompt = self.load_learned_features(prompt_strings, mode)
        l_rewards, l_metrics = self._get_rewards_metrics(learned_prompts, index_prompt)        
        rewards_tensor = l_rewards
        rewards_log["reward"] = torch.mean(l_rewards)
        for k, v in l_metrics.items():
            rewards_log[k] = torch.tensor(v)
            
        # Manual
        if self.m_rewards == None and self.m_metrics == None:
            manual_prompts  = self.load_manual_features(mode)
            self.m_rewards, self.m_metrics = self._get_rewards_metrics(manual_prompts, index_prompt)
            
        rewards_log["reward_manual"] = torch.mean(self.m_rewards)   
        for k, v in self.m_metrics.items():
            rewards_log[k+'_manual'] = torch.tensor(v)
            
        # example = self._tokenizer.decode(learned_prompts["input_ids"][0][:torch.sum(learned_prompts["attention_mask"][0]).item()])
        # print(f'Train: {example} - {rewards_log["reward"].item()}', end='\r')
        
        if mode == 'infer':
            metric_key = max(l_metrics.keys(), key=lambda x: len(x))
            
            if len(self.best_prompts_metrics) < self.top_k:
                self.best_prompts.append(prompt_strings[0])
                self.best_prompts_metrics.append(rewards_log[metric_key].item())
            elif rewards_log[metric_key].item() > min(self.best_prompts_metrics):
                idx = np.argmin(self.best_prompts_metrics)
                self.best_prompts.pop(idx)
                self.best_prompts_metrics.pop(idx)
                self.best_prompts.append(prompt_strings[0])
                self.best_prompts_metrics.append(rewards_log[metric_key].item())
            
            if self.manual_metrics == None:
                self.manual_metrics = self.evaluate_manual()
                
            learned_metrics = self.evaluate_learned([self.best_prompts[np.argmax(self.best_prompts_metrics)]])
            rewards_log[metric_key+'_manual_test'] = self.manual_metrics[metric_key]
            rewards_log[metric_key+'_test'] = learned_metrics[metric_key]
            
            print('=' * 20)
            print('Prompts:', self.best_prompts)
            print(f'Dev  {metric_key}:', self.best_prompts_metrics)
            print('Best prompt:', self.best_prompts[np.argmax(self.best_prompts_metrics)])
            print(f'Learned Test {metric_key}:', learned_metrics[metric_key])
            print(f'Manual  Test {metric_key}:',  self.manual_metrics[metric_key])
            
        
        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log
        

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            target_labels=sources,
            prompts=predictions,
            to_tensor=to_tensor,
            mode=mode)
    
reward_name_to_glue_map = {
    'roberta-glue': RoBERTaGLUEReward,
    'roberta-glue-piece': RoBERTaGLUEReward_piece,
}
