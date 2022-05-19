import os
import sys
sys.path.insert(0,'..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import yaml
import argparse
from argparse import Namespace

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM
)
from model_util import get_optimizer_and_scheduler
from tasks.dataset import FewShotDataset
from tasks.processors import compute_metrics_mapping
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"



parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='data path')
parser.add_argument('--template', type=str, help='Template string')
parser.add_argument('--template_instruction', type=str, help='Template string with instruction')
parser.add_argument('--template_incontext', type=str, help='Template string with in context')
parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--truncate_head', type=bool, default=True)
parser.add_argument('--skip_space', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--incontext', action='store_true')
parser.add_argument('--instruction', action='store_true')
args = parser.parse_args()
print(args)

seed = args.seed 
task_name = args.task

if args.instruction:
    template = args.template_instruction
elif args.incontext:
    template = args.template_incontext
else:
    template = args.template
    
mapping = args.label_map
truncate_head = args.truncate_head


torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device(f'cuda:{args.gpu_id}')
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
generator = AutoModelForMaskedLM.from_pretrained('roberta-large').to(device)


for param in generator.parameters():
    param.requires_grad = False

with open("../configs/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = config['translation']['classification']
config = Namespace(**config)
config.truncate_head = truncate_head
config.task_name = task_name.lower()
config.template = template
config.mapping = mapping
config.data_dir = f"/home/chengping/soft-Q-learning-for-text-generation/tasks/k-shot/{task_name}/16-{seed}"

if args.skip_space:
    config.skip_space = args.skip_space
    
if args.incontext:
    config.use_demo = True
    config.double_demo = False
    config.num_sample = 1

print(config)

testset = FewShotDataset(config, tokenizer=tokenizer, mode="test")
testloader = DataLoader(testset, collate_fn=testset.collate_fn, num_workers=4, pin_memory=True, batch_size=32, shuffle=False)
metrics_fn = compute_metrics_mapping[config.task_name]

os.makedirs(os.path.join('ckpt-inst', task_name), exist_ok=True)
pred_labels, true_labels = [], []
for batch in tqdm(testloader):
    labels = torch.empty_like(batch['input_ids']).fill_(-100).long().to(device)
    labels[range(labels.shape[0]), batch['mask_pos'].squeeze()] = batch['labels_tokens'].to(device)

    with torch.no_grad():
        output = generator(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=labels
        )

    loss, logits = output.loss, output.logits
    logits = logits[range(logits.shape[0]), batch['mask_pos'].squeeze()]
    pred_labels += logits[:, testset.get_labels_tok()].argmax(1).tolist()
    true_labels += batch['labels'].squeeze().tolist()
    
dev_metrics = metrics_fn(config.task_name, np.array(pred_labels), np.array(true_labels))
dev_metrics = dev_metrics['f1'] if 'f1' in dev_metrics else dev_metrics['acc']
print(f'FewShot Performance: {dev_metrics}')

with open(os.path.join('ckpt-inst', task_name, f'result-{seed}.txt'), 'a') as f:
    f.write(f'FewShot Metrics: {dev_metrics}\n')
    