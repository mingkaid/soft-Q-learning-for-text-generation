import os
import re
import pdb
import time
import torch
torch.set_grad_enabled(False)
import argparse
import numpy as np
import pandas as pd
import sacrebleu as scb

from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
from ctc_score import StyleTransferScorer
from tqdm import tqdm


currpath = os.path.abspath(os.getcwd())


def preprocess(text):
    text = re.sub('\s{2,}', ' ', text)
    text = re.sub('(.*?)( )([\.,!?\'])', r'\1\3', text)
    text = re.sub('([a-z])( )(n\'t)', r'\1\3', text)
    text = re.sub('\$ \_', r'$_', text)
    text = re.sub('(\( )(.*?)( \))', r'(\2)', text)
    text = re.sub('(``)( )*(.*?)', r"``\3", text)
    text = re.sub('(.*?)( )*(\'\')', r"\1''", text)
    return text

def postprocess_output(text): 
    text = preprocess(text)
    
    try: 
        end = text.index('"')
    except ValueError: 
        end = len(text)
        
    text = text[:end]
    text = text.strip()
    
    try: 
        end = text.index('.')
    except ValueError: 
        end = len(text)
        
    try: 
        end = min(end, text.index('!'))
    except ValueError: 
        end = end
        
    try: 
        end = min(end, text.index('?'))
    except ValueError: 
        end = end
        
    text = text[:end+1].strip()    
    return text

def load_metrics(args):
    
    # Content
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    ctc_model = StyleTransferScorer(align='E-roberta')
    sbert_sim = lambda hyps, refs: [util.pytorch_cos_sim(sbert_model.encode(hyp), sbert_model.encode(ref)).item() * 100 for hyp, ref in zip(hyps, refs)]
    ctc_sim   = lambda hyps, refs: [ctc_model.score(input_sent=hyp.lower(), hypo=ref.lower(), aspect='preservation') * 2 * 100 if len(hyp) > 0 else 0.0 for hyp, ref in zip(hyps, refs)]
    bleu_sim  = lambda hyps, refs: [scb.sentence_bleu(hypothesis=hyp.lower(), references=[ref.lower()]).score for hyp, ref in zip(hyps, refs)]

    
    # Style    
    our_classifier = pipeline(
                    "sentiment-analysis",
                    model=os.path.join(currpath, "style"),
                    tokenizer='bert-base-uncased',
                    device=args.use_gpu)

    open_classifier = pipeline(
                    "sentiment-analysis",
                    model="siebert/sentiment-roberta-large-english",
                    device=args.use_gpu)
    
    ourC_prob  = lambda hyps, target: [(c['label'] == target) * c['score'] + (c['label'] != target) * (1 - c['score']) for c in our_classifier(hyps, truncation=True)]
    openC_prob = lambda hyps, target: [(c['label'] == target) * c['score'] + (c['label'] != target) * (1 - c['score']) for c in open_classifier(hyps, truncation=True)]
    
    
    # Fluency
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(currpath, "ppl"))
    perplexer = GPT2LMHeadModel.from_pretrained(os.path.join(currpath, "ppl"))
    
    perplexity = lambda hyp: perplexer(
                                input_ids=tokenizer(hyp, return_tensors="pt")['input_ids'],
                                labels=tokenizer(hyp, return_tensors="pt")['input_ids']
                                )[0].item()
    
    
    return {
        'C': {
            'ctc'  : ctc_sim,
            'sbert': sbert_sim,
            'bleu' : bleu_sim
        },
        'S': {
            'our' : {
                'pos': lambda hyps: ourC_prob(hyps, 'LABEL_1'),
                'neg': lambda hyps: ourC_prob(hyps, 'LABEL_0'),
            },
            'open': {
                'pos': lambda hyps: openC_prob(hyps, 'POSITIVE'),
                'neg': lambda hyps: openC_prob(hyps, 'NEGATIVE'),
            },
        },
        'F': perplexity,
    }


def load_model(args): 
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, pad_token='<|endoftext|>')
    generator = pipeline(
                "text-generation", 
                tokenizer=tokenizer, 
                model=args.pretrained_model, 
                device=args.use_gpu)    
    return generator


def load_dataset(args):
    data_pos = [line.strip() for line in open(os.path.join(currpath, "dataset", f'sentiment.{args.use_dataset}.1.preprocess'), 'r')]
    data_neg = [line.strip() for line in open(os.path.join(currpath, "dataset", f'sentiment.{args.use_dataset}.0.preprocess'), 'r')]
    return { 'pos': data_pos[:args.use_dataset_size], 'neg': data_neg[:args.use_dataset_size]}


def load_prompt(args):
    PROMPT = {
        'manual': {
            'p2n' : 'Change the following sentence from positive sentiment to negative sentiment but keep its semantics.\n',
            'n2p' : 'Change the following sentence from negative sentiment to positive sentiment but keep its semantics.\n',
        },
        'learned':{
            'p2n' : 'Ġreflects Ġworthy Unfortunately || Unfortunately',
            'n2p' : 'Ġquestioning ĠWoody Ġcriticisms Ġworries Ġworries',
        },
        'null':{
            'p2n' : '',
            'n2p' : '',
        }
    }

    TEMPLATE = {
        'manual':{
            'p2n' : '{prompt} Sentence Positive: "{sentence_1}" \n Sentence Negative: "',
            'n2p' : '{prompt} Sentence Negative: "{sentence_1}" \n Sentence Positive: "',
        },
        'learned':{
            'p2n' : 'Sentence 1: "{sentence_1}" {prompt} Sentence 2: "',
            'n2p' : 'Sentence 1: "{sentence_1}" {prompt} Sentence 2: "',
        },
        'null':{
            'p2n' : 'Sentence 1: "{sentence_1}" {prompt} Sentence 2: "',
            'n2p' : 'Sentence 1: "{sentence_1}" {prompt} Sentence 2: "',
        }
    }
    
    
    return {
        'p2n': lambda text: TEMPLATE[args.use_prompt]['p2n'].format(sentence_1=text, prompt=PROMPT[args.use_prompt]['p2n']),
        'n2p': lambda text: TEMPLATE[args.use_prompt]['n2p'].format(sentence_1=text, prompt=PROMPT[args.use_prompt]['n2p']), 
    }


def load_reward(args):
    return lambda cs, ps: [(1.0 * c + 1.05 * 100 * p) / (1.0 + 1.05) for c, p in zip(cs, ps)]


def generate(args, sentence, prompt, generator, metrics_fn, reward_fn, style='pos'):
    prompt_sentence = prompt(sentence)
    
    start = time.time()
    max_new_tokens = len(generator.tokenizer(sentence)['input_ids']) * args.max_new_ratio
    
    transfer_sentences = generator([prompt_sentence],
                                    max_new_tokens=max_new_tokens,
                                    pad_token_id=generator.tokenizer.pad_token_id,
                                    num_return_sequences=args.sample_size,
                                    max_length=256,
                                    return_full_text=False)
    use_time = time.time() - start
    
    transfer_sentences = [postprocess_output(s["generated_text"]) for s in transfer_sentences]
    reference_sentences = [sentence for _ in transfer_sentences]
    
    results = {}
    
    # Content
    for name, fn in metrics_fn['C'].items(): results[name] = fn(transfer_sentences, reference_sentences)
        
    # Style
    for name, fn in metrics_fn['S'].items(): results[name] = fn[style](transfer_sentences)
        
    results['reward'] = reward_fn(results[args.use_content_reward], results[args.use_style_reward])
    index = np.array(results['reward']).argmax()
    
    for n, k in results.items():
        results[n] = k[index]
        
    results['input_sentence'] = sentence
    results['output_sentence'] = transfer_sentences[index]
    results['nll'] = metrics_fn['F'](transfer_sentences[index])
    results['time'] = use_time
        
    return results
        
def main(args):
    
    if args.random_seed:
        torch.manual_seed(args.random_seed)
    
    dataset = load_dataset(args)
    print('Load Dataset Complete')
    
    generator = load_model(args)
    print('Load Model Complete')
    
    prompt = load_prompt(args)
    print('Load Prompt Complete')
    
    metrics_fn = load_metrics(args)
    print('Load Metrics Complete')
    
    reward_fn = load_reward(args)
    print('Load Reward Complete')
    
    
    output_path = os.path.join(currpath, 'outputs')
    os.makedirs(output_path, exist_ok=True)
    
    # P2N
    p2n_results = []
    for sentence in tqdm(dataset['pos']): 
        output = generate(args, sentence, prompt['p2n'], generator, metrics_fn, reward_fn, style='neg')
        p2n_results.append(output)
        p2n_results_df = pd.DataFrame(p2n_results)
        p2n_filename = f'{args.sample_size}-p2n-{args.use_prompt}-{args.pretrained_model}-{args.use_style_reward}-{args.random_seed}.csv'
        p2n_results_df.to_csv(os.path.join(output_path, p2n_filename), index=False)
    
    # N2P
    n2p_results = []
    for sentence in tqdm(dataset['neg']): 
        output = generate(args, sentence, prompt['n2p'], generator, metrics_fn, reward_fn, style='pos')
        n2p_results.append(output)
        n2p_results_df = pd.DataFrame(n2p_results)
        n2p_filename = f'{args.sample_size}-n2p-{args.use_prompt}-{args.pretrained_model}-{args.use_style_reward}-{args.random_seed}.csv'
        n2p_results_df.to_csv(os.path.join(output_path, n2p_filename), index=False)
    
    
    results_filename = f'{args.sample_size}-{args.use_prompt}-{args.pretrained_model}-{args.use_style_reward}-{args.random_seed}.txt'
    with open(os.path.join(output_path, results_filename),'w') as f:
        print('P2N', file=f)
        p2n_results_df_avg = p2n_results_df.mean()
        p2n_results_df_avg['fluency'] = np.exp(p2n_results_df_avg['nll'])
        print(p2n_results_df_avg, file=f)
        print('N2P', file=f)
        n2p_results_df_avg = n2p_results_df.mean()
        n2p_results_df_avg['fluency'] = np.exp(n2p_results_df_avg['nll'])
        print(n2p_results_df_avg, file=f)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pretrained_model",   default='distilgpt2', type=str, required=False)
    parser.add_argument("--sample_size",        default=16,           type=int, required=False)
    parser.add_argument("--random_seed",        default=42,           type=int, required=False)
    parser.add_argument("--max_new_ratio",      default=4,            type=int, required=False)
    parser.add_argument("--max_length",         default=256,          type=int, required=False)
    
    parser.add_argument("--use_dataset",        default='dev',        type=str, required=False)    
    parser.add_argument("--use_dataset_size",   default=100,          type=int, required=False)    
    parser.add_argument("--use_content_reward", default='ctc',        type=str, required=False)  
    parser.add_argument("--use_style_reward",   default='open',       type=str, required=False)  
    parser.add_argument("--use_prompt",         default='manual',     type=str, required=False)
    parser.add_argument("--use_gpu",            default=0,            type=int, required=False)
    
    args = parser.parse_args()
    main(args)
    




