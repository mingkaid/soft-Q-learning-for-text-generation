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

from data import load_dataset
from models import StyleTransformer
from utils import tensor2text

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

def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 # +1 for <eos> token
    return lengths

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
    
    three_classifier = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment",
                    device=args.use_gpu)
    
    ourC_prob  = lambda hyps, target: [(c['label'] == target) * c['score'] + (c['label'] != target) * (1 - c['score']) for c in our_classifier(hyps, truncation=True)]
    openC_prob = lambda hyps, target: [(c['label'] == target) * c['score'] + (c['label'] != target) * (1 - c['score']) for c in open_classifier(hyps, truncation=True)]
    threeC_prob = lambda hyps, target: [(c['label'] == target) * c['score'] + (c['label'] != target) * (1 - c['score']) for c in three_classifier(hyps, truncation=True)]
    
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
            'three':{
                'pos': lambda hyps: threeC_prob(hyps, 'LABEL_2'),
                'neg': lambda hyps: threeC_prob(hyps, 'LABEL_0'),
            }
        },
        'F': perplexity,
    }

def load_model(args, vocab): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = StyleTransformer(args, vocab).to(device)
    generator.load_state_dict(torch.load(args.checkpoint))
    generator.eval()
    return generator


def load_reward(args):
    return lambda cs, ps: [(1.0 * c + 1.05 * 100 * p) / (1.0 + 1.05) for c, p in zip(cs, ps)]


def generate(args, sentence, vocab, generator, metrics_fn, reward_fn, style='pos'):
    
    eos_idx = vocab.stoi['<eos>']
    inp_lengths = get_lengths(sentence, eos_idx)
    target_styles = torch.full_like(sentence[:, 0], 1 if style=='pos' else 0)
    start = time.time()
    
    with torch.no_grad():
        _, transfer_sentences = generator(
            sentence, 
            None,
            inp_lengths,
            target_styles,
            generate=True,
            sample=False,
            generate_size=args.sample_size,
            differentiable_decode=False,
            temperature=1.0,
        )
    use_time = time.time() - start
    
    reference_sentences = tensor2text(vocab, sentence.cpu()) * args.sample_size
    transfer_sentences = tensor2text(vocab, transfer_sentences.cpu())

    reference_sentences = [preprocess(t) for t in reference_sentences] 
    transfer_sentences  = [preprocess(t) for t in transfer_sentences] 
    
    results = {}
    
    # Content
    for name, fn in metrics_fn['C'].items(): results[name] = fn(transfer_sentences, reference_sentences)
        
    # Style
    for name, fn in metrics_fn['S'].items(): results[name] = fn[style](transfer_sentences)
        
    results['reward'] = reward_fn(results[args.use_content_reward], results[args.use_style_reward])
    index = np.array(results['reward']).argmax()
    
    for n, k in results.items():
        results[n] = k[index]
        
    results['input_sentence'] = reference_sentences[0]
    results['output_sentence'] = transfer_sentences[index]
    results['nll'] = metrics_fn['F'](transfer_sentences[index])
    results['time'] = use_time
        
    return results


def main(args):
    
    if args.random_seed:
        torch.manual_seed(args.random_seed)
    
    _, dev_iters, test_iters, vocab = load_dataset(args)
    print('Load Dataset Complete')
    
    generator = load_model(args, vocab)
    print('Load Model Complete')
    
    metrics_fn = load_metrics(args)
    print('Load Metrics Complete')
    
    reward_fn = load_reward(args)
    print('Load Reward Complete')
    
    output_path = os.path.join(currpath, 'outputs')
    os.makedirs(output_path, exist_ok=True)
    
    
    # P2N
    p2n_results = []
    for sentence in tqdm(dev_iters.pos_iter): 
        output = generate(args, sentence.text, vocab, generator, metrics_fn, reward_fn, style='neg')
        p2n_results.append(output)
        p2n_results_df = pd.DataFrame(p2n_results)
        p2n_filename = f'{args.sample_size}-p2n-styleTransformer-{args.use_style_reward}-{args.random_seed}.csv'
        p2n_results_df.to_csv(os.path.join(output_path, p2n_filename), index=False)
    
    # N2P
    n2p_results = []
    for sentence in tqdm(dev_iters.neg_iter): 
        output = generate(args, sentence.text, vocab, generator, metrics_fn, reward_fn, style='pos')
        n2p_results.append(output)
        n2p_results_df = pd.DataFrame(n2p_results)
        n2p_filename = f'{args.sample_size}-n2p-styleTransformer-{args.use_style_reward}-{args.random_seed}.csv'
        n2p_results_df.to_csv(os.path.join(output_path, n2p_filename), index=False)
    
    
    results_filename = f'{args.sample_size}-styleTransformer-{args.use_style_reward}-{args.random_seed}.txt'
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
    
    # evaluate
    parser.add_argument("--checkpoint",         default='G.pth',    type=str, required=False)
    parser.add_argument("--batch_size",         default=1 ,           type=int, required=False)
    parser.add_argument("--sample_size",        default=1,            type=int, required=False)
    parser.add_argument("--random_seed",        default=42,           type=int, required=False)
    
    parser.add_argument("--use_dataset",        default='dev',        type=str, required=False)    
    parser.add_argument("--use_dataset_size",   default=100,          type=int, required=False)    
    parser.add_argument("--use_content_reward", default='ctc',        type=str, required=False)  
    parser.add_argument("--use_style_reward",   default='open',       type=str, required=False)  
    parser.add_argument("--use_gpu",            default=0,            type=int, required=False)
    
    # data
    parser.add_argument("--data_path",  default="./data/yelp/")
    parser.add_argument("--min_freq",   default=3, type=int)
    parser.add_argument("--max_length", default=16, type=int)
    
    # model
    parser.add_argument("--load_pretrained_embed", help="whether to load pretrained embeddings.", action="store_true")
    parser.add_argument("--use_gumbel", help="handle discrete part in another way", action="store_true")
    parser.add_argument("-discriminator_method", help="the type of discriminator ('Multi' or 'Cond')", default="Multi")
    parser.add_argument("-embed_size", help="the dimension of the token embedding", default=256, type=int)
    parser.add_argument("-d_model", help="the dimension of Transformer d_model parameter", default=256, type=int)
    parser.add_argument("-head", help="the number of Transformer attention heads", dest="h", default=4, type=int)
    parser.add_argument("-num_styles", help="the number of styles for discriminator", default=2, type=int)
    parser.add_argument("-num_layers", help="the number of Transformer layers", default=4, type=int)
    parser.add_argument("-dropout", help="the dropout factor for the whole model", default=0.1, type=float)
    parser.add_argument("-learned_pos_embed", help="whether to learn positional embedding", default="1", choices=['1', '0', 'True', 'False'])
    parser.add_argument("-inp_drop_prob", help="the initial word dropout rate", default=0.1, type=float)
    parser.add_argument("-temp", help="the initial softmax temperature", default=1.0, type=float)
    args = parser.parse_args()
    main(args)