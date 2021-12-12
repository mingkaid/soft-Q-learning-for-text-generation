import fire
from transformers import pipeline, AutoTokenizer
import sacrebleu as scb
import numpy as np
import pandas as pd
import time
import torch
from tqdm import tqdm

def load_models(device): 
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2', pad_token='<|endoftext|>')
    generator = pipeline(
                "text-generation",
                tokenizer=tokenizer,
                model="distilgpt2",
                device=device)
    TST_CLF_CONFIG = dict(model="./yelp_sentiment_classifier/results-bert-base/checkpoint-10410",
                          tokenizer='bert-base-uncased')
    classifier = pipeline(
                "sentiment-analysis",
                model=TST_CLF_CONFIG['model'],
                tokenizer=TST_CLF_CONFIG['tokenizer'],
                device=device)
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>')
    perplexer = pipeline(
                "text-generation",
                tokenizer=tokenizer,
                model="gpt2",
                device=device)
    return generator, classifier, perplexer


TEMPLATE = 'Sentence 1: "{sentence_1}" {prompt} Sentence 2: "'
def add_input_prompt_to_template(input_sentence, 
                                 prompt_str, 
                                 generator,
                                 template=TEMPLATE): 
    prompt = generator.tokenizer.convert_tokens_to_string(prompt_str.split())
    formatted_prompt = template.format(sentence_1=input_sentence, prompt=prompt)
    return formatted_prompt


def get_input_length(input_sentence, generator): 
    return len(generator.tokenizer(input_sentence)['input_ids'])


def postprocess_output(text): 
    try: 
        end = text.index('"')
    except ValueError: 
        end = len(text)
    text = text[:end]
    
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
        
    return text[:end+1].strip()

NUM_RETURN_SEQUENCES=128
def generate_and_select_output(input_sentence,
                               prompt_str,
                               max_new_tokens,
                               target_label,
                               reward_fn,
                               generator,
                               classifier,
                               perplexer,
                               temperature=1.0,
                               num_return_sequences=NUM_RETURN_SEQUENCES): 
    formatted_prompt = add_input_prompt_to_template(input_sentence,
                                                    prompt_str, 
                                                    generator)
    generator_outputs = generator([formatted_prompt],
                                  max_new_tokens=max_new_tokens,
                                  pad_token_id=50256,
                                  temperature=temperature,
                                  num_return_sequences=num_return_sequences,
                                  # Only return generated text, without the prompt
                                  return_full_text=False)
    generated_texts = []
    for output in generator_outputs: 
        text = output["generated_text"]
        generated_texts.append(postprocess_output(text))

    classes = classifier(generated_texts, truncation=True)
    target = target_label
    correct = [(c['label'] == target) for c in classes]
    probs = [(c['label'] == target) * c['score'] + (c['label'] != target) * (1 - c['score']) for c in classes]

    reference_texts = [input_sentence for _ in generated_texts]
    bleus = [scb.sentence_bleu(hypothesis=x.lower(),
                               references=[y.lower()]) 
             for x, y in zip(generated_texts,
                             reference_texts)]
    bleus = [b.score for b in bleus]
    
#     nll_rewards = compute_nll_reward(generated_texts, perplexer)
#     fluency = np.exp(nll_rewards)
    fluency = [0 for _ in generated_texts]

    sum_rewards = [reward_fn(b, c, p, f) for b, c, p, f in zip(bleus, correct, probs, fluency)]

    idx = np.array(sum_rewards).argmax()
    return {'input_sentence': input_sentence,
            'output_sentence': generated_texts[idx],
            'max_reward': sum_rewards[idx],
            'max_bleu': bleus[idx],
            'max_correct': correct[idx],
            'max_prob': probs[idx],
            'max_perplexity': fluency[idx]}


@torch.no_grad()
def compute_perplexities(
        sentences,
        model,
        tokenizer):

    nlls = []
    for sentence in sentences:
        encodings = tokenizer(
            sentence,
            return_tensors="pt")
        input_ids = encodings.input_ids.to(model.device)
        try:
            # labels **are shifted** inside the model
            outputs = model(
                input_ids,
                labels=input_ids.clone())
            nll = outputs[0]
        except RuntimeError:
            # Could happen when the input is empty
            nll = torch.tensor(float("nan")).to(model.device)

        nlls.append(nll)

    stacked_nlls = torch.stack(nlls, dim=0)
    return stacked_nlls, stacked_nlls.exp()


def compute_nll_reward(sentences, perplexer):
    nlls, _ = compute_perplexities(
        sentences=sentences,
        model=perplexer.model,
        tokenizer=perplexer.tokenizer)
    # When the sentence has just one token,
    # the NLL/perplexity will be `NaN`.
    # Further, we use the negative NLL as the reward
    return (torch.nan_to_num(nlls, nan=10.0).detach()).tolist()


def load_yelp_test_data(sentiment): 
    data = [line.strip() for line in open('/home/yihan.wang/project/mk_sql'
                                          f'/data/yelp-gpt2-control-only/raw/sentiment.test.{sentiment}', 'r')]
    return data


def load_yelp_dev_data(sentiment): 
    data = [line.strip() for line in open('/home/yihan.wang/project/mk_sql'
                                          f'/data/yelp-gpt2-control-only/raw/sentiment.dev.{sentiment}', 'r')]
    return data
    

LOAD_DATA_FNS = {'yelp_positive': lambda: load_yelp_test_data(1),
                 'yelp_negative': lambda: load_yelp_test_data(0),
                 'yelp_positive_dev': lambda: load_yelp_dev_data(1),
                 'yelp_negative_dev': lambda: load_yelp_dev_data(0)}

reward_fn0 = lambda b, c, p, f: (1.0 * b + 1.05 * 100 * (p + p)/2) / (1.0 + 1.05)
reward_fn1 = lambda b, c, p, f: (1.0 * b + 1.05 * 100 * (p + c)/2) / (1.0 + 1.05)
reward_fn2 = lambda b, c, p, f: (1.0 * b + 1.05 * 100 * (c + c)/2) / (1.0 + 1.05)
REWARD_FNS = {'intensity': reward_fn0,
                    'intensity_correct': reward_fn1,
                    'correct': reward_fn2}

PROMPTS = {'yelp_pos_to_neg': 'Ġsinister Ġsinister Ġsinister Ġsinister Ġsinister',
           'yelp_neg_to_pos': 'Ġattribute Ġattribute Ġattribute Ġattribute Ġattribute'}

def main(reward_name, 
         target_reward,
         target_label,
         dataset,
         max_iters,
         prompt_name,
         device=1,
         random_seed=None): 
    if random_seed is not None: torch.manual_seed(random_seed)
    assert target_label in ['LABEL_0', 'LABEL_1']
    
    generator, classifier, perplexer = load_models(device=device)
    
    reward_fn = REWARD_FNS[reward_name]
    data = LOAD_DATA_FNS[dataset]()
    prompt_str = PROMPTS[prompt_name]

    results = []
    for input_sentence in tqdm(data): 
        start = time.time()

        max_new_tokens = get_input_length(input_sentence, generator) * 2
        
        max_reward = 0
        max_output = {}
        i = 0
        while max_reward < target_reward and i < max_iters: 
            output = generate_and_select_output(input_sentence,
                                               prompt_str,
                                               max_new_tokens,
                                               target_label,
                                               reward_fn,
                                               generator,
                                               classifier,
                                               perplexer,)
            if output['max_reward'] > max_reward: 
                max_reward = output['max_reward']
                max_output = output
            i += 1
        max_output.update({'perplexity': np.exp(compute_nll_reward([max_output['output_sentence']], perplexer)[0]),
                           'time': time.time() - start,
                           'iters': i})
        results.append(max_output)

        df_results = pd.DataFrame(results)
        df_results.to_csv('/home/yihan.wang/project/mk_sql/experiments'
                          '/style_transfer_outputs/yelp'
                          f'/{reward_name}-{target_reward}-{target_label}-{dataset}-{max_iters}-{prompt_name}-{random_seed}-outputs.csv',
                          index=False)

    print(df_results.mean())
    

if __name__ == '__main__':
    fire.Fire(main)