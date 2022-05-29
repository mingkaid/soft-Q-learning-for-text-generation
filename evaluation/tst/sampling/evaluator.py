from transformers import AutoTokenizer, pipeline, GPT2LMHeadModel
import pandas as pd
from torch.utils.data import Dataset
import sacrebleu as scb
import numpy as np
from tqdm import tqdm
tqdm.pandas()

class SampleDataset(Dataset):
    def __init__(self, x):
        self.samples = x
        
    def __getitem__(self,index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)

class Evaluator(): 
    def __init__(self, device): 
        self.device = device
        self.classifier = pipeline("sentiment-analysis",
                                   model="/jupyter/prompt-generation/soft-Q-learning-for-text-generation/"
                                         "experiments/yelp_sentiment_classifier/results-bert-base-train-test/checkpoint-8920",
                                   tokenizer='bert-base-uncased',
                                   device=device)

        perplexer_path = "/workspace/soft-Q-learning-for-text-generation/evaluate_styletransformer/ppl"
        self.perplexer_tokenizer = AutoTokenizer.from_pretrained(perplexer_path)
        self.perplexer_model = GPT2LMHeadModel.from_pretrained(perplexer_path).to(device)
        self._load_data()
        
    def _sent_len(self, hyp): 
        return len(self.perplexer_tokenizer(hyp)['input_ids'])
        
    def _sent_nll(self, hyp): 
        input_ids = self.perplexer_tokenizer(hyp, return_tensors='pt')['input_ids'].to(self.device)
        nll = self.perplexer_model(input_ids=input_ids, labels=input_ids)[0].item()
        return nll
    
    def _load_data(self): 
        # Test now on the real test set
        sentence_dict = {}
        data_path = '/jupyter/prompt-generation/soft-Q-learning-for-text-generation/' \
                    'data/yelp-gpt2-control-only/raw-prep/'
        
        filename = 'sentiment.reference.1.preprocess'
        sentence_dict['ref_pos2neg'] = [line.strip() for line in open(data_path + filename)]
        
        filename = 'sentiment.reference.0.preprocess'
        sentence_dict['ref_neg2pos'] = [line.strip() for line in open(data_path + filename)]
        
        self.sentence_dict = sentence_dict
        
    def evaluate_output(self, 
                        task_name, 
                        outputs, 
                        output_col='top_sentence',
                        recon_col='top_recon',
                        evaluate_perplexity=True,
                        evaluate_reference=True,
                        evaluate_classifier=True): 
        assert task_name in ['pos2neg', 'neg2pos']
        
        output_df = pd.DataFrame(outputs)
        summary = {'sum_reward': round(output_df['top_reward'].mean(), 2),
                   'recon': round(output_df['top_recon'].mean(), 2),
                   'self_bleu': round(output_df['top_selfbleu'].mean(), 2)}
        
        if evaluate_reference:
            print('Comparing with reference...')
            ref_sentences = self.sentence_dict[f'ref_{task_name}']
            ref_df = pd.DataFrame({'ref_sentence': ref_sentences,
                                   'sample_id': list(range(len(ref_sentences)))})
            output_ref_df = output_df.merge(ref_df, on='sample_id')
            output_ref_df['ref_bleu'] = (output_ref_df
                                         .progress_apply(
                                             lambda row: (scb.sentence_bleu(hypothesis=row[output_col],
                                                                            references=[row.ref_sentence])  
                                                          .score),
                                             axis=1))
            output_df = output_ref_df
            summary['ref_bleu'] = round(output_df['ref_bleu'].mean(), 2)
        
        if evaluate_classifier: 
            print('Running test classifier')
            target_label_dict = {'pos2neg': 'LABEL_0', 'neg2pos': 'LABEL_1'}
            label = target_label_dict[task_name]

            data = SampleDataset(output_ref_df[output_col].tolist())
            classes = self.classifier(data, truncation=True)
            # correct = [int(c['label'] == label) for c in tqdm(classes)]
            probs = [(c['label'] == label) * c['score'] + (c['label'] != label) * (1 - c['score']) for c in tqdm(classes)]
            output_df['style_prob'] = probs
            
            correct = [int(prob >= 0.5) for prob in probs]
            summary['style_acc'] = round(100 * sum(correct) / len(correct), 2)
        
        if evaluate_perplexity:
            print('Computing perplexity...')
            output_df['sent_len'] = output_df[output_col].progress_apply(self._sent_len)
            output_df['sent_nll'] = output_df['sent_len'] * output_df[output_col].progress_apply(self._sent_nll)
            ppl = np.exp(output_df['sent_nll'].sum() / output_df['sent_len'].sum())
            summary['ppl'] = round(ppl, 2)
        
        
        
        return summary, output_df
        
        