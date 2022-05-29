from transformers import AutoTokenizer, pipeline, GPT2LMHeadModel
from bert_score import BERTScorer
import pandas as pd
from torch.utils.data import Dataset
import sacrebleu as scb
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import json
import fire

class SampleDataset(Dataset):
    def __init__(self, x):
        self.samples = x
        
    def __getitem__(self,index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)

class YelpEvaluator(): 
    def __init__(self, device_id): 
        self.device = device_id
        self.classifier = pipeline("sentiment-analysis",
                                   model="/jupyter/prompt-generation/soft-Q-learning-for-text-generation/"
                                         "experiments/yelp_sentiment_classifier/results-bert-base-train-test/checkpoint-8920",
                                   tokenizer='bert-base-uncased',
                                   device=self.device)

        perplexer_path = "/workspace/soft-Q-learning-for-text-generation/evaluate_styletransformer/ppl"
        self.perplexer_tokenizer = AutoTokenizer.from_pretrained(perplexer_path)
        self.perplexer_model = GPT2LMHeadModel.from_pretrained(perplexer_path).to(self.device)
        
        self.bert_scorer = BERTScorer('roberta-large', 
                                      device=self.device, 
                                      rescale_with_baseline=True, 
                                      lang='en')
        
        self.cola = pipeline('text-classification',
                            model='cointegrated/roberta-large-cola-krishna2020',
                            device=self.device)
        
    def _sent_len(self, hyp): 
        return len(self.perplexer_tokenizer(hyp)['input_ids'])
        
    def _sent_nll(self, hyp): 
        input_ids = self.perplexer_tokenizer(hyp, return_tensors='pt')['input_ids'].to(self.device)
        nll = self.perplexer_model(input_ids=input_ids, labels=input_ids)[0].item()
        return nll
        
    def evaluate_output(self, 
                        hypos_path, 
                        pos2neg_refs_path,
                        neg2pos_refs_path): 
        all_hypos = [json.loads(line.strip()) for line in open(hypos_path)]
        pos2neg_refs = [line.strip() for line in open(pos2neg_refs_path)]
        neg2pos_refs = [line.strip() for line in open(neg2pos_refs_path)]
        
        output_df = pd.DataFrame(all_hypos)
        summary = {'train_reward': round(output_df['train_reward'].mean(), 1),
                   'self_bertscore': round(output_df['self_bertscore'].mean(), 1),
                   'self_bleu': round(output_df['self_bleu'].mean(), 1)}
        
        print('Comparing with reference...')
        output_df.loc[(output_df['target_label'] == 'LABEL_0'), 'ref'] = pos2neg_refs
        output_df.loc[(output_df['target_label'] == 'LABEL_1'), 'ref'] = neg2pos_refs
        output_df['ref_bleu'] = (output_df
                                 .progress_apply(
                                     lambda row: (scb.sentence_bleu(hypothesis=row['selected_hypo'],
                                                                    references=[row['ref']])  
                                                  .score),
                                     axis=1))
        bertscore_f1s = self.bert_scorer.score(output_df['selected_hypo'].tolist(),
                                               output_df['ref'].tolist())[2]
        output_df['ref_bertscore'] = [max(b, 0) for b in (bertscore_f1s * 100).tolist()]
        
        summary['ref_bleu'] = round(output_df['ref_bleu'].mean(), 1)
        summary['ref_bertscore'] = round(output_df['ref_bertscore'].mean(), 1)
        
        print('Running test classifier')
        corrects = []
        probs = []
        for i, c in enumerate(self.classifier(SampleDataset(output_df['selected_hypo'].tolist()), 
                                             batch_size=32, 
                                             truncation=True)): 
            label = output_df.loc[i, 'target_label']
            prob = (c['label'] == label) * c['score'] + (c['label'] != label) * (1 - c['score']) 
            
            probs.append(round(prob * 100, 1))
            corrects.append(int(c['label'] == label))
        output_df['test_style'] = probs
        summary['style_acc'] = round(100 * sum(corrects) / len(corrects), 1)
        
        print('Computing perplexity...')
        output_df['sent_len'] = output_df['selected_hypo'].progress_apply(self._sent_len)
        output_df['sent_nll'] = output_df['sent_len'] * output_df['selected_hypo'].progress_apply(self._sent_nll)
        ppl = np.exp(output_df['sent_nll'].sum() / output_df['sent_len'].sum())
        summary['ppl'] = round(ppl, 1)
        
        print('Computing CoLA acceptability')
        corrects = []
        probs = []
        for i, c in enumerate(self.cola(SampleDataset(output_df['selected_hypo'].tolist()), 
                                             batch_size=32, 
                                             truncation=True)): 
            label = 'LABEL_0'
            prob = (c['label'] == label) * c['score'] + (c['label'] != label) * (1 - c['score']) 
            
            probs.append(round(prob * 100, 1))
            corrects.append(int(c['label'] == label))
        output_df['cola_prob'] = probs
        summary['cola_acc'] = round(100 * sum(corrects) / len(corrects), 1)
        
        print('Computing joint scores')
        joint_scores = (output_df['self_bertscore'] * 
                           (output_df['test_style'] > 50) * 
                           (output_df['cola_prob'] > 50))
        summary['joint_score'] = round(joint_scores.mean(), 1)
        
        print('Computing geometric avg')
        gm = np.exp((np.log(summary['self_bertscore']) + 
                     np.log(summary['style_acc']) + 
                     np.log(summary['cola_acc'])) / 3)
        summary['gm_score'] = round(gm, 1)
        
        return summary, output_df
    
    def evaluate_and_save(self, 
                          hypos_path, 
                          pos2neg_refs_path,
                          neg2pos_refs_path,
                          summary_save_path,
                          results_save_path): 
        summary, result_df = self.evaluate_output(hypos_path, 
                                                  pos2neg_refs_path,
                                                  neg2pos_refs_path)
        json.dump(summary, open(summary_save_path, 'w'))
        result_df.to_csv(results_save_path, index=False)
        
if __name__ == '__main__':
    fire.Fire(YelpEvaluator)
        
        