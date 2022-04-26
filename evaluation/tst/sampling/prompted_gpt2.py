from transformers import AutoTokenizer, pipeline
from base_generator import BaseGenerator
from torch.utils.data import Dataset
from tqdm import tqdm

def postprocess_output(text, end_punct='"', start_punct=None):         
    try: 
        end = text.index(end_punct)
    except ValueError: 
        end = len(text)
    text = text[:end].strip()
    # return text    
    if start_punct is not None: 
        start = text.find(start_punct)
        while start >= 0: 
            text = text[start+1:].strip()
            start = text.find(start_punct)

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

    return text[:end+1].strip().lower()

class SampleDataset(Dataset):
    def __init__(self, x):
        self.samples = x
        
    def __getitem__(self,index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)

class PromptedGPT2Generator(BaseGenerator): 
    def __init__(self, 
                 model_name, 
                 task_prompts,
                 device=None,
                 reward_device=None,
                 generator_device=None,
                 tst_template=None,
                 end_punct=None,
                 start_punct=None,
                 recon_score='bertscore'): 
        assert (((reward_device is not None and generator_device is not None) and device is None) or 
                ((reward_device is None and generator_device is None) and device is not None))        
        super().__init__(device, reward_device=reward_device, recon_score=recon_score)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<|endoftext|>')
        self.generator = pipeline("text-generation",
                                  model=model_name,
                                  tokenizer=tokenizer,
                                  device=generator_device if generator_device is not None else device)
        
        assert 'pos2neg' in task_prompts and 'neg2pos' in task_prompts
        self.start_punct = start_punct
        self.end_punct = end_punct if end_punct is not None else '"'
        self.tst_template = tst_template if tst_template is not None else '{prompt} "{sentence_1}" "'
        
        self.task_prompts = task_prompts
        
    def _load_data(self): 
        # Test now on the real test set
        sentence_dict = {}
        data_path = '/jupyter/prompt-generation/soft-Q-learning-for-text-generation/' \
                    'data/yelp-gpt2-control-only/raw-prep/'
        
        filename = 'sentiment.test_ref.1.preprocess'
        sentence_dict['src_pos2neg'] = [line.strip() for line in open(data_path + filename)]
        
        filename = 'sentiment.test_ref.0.preprocess'
        sentence_dict['src_neg2pos'] = [line.strip() for line in open(data_path + filename)]
        
        self.sentence_dict = sentence_dict
        
    def _model_generate(self, task_name, sample_size, top_k=None, top_p=None, **kwargs): 
        # Assume sample_generate() function already checked that task_name is valid
        source_sentences = self.sentence_dict[f'src_{task_name}']
        
        single_prompt = kwargs.get('single_prompt', False) 
        if single_prompt: prompts = [single_prompt for _ in source_sentences]
        else: prompts = [self.task_prompts[task_name] for _ in source_sentences]

        target_label_dict = {'pos2neg': 'LABEL_0', 'neg2pos': 'LABEL_1'}
        target_label = target_label_dict[task_name]
        
        formatted_prompts = [self.tst_template.format(prompt=p, sentence_1=s) for p, s in zip(prompts, source_sentences)]
        data = SampleDataset(formatted_prompts)
        
        generator_outputs = self.generator(data,
                                            pad_token_id=50256,
                                            top_k=top_k,
                                            top_p=top_p,
                                            num_return_sequences=sample_size,
                                            temperature=1,
                                            # Only return generated text, without the prompt
                                            return_full_text=False)
        
#         generator_outputs = self.generator(data,
#                                            pad_token_id=50256,
#                                            num_beams=32, 
#                                            num_return_sequences=32, 
#                                            do_sample=False,
#                                            # Only return generated text, without the prompt
#                                            return_full_text=False)
        
        output_sentences = []
        for i, output_samples in tqdm(enumerate(generator_outputs), total=len(data)):

            generated_texts = []
            for output in output_samples: 
                text = output["generated_text"]
                generated_texts.append(postprocess_output(text, 
                                                          end_punct=self.end_punct,
                                                          start_punct=self.start_punct))
                
            reference_texts = [source_sentences[i] for _ in generated_texts]
            
            yield i, generated_texts, reference_texts, target_label
        