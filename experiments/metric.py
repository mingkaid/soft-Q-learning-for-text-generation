from generate_style_transfer import *
from sentence_transformers import SentenceTransformer
import random
from typing import *

def sbert_sim(model, src, tgts):
    if type(tgts) is not list:
        tgts = [tgts]
    to_encode = [src] + tgts
    embs = model.encode(to_encode)
    cos_sim = lambda a,b : np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return [cos_sim(embs[0], emb) for emb in embs[1:]]


def evaluate_generated_texts(
                            generated_text: List[str], 
                            clf_model: Any, 
                            recon_model: Any, 
                            reference: str, 
                            target: str):
    classes = clf_model(generated_text, temperature=1.0, truncation = True)
    correct = [(c['label'] == target) for c in classes]
    probs = [(c['label'] == target) * c['score'] + 
                (c['label'] != target) * (1 - c['score']) for c in classes]
    recon_scr = sbert_sim(recon_model, reference.lower(), 
                            [g.lower() for g in generated_text])
    
    # Sacred BLEU Scores
    # reference_texts = [reference] * len(generated_text)
    # bleus = [scb.sentence_bleu(hypothesis=x.lower(),
    #                            references=[y.lower()]) 
    #          for x, y in zip(generated_text,
    #                          reference_texts)]
    # bleus = [b.score for b in bleus]

    reward_list = [(rs + sa) / 2 for rs, sa in zip(recon_scr, probs)]
    logs = [{
            "ref_text": reference,
            "gen_text": gt,
            "target_label": target,
            "score": (rs + sa) / 2, 
            "recon": rs, 
            "clf_acc": sa} for rs, sa, gt in zip(recon_scr, probs, generated_text)]
    return reward_list, logs


def generate_output_with_metric(input_sentence,
                               prompt_str,
                               max_new_tokens,
                               target_label,
                               generator,
                               classifier,
                               recon_model,
                               num_return_sequences=NUM_RETURN_SEQUENCES): 
    formatted_prompt = add_input_prompt_to_template(input_sentence,
                                                    prompt_str, 
                                                    generator)

    generator_outputs = generator([formatted_prompt],
                                  max_new_tokens=max_new_tokens,
                                  pad_token_id=50256,
                                  num_return_sequences=num_return_sequences,
                                  return_full_text=False)

    generated_texts = []
    for output in generator_outputs: 
        text = output["generated_text"]
        generated_texts.append(postprocess_output(text))

    scores, logs = evaluate_generated_texts(generated_texts, 
                        classifier, recon_model, input_sentence, target_label)

    idx = scores.index(max(scores))
    return logs[idx]


if __name__ == "__main__":
    avg = lambda l: sum(l)/len(l)

    prompt_str = "Ġreflects Ġworthy Unfortunately || Unfortunately"
    dataset="yelp_positive"
    random.seed(2)

    generator, classifier, perplexer = load_models(device=0)

    data = LOAD_DATA_FNS[dataset]()
    small = random.sample(data, 100)

    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentiment_clf = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
    
    
    results = []
    target_label = "NEGATIVE"
    for input_sentence in tqdm(small): 
        start = time.time()

        max_new_tokens = get_input_length(input_sentence, generator) * 2
        max_reward = 0
        max_output = {}
        output = generate_output_with_metric(input_sentence,
                                            prompt_str,
                                            max_new_tokens,
                                            target_label,
                                            generator,
                                            sentiment_clf,
                                            sbert_model,
                                            num_return_sequences=16) 
        max_reward = output['score']
        max_output = output

        max_output.update({'perplexity': np.exp(compute_nll_reward([max_output['gen_text']], perplexer)[0]),
                            'time': time.time() - start})
        results.append(max_output)
    print(avg([r["score"] for r in results]))