from math import pi
import os
import click
import torch
import numpy as np
import sacrebleu as scb
from collections import Counter
from collections import defaultdict
from joblib import Parallel, delayed
# from fairseq.data.data_utils import collate_tokens
# from fairseq.models.roberta import RobertaHubInterface
from typing import List, Tuple, Union, Dict, Optional, Callable, Any, cast

from datasets import load_metric
from transformers import (
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    PreTrainedTokenizerBase,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    RobertaForSequenceClassification)
# from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer

from modules import gpt2 as gpt2_modules
from sql.types import FloatTensor
from sql import utils as sql_utils
from sql import misc_utils

from math import cos, pi

try:
    from detoxify import Detoxify
except ModuleNotFoundError:
    Detoxify = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_Xs_Ys_sizes(
        Xs: List,
        Ys: List,
        check_type_is_list: bool = False,
        check_first_element_is_string: bool = True,
) -> None:
    if len(Xs) != len(Ys):
        raise ValueError(
            f"Xs.length = {len(Xs)}, "
            f"Ys.length = {len(Ys)}")

    if check_type_is_list is True:
        if not isinstance(Xs, list) or not isinstance(Ys, list):
            raise ValueError(
                f"Xs.type = {type(Xs)}, "
                f"Ys.type = {type(Ys)}")

    if check_first_element_is_string is True:
        if not isinstance(Xs[0], str) or not isinstance(Ys[0], str):
            raise ValueError(
                f"Xs[0].type = {type(Xs[0])}, "
                f"Ys[0].type = {type(Ys[0])}")


class BLEUReward(object):
    def __init__(self, method: Optional[str] = None) -> None:
        if method is None:
            # `lightning` is marginally better empirically
            method = "lightning"

        self._method = method

    def forward(
            self,
            output_texts: List[str],
            target_texts: List[str],
            to_tensor: bool
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        check_Xs_Ys_sizes(output_texts, target_texts)
        # Using a faster BLEU implementation during training
        # `sacrebleu` is ~3X faster than `lightning`
        # `sacrebleu-parallel` is ~3X faster than `sacrebleu`
        if self._method == "sacrebleu":
            bleus = [
                scb.sentence_bleu(
                    hypothesis=x,
                    references=[y])
                for x, y in zip(
                    output_texts,
                    target_texts)
            ]
            rewards = [b.score for b in bleus]
        elif self._method == "sacrebleu-parallel":
            # two jobs are probably enough for now
            bleus = Parallel(n_jobs=2)(
                delayed(scb.sentence_bleu)(
                    hypothesis=x,
                    references=[y])
                for x, y in zip(
                    output_texts,
                    target_texts)
            )

            rewards = [b.score for b in bleus]
        else:
            rewards = sql_utils.compute_sentence_bleu_batch(
                output_texts=[text.split() for text in output_texts],
                target_texts=[text.split() for text in target_texts],
                method=self._method)

        rewards_log: Dict[str, Any] = {}
        if to_tensor is True:
            return torch.tensor(rewards), rewards_log
        else:
            return rewards, rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            output_texts=predictions,
            target_texts=targets,
            to_tensor=to_tensor)


class ROUGEReward(object):
    def __init__(self, rouge_type: Optional[str] = None) -> None:
        if rouge_type is None:
            rouge_type = "rougeL"

        self._rouge_type = rouge_type
        self._metric = load_metric("rouge")

    def forward(
            self,
            output_texts: List[str],
            target_texts: List[str],
            to_tensor: bool
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        check_Xs_Ys_sizes(output_texts, target_texts)

        results = self._metric.compute(
            predictions=output_texts,
            references=target_texts,
            rouge_types=[self._rouge_type],
            use_agregator=False)

        # The results are list of `Score` tuple
        # and the scale was [0.0, 1.0]
        rewards = [s.fmeasure * 100 for s in results[self._rouge_type]]

        rewards_log: Dict[str, Any] = {}
        if to_tensor is True:
            return torch.tensor(rewards), rewards_log
        else:
            return rewards, rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            output_texts=predictions,
            target_texts=targets,
            to_tensor=to_tensor)


class BleurtReward(object):
    def __init__(self, checkpoint: Optional[str] = None) -> None:
        if checkpoint is None:
            checkpoint = "bleurt-base-128"

        self._metric = load_metric("bleurt", checkpoint)

    def forward(
            self,
            predictions: List[str],
            references: List[str],
            to_tensor: bool
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        check_Xs_Ys_sizes(predictions, references)
        scores_dict = self._metric.compute(
            references=references,
            predictions=predictions)

        # I don't honestly know the range of scores, but
        # looks like they are in [-2, 2], hence this
        # transformation brings it to [0, 100]
        scores = [
            score * 25 + 50 for score in
            scores_dict["scores"]]

        rewards_log: Dict[str, Any] = {}
        if to_tensor is True:
            return torch.tensor(scores), rewards_log
        else:
            return scores, rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            predictions=predictions,
            references=targets,
            to_tensor=to_tensor)


class EntailmentClassifier(object):

    def __init__(
            self,
            task_name: str,
            batch_size: int = 32,
            model_name: Optional[str] = None,
            include_perplexity: bool = True,
    ) -> None:

        if model_name is None:
            model_name = "ynie"

        if model_name not in ["ynie", "fairseq"]:
            raise ValueError

        if include_perplexity is True:
            sql_utils.colorful_warning(
                f"Adding LM-based reward with the "
                f"model trained on {task_name}", bg="blue")

        if model_name == "ynie":
            hf_model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
            tokenizer = (
                AutoTokenizer
                .from_pretrained(hf_model_name))
            model = (
                AutoModelForSequenceClassification
                .from_pretrained(hf_model_name))
        else:
            tokenizer = None
            model = torch.hub.load(
                "pytorch/fairseq",
                "roberta.large.mnli")

        sql_utils.colorful_warning(f"Using {model_name}", bg="blue")

        model.eval()
        model.to(device)
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._model_name = model_name
        self._include_perplexity = include_perplexity
        self._language_model = gpt2_modules.GPT2Model(
            task_name=task_name,
            batch_size=batch_size)

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        # We use the negative NLL as the reward
        return -self._language_model.forward(sentences)

    def _compute_entailment_probs(self, Xs: List[str], Ys: List[str]) -> Tuple[FloatTensor, int]:

        if self._model_name == "ynie":
            entailment_index = 0
            _batch_prediction_fn = (
                lambda _Xs, _Ys: get_NLI_prediction(
                    tokenizer=self._tokenizer,
                    model=self._model,
                    premises=_Xs,
                    hypotheses=_Ys,
                    device=device))
        else:
            entailment_index = 2
            _batch_prediction_fn = (
                lambda _Xs, _Ys: get_NLI_prediction_2(
                    model=self._model,
                    premises=_Xs,
                    hypotheses=_Ys))

        probs = []
        for index in range(0, len(Ys), self._batch_size):
            i_0 = index
            i_1 = index + self._batch_size
            probs.append(
                _batch_prediction_fn(
                    Xs[i_0: i_1],
                    Ys[i_0: i_1]))

        return torch.cat(probs, dim=0), entailment_index

    def _compute_entailment_reward(self, Xs: List[str], Ys: List[str]) -> FloatTensor:
        probs, entailment_index = self._compute_entailment_probs(Xs=Xs, Ys=Ys)
        # We assume rewards are in `[0, 100]`
        return probs[:, entailment_index] * 100

    def forward(self, Xs: List[str], Ys: List[str], to_tensor: bool) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        check_Xs_Ys_sizes(Xs, Ys)
        if isinstance(Xs, np.ndarray):
            Xs = Xs.tolist()
        if isinstance(Ys, np.ndarray):
            Ys = Ys.tolist()

        rewards = self._compute_entailment_reward(Xs=Xs, Ys=Ys)
        rewards_log = {"entailment": rewards.mean()}

        # Adding perplexity if necessary
        if self._include_perplexity is True:
            nll_reward = self._compute_nll_reward(Ys)
            rewards = rewards + nll_reward
            rewards_log["nll"] = nll_reward.mean()

        if to_tensor is True:
            return rewards, rewards_log
        else:
            return rewards.tolist(), rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            Xs=sources,
            Ys=predictions,
            to_tensor=to_tensor)


class EntailmentClassifier2(object):
    """This wraps `EntailmentClassifier` for a special use case"""
    TRAIN_SRC_FNAME = "/export/share/Data/multinli/all/train.sources"
    TRAIN_TGT_FNAME = "/export/share/Data/multinli/all/train.targets"
    VALID_SRC_FNAME = "/export/share/Data/multinli/all/valid.sources"
    VALID_TGT_FNAME = "/export/share/Data/multinli/all/valid.targets"
    RepetitionPenaltyCoef = 5.0

    def __init__(self) -> None:
        # Reuse the `EntailmentClassifier3`, which includes
        # a few useful features for this task as well.
        # We use an easier classifier here
        self._reward_module = EntailmentClassifier3(
            task_name="multinli",
            model_name="fairseq")

        # We will use the original, unprocessed sources, which
        # should be the same with the actual data passed to
        # this function during runtime.
        with open(self.TRAIN_SRC_FNAME) as f:
            train_sources = [d.strip() for d in f.readlines()]
        with open(self.TRAIN_TGT_FNAME) as f:
            train_targets = [d.strip() for d in f.readlines()]
        with open(self.VALID_SRC_FNAME) as f:
            valid_sources = [d.strip() for d in f.readlines()]
        with open(self.VALID_TGT_FNAME) as f:
            valid_targets = [d.strip() for d in f.readlines()]

        self._train_data = {}
        self._valid_data = {}
        # There will be duplicates and overrides. But
        # this represent <0.5% of the cases, so we will
        # ignore it for now.
        for source, target in zip(train_sources, train_targets):
            self._train_data[target] = source
        for source, target in zip(valid_sources, valid_targets):
            self._valid_data[target] = source

    def _repetition_penalty_reward(self, sentences: List[str]) -> FloatTensor:
        penalties = []
        for sentence in sentences:
            penalty = 0
            tokens = sentence.split()
            for _, count in Counter(tokens).items():
                penalty = penalty + count - 1
            penalties.append(penalty)

        # reward = -penalty
        return -torch.tensor(penalties).float().to(device)

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if any([s != "start" for s in sources]):
            raise ValueError

        if mode not in ["train", "infer"]:
            raise ValueError

        if mode == "train":
            data = self._train_data
        if mode == "infer":
            data = self._valid_data
        # We will find corresponding sources w.r.t. targets
        # rather than using original sources. This also made
        # some engineering workload substantially easier.
        corresponding_sources = [data[t] for t in targets]
        rewards, rewards_log = self._reward_module(
            sources=corresponding_sources,
            targets=targets,
            predictions=predictions,
            to_tensor=to_tensor,
            mode=mode)

        repetition_reward = self._repetition_penalty_reward(predictions)
        rewards_log["repetition_reward"] = repetition_reward.mean()
        rewards = rewards + repetition_reward * self.RepetitionPenaltyCoef

        if to_tensor is True:
            return rewards, rewards_log
        else:
            raise NotImplementedError


class EntailmentClassifier3(object):
    """This wraps `EntailmentClassifier` with additional BLEU Rewards"""
    BLEURewardCoef = 1.0

    def __init__(self, task_name: Optional[str] = None, **entailment_kwargs) -> None:
        if task_name is None:
            task_name = "snli"

        # Use `SacreBLEU` here.
        self._bleu_reward_module = BLEUReward(method="sacrebleu")
        self._entailment_reward_module = EntailmentClassifier(
            task_name=task_name,
            include_perplexity=True,
            **entailment_kwargs)

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if to_tensor is False:
            raise NotImplementedError

        # Compute BLEU score w.r.t. sources
        bleu_rewards, bleu_rewards_log = (
            self._bleu_reward_module(
                sources=None,
                targets=sources,
                predictions=predictions,
                to_tensor=to_tensor,
                mode=mode))

        entailment_rewards, entailment_rewards_log = (
            self._entailment_reward_module(
                sources=sources,
                targets=targets,
                predictions=predictions,
                to_tensor=to_tensor,
                mode=mode))

        bleu_rewards = bleu_rewards.to(device)
        rewards = (
            entailment_rewards +
            bleu_rewards * self.BLEURewardCoef)
        rewards_log = misc_utils.unionize_dicts([
            bleu_rewards_log,
            entailment_rewards_log,
            {"bleu": bleu_rewards.mean()}])

        if to_tensor is True:
            return rewards, rewards_log
        else:
            raise NotImplementedError


class GPT2TopicReward(object):
    WORDLISTS_BASE_DIR = "/workspace/soft-Q-learning-for-text-generation/experiments/wordlists"
    PPLM_INPUTS_FILE_NAME = "/workspace/soft-Q-learning-for-text-generation/experiments/pplm-inputs.txt"
    TOPICS = ["legal", "politics", "computers", "space", "religion", "science", "military"]

    def __init__(
            self,
            max_length: int = 60,
            num_return_sequences_train: int = 2,
            num_return_sequences_infer: int = 100,
            topic_scores_aggregator: Optional[Callable[[List[float]], Union[float, np.number]]] = None,
            include_perplexity: bool = True,
            return_intermediate_outputs: bool = False,
    ) -> None:

        if topic_scores_aggregator is None:
            # Use the average by default
            topic_scores_aggregator = lambda scores: np.mean(scores)

        if include_perplexity is True:
            sql_utils.colorful_warning("Adding Perplexity-based Reward", bg="blue")

        sql_utils.colorful_warning(f"max_length={max_length}", bg="blue")

        # https://huggingface.co/gpt2
        # https://huggingface.co/facebook/bart-large-mnli
        self._generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=0)
        self._classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0)

        self._max_length = max_length
        self._num_return_sequences_train = num_return_sequences_train
        self._num_return_sequences_infer = num_return_sequences_infer
        self._topic_scores_aggregator = topic_scores_aggregator
        # `topic_to_candidate_labels_map` is deprecated
        self._topic_to_candidate_labels_map, self._pplm_inputs = (
            self.load_topic_to_candidate_labels_map_and_pplm_inputs())

        # Technically, adding perplexity-based reward will break
        # the scale, but we will ignore this for now since
        # this number is relatively small.
        self._include_perplexity = include_perplexity
        # Do not set is to `True` during training, use it for debugging.
        self._return_intermediate_outputs = return_intermediate_outputs

    def load_topic_to_candidate_labels_map_and_pplm_inputs(self) -> Tuple[Dict[str, List[str]], List[str]]:
        topic_to_candidate_labels_map = {}
        for topic in self.TOPICS:
            file_name = os.path.join(
                self.WORDLISTS_BASE_DIR,
                f"{topic}.txt")

            with open(file_name) as f:
                # There is one file that capitalized all words
                # hence it's likely better to lower case all of
                # them -- with the risk of hurting some words
                topic_to_candidate_labels_map[topic] = [
                    d.strip().lower() for d in f.readlines()]

        with open(self.PPLM_INPUTS_FILE_NAME) as f:
            pplm_inputs = [d.strip() for d in f.readlines()]

        return topic_to_candidate_labels_map, pplm_inputs

    def _format_prompts(self, strings: List[str]) -> List[str]:
        inputs = np.random.choice(
            self._pplm_inputs,
            size=len(strings),
            # we use with-replacement here
            replace=True,).tolist()

        new_strings = [
            self._generator.tokenizer
            .convert_tokens_to_string(s.split())
            for s in strings]

        return [
            f"{s_1} {s_2}" for s_1, s_2
            in zip(new_strings, inputs)]

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        nlls, _ = compute_perplexities(
            sentences=sentences,
            model=self._generator.model,
            tokenizer=self._generator.tokenizer)
        # When the sentence has just one token,
        # the NLL/perplexity will be `NaN`.
        # Further, we use the negative NLL as the reward
        return -torch.nan_to_num(nlls, nan=10.0).mean()

    def _check_classifier_outputs(
            self,
            candidate_labels: List[str],
            classifier_outputs: List[Dict],
    ) -> None:
        for output in classifier_outputs:
            if len(output["scores"]) != len(candidate_labels):
                raise ValueError

    def forward(self, topics: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError

        if mode == "train":
            num_return_sequences = self._num_return_sequences_train
        if mode == "infer":
            num_return_sequences = self._num_return_sequences_infer

        # - List of length `len(prompts)`
        #     - List of length `num_return_sequences`
        #         - Dict of {"generated_text": str}
        formatted_prompts = self._format_prompts(prompts)
        generator_outputs: List[List[Dict[str, Any]]] = self._generator(
            formatted_prompts,
            max_length=self._max_length,
            num_return_sequences=num_return_sequences,
            # Only return generated text, without the prompt
            return_full_text=False)

        all_classifier_outputs = []
        rewards: List[FloatTensor] = []
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for batch_index in range(len(prompts)):
            generated_texts = [
                output["generated_text"] for output in
                generator_outputs[batch_index]]

            # - List of length `len(generated_texts)`
            #     - Dict of {
            #         "labels": List of length `num_topics`,
            #         "scores": List of length `num_topics`,
            #         "sequence": str,
            #     }
            try:
                topic = topics[batch_index]
                classifier_outputs = self._classifier(
                    generated_texts,
                    candidate_labels=[topic],
                    multi_label=True)

                self._check_classifier_outputs(
                    candidate_labels=[topic],
                    classifier_outputs=classifier_outputs)

                _reward_list = [
                    self._topic_scores_aggregator(output["scores"])
                    for output in classifier_outputs]

                # We assume rewards are in `[0, 100]`
                reward = torch.tensor(_reward_list).float().mean() * 100
                quantities_to_log["topic"].append(reward)
                if self._include_perplexity is True:
                    nll_reward = (
                        self._compute_nll_reward(
                            sentences=generated_texts))
                    reward = reward + nll_reward
                    quantities_to_log["nll"].append(nll_reward)

                rewards.append(reward)
                all_classifier_outputs.append(classifier_outputs)

            except ValueError as err:
                # This happens when the generated text itself includes the
                # `</s>` token, which does happen and will cause the classifier to fail.
                # So we just ignore this error and give a score of zero for this batch.
                if str(err) != "All examples must have the same number of <eos> tokens.":
                    raise err

                click.secho("Encountered an error, skipping ...", bg="red")
                rewards.append(torch.tensor(0.).to(device))

        rewards_tensor = torch.stack(rewards)
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        if self._return_intermediate_outputs is True:
            rewards_log["quantities_to_log"] = quantities_to_log  # type: ignore
            rewards_log["formatted_prompts"] = formatted_prompts  # type: ignore
            rewards_log["generator_outputs"] = generator_outputs  # type: ignore
            rewards_log["all_classifier_outputs"] = all_classifier_outputs  # type: ignore

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
            topics=sources,
            prompts=predictions,
            to_tensor=to_tensor,
            mode=mode)
    
class GPT2BLEUReward(object):
    # WORDLISTS_BASE_DIR = "/workspace/soft-Q-learning-for-text-generation/experiments/wordlists"
    TST_TEMPLATES_FILE_NAME = "/workspace/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task-no-quotes.txt"
    # TOPICS = ["legal", "politics", "computers", "space", "religion", "science", "military"]

    def __init__(
            self,
            max_length: int = 60,
            num_return_sequences_train: int = 2,
            num_return_sequences_infer: int = 100,
            # topic_scores_aggregator: Optional[Callable[[List[float]], Union[float, np.number]]] = None,
            include_perplexity: bool = True,
            return_intermediate_outputs: bool = False,
    ) -> None:

#         if topic_scores_aggregator is None:
#             # Use the average by default
#             topic_scores_aggregator = lambda scores: np.mean(scores)

        if include_perplexity is True:
            sql_utils.colorful_warning("Adding Perplexity-based Reward", bg="blue")

        sql_utils.colorful_warning(f"max_length={max_length}", bg="blue")

        # https://huggingface.co/gpt2
        # https://huggingface.co/facebook/bart-large-mnli
        self._generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=0)
#         self._classifier = pipeline(
#             "zero-shot-classification",
#             model="facebook/bart-large-mnli",
#             device=0)

        self._max_length = max_length
        self._num_return_sequences_train = num_return_sequences_train
        self._num_return_sequences_infer = num_return_sequences_infer
#         self._topic_scores_aggregator = topic_scores_aggregator
        # `topic_to_candidate_labels_map` is deprecated
#         self._topic_to_candidate_labels_map, self._pplm_inputs = (
#             self.load_topic_to_candidate_labels_map_and_pplm_inputs())
        self._tst_templates = self.load_tst_templates()

        # Technically, adding perplexity-based reward will break
        # the scale, but we will ignore this for now since
        # this number is relatively small.
        self._include_perplexity = include_perplexity
        # Do not set is to `True` during training, use it for debugging.
        self._return_intermediate_outputs = return_intermediate_outputs
        
    def load_tst_templates(self) -> List[str]:
        with open(self.TST_TEMPLATES_FILE_NAME) as f: 
            tst_templates = [d.strip() for d in f.readlines()]
        return tst_templates

#     def load_topic_to_candidate_labels_map_and_pplm_inputs(self) -> Tuple[Dict[str, List[str]], List[str]]:
#         topic_to_candidate_labels_map = {}
#         for topic in self.TOPICS:
#             file_name = os.path.join(
#                 self.WORDLISTS_BASE_DIR,
#                 f"{topic}.txt")

#             with open(file_name) as f:
#                 # There is one file that capitalized all words
#                 # hence it's likely better to lower case all of
#                 # them -- with the risk of hurting some words
#                 topic_to_candidate_labels_map[topic] = [
#                     d.strip().lower() for d in f.readlines()]

#         with open(self.PPLM_INPUTS_FILE_NAME) as f:
#             pplm_inputs = [d.strip() for d in f.readlines()]

#         return topic_to_candidate_labels_map, pplm_inputs

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._generator.tokenizer
                .convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, source_strings: List[str], prompt_strings: List[str]) -> List[str]:
        templates = np.random.choice(
            self._tst_templates,
            size=len(prompt_strings),
            # we use with-replacement here
            replace=True,).tolist()
        # print(templates)

        return [
            t.format(sentence_1=s_1, prompt=p) for t, s_1, p
            in zip(templates, source_strings, prompt_strings)]

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        nlls, _ = compute_perplexities(
            sentences=sentences,
            model=self._generator.model,
            tokenizer=self._generator.tokenizer)
        # When the sentence has just one token,
        # the NLL/perplexity will be `NaN`.
        # Further, we use the negative NLL as the reward
        return -torch.nan_to_num(nlls, nan=10.0).mean()

#     def _check_classifier_outputs(
#             self,
#             candidate_labels: List[str],
#             classifier_outputs: List[Dict],
#     ) -> None:
#         for output in classifier_outputs:
#             if len(output["scores"]) != len(candidate_labels):
#                 raise ValueError

    def forward(self, sources: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError

        if mode == "train":
            num_return_sequences = self._num_return_sequences_train
        if mode == "infer":
            num_return_sequences = self._num_return_sequences_infer

        # - List of length `len(prompts)`
        #     - List of length `num_return_sequences`
        #         - Dict of {"generated_text": str}
        source_strings = self._convert_tokens_to_string(sources)
        prompt_strings = self._convert_tokens_to_string(prompts)
        formatted_prompts = self._format_prompts(source_strings, prompt_strings)
        
#         eos_token_id = (self._generator
#                         .tokenizer
#                         .convert_tokens_to_ids(['"',
        generator_outputs: List[List[Dict[str, Any]]] = self._generator(
            formatted_prompts,
            max_length=self._max_length,
            num_return_sequences=num_return_sequences,
            # Only return generated text, without the prompt
            return_full_text=False)

        # all_classifier_outputs = []
        rewards: List[FloatTensor] = []
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for batch_index in range(len(prompts)):
            generated_texts = [
                output["generated_text"] for output in
                generator_outputs[batch_index]]
            
#             generated_texts = []
#             for output in generator_outputs[batch_index]: 
#                 text = output["generated_text"]
#                 try: 
#                     end = text.index('"')
#                 except ValueError: 
#                     end = len(text)
#                 generated_texts.append(text[:end])
            
            if mode == "infer": 
                print(f"Sentence 1: {source_strings[batch_index]};",
                      f"Prompt: {prompt_strings[batch_index]};",
                      f"Sentence 2: {generated_texts[0]}")

            # - List of length `len(generated_texts)`
            #     - Dict of {
            #         "labels": List of length `num_topics`,
            #         "scores": List of length `num_topics`,
            #         "sequence": str,
            #     }
            try:
                reference_texts = [source_strings[batch_index] for _ in generator_outputs[batch_index]]
                
                check_Xs_Ys_sizes(generated_texts, reference_texts)
                
                # Using a faster BLEU implementation during training
                # `sacrebleu` is ~3X faster than `lightning`
                # `sacrebleu-parallel` is ~3X faster than `sacrebleu`
                bleus = [
                    scb.sentence_bleu(
                        hypothesis=x,
                        references=[y])
                    for x, y in zip(
                        generated_texts,
                        reference_texts)
                ]
                bleu_rewards = [b.score for b in bleus]
                
                reward = torch.tensor(bleu_rewards).float().mean()
                quantities_to_log["bleu"].append(reward)
                
                if self._include_perplexity is True:
                    nll_reward = (
                        self._compute_nll_reward(
                            sentences=generated_texts))
                    reward = reward + nll_reward
                    quantities_to_log["nll"].append(nll_reward)

                rewards.append(reward)
                # all_classifier_outputs.append(classifier_outputs)

            except ValueError as err:
                # This happens when the generated text itself includes the
                # `</s>` token, which does happen and will cause the classifier to fail.
                # So we just ignore this error and give a score of zero for this batch.
                if str(err) != "All examples must have the same number of <eos> tokens.":
                    raise err

                click.secho("Encountered an error, skipping ...", bg="red")
                rewards.append(torch.tensor(0.).to(device))

        rewards_tensor = torch.stack(rewards)
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        if self._return_intermediate_outputs is True:
            rewards_log["quantities_to_log"] = quantities_to_log  # type: ignore
            rewards_log["formatted_prompts"] = formatted_prompts  # type: ignore
            rewards_log["generator_outputs"] = generator_outputs  # type: ignore
            # rewards_log["all_classifier_outputs"] = all_classifier_outputs  # type: ignore

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
            sources=sources,
            prompts=predictions,
            to_tensor=to_tensor,
            mode=mode)
    
class GPT2BLEUNoInputReward(object):
    TST_TEMPLATES_FILE_NAME = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task.txt"

    def __init__(
            self,
            max_length: int = 60,
            num_return_sequences_train: int = 2,
            num_return_sequences_infer: int = 100,
            # topic_scores_aggregator: Optional[Callable[[List[float]], Union[float, np.number]]] = None,
            include_perplexity: bool = True,
            return_intermediate_outputs: bool = False,
    ) -> None:

        if include_perplexity is True:
            sql_utils.colorful_warning("Adding Perplexity-based Reward", bg="blue")

        sql_utils.colorful_warning(f"max_length={max_length}", bg="blue")

        # https://huggingface.co/gpt2
        # https://huggingface.co/facebook/bart-large-mnli
        self._generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=0)

        self._max_length = max_length
        self._num_return_sequences_train = num_return_sequences_train
        self._num_return_sequences_infer = num_return_sequences_infer
        self._tst_templates = self.load_tst_templates()
        self._tst_inputs = self._load_tst_inputs()
        self._tst_inputs_idx = {'train': 0, 'infer': 0}        

        # Technically, adding perplexity-based reward will break
        # the scale, but we will ignore this for now since
        # this number is relatively small.
        self._include_perplexity = include_perplexity
        # Do not set is to `True` during training, use it for debugging.
        self._return_intermediate_outputs = return_intermediate_outputs
        
    def load_tst_templates(self) -> List[str]:
        with open(self.TST_TEMPLATES_FILE_NAME) as f: 
            tst_templates = [d.strip() for d in f.readlines()]
        return tst_templates
    
    def _load_tst_inputs(self) -> Dict[Tuple[str], List[str]]: 
        tst_inputs = {}
        # tokenizer = self._generator.tokenizer
        filepath_train_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.0"
        filepath_train_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.1"
        filepath_dev_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.0"
        filepath_dev_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.1"
        
        with open(filepath_train_0) as f: 
            sentences_train_0 = [line.strip() for line in f]
        with open(filepath_train_1) as f: 
            sentences_train_1 = [line.strip() for line in f]
        with open(filepath_dev_0) as f: 
            sentences_dev_0 = [line.strip() for line in f]
        with open(filepath_dev_1) as f: 
            sentences_dev_1 = [line.strip() for line in f]
            
        import random
        sentences_train = sentences_train_0 + sentences_train_1
        random.seed(0)
        random.shuffle(sentences_train)
        tst_inputs['train'] = sentences_train
        tst_inputs['infer'] = sentences_dev_0[:5] + sentences_dev_1[:5]
        return tst_inputs

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._generator.tokenizer
                .convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, source_strings: List[str], prompt_strings: List[str]) -> List[str]:
#         templates = np.random.choice(
#             self._tst_templates,
#             size=len(prompt_strings),
#             # we use with-replacement here
#             replace=True,).tolist()
        # print(templates)
        template = self._tst_templates[0]

        return [
            template.format(sentence_1=s_1, prompt=p) for s_1, p
            in zip(source_strings, prompt_strings)]

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        nlls, _ = compute_perplexities(
            sentences=sentences,
            model=self._generator.model,
            tokenizer=self._generator.tokenizer)
        # When the sentence has just one token,
        # the NLL/perplexity will be `NaN`.
        # Further, we use the negative NLL as the reward
        return -torch.nan_to_num(nlls, nan=10.0).mean()
    
    def _get_inputs(self, mode: str, batch_size: int): 
        data = self._tst_inputs[mode]
        idx = self._tst_inputs_idx[mode]
        inputs = []
        for _ in range(batch_size): 
            sentence = 'thank you for a five star service .'
#             inputs.append(data[idx])
            inputs.append(sentence)
            idx += 1
            idx %= len(data)
        self._tst_inputs_idx[mode] = idx
        return inputs

    def forward(self, control_codes: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError

        if mode == "train":
            num_return_sequences = self._num_return_sequences_train
        if mode == "infer":
            num_return_sequences = self._num_return_sequences_infer

        # - List of length `len(prompts)`
        #     - List of length `num_return_sequences`
        #         - Dict of {"generated_text": str}
        batch_size = len(control_codes)
        source_strings = self._get_inputs(mode, batch_size)
        prompt_strings = self._convert_tokens_to_string(prompts)
        formatted_prompts = self._format_prompts(source_strings, prompt_strings)
        
        generator_outputs: List[List[Dict[str, Any]]] = self._generator(
            formatted_prompts,
            max_length=self._max_length,
            num_return_sequences=num_return_sequences,
            # Only return generated text, without the prompt
            return_full_text=False)

        rewards: List[FloatTensor] = []
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for batch_index in range(len(prompts)):
#             generated_texts = [
#                 output["generated_text"] for output in
#                 generator_outputs[batch_index]]
            
            generated_texts = []
            for output in generator_outputs[batch_index]: 
                text = output["generated_text"]
                try: 
                    end = text.index('"')
                except ValueError: 
                    end = len(text)
                generated_texts.append(text[:end])
            
            if mode == "infer": 
                print(f"Formatted Prompt: {formatted_prompts[batch_index]};",
                      f"Output: {generated_texts[0]}")

            # - List of length `len(generated_texts)`
            #     - Dict of {
            #         "labels": List of length `num_topics`,
            #         "scores": List of length `num_topics`,
            #         "sequence": str,
            #     }
            try:
                reference_texts = [source_strings[batch_index] for _ in generator_outputs[batch_index]]
                
                check_Xs_Ys_sizes(generated_texts, reference_texts)
                
                # Using a faster BLEU implementation during training
                # `sacrebleu` is ~3X faster than `lightning`
                # `sacrebleu-parallel` is ~3X faster than `sacrebleu`
                bleus = [
                    scb.sentence_bleu(
                        hypothesis=x,
                        references=[y])
                    for x, y in zip(
                        generated_texts,
                        reference_texts)
                ]
                bleu_rewards = [b.score for b in bleus]
                
                reward = torch.tensor(bleu_rewards).float().mean()
                quantities_to_log["bleu"].append(reward)
                
                if self._include_perplexity is True:
                    nll_reward = (
                        self._compute_nll_reward(
                            sentences=generated_texts))
                    reward = reward + nll_reward
                    quantities_to_log["nll"].append(nll_reward)

                rewards.append(reward)

            except ValueError as err:
                # This happens when the generated text itself includes the
                # `</s>` token, which does happen and will cause the classifier to fail.
                # So we just ignore this error and give a score of zero for this batch.
                if str(err) != "All examples must have the same number of <eos> tokens.":
                    raise err

                click.secho("Encountered an error, skipping ...", bg="red")
                rewards.append(torch.tensor(0.).to(device))

        rewards_tensor = torch.stack(rewards)
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        if self._return_intermediate_outputs is True:
            rewards_log["quantities_to_log"] = quantities_to_log  # type: ignore
            rewards_log["formatted_prompts"] = formatted_prompts  # type: ignore
            rewards_log["generator_outputs"] = generator_outputs  # type: ignore
            # rewards_log["all_classifier_outputs"] = all_classifier_outputs  # type: ignore

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
            control_codes=sources,
            prompts=predictions,
            to_tensor=to_tensor,
            mode=mode)
    
class GPT2SentimentNoInputReward(object):
    TST_TEMPLATES_FILE_NAME = "/workspace/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task.txt"
    # TST_TEMPLATES_FILE_NAME = "/workspace/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task-no-source.txt"
    TST_CLF_CONFIG = dict(model=("/workspace/soft-Q-learning-for-text-generation/experiments/yelp_sentiment_classifier/"
                                 "results-bert-base/checkpoint-10410/"),
                          tokenizer='bert-base-uncased')

    def __init__(
            self,
            max_length: int = 60,
            num_return_sequences_train: int = 2,
            num_return_sequences_infer: int = 100,
            # topic_scores_aggregator: Optional[Callable[[List[float]], Union[float, np.number]]] = None,
            include_perplexity: bool = True,
            return_intermediate_outputs: bool = False,
    ) -> None:

        if include_perplexity is True:
            sql_utils.colorful_warning("Adding Perplexity-based Reward", bg="blue")

        sql_utils.colorful_warning(f"max_length={max_length}", bg="blue")

        # https://huggingface.co/gpt2
        # https://huggingface.co/facebook/bart-large-mnli
        self._generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=0)
        self._classifier = pipeline(
            "sentiment-analysis",
            model=self.TST_CLF_CONFIG['model'],
            tokenizer=self.TST_CLF_CONFIG['tokenizer'],
            device=0)

        self._max_length = max_length
        self._num_return_sequences_train = num_return_sequences_train
        self._num_return_sequences_infer = num_return_sequences_infer
        self._tst_templates = self.load_tst_templates()
        self._tst_inputs = self._load_tst_inputs()
        self._tst_inputs_idx = {('train', 'LABEL_0'): 0, 
                                ('train', 'LABEL_1'): 0,
                                ('infer', 'LABEL_0'): 0,
                                ('infer', 'LABEL_1'): 0}

        # Technically, adding perplexity-based reward will break
        # the scale, but we will ignore this for now since
        # this number is relatively small.
        self._include_perplexity = include_perplexity
        # Do not set is to `True` during training, use it for debugging.
        self._return_intermediate_outputs = return_intermediate_outputs
        
    def load_tst_templates(self) -> List[str]:
        with open(self.TST_TEMPLATES_FILE_NAME) as f: 
            tst_templates = [d.strip() for d in f.readlines()]
        return tst_templates
    
    def _load_tst_inputs(self) -> Dict[Tuple[Any], List[str]]: 
        tst_inputs = {}
        # tokenizer = self._generator.tokenizer
        filepath_train_0 = "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.0"
        filepath_train_1 = "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.1"
        filepath_dev_0 = "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.0"
        filepath_dev_1 = "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.1"
        
        with open(filepath_train_0) as f: 
            sentences_train_0 = [line.strip() for line in f]
        with open(filepath_train_1) as f: 
            sentences_train_1 = [line.strip() for line in f]
        with open(filepath_dev_0) as f: 
            sentences_dev_0 = [line.strip() for line in f]
        with open(filepath_dev_1) as f: 
            sentences_dev_1 = [line.strip() for line in f]
            
        tst_inputs[('train', 'LABEL_0')] = sentences_train_1
        tst_inputs[('train', 'LABEL_1')] = sentences_train_0
        tst_inputs[('infer', 'LABEL_0')] = sentences_dev_1[:5]
        tst_inputs[('infer', 'LABEL_1')] = sentences_dev_0[:5]
        
        return tst_inputs

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._generator.tokenizer
                .convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, source_strings: List[str], prompt_strings: List[str]) -> List[str]:
        template = self._tst_templates[0]
        
#         return [
#             template.format(prompt=p) for s_1, p
#             in zip(source_strings, prompt_strings)]

        return [
            template.format(sentence_1=s_1, prompt=p) for s_1, p
            in zip(source_strings, prompt_strings)]

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        nlls, _ = compute_perplexities(
            sentences=sentences,
            model=self._generator.model,
            tokenizer=self._generator.tokenizer)
        # When the sentence has just one token,
        # the NLL/perplexity will be `NaN`.
        # Further, we use the negative NLL as the reward
        return -torch.nan_to_num(nlls, nan=10.0).mean()
    
    def _get_inputs(self, mode: str, target_labels: List[str]): 
        # data_0 = self._tst_inputs[(mode, 'LABEL_0')]
        # data_1 = self._tst_inputs[(mode, 'LABEL_1')]
        
        # idx_0 = self._tst_inputs_idx[(mode, 'LABEL_0')]
        # idx_1 = self._tst_inputs_idx[(mode, 'LABEL_1')]
        
        inputs = []
        for i, label in enumerate(target_labels): 
            idx = self._tst_inputs_idx[(mode, label)]
            data = self._tst_inputs[(mode, label)]
            
            inputs.append(data[idx])
            idx += 1
            idx %= len(data)
            self._tst_inputs_idx[(mode, label)] = idx
        
        return inputs

    def forward(self, target_labels: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError
        assert all([label in ['LABEL_0', 'LABEL_1'] for label in target_labels])

        if mode == "train":
            num_return_sequences = self._num_return_sequences_train
        if mode == "infer":
            num_return_sequences = self._num_return_sequences_infer

        # - List of length `len(prompts)`
        #     - List of length `num_return_sequences`
        #         - Dict of {"generated_text": str}
        source_strings = self._get_inputs(mode, target_labels)
        prompt_strings = self._convert_tokens_to_string(prompts)
        formatted_prompts = self._format_prompts(source_strings, prompt_strings)
        
        generator_outputs: List[List[Dict[str, Any]]] = self._generator(
            formatted_prompts,
            max_length=self._max_length,
            num_return_sequences=num_return_sequences,
            # Only return generated text, without the prompt
            return_full_text=False)

        rewards: List[FloatTensor] = []
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for batch_index in range(len(prompts)):
            # generated_texts = [
            #     output["generated_text"] for output in
            #     generator_outputs[batch_index]]
            
            generated_texts = []
            for output in generator_outputs[batch_index]: 
                text = output["generated_text"]
                try: 
                    end = text.index('"')
                except ValueError: 
                    end = len(text)
                generated_texts.append(text[:end])
            
            if mode == "infer": 
                print(f"Formatted Prompt: {formatted_prompts[batch_index]};",
                      f"Output: {generated_texts[0]}")

            # - List of length `len(generated_texts)`
            #     - Dict of {
            #         "labels": List of length `num_topics`,
            #         "scores": List of length `num_topics`,
            #         "sequence": str,
            #     }
            try:
                reference_texts = [source_strings[batch_index] for _ in generator_outputs[batch_index]]
                
                check_Xs_Ys_sizes(generated_texts, reference_texts)
                
                # Using a faster BLEU implementation during training
                # `sacrebleu` is ~3X faster than `lightning`
                # `sacrebleu-parallel` is ~3X faster than `sacrebleu`
#                 bleus = [
#                     scb.sentence_bleu(
#                         hypothesis=x,
#                         references=[y])
#                     for x, y in zip(
#                         generated_texts,
#                         reference_texts)
#                 ]
#                 bleu_rewards = [b.score for b in bleus]
                
#                 reward = torch.tensor(bleu_rewards).float().mean()
#                 quantities_to_log["bleu"].append(reward)

                classes = self._classifier(generated_texts, truncation=True)
                label = target_labels[batch_index]
                correct = [100 * (c['label'] == label) for c in classes]
                acc = torch.tensor(correct).float().mean()
                reward = acc
                quantities_to_log['acc'].append(acc)
                if label == 'LABEL_0': quantities_to_log['acc_0'].append(acc)
                elif label == 'LABEL_1': quantities_to_log['acc_1'].append(acc)
                
                if self._include_perplexity is True:
                    nll_reward = (
                        self._compute_nll_reward(
                            sentences=generated_texts))
                    reward = reward + nll_reward
                    quantities_to_log["nll"].append(nll_reward)

                rewards.append(reward)

            except ValueError as err:
                # This happens when the generated text itself includes the
                # `</s>` token, which does happen and will cause the classifier to fail.
                # So we just ignore this error and give a score of zero for this batch.
                if str(err) != "All examples must have the same number of <eos> tokens.":
                    raise err

                click.secho("Encountered an error, skipping ...", bg="red")
                rewards.append(torch.tensor(0.).to(device))

        rewards_tensor = torch.stack(rewards)
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        if self._return_intermediate_outputs is True:
            rewards_log["quantities_to_log"] = quantities_to_log  # type: ignore
            rewards_log["formatted_prompts"] = formatted_prompts  # type: ignore
            rewards_log["generator_outputs"] = generator_outputs  # type: ignore
            # rewards_log["all_classifier_outputs"] = all_classifier_outputs  # type: ignore

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
    
from ctc_score import StyleTransferScorer

from bert_score import BERTScorer

from transformers import (LogitsProcessor, 
                          LogitsProcessorList, 
                          TextClassificationPipeline,
                          AutoModelForSequenceClassification, 
                          AutoTokenizer)

import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        
        
train_0_selection = [ 27498,   4509,  77038, 150017, 109602, 135287,  86914,  50901,
        58031, 146386,  83873, 144006,  32597,  78170,  55648,  83065,
        42021,  30683, 154906,  95993,  49274,  55177, 122670,  32142,
       131667,  45763,  83484, 162121,  28636, 173058,   6302, 165094,
        17587, 176237,  81262,  77135, 107024, 176086,   8599,  96121,
       113907,  29813,  67358,  13240,  60101, 147802,  96902,  15058,
        12838,  71334,  48698,  58335,  63514,  16837,  24003,  56136,
        24992,  61916, 164576, 152960,  20114,  43580,  23216, 166835,
       118151,  11185,  82050,  60604, 108569,  72188,  92212,  66694,
       105051, 142463,  64145, 171007,  77161, 155460,  38798, 160594,
        94212,  51143,  11848, 170350,  68540,  82013,  25503,  82413,
       154590,  51857, 128999,  61064, 101699,  71620,  15952, 165020,
       115723,  44356,  12066,  48330,  41733,   5862,   5997,   5440,
       167794, 172628, 157669,  66318,  96978, 145128, 141914,  99683,
        71596,  57663, 149468,  92773, 117626,  26979, 138122, 175299,
        18191, 158748,   5856,  41350,  52981,  29155, 159250,  43482,
       176111,  42615, 166952, 157514,  66746,   5809, 173067, 149543,
       138226,  28185,  84952,  49257, 155480,  80843, 136911,  85816,
       119914, 151619,  47023,  58999,  82810,  18162, 104847, 173485,
       150771,  42221,  57717,  88784,  98146,  68414, 130348, 113812,
        59409,  40094,  11107, 170918, 175621,  77945, 173838, 103439,
        62950, 148182, 145277, 154233, 156491,  54367,  95341, 135187,
        91596, 165584, 147841, 170200,  52518,  36338,  71915,  85078,
        68924,   7333,  70820,  58589,  18579, 109000, 130088, 123361,
       169156, 166493,  17201,  95369,  31029,  73969,  14357, 170232,
       138760,  61393,  47882, 107661, 155268, 168869, 171167, 116628,
       174620,  61708, 138202,   5026,  15779,  94156, 159325,    957,
       126534,  37996,  49599, 128671,  41868,  37513, 126629, 168215,
       124328, 106448, 155013,  28549,  55847,  26235, 114982, 156836,
        91746,  15125,  74650, 135605,  69565,  31495,   7850,  88208,
       135031,  74460,  26140,  92796,  36146,  82934,  35023,   9958,
        43309, 132293,  43549, 162731,  55329, 157351,  83082,  42227,
        27564,  43478,  69474, 149986,  77505,  56704,   7852,   5300,
       103225,  86465,  53024, 169906,  45686,  11109,  65493,  90043,
        39411, 172615, 108338, 158455,  96158, 136162, 175644,  27963,
       118056, 148988,   6691, 133583,  31962, 140405,  58434, 174711,
       124722,   8797, 153914,  79256,  98794,  81308, 171620, 132506,
       143478, 108851,  87588,  46529, 140425,  78718,  55283, 143581,
        49135,  85684,  18926,   3140,  40915,  40649, 130546, 163328,
       145208,  60819, 156483, 155505,  51401, 102787,  18456,  56712,
       105983,  39810,  82248, 108902,  80189,  15874, 100602,  88656,
        66171, 146550, 142181,  97854, 100398, 175083, 166462, 123230,
        63761, 151016,  93058,   1564, 115643,  62527,   7314,    565,
        87262,  59255,    867, 160232,  84592,  99202, 104681,  97525,
        96260, 143038,  67253,  86713, 105763,  35134,  24374,  86210,
        18630, 111067,  82191,  84144, 157811, 101684,  49800, 167683,
        17780,  63054, 105274,  77500, 165994,  85813,  10736, 103499,
       115935, 101027, 125853, 129362, 142527,  53176, 138530,  10987,
        79991, 132021, 175530,  38121,  10630,  24148, 100180,  94230,
        77224, 107902, 168658, 138131, 167355,  85354, 154259, 138419,
        96420,  90081,  56633, 162282,  77356, 124891, 118459, 111392,
        31169, 110609,  10258, 173313,   5019,  99980,  99195,  83175,
       131196,  53996,  97648,   7806, 140435, 129701, 143899, 152586,
        98686, 100361,    337, 124079, 103432, 146740,  61228, 176738,
        63742,  86448, 159253,  82163, 150295,  42932,  82827,  51740,
       109601, 158284, 175721, 101750,  33142,  74533,   9535, 113333,
       136281,  64472, 172918, 157476,  91951, 112875,  39285, 116384,
        44510, 131142,  70454,   8974,  42632, 142186,  85582,  87774,
        67836, 115710,  56891, 119043,  65222, 173038, 117152, 137898,
        26109, 111370,  24461,  30761,  20887,  96047,  55298, 148365,
        84305,  78819,  78401, 174759,  83863,  39141, 106976,  29661,
       127983,   3862, 135391, 122007, 132109,  39052, 160669, 139982,
        56885, 146695,  83694,  40671]

train_1_selection = [241266,  44231, 235186, 128155,  84469, 119886,  73860, 253558,
       184159,  97094, 110431, 105040, 170276,  70690, 186078, 237257,
       256509, 172457,  15700, 256161,  92162, 201093,  49406, 133049,
       184789, 255505, 143377,   5277, 255319, 150640, 143161,  30866,
       154364, 138123,  20230, 259144,   1988, 264093,  39249,  55195,
        73822, 106740,  32443, 208219, 150782, 196292,  74768, 103265,
       183722,  64278, 243898,  87209, 107538, 253736, 224129,  74716,
       242412, 217246,  62031,  68743, 162349, 242451, 226795, 113443,
        80709, 165904, 196423, 198815, 143744,  20809,  62766, 179510,
       177938,  45284,  45395, 117796, 234801, 181297,  97879,  96916,
        21903, 196077, 209302,  31603, 165318, 149545,  40384, 197509,
        70488,  93200,  27756, 177492,  40587, 131517,  17733, 199221,
        60692, 162167, 208085, 180057, 123359,  39571, 204713,  63426,
        66331,  96961, 107948, 186860,  29477, 108538,  50453, 248504,
        92575, 179109, 167868, 231429, 101301, 242411, 233148,  74984,
         7394, 155078, 137531, 233051, 217693, 151083,  91661, 147661,
       139274,   7396, 168672,  39699,  47838,  75145,  73863,   1821,
        68298,  63900, 112238, 109818,  12204,  60253, 149880, 193967,
       248688, 204152, 219583,  35814, 127935,  62605, 116382, 173099,
        86916,  35547, 116314,  85645, 261244, 152716, 248796, 240245,
        76285,  50622,  45787, 233223,  90106, 167785, 129004, 204125,
       244634, 202061, 180924,  65228, 246637,  93476, 190824,  49910,
       159879,  83186, 192924, 159676, 200154,  69605, 197511,  70647,
        29578, 189387,   9593, 158990,  90771,  73406,  88739,  24222,
        83258, 132100, 146691, 143862, 192273, 198754, 256978, 138845,
       210747, 107820, 235181, 171791,  97800,   9448,  26497, 127907,
       178528,   3834, 255542,  89279,  95521,  90330, 149829, 106889,
       118887, 219515, 257671,  51537,  54963, 258710, 139200, 258098,
        42060, 184105, 155628,  66141,  79246, 128142, 102859, 259123,
        34902, 232247,  73239,  36426, 253537, 217515, 250995,  45363,
       223013, 144158, 250589, 242812, 131082, 107900, 143967,  58237,
        20438,  15818, 251722,  55448, 115804,  93141, 231572, 250187,
       251215,  72804, 169125,  54460,  95031, 112829, 183070, 248171,
       261081,  57474, 124248, 132220,  88030,  45705, 107493, 261768,
       194233,  34451, 216561,  74921, 165172, 259608, 126623,  27476,
       212391, 138834, 158867, 225366, 110459,  73084, 104676,  69317,
        94426, 266285,   9037,  41061,  92258, 252514, 184030, 246339,
       192288,  33482,   8136, 154926,  11725,  20442, 113503, 245483,
        57914, 213633, 145997,  48757,  39962, 111230,  45585, 264812,
       202761,  53083, 251285, 107557,  70692,  51220, 137497,  74640,
       161427, 210217, 266416, 121841, 257052, 208823, 259001, 244708,
        78584,  26309, 230758,  18954,  39959, 228367, 195858, 110157,
           91,  28893, 110691, 191806, 187284,  54145,  29207,   2122,
        71835, 102246,  35922, 138898,  90339, 123098, 163664,  36793,
        60114, 117021, 129552,  42698, 177343,   5172, 175403, 110631,
       228428,   9144,  24759, 241404, 257346, 165692,  83131,    915,
       262429, 205354,  71722,  58987,  79116, 102736, 226170,  17450,
       179863,  94395, 159890, 202015, 139001, 154397, 213694, 146428,
       165695, 237526,  50258, 215755, 171272,  12644, 179226,   3938,
       224430, 131779,  80550, 251898,  84159,  18437, 242904, 125169,
        30865,  55401, 161074, 150490, 159713, 124104, 174059,  94483,
       158967, 238129, 207998, 181772, 214754,  16071, 197381,  47281,
        37996,  58073, 252088, 105790,  70970, 261555,  10640,  66349,
       233827,  20946, 177476, 162185,  77957, 235142,  12207, 119585,
       141200, 119950, 183737, 156148, 110092, 253248,  50255, 138739,
       227825, 254110, 158414,  27446, 231904, 187763, 123838, 145120,
       228971, 262956,  35221, 100288,  31653, 266988, 114291, 127908,
       111092,  46629, 230059, 153192,  89672, 158605, 219242, 180063,
       189810,  16773,  54204, 115703, 109197,   7723, 222288,  50761,
       124831, 126725, 118506, 129568, 224566, 256385,  64630,  74495,
       124042,  64225, 119055, 152925,  11045, 139912, 105826,  11114,
       144642, 252553,  96435,  72604, 242925, 185539,   2097, 240749,
       190902, 262115, 109597, 189027]
        
        
import itertools
class GPT2SentimentBLEUNoInputReward(object):
    TST_TEMPLATES_FILE_NAME = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task-prefix.txt"
#     TST_TEMPLATES_FILE_NAME = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task.txt"
#     TST_TEMPLATES_FILE_NAME = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task-prefix.txt"
#     TST_TEMPLATES_FILE_NAME = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task-infix-no-prompt.txt"
#     TST_TEMPLATES_FILE_NAME = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task-prefix-no-prompt.txt"
    END_PUNCT = '"'
    TST_CLF_CONFIG = dict(model=("/jupyter/prompt-generation/soft-Q-learning-for-text-generation/experiments/yelp_sentiment_classifier/"
                                    "results-bert-base/checkpoint-10410"),
                          tokenizer='bert-base-uncased')

    def __init__(
            self,
            max_length: int = 30,
            num_return_sequences_train: int = 128,
            num_return_sequences_infer: int = 128,
            # topic_scores_aggregator: Optional[Callable[[List[float]], Union[float, np.number]]] = None,
            include_perplexity: bool = False,
            return_intermediate_outputs: bool = False,
    ) -> None:

        if include_perplexity is True:
            sql_utils.colorful_warning("Adding Perplexity-based Reward", bg="blue")

        sql_utils.colorful_warning(f"max_length={max_length}", bg="blue")

        # https://huggingface.co/gpt2
        # https://huggingface.co/facebook/bart-large-mnli
        # generator_model = 'gpt2-xl'
        # generator_model = 'gpt2-large'
        # generator_model = 'gpt2-medium'
        generator_model = 'distilgpt2'
        generator_device = 0
        evaluator_device = 0
        tokenizer = AutoTokenizer.from_pretrained(generator_model, pad_token='<|endoftext|>')
        self._generator = pipeline(
            "text-generation",
            model=generator_model,
            tokenizer=tokenizer,
            device=generator_device)
        self._classifier = pipeline(
            "sentiment-analysis",
            model=self.TST_CLF_CONFIG['model'],
            tokenizer=self.TST_CLF_CONFIG['tokenizer'],
            device=evaluator_device,)
            #return_all_scores=True)
#         self._classifier = pipeline(
#             "sentiment-analysis",
#             model='cardiffnlp/twitter-roberta-base-sentiment-latest',
#             device=evaluator_device,
#             return_all_scores=True)

        # MOD for roberta
        # self.roberta_clf = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
        # self._classifier = self.roberta_classifier

        self._max_length = max_length
        self._num_return_sequences_train = num_return_sequences_train
        self._num_return_sequences_infer = num_return_sequences_infer
        self._tst_templates = self.load_tst_templates()
        

        ### Modification starts ###
        self._warm_up_reward = False
        self._counter = 0
        self.tokens_explored = set()
        self.temp_input_0 = "they are all really friendly and the vets are knowledgable and patient ."
        self.temp_input_1 = "thank you for a five star service ."
        self.temp_input_2 = "the mojitos are deliciously made with fresh fruit ."
        self.temp_input_3 = "someone should buy this place and turn it into what it should be."
        self.temp_input_4 = "we were ignored until i grabbed a table in the back."
        self.temp_input_5 = "thank you for the meal ."
        self.temp_input_6 = "manager actually has respect for the customer instead of ignoring them ."
        
        self.dataset_inputs = ['the carts are in excellent shape, all electric and all equipped with gps.',
                               'challenging but fun course!',
                               'beautiful views and lots of variety of length and layout of holes.',
                               "i'll definitely be back!",
                               'the service and prices were great.',
                               'i had the buffalo chicken sandwich and it was delicious.',
                               'a cool bar off the beaten path that is a worth a trip.',
                               'awesome drink specials during happy hour.',
                               'fantastic wings that are crispy and delicious, wing night on tuesday and thursday!',
                               'the sandwiches are always amazing just as i remember.',
                               'the staff is amazing and friendly.',
                               'great place for lunch as well.',
                               'friendly staff, good food, great beer selection, and relaxing atmosphere.',
                               "great wings and the buffalo chicken pizza is the best i've had.",
                               'the sandwiches are all on thick cut italian bread and fresh.',
                               'if we ever get to the pittsburgh area again, we will go back!']
        
#         self.dataset_inputs = ['the carts are in excellent shape, all electric and all equipped with gps.',
#                                'challenging but fun course!',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                "i'll definitely be back!",
#                                'the service and prices were great.',
#                                'i had the buffalo chicken sandwich and it was delicious.',
#                                'a cool bar off the beaten path that is a worth a trip.',
#                                'awesome drink specials during happy hour.',]

        self.dataset_inputs = ['i was sadly mistaken.',
                               'so on to the hoagies, the italian is general run of the mill.',
                               'minimal meat and a ton of shredded lettuce.',
                               'nothing really special & not worthy of the $_num_ price tag.',
                               'second, the steak hoagie, it is atrocious.',
                               'i had to pay $_num_ to add cheese to the hoagie.',
                               'she told me there was a charge for the dressing on the side.',
                               'are you kidding me?',
                               'i was not going to pay for the dressing on the side.',
                               'i ordered it without lettuce, tomato, onions, or dressing.',
                               'are you kidding me?',
                               'i paid $_num_ to add sauted mushrooms, onions, and cheese.',
                               'in this case, never.',
                               '(the hoagie bun was better than average.)',
                               'wake up or you are going to lose your business.',
                               'this place has none of them.']

        import itertools
        self.n_repeats = 4
        self.dataset_inputs = list(itertools.chain(*[[s for _ in range(self.n_repeats)] for s in self.dataset_inputs]))
        # print(self.dataset_inputs)
        
#         self.dataset_inputs = ['challenging but fun course!',
#                                "i'll definitely be back!",
#                                'the service and prices were great.',
#                                'i had the buffalo chicken sandwich and it was delicious.',
#                                'thank you for a five star service.',
#                                # 'a cool bar off the beaten path that is a worth a trip.',
#                                'awesome drink specials during happy hour.',
#                                'fantastic wings that are crispy and delicious, wing night on tuesday and thursday!',
#                                'the sandwiches are always amazing just as i remember.']
        
        # self.dataset_inputs = ['this is good.']
    
#         self.dataset_inputs = ['thank you for a five star service.',
#                                'this is good.']
        
        # self.dataset_inputs = ['thank you for a five star service.']
        
        # self.dataset_inputs = ['the service and prices were great.']
        
#         self.dataset_inputs = ['thank you for a five star service.',
#                                'the service and prices were great.']

        # self.dataset_inputs = ["i'll definitely be back!"]
    
#         self.dataset_inputs = ["i'll definitely be back!",
#                                'this is good.']
        
#         self.dataset_inputs = ["i'll definitely be back!",
#                                'this is good.',
#                                'the prices were great.',
#                                'great place for lunch as well.']
        
#         self.dataset_inputs = ["i'll definitely be back!",
#                                'this is good.',
#                                'the service and prices were great.',
#                                'great place for lunch as well.']

        # self.dataset_inputs = ['the carts are in excellent shape, all electric and all equipped with gps.']
        
        # self.dataset_inputs = ['i had the buffalo chicken sandwich and it was delicious.']
        
        # self.dataset_inputs = ['a cool bar off the beaten path that is a worth a trip.']
        
        # self.dataset_inputs = ['challenging but fun course!']
        
        # self.dataset_inputs = ['awesome drink specials during happy hour.']
        
        # self.dataset_inputs = ['the service and prices were great.']
        
        # self.dataset_inputs = ['beautiful views and lots of variety of length and layout of holes.']
        
#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'beautiful views and lots of variety of length and layout of holes.']
        
#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'the service and prices were great.']
        
#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'the service and prices were great.',
#                                'beautiful views and lots of variety of length and layout of holes.']

#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'the service and prices were great.',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                'challenging but fun course!']
    
#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'awesome drink specials during happy hour.',
#                                'awesome drink specials during happy hour.',
#                                'awesome drink specials during happy hour.',
#                                'the service and prices were great.',
#                                'the service and prices were great.',
#                                'the service and prices were great.',
#                                'the service and prices were great.',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                'beautiful views and lots of variety of length and layout of holes.',
#                                'challenging but fun course!',
#                                'challenging but fun course!',
#                                'challenging but fun course!',
#                                'challenging but fun course!']

#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'thank you for a five star service.']
    
#         self.dataset_inputs = ['awesome drink specials during happy hour.',
#                                'the service and prices were great.',
#                                'thank you for a five star service.']

        # self.dataset_inputs = ['the prices were great.']
        
        # self.dataset_inputs = ['the service and prices were good.']
        
        # self.dataset_inputs = ['great place for lunch as well.']
        
#         self.dataset_inputs = ['great place for lunch as well.',
#                                'this is good.']
        
#         self.dataset_inputs = ['thank you for a five star service.',
#                                'this is good.',
#                                'the service and prices were great.',
#                                'great place for lunch as well.']
        
        self.temp_input = 'this is good.'
        # self.sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # self.ctc_scorer = StyleTransferScorer(align='E-roberta')
        self._bert_scorer = BERTScorer('roberta-large', device=evaluator_device, rescale_with_baseline=True, lang='en')
        
        self._tst_inputs = self._load_tst_inputs()
        self._tst_inputs_idx = {('train', 'LABEL_0'): 0, 
                                ('train', 'LABEL_1'): 0,
                                ('infer', 'LABEL_0'): 0,
                                ('infer', 'LABEL_1'): 0}
        
#         self._tst_input_reward_ranges = defaultdict(tuple)
#         self._tst_input_reward_meanstd = defaultdict(RunningMeanStd)
        
#         masked_token_ids = self._get_masked_token_ids(tokenizer)
#         class TokenIdMasker(LogitsProcessor): 
#             MASKED_TOKEN_IDS = masked_token_ids
#             def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor: 
#                 scores[:, self.MASKED_TOKEN_IDS] = float('-inf')
#                 return scores

#         self.logits_processor_list = LogitsProcessorList([TokenIdMasker()])
   
        self.sample_size = 4
        self.temperature = 1.0
        self.sample_size_annealing = False
        self.temperature_annealing = False
        self.val_sample_sizes = [1, 16, 64]
        ### Modification ends ###

        # Technically, adding perplexity-based reward will break
        # the scale, but we will ignore this for now since
        # this number is relatively small.
        self._include_perplexity = include_perplexity
        # Do not set is to `True` during training, use it for debugging.
        self._return_intermediate_outputs = return_intermediate_outputs
        print(f'Model Input Max Length = {self._generator.model.config.max_length}')
        
    def _get_masked_token_ids(self, tokenizer): 
        # Get the set of all vocabulary words from the Yelp training set
        train_sentences = []
        with open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                  'data/yelp-gpt2-control-only/raw-prep/sentiment.train.0.preprocess') as f: 
            train_sentences += [line.strip() for line in f]
        with open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                  'data/yelp-gpt2-control-only/raw-prep/sentiment.train.1.preprocess') as f: 
            train_sentences += [line.strip() for line in f]
            
        used_vocab = set()
        input_ids = tokenizer(train_sentences)['input_ids']
        
        for token_ids in input_ids: 
            used_vocab |= set(token_ids)
        print(len(used_vocab))
        used_token_ids = list(used_vocab)
        used_tokens = tokenizer.convert_ids_to_tokens(used_token_ids)
        
        all_tokens = list(tokenizer.vocab.keys())
        masked_tokens = list(set(all_tokens) - set(used_tokens))
        masked_tokens.remove('<|endoftext|>')
        masked_tokens.remove('"')
        masked_token_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
        return masked_token_ids
        

    ### Mod for sbert score ###    
    def sbert_sim(self, src, tgts):
        if type(tgts) is not list:
            tgts = [tgts]
        to_encode = [src] + tgts
        embs = self.sbert.encode(to_encode, show_progress_bar=False)
        cos_sim = lambda a,b : np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return [cos_sim(embs[0], emb) for emb in embs[1:]]

    ### Mod for roberta sentiment classifier
    def roberta_classifier(self, inputs, **kwargs):
        res = self.roberta_clf(inputs, **kwargs)
        for c in res:
            if c["label"] == "POSITIVE":
                c["label"] = "LABEL_1"
            else:
                c["label"] = "LABEL_0"
        return res

    def load_tst_templates(self) -> List[str]:
#         with open(self.TST_TEMPLATES_FILE_NAME) as f: 
#             tst_templates = [d.strip() for d in f.readlines()]
        # temp_tst_template = 'Sentence 1: "{sentence_1}" {prompt} Sentence 2: "'
        temp_tst_template = '{prompt} "{sentence_1}" "'
        temp_tst_template_null = '"{sentence_1}" "'
        temp_tst_template_manual = 'Change the following sentence from positive sentiment to negative sentiment but keep its semantics. Sentence Positive: "{sentence_1}" Sentence Negative: "'
        temp_tst_template_sentence = '{prompt} Sentence 1: "{sentence_1}" Sentence 2: "'
        temp_tst_template_input = '{prompt} Input: "{sentence_1}" Output: "'
        temp_tst_template_infix = '"{sentence_1}" {prompt} "'
        temp_tst_template_curly = '{prompt} {{{sentence_1}}} {{'
        return [temp_tst_template, 
                temp_tst_template_null, 
                temp_tst_template_manual, 
                temp_tst_template_sentence,
                temp_tst_template_input,
                temp_tst_template_infix,
                temp_tst_template_curly]
    
    def _load_tst_inputs(self) -> Dict[Tuple[Any], List[str]]: 
        tst_inputs = {}
        # tokenizer = self._generator.tokenizer
        filepath_train_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw-prep/sentiment.train.0.preprocess"
        filepath_train_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw-prep/sentiment.train.1.preprocess"
        filepath_dev_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw-prep/sentiment.dev.0.preprocess"
        filepath_dev_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw-prep/sentiment.dev.1.preprocess"
        
        filepath_test_ref_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw-prep/sentiment.test_ref.0.preprocess"
        filepath_test_ref_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw-prep/sentiment.test_ref.1.preprocess"
        
        with open(filepath_train_0) as f: 
            sentences_train_0 = [line.strip() for line in f]
        with open(filepath_train_1) as f: 
            sentences_train_1 = [line.strip() for line in f]
            
        with open(filepath_dev_0) as f: 
            sentences_dev_0 = [line.strip() for line in f]
        with open(filepath_dev_1) as f: 
            sentences_dev_1 = [line.strip() for line in f]
            
        with open(filepath_test_ref_0) as f: 
            sentences_test_ref_0 = [line.strip() for line in f]
        with open(filepath_test_ref_1) as f: 
            sentences_test_ref_1 = [line.strip() for line in f]
            
        # idx = 43
        idx = 0
        # size = len(self.dataset_inputs)
        # size = 16
        # size = 500
        size = 10000
        
        sentences_train_0_sample = np.array(sentences_train_0)[train_0_selection].tolist()
        sentences_train_1_sample = np.array(sentences_train_1)[train_1_selection].tolist()
        
        
        # sentences_train_0_sample = list(np.random.choice(sentences_train_0, size=size, replace=False))
        # sentences_train_1_sample = list(np.random.choice(sentences_train_1, size=size, replace=False))
        # tst_inputs[('train', 'LABEL_0')] = sentences_test_ref_1[idx:(idx+size)]
        size = len(sentences_train_1)
        tst_inputs[('train', 'LABEL_0')] = sentences_train_1[idx:(idx+size)]
        # tst_inputs[('train', 'LABEL_0')] = sentences_train_1_sample
        tst_inputs[('train', 'LABEL_0')] = list(itertools.chain(*[[s for _ in range(self.n_repeats)] \
                                                                   for s in tst_inputs[('train', 'LABEL_0')]]))
        # tst_inputs[('train', 'LABEL_1')] = sentences_test_ref_0[idx:(idx+size)]
        size = len(sentences_train_0)
        tst_inputs[('train', 'LABEL_1')] = sentences_train_0[idx:(idx+size)]
        # tst_inputs[('train', 'LABEL_1')] = sentences_train_0_sample
        tst_inputs[('train', 'LABEL_1')] = list(itertools.chain(*[[s for _ in range(self.n_repeats)] \
                                                                   for s in tst_inputs[('train', 'LABEL_1')]]))
        # tst_inputs[('train', 'LABEL_0')] = sentences_train_1[idx:]
        # tst_inputs[('train', 'LABEL_1')] = sentences_train_0[idx:]
        test_size = 16
        tst_inputs[('infer', 'LABEL_0')] = sentences_dev_1[idx:(idx+test_size)]
        tst_inputs[('infer', 'LABEL_1')] = sentences_dev_0[idx:(idx+test_size)]
        
        return tst_inputs

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        # return tokens
        return [self._generator.tokenizer
                .convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, source_strings: List[str], prompt_strings: List[str]) -> List[str]:
        template = self._tst_templates[0]
        
#         return [
#             template.format(prompt=p) for s_1, p
#             in zip(source_strings, prompt_strings)]

        return [
            template.format(sentence_1=s_1, prompt=p) for s_1, p
            in zip(source_strings, prompt_strings)]

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        nlls, _ = compute_perplexities(
            sentences=sentences,
            model=self._generator.model,
            tokenizer=self._generator.tokenizer)
        # When the sentence has just one token,
        # the NLL/perplexity will be `NaN`.
        # Further, we use the negative NLL as the reward
        return -torch.nan_to_num(nlls, nan=10.0).mean()
    
    def _get_inputs(self, mode: str, target_labels: List[str]): 
        # data_0 = self._tst_inputs[(mode, 'LABEL_0')]
        # data_1 = self._tst_inputs[(mode, 'LABEL_1')]
        
        # idx_0 = self._tst_inputs_idx[(mode, 'LABEL_0')]
        # idx_1 = self._tst_inputs_idx[(mode, 'LABEL_1')]
        
        inputs = []
        indices = []
        
        nunique_inputs = len(set(self.dataset_inputs))
        for i, label in enumerate(target_labels): 
            idx = self._tst_inputs_idx[(mode, label)]
            data = self._tst_inputs[(mode, label)]
            
            # inputs.append(self.temp_input)
            # inputs.append(data[idx])
            if mode == 'train': 
                # inputs.append('thank you for a five star service.')
                # inputs.append(self.temp_input)
                # inputs.append(self.dataset_inputs[idx])
                inputs.append(data[idx])
                indices.append(int(idx // self.n_repeats))
            else: 
                # inputs.append(self.dataset_inputs[idx])
                inputs.append(data[idx])
                indices.append(int(idx))
            # print(idx)
            
                
            # inputs.append(self.dataset_inputs[idx])
            idx += 1
            idx %= len(data)
            self._tst_inputs_idx[(mode, label)] = idx
        
        return indices, inputs


    def evaluate_generated_texts(self,
                            generated_text: List[str], 
                            clf_model: Any, 
                            recon_model: Any, 
                            reference: str, 
                            target: str):
        classes = clf_model(generated_text, temperature=1.0, truncation = True)
        correct = [(c['label'] == target) for c in classes]
        probs = [(c['label'] == target) * c['score'] + 
                    (c['label'] != target) * (1 - c['score']) for c in classes]
        recon_scr = self.sbert_sim(reference.lower(), [g.lower() for g in generated_text])

        reward_list = [(rs + sa) / 2 for rs, sa in zip(recon_scr, probs)]
        sum_rewards = [(b * 100 + 1.05 * 100 * c) / (1 + 1) for b, c, p in zip(recon_scr, correct, probs)]
        logs = [{
                "ref_text": reference,
                "gen_text": gt,
                "target_label": target,
                "score": (rs + sa) / 2, 
                "recon": rs, 
                "clf_acc": sa} for rs, sa, gt in zip(recon_scr, probs, generated_text)]
        return sum_rewards, logs
    
    def postprocess_output(self, text, end_punct='"', start_punct=None): 
        # end_punct='}'
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


    def forward(self, 
                target_labels: List[str], 
                prompts: List[str],
                to_tensor: bool, 
                mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError
        assert all([label in ['LABEL_0', 'LABEL_1'] for label in target_labels])

        if mode == "train":
            num_return_sequences = self._num_return_sequences_train
        if mode == "infer":
            num_return_sequences = self._num_return_sequences_infer

        # - List of length `len(prompts)`
        #     - List of length `num_return_sequences`
        #         - Dict of {"generated_text": str}
        self.tokens_explored = self.tokens_explored.union(*[set(p.split()) for p in prompts])
#         print(len(self.tokens_explored) + ' tokens explored')
        source_indices, source_strings = self._get_inputs(mode, target_labels)
        # print(source_strings)
        # print('Reward:', source_strings)
        prompt_strings = self._convert_tokens_to_string(prompts)
        formatted_prompts = self._format_prompts(source_strings, prompt_strings)
        # input_length = len(self._generator.tokenizer(source_strings[0])['input_ids']) * 1.0
#         input_length = max([len(sent) for sent in self._generator.tokenizer(source_strings)['input_ids']]) + 5

        if mode == 'infer': 
            print('Infer Reward Inputs:', source_strings)

        if mode == 'train': 
            self._counter += 1
#         print(input_length)

        from torch.utils.data import Dataset
        class MyDataset(Dataset):
            def __init__(self, x):
                self.samples = x

            def __getitem__(self,index):
                return self.samples[index]

            def __len__(self):
                return len(self.samples)

        n_reward = 32
        k_reward = 4
        N = n_reward * k_reward
        X = MyDataset(formatted_prompts)
        generator_outputs: List[List[Dict[str, Any]]] = self._generator(
            # formatted_prompts,
            X,
            # max_length=self._max_length,
            # max_new_tokens=input_length,
            pad_token_id=50256,
            top_k=10,
            temperature=1,
            # logits_processor=self.logits_processor_list,
            # top_p=0.6,
            # num_beams=N,
            # do_sample=False,
            # no_repeat_ngram_size=2,
            num_return_sequences=N,
            # Only return generated text, without the prompt
            return_full_text=False)
            
#         generator_outputs: List[List[Dict[str, Any]]] = self._generator(
#             # formatted_prompts,
#             X,
#             # max_length=self._max_length,
#             # max_new_tokens=input_length,
#             pad_token_id=50256,
#             # top_k=10,
#             # temperature=1,
#             # logits_processor=self.logits_processor_list,
#             # top_p=0.6,
#             num_beams=N,
#             do_sample=False,
#             # no_repeat_ngram_size=2,
#             num_return_sequences=N,
#             # Only return generated text, without the prompt
#             return_full_text=False)

        rewards: List[FloatTensor] = []
        input_rewards: Dict(str, List(float)) = defaultdict(list)
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for batch_index, out in enumerate(generator_outputs):
        #range(len(prompts)):
                        
#             input_length = len(self._generator.tokenizer(source_strings[batch_index])['input_ids']) + 5
#             print(input_length)
#             generator_outputs: List[List[Dict[str, Any]]] = self._generator(
#                 formatted_prompts[batch_index],
#                 max_new_tokens=input_length,
#                 pad_token_id=50256,
#                 top_k=10,
#                 logits_processor=self.logits_processor_list,
#                 num_return_sequences=16,
#                 temperature=self.temperature,
#                 # Only return generated text, without the prompt
#                 return_full_text=False)
            
            generated_texts = []
            # for output in generator_outputs[batch_index]: 
            for output in out: 
#             for output in generator_outputs: 
                text = output["generated_text"]
#                 try: 
#                     end = text.index(self.END_PUNCT)
#                 except ValueError: 
#                     end = len(text)
#                 generated_texts.append(text[:end])
                # end_punct = '}'
                end_punct = '"'
                # start_punct = '{'
                start_punct = None
                generated_texts.append(self.postprocess_output(text, 
                                                               end_punct=end_punct,
                                                               start_punct=start_punct))
            
            # if mode == "infer": 
            #     print(f"Formatted Prompt: {formatted_prompts[batch_index]};",
            #           f"Output: {generated_texts[0]}")

            # - List of length `len(generated_texts)`
            #     - Dict of {
            #         "labels": List of length `num_topics`,
            #         "scores": List of length `num_topics`,
            #         "sequence": str,
            #     }
            try:
                # reference_texts = [source_strings[batch_index] for _ in generator_outputs[batch_index]]
#                 reference_texts = [source_strings[batch_index] for _ in generator_outputs]
                reference_texts = [source_strings[batch_index] for g in generated_texts]
                # generated_texts = [g for g in generated_texts if len(g) > 0]
                
                check_Xs_Ys_sizes(generated_texts, reference_texts)
                
                # Using a faster BLEU implementation during training
                # `sacrebleu` is ~3X faster than `lightning`
                # `sacrebleu-parallel` is ~3X faster than `sacrebleu`
                
#                 bleus = [
#                     scb.sentence_bleu(
#                         hypothesis=x,
#                         references=[y])
#                     for x, y in zip(
#                         generated_texts,
#                         reference_texts)
#                 ]
#                 eps = 1e-3
#                 bleu_rewards = [b.score + eps for b in bleus]
#                 mean_bleu = torch.tensor(bleu_rewards).float().mean()
#                 quantities_to_log['bleu'].append(mean_bleu)

                ### The bleus here are temporarily sbert scores ###
#                 sberts = self.sbert_sim(reference_texts[0].lower(), 
#                             [g.lower() for g in generated_texts])
#                 sbert_rewards = [s * 100 for s in sberts]
#                 sbert = torch.tensor(sbert_rewards).float().mean()
#                 quantities_to_log["sbert"].append(sbert)

                compute_bertscore = True
                if compute_bertscore:
                    bertscore_f1 = self._bert_scorer.score(generated_texts, 
                                                           reference_texts)[2]
                    bertscore_rewards = [max(b, 0) for b in (bertscore_f1 * 100).tolist()]
                    bertscore = torch.tensor(bertscore_rewards).float().mean()
                    quantities_to_log['bertscore'].append(bertscore)
                
#                 generated_encoding = (self._generator
#                                            .tokenizer(generated_texts, 
#                                                       padding=True,
#                                                       return_tensors='pt'))
#                 generated_lengths = generated_encoding['attention_mask'].sum(dim=1).tolist()
                
#                 reference_encoding = (self._generator
#                                            .tokenizer(reference_texts, 
#                                                       padding=True,
#                                                       return_tensors='pt'))
#                 reference_lengths = reference_encoding['attention_mask'].sum(dim=1).tolist()
#                 length_penalties = [np.exp(1 - max(g, r) / min(g, r)) if min(g, r) > 0 else 0 \
#                                     for g, r in zip(generated_lengths, reference_lengths)]
                
                compute_recon = True
                if compute_recon:
                    # recon_rewards = [lp**0.25 * b for lp, b in zip(length_penalties, bertscore_rewards)]
                    recon_rewards = bertscore_rewards
                    # recon_rewards = [(3 * bert + bleu) / 4 for bert, bleu in zip(bertscore_rewards, bleu_rewards)]
                    # recon_rewards = bleu_rewards
                    # recon_rewards = sbert_rewards
                    recon = torch.tensor(recon_rewards).float().mean()
                    quantities_to_log["recon"].append(recon)
                
                compute_sentiment = True
                if compute_sentiment:
                    X_output = MyDataset(generated_texts)
                    classes = self._classifier(X_output, truncation=True)
                    # print(clf_results)
                    label = target_labels[batch_index]
                    # label = 'Negative' if label == 'LABEL_0' else 'Positive' if label == 'LABEL_1' else None
                    # print(label)

    #                 # class_label_mapping = ['Negative', 'Neutral', 'Positive']
    #                 class_label_mapping = ['LABEL_0', 'LABEL_1']
    #                 classes = [np.argmax([d['score'] for d in result]) for result in clf_results]
    #                 # print(classes)
    #                 class_labels = [class_label_mapping[c] for c in classes]
    #                 correct = [c == label for c in class_labels]
    #                 # print(correct)

    #                 probs = [{d['label']: d['score'] for d in result}[label] for result in clf_results]
    #                 # print(probs)


                    correct = [(c['label'] == label) for c in classes]
                    acc = torch.tensor(correct).float().mean()
                    quantities_to_log['acc'].append(acc)
                    probs = [(c['label'] == label) * c['score'] + (c['label'] != label) * (1 - c['score']) for c in classes]
                    style = torch.tensor(probs).float().mean()
                    quantities_to_log['style'].append(style)
                
                compute_sum_reward = True
                if compute_sum_reward:
                    if self._warm_up_reward:
                        sum_rewards = [b for b, c, p in zip(recon_rewards, correct, probs)]
                        reward = torch.tensor(sum_rewards).float().mean()
                        if reward > 80 and self._counter >= 1000:
                            self._warm_up_reward = False
                            print("Warm up reward ends")
                            top_index = 0
                    else:
                        recon_weight = 1
                        style_weight = 1
    #                     sum_rewards = [(recon_weight * r + style_weight * 100 * c) / (recon_weight + style_weight) \
    #                                    for r, c, p in zip(recon_rewards, correct, probs)]
                        sum_rewards = [(recon_weight * r + style_weight * 100 * p) / (recon_weight + style_weight) \
                                       for r, c, p in zip(recon_rewards, correct, probs)]
                        # Perform reward clipping
                        reward_clipping = False
                        if reward_clipping: 
                            clip_reward = 85
                            sum_rewards = [min(r, clip_reward) for r in sum_rewards]

                        # Monte Carlo k_reward times and average
                        mc_avg = True
                        if mc_avg:
                            l = len(sum_rewards)
                            k = k_reward
                            segmented_sum_rewards = [sum_rewards[i*l//k:(i+1)*l//k] for i in range(k)]

                            mean_sum_reward = torch.tensor(segmented_sum_rewards).float().mean()
                            values, indices = torch.tensor(segmented_sum_rewards).float().max(axis=1)

                            list_values = [segmented_sum_rewards[i][index] for i, index in enumerate(indices)]
                            input_rewards[reference_texts[0]] += list_values
                            max_sum_reward = values.mean()

                            max_reward_value = max(list_values)

                        comb_avg = False
                        assert not (mc_avg and comb_avg)
                        if comb_avg: 
                            sorted_sum_rewards = list(sorted(sum_rewards))
                            n = n_reward
                            # N = n_reward*k_reward
                            N = N
                            NCn = math.comb(N, n)

                            max_sum_reward = sum([math.comb(i-1, n-1) * sum_rewards[i-1] / NCn for i in range(n, N+1)])
                            max_sum_reward = torch.tensor(max_sum_reward).float()
                            input_rewards[reference_texts[0]] += [max_sum_reward]
                            max_reward_value = max(sum_rewards)
                        
                    
                        # mean_sum_reward = torch.tensor(sum_rewards).float().mean()
                        # max_sum_reward = torch.tensor(sum_rewards).float().max()

    #                     log_sum_rewards = [(recon_weight * np.log(r / 100) + style_weight * np.log(p)) / (recon_weight + style_weight) \
    #                                    for r, c, p in zip(recon_rewards, correct, probs)]
    #                     # print(log_sum_rewards)
    #                     mean_log_sum_reward = torch.tensor(log_sum_rewards).float().nanmean()
    #                     max_log_sum_reward = torch.nan_to_num(torch.tensor(log_sum_rewards), nan=-100).float().max()

    #                     prod_rewards = [(b * c) for b, c, p in zip(recon_rewards, correct, probs)]
    #                     mean_prod_reward = torch.tensor(prod_rewards).float().mean()
    #                     max_prod_reward = torch.tensor(prod_rewards).float().max()

                        # top_index = 0
                        # top_index = sum_rewards.index(max_sum_reward)
                        # list_values = [segmented_sum_rewards[i][index] for i, index in enumerate(indices)]
                        # input_rewards[reference_texts[0]] += list_values

                        top_index = sum_rewards.index(max_reward_value)
                        # top_index = log_sum_rewards.index(max_log_sum_reward)
                        # top_index = prod_rewards.index(max_prod_reward)
                        # reward = recon
                        # reward = mean_bleu
                        # reward = sbert
                        # reward = ctc
                        # reward = bertscore
                        # reward = max_log_sum_reward
                        # reward = mean_log_sum_reward
                        # reward = max_sum_reward * max(torch.exp(1.5 * (max_sum_reward - 80) / 100), 1)

                        reward = max_sum_reward
                        # clip_reward = 85
                        # reward = torch.min(torch.tensor([reward, clip_reward]))
                        
                        quantities_to_log["sum_reward"].append(max_sum_reward)
                        mean_reward = torch.tensor(sum_rewards).float().mean()
                        quantities_to_log["mean_reward"].append(mean_reward)
                        top_recon = torch.tensor(recon_rewards[top_index]).float()
                        quantities_to_log["top_recon"].append(top_recon)
                        # top_acc = torch.tensor(correct[top_index]).float()
                        # quantities_to_log["top_acc"].append(top_acc)
                        top_style = torch.tensor(probs[top_index]).float()
                        quantities_to_log["top_style"].append(top_style)
                        
                        print(self._counter, '|',
                              prompts[batch_index], '|', 
                              formatted_prompts[batch_index], '|', 
                              generated_texts[top_index], '|', 
                              # 'SBERT:', round(sbert_rewards[top_index], 2), '|',
                              # 'BLEU:', round(bleu_rewards[top_index], 2), '|',
                              'BERTScore:', round(bertscore_rewards[top_index], 2), '|',
                              # 'ACC:', round(correct[top_index], 2), '|',
                              'STYLE:', round(probs[top_index], 2), '|',
                              'Sum Reward:', round(sum_rewards[top_index], 2), '|',
                              'Reward:', round(reward.item(), 2))
                            
                else:
                    reward = bertscore

                # idx = self.dataset_inputs.index(reference_texts[0])
                # idx_2 = source_indices[batch_index]
                # assert idx == idx_2, f'{idx} {idx_2}'
                # print(idx_2, len(self.dataset_inputs))
                # quantities_to_log[f'example_{idx_2}_sum_reward'].append(max_sum_reward)
                
                input_zscore = False
                if input_zscore:
                    est_mean = self._tst_input_reward_meanstd[reference_texts[0]].mean
                    est_std = np.sqrt(self._tst_input_reward_meanstd[reference_texts[0]].var)
                    quantities_to_log[f'example_{idx%len(self.dataset_inputs)}_est_mean'].append(torch.tensor(est_mean).float())
                    quantities_to_log[f'example_{idx%len(self.dataset_inputs)}_est_std'].append(torch.tensor(est_std).float())
                    
                min_max_norm = False
                if min_max_norm:
                    # min_max_warmup_steps = 10
                    min_max_warmup_steps = float('inf')
                    if self._counter <= min_max_warmup_steps: 
                        if reference_texts[0] not in self._tst_input_reward_ranges: 
                            print(f'creating entry for {reference_texts[0]}')
                            self._tst_input_reward_ranges[reference_texts[0]] = (min(list_values), max(list_values))
                        prev_min, prev_max = self._tst_input_reward_ranges[reference_texts[0]]

                        curr_min = min(prev_min, min(list_values))
                        alpha = 0.1
                        # alpha = 1
                        if max(list_values) > prev_max: 
                            curr_max = alpha * max(list_values) + (1 - alpha) * prev_max
                        else: 
                            curr_max = prev_max
                        # curr_min, curr_max = min(prev_min, min(list_values)), max(prev_max, max(list_values))
                        self._tst_input_reward_ranges[reference_texts[0]] = (prev_min, curr_max)

                    relative_reward = True
                    input_min, input_max = self._tst_input_reward_ranges[reference_texts[0]]
                    if relative_reward: reward = (max_sum_reward - input_min) / (input_max - input_min)
                    
                    quantities_to_log[f'example_{idx%len(self.dataset_inputs)}_min'].append(torch.tensor(input_min).float())
                    quantities_to_log[f'example_{idx%len(self.dataset_inputs)}_max'].append(torch.tensor(input_max).float())
                
                
                # quantities_to_log["mean_prod_reward"].append(mean_prod_reward)
                # quantities_to_log["max_prod_reward"].append(max_prod_reward)
                
                # quantities_to_log["mean_log_sum_reward"].append(mean_log_sum_reward)
                # quantities_to_log["max_log_sum_reward"].append(max_log_sum_reward)
                
                # quantities_to_log["num_input_ranges"].append(torch.tensor(len(self._tst_input_reward_ranges)).float())
                
                

                # quantities_to_log["sample_size"].append(torch.tensor(self.sample_size).float())
                quantities_to_log["num_tokens_explored"].append(torch.tensor(len(self.tokens_explored)).float())
                
                # print(source_indices)
                # print(self.dataset_inputs)
                # print(reference_texts[0])
                
                if self._include_perplexity is True:
                    nll_reward = (
                        self._compute_nll_reward(
                            sentences=generated_texts))
                    reward = reward + nll_reward
                    quantities_to_log["nll"].append(nll_reward)

                # print(source_strings)
                
                
                
                rewards.append(reward)

            except ValueError as err:
                # This happens when the generated text itself includes the
                # `</s>` token, which does happen and will cause the classifier to fail.
                # So we just ignore this error and give a score of zero for this batch.
                if str(err) != "All examples must have the same number of <eos> tokens.":
                    raise err

                click.secho("Encountered an error, skipping ...", bg="red")
                rewards.append(torch.tensor(0.).to(device))

        # Record prompt performance at different sample size in validation time.
#         if mode == "infer":
#             for sample_size in self.val_sample_sizes:
#                 val_gen_outputs: List[List[Dict[str, Any]]] = self._generator(
#                     formatted_prompts,
#                     max_new_tokens=input_length,
#                     pad_token_id=50256,
#                     num_return_sequences=sample_size,
#                     temperature=self.temperature,
#                     top_k=10,
#                     return_full_text=False)
                
#                 top_rewards = []
#                 mean_rewards = []
#                 for batch_index in range(len(prompts)):
#                     generated_texts = []
#                     for output in val_gen_outputs[batch_index]: 
#                         text = output["generated_text"]
#                         try: 
#                             end = text.index('"')
#                         except ValueError: 
#                             end = len(text)
#                         generated_texts.append(text[:end])
#                     sum_rewards, logs = self.evaluate_generated_texts(
#                         generated_texts,
#                         self._classifier,
#                         self.sbert,
#                         source_strings[batch_index],
#                         target_labels[batch_index]
#                         )
#                     top_reward = torch.tensor(sum_rewards).float().max()
#                     top_rewards.append(top_reward)
#                     mean_reward = torch.tensor(sum_rewards).float().mean()
#                     mean_rewards.append(mean_reward)
#                     quantities_to_log[f"{sample_size}_top_reward"].append(top_reward)
#                     quantities_to_log[f"{sample_size}_mean_reward"].append(mean_reward)
#                 top_reward_var = torch.var(torch.tensor(top_rewards))
#                 mean_reward_var = torch.var(torch.tensor(mean_rewards))
#                 quantities_to_log[f"{sample_size}_top_reward_var"].append(top_reward_var)
#                 quantities_to_log[f"{sample_size}_mean_reward_var"].append(mean_reward_var)

        rewards_tensor = torch.stack(rewards)
    
        batch_zscore = True
        if batch_zscore:
            by_input = True
            improved = True
            if by_input and not improved:
                source_indices_tensor = torch.tensor(source_indices)
                M = torch.zeros(source_indices_tensor.max() + 1, len(rewards_tensor))
                M[source_indices_tensor, torch.arange(len(rewards_tensor))] = 1
                M = torch.nn.functional.normalize(M, p=1, dim=1)
                idx_mean = M @ rewards_tensor
                SS = (rewards_tensor - idx_mean[source_indices_tensor])**2
                idx_std = M @ SS
                print(idx_mean)
                print(idx_std)
                
                rewards_tensor = (rewards_tensor - idx_mean.float()[source_indices_tensor]) / \
                                 (idx_std.float()[source_indices_tensor] + 1e-4)
                
            elif by_input and improved: 
                input_max_reward_means = {k: np.mean(v) for k, v in input_rewards.items()}
                input_max_reward_stds = {k: np.std(v) for k, v in input_rewards.items()}
                idx_means = torch.tensor([input_max_reward_means[s] for s in source_strings]).float()
                idx_stds = torch.tensor([input_max_reward_stds[s] for s in source_strings]).float()
                print(idx_means)
                print(idx_stds)
                
                rewards_tensor = (rewards_tensor - idx_means) / (idx_stds + 1e-4)
                
                
            else:
                rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-4)
            
        input_zscore = False
        input_zscore = input_zscore
        if input_zscore: 
            for s in set(source_strings):
                print(s, input_rewards[s])
                self._tst_input_reward_meanstd[s].update(np.array(input_rewards[s]))
            reward_means = torch.tensor([self._tst_input_reward_meanstd[s].mean for s in source_strings])
            reward_stds = torch.tensor([np.sqrt(self._tst_input_reward_meanstd[s].var) for s in source_strings])
            # print(reward_means, reward_stds)
            rewards_tensor = (rewards_tensor - reward_means) / (reward_stds + 1e-4)
            # print(rewards_tensor)
            rewards_tensor = rewards_tensor.float()
            
        quantities_to_log["reward_var"].append(torch.var(torch.tensor(rewards)))
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        self._return_intermediate_outputs = False
        if self._return_intermediate_outputs is True:
            rewards_log["quantities_to_log"] = quantities_to_log  # type: ignore
            rewards_log["formatted_prompts"] = formatted_prompts  # type: ignore
            rewards_log["generator_outputs"] = generator_outputs  # type: ignore
            # rewards_log["all_classifier_outputs"] = all_classifier_outputs  # type: ignore

        # Exponential Decay Annealing
        # if self._counter % 1000 == 0 and self.sample_size > 16:
        #     self.sample_size = int(self.sample_size / 2)

        # Piecewise Linear Annealing with warmup
        if self.sample_size_annealing and self._counter > 500 and \
             self._counter % 30 == 0 and self.sample_size > 16:
            self.sample_size = int(self.sample_size - 1)

        # Cosine Annealing with warmup
        # if self.sample_size_annealing and self._counter > 500 and \
        #     self.sample_size > 16:
        #     self.sample_size = int(16 + 56 * (1 + cos(pi * ((self._counter - 500) / 5000))))

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
    
import pandas as pd
@torch.no_grad()
def compute_triggered_output_perplexities(
        triggers: List[str],
        sentences: List[str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
) -> Tuple[FloatTensor, FloatTensor]:

    nlls = []
    for trigger, sentence in zip(triggers, sentences):
        # print(trigger, sentence)
        tensor_trigger = tokenizer(trigger, return_tensors='pt').input_ids.to(device)
        tensor_sentence = tokenizer(sentence, return_tensors='pt').input_ids.to(device)
        
        mask_out = -100 * torch.ones_like(tensor_trigger)
        lm_input = torch.cat((tensor_trigger, tensor_sentence), dim=1)
        mask_and_target = torch.cat((mask_out, tensor_sentence), dim=1)
        
        try:
            # labels **are shifted** inside the model
            outputs = model(
                lm_input,
                labels=mask_and_target)
            nll = outputs[0]
        except RuntimeError:
            # Could happen when the input is empty
            nll = torch.tensor(float("nan")).to(device)

        nlls.append(nll)

    stacked_nlls = torch.stack(nlls, dim=0)
    return stacked_nlls, stacked_nlls.exp()

class GPT2TriggerReward(object):
    TST_TEMPLATES_FILE_NAME = "/jupyter/prompt-generationsoft-Q-learning-for-text-generation/experiments/tst-templates-no-task.txt"
    TST_CLF_CONFIG = dict(model=("/jupyter/prompt-generationsoft-Q-learning-for-text-generation/experiments/yelp_sentiment_classifier/"
                                 "results-bert-base/checkpoint-10410/"),
                          tokenizer='bert-base-uncased')

    def __init__(
            self,
            max_length: int = 60,
            num_return_sequences_train: int = 2,
            num_return_sequences_infer: int = 100,
            # topic_scores_aggregator: Optional[Callable[[List[float]], Union[float, np.number]]] = None,
            include_perplexity: bool = True,
            return_intermediate_outputs: bool = False,
    ) -> None:

        if include_perplexity is True:
            sql_utils.colorful_warning("Adding Perplexity-based Reward", bg="blue")

        sql_utils.colorful_warning(f"max_length={max_length}", bg="blue")

        # https://huggingface.co/gpt2
        # https://huggingface.co/facebook/bart-large-mnli
        self._generator = pipeline(
            "text-generation",
            model="gpt2",
            device=0)

        self._max_length = max_length
        self._num_return_sequences_train = num_return_sequences_train
        self._num_return_sequences_infer = num_return_sequences_infer
        self._tst_templates = self.load_tst_templates()
        self._tst_inputs = self._load_tst_inputs()
        self._tst_inputs_idx = {('train', 'trigger'): 0, 
                                ('infer', 'trigger'): 0}

        # Technically, adding perplexity-based reward will break
        # the scale, but we will ignore this for now since
        # this number is relatively small.
        self._include_perplexity = include_perplexity
        # Do not set is to `True` during training, use it for debugging.
        self._return_intermediate_outputs = return_intermediate_outputs
        
    def load_tst_templates(self) -> List[str]:
#         with open(self.TST_TEMPLATES_FILE_NAME) as f: 
#             tst_templates = [d.strip() for d in f.readlines()]
#         return tst_templates
        # temp_tst_template = '{prompt} Sentence 1: "{sentence_1}" Sentence 2: "'
        temp_tst_template = '{prompt} "{sentence_1}" "'
        return [temp_tst_template]
        
    
    def _load_tst_inputs(self) -> Dict[Tuple[Any], List[str]]: 
        tst_inputs = {}
        
        data_path = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-trigger/yelp-gold.csv"
        df = pd.read_csv(data_path)
        cutoff = int(0.8 * len(df))
        
        df_train = df.iloc[:cutoff]
        df_infer = df.iloc[cutoff:]
            
        tst_inputs[('train', 'trigger')] = (df_train.text_1.tolist(), df_train.text_2.tolist())
        tst_inputs[('infer', 'trigger')] = (df_infer.text_1.tolist(), df_infer.text_2.tolist())
        
        return tst_inputs

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._generator.tokenizer
                .convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, 
                        source_strings: List[str], 
                        prompt_strings: List[str]) -> List[str]:
        template = self._tst_templates[0]
        
#         return [
#             template.format(prompt=p) for s_1, p
#             in zip(source_strings, prompt_strings)]

        return [
            template.format(sentence_1=s_1, prompt=p) for s_1, p
            in zip(source_strings, prompt_strings)]

    def _compute_nll_reward(self, triggers: List[str], sentences: List[str]) -> FloatTensor:
        nlls, _ = compute_triggered_output_perplexities(
            triggers=triggers,
            sentences=sentences,
            model=self._generator.model,
            tokenizer=self._generator.tokenizer)
        # When the sentence has just one token,
        # the NLL/perplexity will be `NaN`.
        # Further, we use the negative NLL as the reward
        return -torch.nan_to_num(nlls, nan=10.0).mean()
    
    def _get_inputs_and_targets(self, mode: str, target_labels: List[str]): 
        # data_0 = self._tst_inputs[(mode, 'LABEL_0')]
        # data_1 = self._tst_inputs[(mode, 'LABEL_1')]
        
        # idx_0 = self._tst_inputs_idx[(mode, 'LABEL_0')]
        # idx_1 = self._tst_inputs_idx[(mode, 'LABEL_1')]
        
        inputs, targets = [], []
        for i, label in enumerate(target_labels): 
            # idx = self._tst_inputs_idx[(mode, label)]
            # text_1, text_2 = self._tst_inputs[(mode, label)]
            
            text_1 = ['thank you for a five star service .']
            text_2 = ['sorry for a five star service .']
            
            inputs.append(text_1[0])
            targets.append(text_2[0] + '"')
            # idx += 1
            # idx %= len(text_1)
            # self._tst_inputs_idx[(mode, label)] = idx
        
        return inputs, targets

    def forward(self, target_labels: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError
        # assert all([label in ['trigger'] for label in target_labels])

        if mode == "train":
            num_return_sequences = self._num_return_sequences_train
        if mode == "infer":
            num_return_sequences = self._num_return_sequences_infer

        # - List of length `len(prompts)`
        #     - List of length `num_return_sequences`
        #         - Dict of {"generated_text": str}
        source_strings, target_strings = self._get_inputs_and_targets(mode, target_labels)
        prompt_strings = self._convert_tokens_to_string(prompts)
        formatted_prompts = self._format_prompts(source_strings, prompt_strings)
        
#         generator_outputs: List[List[Dict[str, Any]]] = self._generator(
#             formatted_prompts,
#             max_length=self._max_length,
#             num_return_sequences=num_return_sequences,
#             # Only return generated text, without the prompt
#             return_full_text=False)

        rewards: List[FloatTensor] = []
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for formatted_prompt, target in zip(formatted_prompts, target_strings): 
            nll_reward = (
                    self._compute_nll_reward(
                        triggers=[formatted_prompt],
                        sentences=[target]))
            reward = nll_reward
            quantities_to_log["nll"].append(nll_reward)
            
            print(formatted_prompt + target + '"', 'NLL:', nll_reward.item())

            rewards.append(reward)
            
        rewards_tensor = torch.stack(rewards)
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        if self._return_intermediate_outputs is True:
            rewards_log["quantities_to_log"] = quantities_to_log  # type: ignore
            rewards_log["formatted_prompts"] = formatted_prompts  # type: ignore
            rewards_log["generator_outputs"] = generator_outputs  # type: ignore
            # rewards_log["all_classifier_outputs"] = all_classifier_outputs  # type: ignore

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
    
class GPT2SentimentBERTScoreNoInputReward(object):
    TST_TEMPLATES_FILE_NAME = "/workspace/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task.txt"
    # TST_TEMPLATES_FILE_NAME = "/workspace/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task-no-source.txt"
    TST_CLF_CONFIG = dict(model=("/workspace/soft-Q-learning-for-text-generation/experiments/yelp_sentiment_classifier/"
                                 "results-bert-base/checkpoint-10410/"),
                          tokenizer='bert-base-uncased')

    def __init__(
            self,
            max_length: int = 60,
            num_return_sequences_train: int = 2,
            num_return_sequences_infer: int = 100,
            include_perplexity: bool = True,
            return_intermediate_outputs: bool = False,
    ) -> None:

        if include_perplexity is True:
            sql_utils.colorful_warning("Adding Perplexity-based Reward", bg="blue")

        sql_utils.colorful_warning(f"max_length={max_length}", bg="blue")

        # https://huggingface.co/gpt2
        # https://huggingface.co/facebook/bart-large-mnli
        self._generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=0)
        self._classifier = pipeline(
            "sentiment-analysis",
            model=self.TST_CLF_CONFIG['model'],
            tokenizer=self.TST_CLF_CONFIG['tokenizer'],
            device=0)
        self._bert_scorer = BERTScorer('bert-base-uncased', device=0)

        self._max_length = max_length
        self._num_return_sequences_train = num_return_sequences_train
        self._num_return_sequences_infer = num_return_sequences_infer
        self._tst_templates = self.load_tst_templates()
        self._tst_inputs = self._load_tst_inputs()
        self._tst_inputs_idx = {('train', 'LABEL_0'): 0, 
                                ('train', 'LABEL_1'): 0,
                                ('infer', 'LABEL_0'): 0,
                                ('infer', 'LABEL_1'): 0}

        # Technically, adding perplexity-based reward will break
        # the scale, but we will ignore this for now since
        # this number is relatively small.
        self._include_perplexity = include_perplexity
        # Do not set is to `True` during training, use it for debugging.
        self._return_intermediate_outputs = return_intermediate_outputs
        
    def load_tst_templates(self) -> List[str]:
        with open(self.TST_TEMPLATES_FILE_NAME) as f: 
            tst_templates = [d.strip() for d in f.readlines()]
        return tst_templates
    
    def _load_tst_inputs(self) -> Dict[Tuple[Any], List[str]]: 
        tst_inputs = {}
        # tokenizer = self._generator.tokenizer
        filepath_train_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.0"
        filepath_train_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.1"
        filepath_dev_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.0"
        filepath_dev_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.1"
        
        with open(filepath_train_0) as f: 
            sentences_train_0 = [line.strip() for line in f]
        with open(filepath_train_1) as f: 
            sentences_train_1 = [line.strip() for line in f]
        with open(filepath_dev_0) as f: 
            sentences_dev_0 = [line.strip() for line in f]
        with open(filepath_dev_1) as f: 
            sentences_dev_1 = [line.strip() for line in f]
            
        tst_inputs[('train', 'LABEL_0')] = sentences_train_1
        tst_inputs[('train', 'LABEL_1')] = sentences_train_0
        tst_inputs[('infer', 'LABEL_0')] = sentences_dev_1[:5]
        tst_inputs[('infer', 'LABEL_1')] = sentences_dev_0[:5]
        
        return tst_inputs

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._generator.tokenizer
                .convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, source_strings: List[str], prompt_strings: List[str]) -> List[str]:
        template = self._tst_templates[0]
        
#         return [
#             template.format(prompt=p) for s_1, p
#             in zip(source_strings, prompt_strings)]

        return [
            template.format(sentence_1=s_1, prompt=p) for s_1, p
            in zip(source_strings, prompt_strings)]

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        nlls, _ = compute_perplexities(
            sentences=sentences,
            model=self._generator.model,
            tokenizer=self._generator.tokenizer)
        # When the sentence has just one token,
        # the NLL/perplexity will be `NaN`.
        # Further, we use the negative NLL as the reward
        return -torch.nan_to_num(nlls, nan=10.0).mean()
    
    def _get_inputs(self, mode: str, target_labels: List[str]): 
        # data_0 = self._tst_inputs[(mode, 'LABEL_0')]
        # data_1 = self._tst_inputs[(mode, 'LABEL_1')]
        
        # idx_0 = self._tst_inputs_idx[(mode, 'LABEL_0')]
        # idx_1 = self._tst_inputs_idx[(mode, 'LABEL_1')]
        
        inputs = []
        for i, label in enumerate(target_labels): 
            idx = self._tst_inputs_idx[(mode, label)]
            data = self._tst_inputs[(mode, label)]
            
            inputs.append(data[0])
            idx += 1
            idx %= len(data)
            self._tst_inputs_idx[(mode, label)] = idx
        
        return inputs

    def forward(self, target_labels: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError
        assert all([label in ['LABEL_0', 'LABEL_1'] for label in target_labels])

        if mode == "train":
            num_return_sequences = self._num_return_sequences_train
        if mode == "infer":
            num_return_sequences = self._num_return_sequences_infer

        # - List of length `len(prompts)`
        #     - List of length `num_return_sequences`
        #         - Dict of {"generated_text": str}
        source_strings = self._get_inputs(mode, target_labels)
        prompt_strings = self._convert_tokens_to_string(prompts)
        formatted_prompts = self._format_prompts(source_strings, prompt_strings)
        
        generator_outputs: List[List[Dict[str, Any]]] = self._generator(
            formatted_prompts,
            max_length=self._max_length,
            num_return_sequences=num_return_sequences,
            # Only return generated text, without the prompt
            return_full_text=False)

        rewards: List[FloatTensor] = []
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for batch_index in range(len(prompts)):
            # generated_texts = [
            #     output["generated_text"] for output in
            #     generator_outputs[batch_index]]
            
            generated_texts = []
            for output in generator_outputs[batch_index]: 
                text = output["generated_text"]
                try: 
                    end = text.index('"')
                except ValueError: 
                    end = len(text)
                generated_texts.append(text[:end])
            
            if mode == "infer": 
                print(f"Formatted Prompt: {formatted_prompts[batch_index]};",
                      f"Output: {generated_texts[0]}")

            # - List of length `len(generated_texts)`
            #     - Dict of {
            #         "labels": List of length `num_topics`,
            #         "scores": List of length `num_topics`,
            #         "sequence": str,
            #     }
            try:
                reference_texts = [source_strings[batch_index] for _ in generator_outputs[batch_index]]
                
                check_Xs_Ys_sizes(generated_texts, reference_texts)
                
                # Using a faster BLEU implementation during training
                # `sacrebleu` is ~3X faster than `lightning`
                # `sacrebleu-parallel` is ~3X faster than `sacrebleu`
                bleus = [
                    scb.sentence_bleu(
                        hypothesis=x,
                        references=[y])
                    for x, y in zip(
                        generated_texts,
                        reference_texts)
                ]
                bleu_rewards = [b.score for b in bleus]
                
                bleu = torch.tensor(bleu_rewards).float().mean()
                quantities_to_log["bleu"].append(bleu)
                
                bertscore_f1 = self._bert_scorer.score(generated_texts, 
                                                       reference_texts)[2]
                bertscore_f1 = bertscore_f1.float().mean()
                quantities_to_log['bertscore_f1'].append(bertscore_f1)

                classes = self._classifier(generated_texts, truncation=True)
                label = target_labels[batch_index]
                correct = [(c['label'] == label) for c in classes]
                acc = torch.tensor(correct).float().mean()
                quantities_to_log['acc'].append(acc)
                
                
                # style_rewards = [100 * ((c['label'] == label) * c['score'] + (c['label'] != label) * (1 - c['score']))\
                #                   for c in classes]
                # style_strength = torch.tensor(style_rewards).float().mean()
                # quantities_to_log['style_strength'].append(style_strength)
                
                # f1_rewards = [2 * (b * a) / (b + a) for b, a in zip(bleu_rewards, style_rewards)]
                # reward = torch.tensor(f1_rewards).float().mean()
                # quantities_to_log["f1"].append(reward)
                reward = (bertscore_f1**2) * acc * bleu
                quantities_to_log['prod_reward'].append(reward)
                
                if label == 'LABEL_0': quantities_to_log['acc_0'].append(acc)
                elif label == 'LABEL_1': quantities_to_log['acc_1'].append(acc)
                
                if self._include_perplexity is True:
                    nll_reward = (
                        self._compute_nll_reward(
                            sentences=generated_texts))
                    reward = reward + nll_reward
                    quantities_to_log["nll"].append(nll_reward)

                rewards.append(reward)

            except ValueError as err:
                # This happens when the generated text itself includes the
                # `</s>` token, which does happen and will cause the classifier to fail.
                # So we just ignore this error and give a score of zero for this batch.
                if str(err) != "All examples must have the same number of <eos> tokens.":
                    raise err

                click.secho("Encountered an error, skipping ...", bg="red")
                rewards.append(torch.tensor(0.).to(device))

        rewards_tensor = torch.stack(rewards)
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        if self._return_intermediate_outputs is True:
            rewards_log["quantities_to_log"] = quantities_to_log  # type: ignore
            rewards_log["formatted_prompts"] = formatted_prompts  # type: ignore
            rewards_log["generator_outputs"] = generator_outputs  # type: ignore
            # rewards_log["all_classifier_outputs"] = all_classifier_outputs  # type: ignore

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
    
class GPT2BLEUSentimentReward(object):
    TST_CLF_DIR = ("/workspace/soft-Q-learning-for-text-generation/experiments/yelp_sentiment_classifier/"
                   "results-bert-base/checkpoint-10410/")
    TST_CLF_MODELNAME = 'bert-base-uncased'
    TST_TEMPLATES_FILE_NAME = "/workspace/soft-Q-learning-for-text-generation/experiments/tst-templates-no-task.txt"
    TST_TARGET_TO_LABEL_MAP = {'negative': 'LABEL_0', 'positive': 'LABEL_1'}
#     TST_INPUTS_FILE_NAME_MAP = {('train', 'negative'): "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.0", 
#                                 ('train', 'positive'): "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.1",
#                                 ('infer', 'negative'): "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.0", 
#                                 ('infer', 'positive'): "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.1"}

    def __init__(
            self,
            max_length: int = 60,
            num_sampled_inputs_train: int = 12,
            num_sampled_inputs_infer: int = 10,
            num_return_sequences_train: int = 6,
            num_return_sequences_infer: int = 100,
            include_perplexity: bool = True,
            include_classifier: bool = False,
            include_bleu: bool = True,
            return_intermediate_outputs: bool = False,
    ) -> None:

        if include_perplexity is True:
            sql_utils.colorful_warning("Adding Perplexity-based Reward", bg="blue")

        sql_utils.colorful_warning(f"max_length={max_length}", bg="blue")

        # https://huggingface.co/gpt2
        # https://huggingface.co/facebook/bart-large-mnli
        self._generator = pipeline(
            "text-generation",
            model="distilgpt2",
            device=0)
        self._classifier = pipeline(
            "sentiment-analysis",
            model=self.TST_CLF_DIR,
            tokenizer=self.TST_CLF_MODELNAME,
            device=0)

        self._max_length = max_length
        self._num_sampled_inputs_train = num_sampled_inputs_train
        self._num_sampled_inputs_infer = num_sampled_inputs_infer
        self._num_return_sequences_train = num_return_sequences_train
        self._num_return_sequences_infer = num_return_sequences_infer
        self._tst_templates = self._load_tst_templates()
        self._tst_inputs = self._load_tst_inputs()
        self._train_idx = 0

        # Technically, adding perplexity-based reward will break
        # the scale, but we will ignore this for now since
        # this number is relatively small.
        self._include_perplexity = include_perplexity
        
        # Allow us to switch the classifier and bleu rewards on and off
        # But at least one must be on, so we can have something substantial to learn
        assert include_classifier or include_bleu
        self._include_classifier = include_classifier
        self._include_bleu = include_bleu
        
        # Do not set is to `True` during training, use it for debugging.
        self._return_intermediate_outputs = return_intermediate_outputs
                                
    def _load_tst_inputs(self) -> Dict[Tuple[str], List[str]]: 
        tst_inputs = {}
        # tokenizer = self._generator.tokenizer
        filepath_train_0 = "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.0"
        filepath_train_1 = "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.1"
        filepath_dev_0 = "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.0"
        filepath_dev_1 = "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.1"
        
        with open(filepath_train_0) as f: 
            sentences_train_0 = [line.strip() for line in f]
        with open(filepath_train_1) as f: 
            sentences_train_1 = [line.strip() for line in f]
        with open(filepath_dev_0) as f: 
            sentences_dev_0 = [line.strip() for line in f]
        with open(filepath_dev_1) as f: 
            sentences_dev_1 = [line.strip() for line in f]
            
        import random
        sentences_train = sentences_train_0 + sentences_train_1
        random.shuffle(sentences_train)
        tst_inputs['train'] = sentences_train
        tst_inputs['infer'] = sentences_dev_0[:5] + sentences_dev_1[:5]
        
#         for (mode, sentiment), filepath in self.TST_INPUTS_FILE_NAME_MAP.items(): 
#             with open(filepath, 'r') as fr: 
#                 sentences = [line.strip() for line in fr]
#             tst_inputs[(mode, sentiment)] = sentences
        return tst_inputs
        
    def _load_tst_templates(self) -> List[str]:
        with open(self.TST_TEMPLATES_FILE_NAME) as f: 
            tst_templates = [d.strip() for d in f.readlines()]
        return tst_templates

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._generator.tokenizer
                .convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, 
                        input_strings: List[str], 
                        prompt_strings: List[str]) -> List[str]:
        # templates = [self._tst_templates[0] for _ in source_strings]
        template = self._tst_templates[0]
        # print(templates)

        return [
            template.format(sentence_1=s_1, prompt=p) for s_1, p
            in zip(input_strings, prompt_strings)]

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        nlls, _ = compute_perplexities(
            sentences=sentences,
            model=self._generator.model,
            tokenizer=self._generator.tokenizer)
        # When the sentence has just one token,
        # the NLL/perplexity will be `NaN`.
        # Further, we use the negative NLL as the reward
        return -torch.nan_to_num(nlls, nan=10.0).mean()
    
    def _get_inputs(self, 
                    mode: str, 
                    num_sampled_inputs: int,
                    control_codes: List[str], 
                    prompts: List[str]) -> Tuple[List[str]]: 
        new_target_styles = []
        new_prompts = []
        new_inputs = []
        for control_code, prompt in zip(control_codes, prompts): 
            assert 'positive' in control_code or 'negative' in control_code or 'reconstruct' in control_code
            #             target_style = 'positive' if 'positive' in control_code else 'negative'

            #             inputs = np.random.choice(
            #                 self._tst_inputs[(mode, target_style)],
            #                 size=num_sampled_inputs,
            #                 replace=False,).tolist()
            #             new_inputs += inputs
            #             new_target_styles += [target_style] * num_sampled_inputs
            if mode == 'train': 
                first_inputs = self._tst_inputs['train'][self._train_idx:(self._train_idx+num_sampled_inputs)]
                self._train_idx += num_sampled_inputs
                
                if len(first_inputs) == num_sampled_inputs: 
                    new_inputs = first_inputs
                else: 
                    num_remaining = num_sampled_inputs - len(first_inputs)
                    self._train_idx %= len(inputs)
                    more_inputs = self._tst_inputs['train'][:self._train_idx]
                    new_inputs = first_inputs + more_inputs
            elif mode == 'infer': 
                new_inputs = self._tst_inputs['infer']
                
            assert len(new_inputs) == num_sampled_inputs
            new_prompts += [prompt] * num_sampled_inputs
            
        return new_target_styles, new_prompts, new_inputs
                

    def forward(self, control_codes: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError

        if mode == "train":
            num_sampled_inputs = self._num_sampled_inputs_train
            num_return_sequences = self._num_return_sequences_train
        if mode == "infer":
            num_sampled_inputs = self._num_sampled_inputs_infer
            num_return_sequences = self._num_return_sequences_infer

        # - List of length `len(prompts)`
        #     - List of length `num_return_sequences`
        #         - Dict of {"generated_text": str}
        # source_sentences = [' '.join(s.split(' ')[3:]) for s in sources]
        # source_strings = self._convert_tokens_to_string(source_sentences)
        
        new_target_styles, new_prompts, new_input_strings = self._get_inputs(mode, 
                                                                             num_sampled_inputs, 
                                                                             control_codes, 
                                                                             prompts)
        # print(new_target_styles)
        # print(new_prompts)
        # print(new_input_strings)
        new_prompt_strings = self._convert_tokens_to_string(new_prompts)
        formatted_prompts = self._format_prompts(new_input_strings, new_prompt_strings)
        # print(formatted_prompts)
        
        # target_style_strings = [self.TST_TARGET_TO_LABEL_MAP[t] for t in new_target_styles]
        
        generator_outputs: List[List[Dict[str, Any]]] = self._generator(
            formatted_prompts,
            max_length=self._max_length,
            num_return_sequences=num_return_sequences,
            # Only return generated text, without the prompt
            return_full_text=False)
        # print([output[0]["generated_text"] for output in generator_outputs])

        all_classifier_outputs = []
        rewards: List[FloatTensor] = []
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for batch_index in range(len(formatted_prompts)):
#             generated_texts = [
#                 output["generated_text"] for output in
#                 generator_outputs[batch_index]]
            
            
            generated_texts = []
            for output in generator_outputs[batch_index]: 
                text = output["generated_text"]
                try: 
                    end = text.index('"')
                except ValueError: 
                    end = len(text)
                generated_texts.append(text[:end])
            
            if mode == "infer": 
                print("Formatted Prompt and Generated Text: " + formatted_prompts[batch_index] + generated_texts[0])

            # - List of length `len(generated_texts)`
            #     - Dict of {
            #         "labels": List of length `num_topics`,
            #         "scores": List of length `num_topics`,
            #         "sequence": str,
            #     }
            try:
                reference_texts = [new_input_strings[batch_index] for _ in generator_outputs[batch_index]]
                
                check_Xs_Ys_sizes(generated_texts, reference_texts)
                
                if self._include_bleu: 
                
                    # Using a faster BLEU implementation during training
                    # `sacrebleu` is ~3X faster than `lightning`
                    # `sacrebleu-parallel` is ~3X faster than `sacrebleu`
                    bleus = [
                        scb.sentence_bleu(
                            hypothesis=x,
                            references=[y])
                        for x, y in zip(
                            generated_texts,
                            reference_texts)
                    ]
                    bleu_rewards = [b.score for b in bleus]
                else: 
                    bleu_rewards = [0 for b in generated_texts]
                
                reward = torch.tensor(bleu_rewards).float().mean()
                quantities_to_log["bleu"].append(reward)
                
                if self._include_classifier: 
                    classes = self._classifier(generated_texts, truncation=True)
                    label = target_style_strings[batch_index]
                    correct = [100 * (c['label'] == label) for c in classes]
                    acc = torch.tensor(correct).float().mean()
                    reward = reward + acc
                    quantities_to_log['acc'].append(acc)
                    
                
                
                if self._include_perplexity is True:
                    nll_reward = (
                        self._compute_nll_reward(
                            sentences=generated_texts))
                    reward = reward + nll_reward
                    quantities_to_log["nll"].append(nll_reward)

                rewards.append(reward)
                # all_classifier_outputs.append(classifier_outputs)

            except ValueError as err:
                # This happens when the generated text itself includes the
                # `</s>` token, which does happen and will cause the classifier to fail.
                # So we just ignore this error and give a score of zero for this batch.
                if str(err) != "All examples must have the same number of <eos> tokens.":
                    raise err

                click.secho("Encountered an error, skipping ...", bg="red")
                rewards.append(torch.tensor(0.).to(device))
                
        assert len(rewards) / num_sampled_inputs == len(prompts)
        rewards = [sum(rewards[i*num_sampled_inputs:(i+1)*num_sampled_inputs]) / num_sampled_inputs \
                   for i in range(len(prompts))]
        rewards_tensor = torch.stack(rewards)
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items())

        if self._return_intermediate_outputs is True:
            rewards_log["quantities_to_log"] = quantities_to_log  # type: ignore
            rewards_log["formatted_prompts"] = formatted_prompts  # type: ignore
            rewards_log["generator_outputs"] = generator_outputs  # type: ignore
            # rewards_log["all_classifier_outputs"] = all_classifier_outputs  # type: ignore

        if to_tensor is True:
            # print(rewards_tensor)
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
            control_codes=sources,
            prompts=predictions,
            to_tensor=to_tensor,
            mode=mode)

class GPT2ClassifierReward(object):

    def __init__(
            self,
            max_length: int = 30,
            num_return_sequences_train: int = 128,
            num_return_sequences_infer: int = 128,
            include_perplexity: bool = False,
            return_intermediate_outputs: bool = True,
    ) -> None:

        self.device = torch.device('cuda')
        self._tokenizer = AutoTokenizer.from_pretrained('distilgpt2', pad_token='<|endoftext|>')
        self._generator = GPT2LMHeadModel.from_pretrained('distilgpt2').to(self.device)
        self._generator.config.pad_token_id = self._tokenizer.pad_token_id

        self._max_length = max_length
        self._tst_templates = self.load_tst_templates()
        self._tst_inputs = self._load_tst_inputs()
        self._tst_inputs_idx = {('train', 'LABEL_0'): 0, 
                                ('train', 'LABEL_1'): 0,
                                ('infer', 'LABEL_0'): 0,
                                ('infer', 'LABEL_1'): 0}
        
        ### Modification starts ###
        self._counter = 0
    
        # self.train_input_pos = "this place was very good."
        # self.train_input_neg = "terrible service."
        
        self.train_input_pos = "thank you for a five star service ."
        self.train_input_neg = "thank you for the meal"
        
        # self.pos_verbalizer_candidate = ['\u0120positive', '\u0120great'   ,'\u0120good', '\u0120wonderful', '\u0120delicious']
        # self.neg_verbalizer_candidate = ['\u0120negative', '\u0120terrible','\u0120bad' , '\u0120bad', '\u0120bad']
        
        self.pos_verbalizer_candidate = ['Ġsorry', 'sorry']
        self.neg_verbalizer_candidate = ['Ġthank', 'thank']
        
        self.pos_verbalizer = self.pos_verbalizer_candidate[1]
        self.neg_verbalizer = self.neg_verbalizer_candidate[1]
        
        # self.pos_verbalizer = self.pos_verbalizer_candidate[3]
        # self.neg_verbalizer = self.neg_verbalizer_candidate[3]
        
        self.pos_id = self._tokenizer.convert_tokens_to_ids(self.pos_verbalizer)
        self.neg_id = self._tokenizer.convert_tokens_to_ids(self.neg_verbalizer)

    def load_tst_templates(self) -> List[str]:
        # temp_tst_template = "{sentence_1} {prompt}"
        # temp_tst_template = 'Sentence 1: "{sentence_1}" {prompt} Sentence 2: "'
        temp_tst_template = '{prompt} "{sentence_1}" "'
        # temp_tst_template = 'Sentence 1: "{sentence_1}" Sentence 2: "'
        # temp_tst_template = 'Sentence 1: "{sentence_1}" apologizing apologized apologies apologies apology Sentence 2: "'
        
        # temp_tst_template = "{sentence_1} It was"
        # temp_tst_template = "{sentence_1} All in all"
        # temp_tst_template = "Review: {sentence_1} Is the review positive or negative?"
        return [temp_tst_template]
    
    def _load_tst_inputs(self) -> Dict[Tuple[Any], List[str]]: 
        tst_inputs = {}
        
        filepath_train_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.0"
        filepath_train_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.train.1"
        filepath_dev_0 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.0"
        filepath_dev_1 = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only/raw/sentiment.dev.1"
        
        with open(filepath_train_0) as f: 
            sentences_train_0 = [line.strip() for line in f]
        with open(filepath_train_1) as f: 
            sentences_train_1 = [line.strip() for line in f]
        with open(filepath_dev_0) as f: 
            sentences_dev_0 = [line.strip() for line in f]
        with open(filepath_dev_1) as f: 
            sentences_dev_1 = [line.strip() for line in f]
            
        idx = 43
        tst_inputs[('train', 'LABEL_0')] = sentences_train_1[idx:]
        tst_inputs[('train', 'LABEL_1')] = sentences_train_0[idx:]
        tst_inputs[('infer', 'LABEL_0')] = sentences_train_1[idx:(idx+10)]
        tst_inputs[('infer', 'LABEL_1')] = sentences_train_0[idx:(idx+10)]
        
        return tst_inputs

    def _convert_tokens_to_string(self, tokens: List[str]) -> List[str]: 
        return [self._tokenizer.convert_tokens_to_string(s.split())
                for s in tokens]

    def _format_prompts(self, source_strings: List[str], prompt_strings: List[str]) -> List[str]:
        template = self._tst_templates[0]
        return [
            template.format(sentence_1=s, prompt=p) for s_1, p
            in zip(source_strings, prompt_strings) for s in s_1
        ]

    
    def _get_inputs(self, mode: str, target_labels: List[str]): 
    
        inputs = []
        for i, label in enumerate(target_labels): 
            idx = self._tst_inputs_idx[(mode, label)]
            p_data = self._tst_inputs[(mode, 'LABEL_0')]
            n_data = self._tst_inputs[(mode, 'LABEL_1')]

            if mode == 'train':
                inputs.append([self.train_input_pos, self.train_input_neg])
            elif mode == 'infer':
                inputs.append([p_data[i], n_data[i]])
                # inputs.append([self.train_input_pos, self.train_input_neg])
                
            idx += 1
            idx %= len(p_data)
            self._tst_inputs_idx[(mode, label)] = idx
        
        return inputs
    
    def _get_probs(self, texts):
        input_ids  = self._tokenizer(texts, padding='longest', truncation=True, return_tensors="pt", add_special_tokens=True).input_ids
        batch_size = len(texts)
        seq_len    = torch.ne(input_ids, self._tokenizer.pad_token_id).sum(-1) - 1
        
        with torch.no_grad():
            logits = self._generator(input_ids=input_ids.to(device)).logits
            logits = logits[range(batch_size), seq_len] 
            
        probs  = torch.softmax(logits, -1)
        return probs
    
    def forward(self, target_labels: List[str], prompts: List[str], to_tensor: bool, mode: str) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError
            
        source_strings = self._get_inputs(mode, target_labels)
        prompt_strings = self._convert_tokens_to_string(prompts)
        formatted_prompts = self._format_prompts(source_strings, prompt_strings)
        probs = self._get_probs(formatted_prompts)
        # len(prompts) == 6
        # len(formatted_prompts) == 12 (pos + neg)
        
        rewards = []
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
            
        for batch_index in range(len(prompts)):
   
            pos_probs = probs[batch_index * 2    ]
            neg_probs = probs[batch_index * 2 + 1]
            
            pos_pos_prob, pos_neg_prob = pos_probs[self.pos_id], pos_probs[self.neg_id]
            neg_pos_prob, neg_neg_prob = neg_probs[self.pos_id], neg_probs[self.neg_id]
            
            # UnBounded
            # pos_reward = ((pos_pos_prob - pos_neg_prob) / (pos_neg_prob))
            # neg_reward = ((neg_neg_prob - neg_pos_prob) / (neg_pos_prob))
            
            # Bounded
            pos_reward = ((pos_pos_prob - pos_neg_prob) / (pos_neg_prob + pos_pos_prob))
            neg_reward = ((neg_neg_prob - neg_pos_prob) / (neg_pos_prob + neg_neg_prob))
            
            # reward = (pos_reward + neg_reward) / 2 * 100
            reward = (pos_reward) * 100
            
            quantities_to_log["pp"].append(pos_pos_prob.item())
            quantities_to_log["pn"].append(pos_neg_prob.item())
            quantities_to_log["nn"].append(neg_neg_prob.item())
            quantities_to_log["np"].append(neg_pos_prob.item())
            quantities_to_log["reward"].append(reward.item())
            
            self._counter += 1
        
            print(prompts[batch_index], '\n', 
                  formatted_prompts[batch_index * 2], '\n', 
                  formatted_prompts[batch_index * 2 + 1], '\n', 
                  'PP:', pos_pos_prob.item(), '|',
                  'PN:', pos_neg_prob.item(), '|',
                  'NN:', neg_neg_prob.item(), '|',
                  'NP:', neg_pos_prob.item(), '|',
                  'Reward:', round(reward.item(), 2))
            
            rewards.append(reward)
            
        rewards_tensor = torch.stack(rewards)
        
        rewards_log = dict(
            (reward_key, torch.mean(torch.tensor(reward_vals)))
            for reward_key, reward_vals in quantities_to_log.items())
        
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

class PrefixSentimentClassifier(object):
    def __init__(self) -> None:
        self._classifier = pipeline("sentiment-analysis", device=0)

    def forward(self, prefixes: List[str], sentences: List[str], to_tensor: bool) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        check_Xs_Ys_sizes(prefixes, sentences)

        new_sentences = [
            f"{prefix} {sentence}"
            for prefix, sentence in
            zip(prefixes, sentences)]
        raw_outputs = self._classifier(new_sentences)
        rewards = [
            output["score"] * 100
            if output["label"] == "POSITIVE"
            else (1 - output["score"]) * 100
            for output in raw_outputs]

        rewards_log: Dict[str, Any] = {}
        if to_tensor is True:
            return torch.tensor(rewards), rewards_log
        else:
            return rewards, rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            prefixes=predictions,
            sentences=targets,
            to_tensor=to_tensor)


class ToxificationClassifier(object):
    def __init__(self) -> None:
        self._model = Detoxify("original", device="cuda")

    def forward(self, Xs: List[str], to_tensor: bool) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if isinstance(Xs, np.ndarray):
            Xs = Xs.tolist()

        outputs = self._model.predict(Xs)
        outputs = [(1 - score) * 100 for score in outputs["toxicity"]]

        rewards_log: Dict[str, Any] = {}
        if to_tensor is True:
            return torch.tensor(outputs), rewards_log
        else:
            return outputs, rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(
            Xs=predictions,
            to_tensor=to_tensor)


reward_name_to_cls_map = {
    "bleu": BLEUReward,
    "rouge": ROUGEReward,
    "bleurt": BleurtReward,
    # "entailment": EntailmentClassifier,
    "entailment2": EntailmentClassifier2,
    "entailment3": EntailmentClassifier3,
    "gpt2-topic": GPT2TopicReward,
    "sentiment": PrefixSentimentClassifier,
    "toxicity": ToxificationClassifier,
    "gpt2-bleu": GPT2BLEUReward,
    "gpt2-bleu-no-input": GPT2BLEUNoInputReward,
    "gpt2-sentiment-no-input": GPT2SentimentNoInputReward,
    "gpt2-sentiment-bleu-no-input": GPT2SentimentBLEUNoInputReward,
    "gpt2-sentiment-bertscore-no-input": GPT2SentimentBERTScoreNoInputReward,
    'gpt2-trigger': GPT2TriggerReward,
    "gpt2-bleu-sentiment": GPT2BLEUSentimentReward,
    'gpt2-classifier': GPT2ClassifierReward,
}


@torch.no_grad()
def compute_perplexities(
        sentences: List[str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
) -> Tuple[FloatTensor, FloatTensor]:

    nlls = []
    for sentence in sentences:
        encodings = tokenizer(
            sentence,
            return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        try:
            # labels **are shifted** inside the model
            outputs = model(
                input_ids,
                labels=input_ids.clone())
            nll = outputs[0]
        except RuntimeError:
            # Could happen when the input is empty
            nll = torch.tensor(float("nan")).to(device)

        nlls.append(nll)

    stacked_nlls = torch.stack(nlls, dim=0)
    return stacked_nlls, stacked_nlls.exp()


@torch.no_grad()
def get_NLI_prediction(
        tokenizer: PreTrainedTokenizerFast,
        model: RobertaForSequenceClassification,
        premises: List[str],
        hypotheses: List[str],
        device: torch.device,
        max_length: int = 256,
) -> FloatTensor:

    tokenized_inputs = tokenizer(
        premises,
        hypotheses,
        max_length=max_length,
        return_token_type_ids=True,
        truncation=True,
        padding=True)

    input_ids = (
        torch.Tensor(tokenized_inputs["input_ids"])
        .long()
        .to(device)
    )
    token_type_ids = (
        torch.Tensor(tokenized_inputs["token_type_ids"])
        .long()
        .to(device)
    )
    attention_mask = (
        torch.Tensor(tokenized_inputs["attention_mask"])
        .long()
        .to(device)
    )

    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=None)

    predicted_probability = torch.softmax(outputs.logits, dim=1)
    # predicted_index = torch.argmax(predicted_probability)
    # predicted_probability = predicted_probability.tolist()
    return predicted_probability


# @torch.no_grad()
# def get_NLI_prediction_2(
#         model: RobertaHubInterface,
#         premises: List[str],
#         hypotheses: List[str],
# ) -> FloatTensor:
#     """https://github.com/pytorch/fairseq/tree/master/examples/roberta"""
#     batch = collate_tokens([
#         model.encode(premise, hypothesis)
#         for premise, hypothesis
#         in zip(premises, hypotheses)], pad_idx=1)

#     logits = model.predict(
#         "mnli", batch,
#         return_logits=True)

#     return torch.nn.functional.softmax(logits, dim=-1)


def load_paraphrase_generator() -> Tuple[PegasusForConditionalGeneration, PegasusTokenizer]:
    model_name = "tuner007/pegasus_paraphrase"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer


@torch.no_grad()
def generate_paraphrases(
    model: PegasusForConditionalGeneration,
    tokenizer: PegasusTokenizer,
    input_text: str,
    num_return_sequences: int,
    num_beams: int,
    source_max_length: int = 60,
    target_max_length: int = 60,
) -> List[str]:

    batch = tokenizer(
        [input_text],
        truncation=True,
        padding="longest",
        max_length=source_max_length,
        return_tensors="pt").to(device)

    translated = model.generate(
        **batch,
        max_length=target_max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        temperature=1.5)

    tgt_text = tokenizer.batch_decode(
        translated,
        skip_special_tokens=True)

    return tgt_text


def tokenize_string_HF(
        sentences: List[str],
        tokenizer: PreTrainedTokenizerBase
) -> List[str]:
    """Tokenize strings using HF's tokenizer and white-space join them. This is mainly
       used in settings where we need to use HF's tokens in our models which require
       white-space tokenizable."""
    new_sentences = []
    for sentence in sentences:
        token_ids = tokenizer(sentence)["input_ids"]
        new_sentence = tokenizer.convert_ids_to_tokens(token_ids)
        new_sentences.append(" ".join(new_sentence))
        # The line below will recover the original sentence
        # gpt2_tokenizer.convert_tokens_to_string(new_sentence)
    return new_sentences


def _get_compute_perplexities_fn(
        task_name: str,
) -> Tuple[Callable[[List[str]], Tuple[FloatTensor, FloatTensor]],
           PreTrainedModel,
           PreTrainedTokenizerFast]:

    if task_name not in ["snli", "multinli"]:
        raise ValueError

    if task_name == "multinli":
        model_path = "/export/share/Experiments/20210515/checkpoints-from-colab/gpt2-multinli-10epochs/"

    if task_name == "snli":
        model_path = "/export/share/Experiments/20210515/checkpoints-from-colab/gpt2-snli-10epochs/"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelWithLMHead.from_pretrained(model_path)
    model.cuda()
    model.eval()

    @torch.no_grad()
    def _wrapped_fn(sentences: List[str]) -> Tuple[FloatTensor, FloatTensor]:
        return compute_perplexities(
            sentences=sentences,
            model=model,
            tokenizer=tokenizer)

    return _wrapped_fn, model, tokenizer
