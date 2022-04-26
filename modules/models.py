# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import json
import torch
import torch.nn as nn
import texar.torch as tx
from functools import partial
from typing import List, Tuple, Union, Dict, Optional, Callable, Any, cast

from configs.models import (
    config_model_transformers_small)

from sql.utils import ForwardMode
from sql.types import (
    BatchType,
    FloatTensor,
    LongTensor)

from transformers import pipeline, AutoTokenizer

def _build_gpt2_vocab_mlp(out_dim, in_dim=768, device=0): 
#     W1 = nn.Linear(in_dim, 1024)
#     A = nn.ReLU()
#     W2 = nn.Linear(1024, 64)
#     O = nn.Linear(64, out_dim)
#     return nn.Sequential(W1, A, W2, O)

#     W1 = nn.Linear(in_dim, 1024)
#     A1 = nn.ReLU()
#     W2 = nn.Linear(1024, 512)
#     A2 = nn.ReLU()
#     W3 = nn.Linear(512, 256)
#     A3 = nn.ReLU()
#     W4 = nn.Linear(256, 64)
#     O = nn.Linear(64, out_dim)
    
    W1 = nn.Linear(in_dim, 2048)
    A1 = nn.ReLU()
    # D1 = nn.Dropout(p=0.1)
    W2 = nn.Linear(2048, out_dim)
    # N2 = nn.LayerNorm(768)
    # O = nn.Linear(64, out_dim)
    
    # return nn.Sequential(W1, A1, W2, N2)
    return nn.Sequential(W1, A1, W2)

def _build_self_vocab_mlp(out_dim, in_dim=768, device=0): 
    W1 = nn.Linear(in_dim, 256)
    A1 = nn.ReLU()
    O = nn.Linear(256, out_dim)
    return nn.Sequential(W1, A1, O)


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    r"""Adapted from
    https://github.com/openai/gpt-2/blob/master/src/sample.py#L63-L77
    """
    if k == 0:
        # no truncation
        return logits

    values, _ = torch.topk(logits, k=k)
    min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_values,
        torch.full_like(logits, float('-inf')), logits)

def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    r"""Adapted from
    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317#file-top-k-top-p-py-L16-L27"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the
    # threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    for idx in range(logits.size(0)):
        batch_indices = sorted_indices[idx, sorted_indices_to_remove[idx]]
        logits[idx, batch_indices] = float("-inf")
    return logits

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
class GPT2ConditionedMLP(nn.Module): 
    input_template = 'Sentence 1: "{sentence_1}"'
    output_template = 'Sentence 2: "'
    sentence_1 = 'thank you for a five star service .'
    sentence_2 = "thank you for the meal ."
    temp_input_2 = "the mojitos are deliciously made with fresh fruit ."
    temp_input_0 = "they are all really friendly and the vets are knowledgable and patient ."
    temp_input_3 = "someone should buy this place and turn it into what it should be."
    temp_input_6 = "manager actually has respect for the customer instead of ignoring them ."
    
    sentence_1 = 'thank you for a five star service .'
#     sentence_1 = 'classification'
#     sentence_1 = 'Prompt: '

    def __init__(self, 
                 train_data: tx.data.PairedTextData,
                 max_source_length: int,
                 max_decoding_length: int,
                 config_name: str,
                 device=0) -> None: 
        super().__init__()
        
        if config_name not in ['gpt2_conditioned_mlp']: 
            raise ValueError
#         if config_name == 'gpt2_conditioned_mlp': 
#             config_model = config_model_gpt2_conditioned_mlp
            
        # self.config_model = config_model
        self.device = device

        self.max_source_length = max_source_length
        self.max_decoding_length = max_decoding_length
        
        self.source_vocab = train_data.source_vocab
        self.target_vocab = train_data.target_vocab
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id
        
        model = 'distilgpt2'
        # model = 'gpt2-medium'
        self.tokenizer = AutoTokenizer.from_pretrained(model, pad_token='<|endoftext|>')
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model=model,
                                  device=self.device)
        for param in self.generator.model.parameters():
            param.requires_grad = False
        
        mode = 'gpt2_vocab'
        # mode = 'self_vocab'
        # mode = 'prep_vocab'
        if mode == 'gpt2_vocab': 
            model_sizes = {'distilgpt2': 768,
                           'gpt2-medium': 1024,
                           'gpt2-large': 1280}
            model_dim = model_sizes[model]
            self.mlp = _build_gpt2_vocab_mlp(model_dim, in_dim=model_dim).to(self.device)
            self._mlp_forward = self._gpt2_vocab_mlp_forward
            self.valid_token_ids = None
        elif mode == 'self_vocab': 
            # self.mlp = _build_self_vocab_mlp(self.target_vocab_size - 4).to(self.device)
            self.mlp = _build_self_vocab_mlp(self.target_vocab_size).to(self.device)
            self._mlp_forward = self.mlp
            # self._mlp_forward = self._self_vocab_mlp_forward
        elif mode == 'prep_vocab': 
            self.mlp = _build_gpt2_vocab_mlp(self.target_vocab_size).to(self.device)
            self._mlp_forward = self._gpt2_vocab_mlp_forward
            self.valid_token_ids = json.load(open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_negative_prep'))
            self.valid_token_ids = json.load(open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_3_tokens'))
            self.valid_token_ids = json.load(open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_3_tokens_rep_100'))
            self.valid_token_ids = json.load(open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_1k_random_tokens'))
            self.valid_token_ids = json.load(open('/jupyter/prompt-generation/soft-Q-learning-for-text-generation/'
                                                  'experiments/valid_gpt2_token_ids.yelp_3_tokens_1'))

            
#         self.dataset_inputs = ['the carts are in excellent shape, all electric and all equipped with gps. the',
#                                'challenging but fun course! the',
#                                'beautiful views and lots of variety of length and layout of holes. the',
#                                "i'll definitely be back! the",
#                                'the service and prices were great. the',
#                                'i had the buffalo chicken sandwich and it was delicious. the',
#                                'a cool bar off the beaten path that is a worth a trip. the',
#                                'awesome drink specials during happy hour. the',
#                                'fantastic wings that are crispy and delicious, wing night on tuesday and thursday! the',
#                                'the sandwiches are always amazing just as i remember. the',
#                                'the staff is amazing and friendly. the',
#                                'great place for lunch as well. the',
#                                'friendly staff, good food, great beer selection, and relaxing atmosphere. the',
#                                "great wings and the buffalo chicken pizza is the best i've had. the",
#                                'the sandwiches are all on thick cut italian bread and fresh. the',
#                                'if we ever get to the pittsburgh area again, we will go back! the']
        
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

        self.dataset_inputs = [' ']

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
#         self.dataset_inputs = list(itertools.chain(*[[s, s, s, s] for s in self.dataset_inputs]))
#         print(self.dataset_inputs)
    
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
        # self.temp_input = ' '
        # self.temp_input = 'Prompt:'
        # self.temp_input = 'classification'
        # self.temp_input = self.sentence_1
#         self.temp_input = self.sentence_2
#         self.temp_input = self.temp_input_2
#         self.temp_input = self.temp_input_0
        # self.temp_input = self.temp_input_6
        self._tst_inputs = self._load_tst_inputs()
        self._tst_inputs_idx = {('train', 'LABEL_0'): 0, 
                                ('train', 'LABEL_1'): 0,
                                ('infer', 'LABEL_0'): 0,
                                ('infer', 'LABEL_1'): 0}
        self.fluent = False
        self.logit_bias = -10
        # self.logit_bias = nn.Parameter(torch.tensor(-10.))
        self.normalize_mlp_output = False
        

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.mlp.apply(init_weights)
    
    
    def _gpt2_vocab_mlp_forward(self, state): 
        # print(self.mlp.device)
        # print(state)
        mlp_output = self.mlp(state)
        
        if self.normalize_mlp_output:
            mlp_output = mlp_output / torch.linalg.norm(mlp_output, dim=-1).unsqueeze(-1)
            
        # print('Norm:', torch.linalg.norm(mlp_output, dim=-1).mean().item())
        logits = self.generator.model.lm_head(mlp_output)
        if self.valid_token_ids is not None: 
            logits = logits[:, self.valid_token_ids]
            
        if self.fluent: 
            plm_logits = self.generator.model.lm_head(state)
            values, _ = torch.topk(plm_logits, k=20)
            min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                plm_logits < min_values,
                torch.full_like(logits, float('-inf')), logits)
        # print(logits.shape)
        zeros = torch.ones_like(logits)[:, :4] * float('-inf')
        # print(zeros)
        modified_logits = torch.cat([zeros, logits], dim=-1)
        # print(modified_logits.shape)
        return modified_logits
    
    def _self_vocab_mlp_forward(self, state): 
        logits = self.mlp(state)
        zeros = torch.ones_like(logits)[:, :4] * float('-inf')
        # print(zeros)
        modified_logits = torch.cat([zeros, logits], dim=-1)
        # print(modified_logits.shape)
        return modified_logits
    
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
        tst_inputs[('train', 'LABEL_0')] = sentences_train_1[idx:(idx+size)]
        # tst_inputs[('train', 'LABEL_0')] = sentences_train_1_sample
        tst_inputs[('train', 'LABEL_0')] = list(itertools.chain(*[[s for _ in range(self.n_repeats)] \
                                                                   for s in tst_inputs[('train', 'LABEL_0')]]))
        # tst_inputs[('train', 'LABEL_1')] = sentences_test_ref_0[idx:(idx+size)]
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
        
    def decode_teacher_forcing(self,
                               batch: BatchType,
                               last_token_hidden_state,
                               past_key_values) \
    -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        
        state = last_token_hidden_state
        sample_ids, sample_logits = batch['target_text_ids'][:, 1:], []
        #print(sample_ids)
        for i in range(self.max_decoding_length): 
            # logits = self.mlp(state)
            logits = self._mlp_forward(state)
            logits = logits + self.logit_bias
            
            actions = sample_ids[:, i]
            tokens = self.target_vocab.map_ids_to_tokens_py(actions.tolist()).tolist()
#             tokens = self.generator.tokenizer.convert_ids_to_tokens(actions.tolist())
            # if i == 0: print(tokens)
            
            # sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))
            
            tokens = [self.generator.tokenizer.convert_tokens_to_string([t]) \
                      for t in tokens]
            # tokens = [' ' + t for t in tokens]
            token_encoding = (self.generator
                               .tokenizer(tokens, 
                                          padding=True,
                                          return_tensors='pt')
                               .to(self.device))
            input_ids = token_encoding['input_ids']
            input_lengths = token_encoding['attention_mask'].sum(dim=1)

            next_outputs = (self.generator.model
                            .transformer(input_ids, 
                                         past_key_values=past_key_values, 
                                         use_cache=True))
            state = next_outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                   (input_lengths - 1)]
            past_key_values = next_outputs.past_key_values
            
        # sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
            
        decoder_output = tx.modules.TransformerDecoderOutput(
            logits=sample_logits,
            sample_id=sample_ids
        )
        return decoder_output, None
    
    def decode_sampling(self,
                        batch: BatchType,
                        last_token_hidden_state,
                        past_key_values,
                        top_k: Optional[int] = None,
                        top_p: Optional[float] = None) \
    -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        if top_k is not None and top_p is not None:
            raise ValueError
            
        print(self.logit_bias)
            
        state = last_token_hidden_state
        prompt_tokens, sample_ids, sample_logits = [], [], []
        for i in range(self.max_decoding_length): 
            # logits = self.mlp(state)
            logits = self._mlp_forward(state)
            logits = logits + self.logit_bias
            # print(state.min().item(), state.max().item())
            print(logits[:, 4:].min().item(), logits.max().item())
            # print(logits.min().item(), logits.max().item())
            
            if top_k is not None: sampling_logits = _top_k_logits(logits, k=top_k)
            elif top_p is not None: sampling_logits = _top_p_logits(logits, p=top_p)
            else: sampling_logits = logits
            
            # print(sampling_logits)
            actions = (torch.distributions.categorical
                       .Categorical(logits=sampling_logits)
                       .sample())
            # print(actions)
            tokens = self.target_vocab.map_ids_to_tokens_py(actions.tolist()).tolist()
#             tokens = self.generator.tokenizer.convert_ids_to_tokens(actions.tolist())
            # if i == 0: print(tokens)
            
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))
            
            tokens = [self.generator.tokenizer.convert_tokens_to_string([t]) \
                      for t in tokens]
            # print(tokens)
            
            token_encoding = (self.generator
                               .tokenizer(tokens, 
                                          padding=True,
                                          return_tensors='pt')
                               .to(self.device))
            input_ids = token_encoding['input_ids']
            input_lengths = token_encoding['attention_mask'].sum(dim=1)

            next_outputs = (self.generator.model
                            .transformer(input_ids, 
                                         past_key_values=past_key_values, 
                                         use_cache=True))
            state = next_outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                   (input_lengths - 1)]
            past_key_values = next_outputs.past_key_values
            
        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
        # print(sample_logits.min().item(), sample_logits.max().item())
            
        decoder_output = tx.modules.TransformerDecoderOutput(
            logits=sample_logits,
            sample_id=sample_ids
        )
        return (decoder_output, 
                torch.tensor([self.max_decoding_length \
                              for _ in range(sample_ids.shape[0])]).to(self.device)
               )

    def decode_greedy(
            self,
            batch: BatchType,
            last_token_hidden_state,
            past_key_values,
            corruption_p: Optional[float] = None,
            input_mode='train',
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        state = last_token_hidden_state
        prompt_tokens, sample_ids, sample_logits = [], [], []
        for i in range(self.max_decoding_length): 
            # logits = self.mlp(state) # [batch_size, vocab_size]
            logits = self._mlp_forward(state)
            if input_mode == 'infer': 
                print('State:', state)
                print('Logits:', logits)
            
            # actions = torch.distributions.categorical.Categorical(logits).sample() # [batch_size]
            actions = logits.argmax(dim=-1) # [batch_size]
            tokens = self.target_vocab.map_ids_to_tokens_py(actions.tolist()).tolist()
            # tokens = self.generator.tokenizer.convert_ids_to_tokens(actions.tolist())
            
            sample_ids.append(actions.unsqueeze(dim=1)) # [batch_size, 1]
            sample_logits.append(logits.unsqueeze(dim=1)) # [batch_size, 1, vocab_size]
            
            tokens = [self.generator.tokenizer.convert_tokens_to_string([t]) \
                      for t in tokens]
            # tokens = [' ' + t for t in tokens]
#             input_ids = (self.generator
#                          .tokenizer(tokens, 
#                                     return_tensors='pt')
#                          ['input_ids']
#                          .to(0))
            token_encoding = (self.generator
                               .tokenizer(tokens, 
                                          padding=True,
                                          return_tensors='pt')
                               .to(self.device))
            input_ids = token_encoding['input_ids']
            input_lengths = token_encoding['attention_mask'].sum(dim=1)

            next_outputs = (self.generator.model
                            .transformer(input_ids, 
                                         past_key_values=past_key_values, 
                                         use_cache=True))
            # state = next_outputs.last_hidden_state[:, -1, :]
            state = next_outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                   (input_lengths - 1)]
            past_key_values = next_outputs.past_key_values
            
        sample_ids = torch.cat(sample_ids, dim=1) # [batch_size, prompt_length]
        sample_logits = torch.cat(sample_logits, dim=1) # [batch_size, prompt_length, vocab_size]
            
        decoder_output = tx.modules.TransformerDecoderOutput(
            logits=sample_logits,
            sample_id=sample_ids
        )
        return {
                "sample_id": (
                    decoder_output
                    .sample_id
                    .unsqueeze(dim=-1)
                )
            }
    
    def _get_inputs(self, mode: str, target_labels: List[str]): 
        # data_0 = self._tst_inputs[(mode, 'LABEL_0')]
        # data_1 = self._tst_inputs[(mode, 'LABEL_1')]
        
        # idx_0 = self._tst_inputs_idx[(mode, 'LABEL_0')]
        # idx_1 = self._tst_inputs_idx[(mode, 'LABEL_1')]
        
        inputs = []
        for i, label in enumerate(target_labels): 
            idx = self._tst_inputs_idx[(mode, label)]
            data = self._tst_inputs[(mode, label)]
            
            # inputs.append(self.temp_input)
            if mode == 'train': 
                # inputs.append('thank you for a five star service.')
                # inputs.append(self.temp_input)
                # inputs.append(self.dataset_inputs[idx])
                inputs.append(data[idx])
            else: 
                # inputs.append(self.dataset_inputs[idx])
                inputs.append(data[idx])
                
            # inputs.append('Prompt:')
            # inputs.append(self.dataset_inputs[idx])
            # inputs.append(data[idx])
            idx += 1
            idx %= len(data)
            self._tst_inputs_idx[(mode, label)] = idx
        
        return inputs
    
    def forward(self,
                batch: BatchType,
                mode: ForwardMode,
                **kwargs) -> Union[Tuple[tx.modules.TransformerDecoderOutput, 
                                         LongTensor], 
                                   Dict]:        
        # print(batch['source_text'])
        target_labels = [t[0] for t in batch['source_text']]
        if mode in [ForwardMode.INFER]: 
            input_mode = 'infer'
        else: 
            input_mode = 'train'
        input_texts = self._get_inputs(input_mode, target_labels)
        # print(input_texts)
        
        if input_mode == 'infer': 
            print('Infer Inputs:', input_texts)
        
        # print('Model:', input_texts)
        
        # input_texts = [self.temp_input for _ in range(len(batch['source_text']))]
        
        token_encoding = (self.generator
                          .tokenizer(input_texts, 
                                     padding=True,
                                     return_tensors='pt')
                          .to(self.device))
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        outputs = (self.generator.model
                   .transformer(input_ids, use_cache=True))
        last_token_hidden_state = outputs.last_hidden_state[np.arange(input_ids.shape[0]), 
                                                            (input_lengths - 1)]
        past_key_values = outputs.past_key_values
        
#         input_ids = (self.generator
#                      .tokenizer(input_texts, 
#                                 return_tensors='pt')['input_ids']
#                      .to(0))
#         outputs = (self.generator.model
#                    .transformer(input_ids, use_cache=True))
#         last_token_hidden_state = outputs.last_hidden_state[:, -1, :]
#         past_key_values = outputs.past_key_values
        
        if mode in [ForwardMode.MLE, ForwardMode.SQL_OFF_GT]:
            return self.decode_teacher_forcing(
                batch=batch,
                last_token_hidden_state=last_token_hidden_state,
                past_key_values=past_key_values,
                **kwargs)

        if mode in [ForwardMode.PG, ForwardMode.SQL_ON]:
            return self.decode_sampling(
                batch=batch,
                last_token_hidden_state=last_token_hidden_state,
                past_key_values=past_key_values,
                **kwargs)

        if mode in [ForwardMode.INFER]:
            return self.decode_greedy(
                batch=batch,
                last_token_hidden_state=last_token_hidden_state,
                past_key_values=past_key_values,
                input_mode=input_mode,
                **kwargs)

        raise ValueError(f"Unknown mode {mode}")


class Transformer(nn.Module):
    r"""A standalone sequence-to-sequence Transformer model, from "Attention
    Is All You Need". The Transformer model consists of the word embedding
    layer, position embedding layer, an encoder and a decoder. Both encoder
    and decoder are stacks of self-attention layers followed by feed-forward
    layers. See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)
    for the full description of the model.
    """

    def __init__(
            self,
            train_data: tx.data.PairedTextData,
            max_source_length: int,
            max_decoding_length: int,
            config_name: str,
    ) -> None:
        super().__init__()

        if config_name not in ["transformer_small"]:
            raise ValueError

        if config_name == "transformer_small":
            config_model = config_model_transformers_small

        self.config_model = config_model
        self.max_source_length = max_source_length
        self.max_decoding_length = max_decoding_length

        self.source_vocab = train_data.source_vocab
        self.target_vocab = train_data.target_vocab
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id

        self.source_embedder = tx.modules.WordEmbedder(
            vocab_size=self.source_vocab_size,
            hparams=self.config_model.emb)

        self.target_embedder = tx.modules.WordEmbedder(
            vocab_size=self.target_vocab_size,
            hparams=self.config_model.emb)

        self.source_pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.max_source_length,
            hparams=self.config_model.position_embedder_hparams)

        self.target_pos_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=self.max_decoding_length,
            hparams=self.config_model.position_embedder_hparams)

        self.encoder = tx.modules.TransformerEncoder(
            hparams=self.config_model.encoder)
        self.decoder = tx.modules.TransformerDecoder(
            token_pos_embedder=partial(
                self._embedding_fn,
                source_or_target="target"),
            vocab_size=self.target_vocab_size,
            output_layer=self.target_embedder.embedding,
            hparams=self.config_model.decoder)

    def _embedding_fn(
            self,
            tokens: LongTensor,
            positions: LongTensor,
            source_or_target: str,
    ) -> FloatTensor:
        if source_or_target not in ["source", "target"]:
            raise ValueError

        if source_or_target == "source":
            word_embed = self.source_embedder(tokens)
            pos_embed = self.source_pos_embedder(positions)
        if source_or_target == "target":
            word_embed = self.target_embedder(tokens)
            pos_embed = self.target_pos_embedder(positions)

        scale = self.config_model.hidden_dim ** 0.5
        return word_embed * scale + pos_embed

    def decode_teacher_forcing(
            self,
            batch: BatchType,
            memory: FloatTensor
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        decoder_outputs = self.decoder(
            memory=memory,
            memory_sequence_length=batch["source_length"],
            inputs=batch["target_text_ids"][:, :-1],
            sequence_length=batch["target_length"] - 1,
            decoding_strategy="train_greedy")

        # label_lengths = (labels != 0).long().sum(dim=1)
        # We don't really need `sequence_lengths` here
        return decoder_outputs, None

    def decode_greedy(
            self,
            batch: BatchType,
            memory: FloatTensor,
            corruption_p: Optional[float] = None,
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:

        start_tokens = memory.new_full(
            batch["target_length"].size(),
            self.bos_token_id,
            dtype=torch.int64)

        helper = None
        if corruption_p is not None:
            raise NotImplementedError("Deprecated")

        return self.decoder(
            start_tokens=start_tokens,
            end_token=self.eos_token_id,
            helper=helper,
            memory=memory,
            memory_sequence_length=batch["source_length"],
            decoding_strategy="infer_greedy",
            # Probably will hurt the longest sequence,
            # but probably better learning
            max_decoding_length=min(
                self.max_decoding_length,
                batch["target_length"].max().item() - 1))

    def decode_sampling(
            self,
            batch: BatchType,
            memory: FloatTensor,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
    ) -> Tuple[tx.modules.TransformerDecoderOutput, LongTensor]:
        if top_k is not None and top_p is not None:
            raise ValueError

        start_tokens = memory.new_full(
            batch["target_length"].size(),
            self.bos_token_id,
            dtype=torch.int64)

        helper = None
        if top_k is not None:
            helper = tx.modules.TopKSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                top_k=top_k)

        if top_p is not None:
            helper = tx.modules.TopPSampleEmbeddingHelper(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                p=top_p)

        decoder_output = self.decoder(
            start_tokens=start_tokens,
            end_token=self.eos_token_id,
            helper=helper,
            memory=memory,
            memory_sequence_length=batch["source_length"],
            decoding_strategy="infer_sample",
            # Probably will hurt the longest sequence,
            # but probably better learning
            max_decoding_length=min(
                self.max_decoding_length,
                batch["target_length"].max().item() - 1))
        
        sample_logits = decoder_output[0].logits
        print(sample_logits.min().item(), sample_logits.max().item())
        return decoder_output

    def decode_beam_search(
            self,
            batch: BatchType,
            memory: FloatTensor,
            beam_width: int,
            corruption_p: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:

        # Only greedy decoding is support for this as of now.
        if corruption_p is not None:
            if beam_width != 1:
                raise NotImplementedError

        # when `beam_width in [None, 1]`, `self.decoder`
        # will switch to default decoding mode, which is
        # not necessarily what we want. Instead, let's
        # explicitly call greedy-decoding.
        # https://sourcegraph.com/github.com/asyml/texar-pytorch/-/blob/texar/torch/modules/decoders/rnn_decoders.py#L717:9
        if beam_width > 1:

            start_tokens = memory.new_full(
                batch["target_length"].size(),
                self.bos_token_id,
                dtype=torch.int64)

            return self.decoder(
                start_tokens=start_tokens,
                end_token=self.eos_token_id,
                memory=memory,
                memory_sequence_length=batch["source_length"],
                beam_width=beam_width,
                max_decoding_length=self.max_decoding_length)

        else:
            infer_outputs, _ = self.decode_greedy(
                batch=batch,
                memory=memory,
                corruption_p=corruption_p)

            return {
                "sample_id": (
                    infer_outputs
                    .sample_id
                    .unsqueeze(dim=-1)
                )
            }

    def forward(
            self,
            batch: BatchType,
            mode: ForwardMode,
            **kwargs,
    ) -> Union[Tuple[tx.modules.TransformerDecoderOutput, LongTensor], Dict]:

        # Text sequence length excluding padding
        if not (batch["source_length"] == (batch["source_text_ids"] != 0).int().sum(dim=1)).all():
            raise ValueError

        positions: LongTensor = (
            torch.arange(
                batch["source_length"].max(),  # type: ignore
                dtype=torch.long,
                device=batch["source_text_ids"].device)
            .unsqueeze(0)
            .expand(batch["source_text_ids"].size(0), -1)
        )

        encoder_output = self.encoder(
            inputs=self._embedding_fn(
                tokens=batch["source_text_ids"],
                positions=positions,
                source_or_target="source"),
            sequence_length=batch["source_length"])

        if mode in [ForwardMode.MLE, ForwardMode.SQL_OFF_GT]:
            return self.decode_teacher_forcing(
                batch=batch,
                memory=encoder_output)

        if mode in [ForwardMode.PG, ForwardMode.SQL_ON]:
            return self.decode_sampling(
                batch=batch,
                memory=encoder_output,
                **kwargs)

        if mode in [ForwardMode.INFER]:
            return self.decode_beam_search(
                batch=batch,
                memory=encoder_output,
                **kwargs)

        raise ValueError(f"Unknown mode {mode}")
