gpu_id=2
for seed in 13 21 42 87 100
do
    python mlm-evaluate.py \
        --task "SST-2" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --template_instruction "*cls*In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. *sent_0*_It_was*mask*.*sep+*" \
        --template_incontext "*cls**sent_0*_It_was*mask*.*sep+**sent_1*_It_was*label_0*.*sep+**sent_2*_It_was*label_1*.*sep+*" \
        --label-map "{'0':'terrible','1':'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "mr" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --template_instruction "*cls*In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. *sent_0*_It_was*mask*.*sep+*" \
        --template_incontext "*cls**sent_0*_It_was*mask*.*sep+**sent_1*_It_was*label_0*.*sep+**sent_2*_It_was*label_1*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "cr" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --template_instruction "*cls*In this task, you are given sentences from customer reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative. *sent_0*_It_was*mask*.*sep+*" \
        --template_incontext "*cls**sent_0*_It_was*mask*.*sep+**sent_1*_It_was*label_0*.*sep+**sent_2*_It_was*label_1*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "sst-5" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --template_instruction "*cls*In this task, you are given sentences from movie reviews. Based on the given review, classify it to one of the five classes: (1) terrible, (2) bad, (3) okay, (4) good, and (5) great. *sent_0*_It_was*mask*.*sep+*" \
        --template_incontext "*cls**sent_0*_It_was*mask*.*sep+**sent_1*_It_was*label_0*.*sep+**sent_2*_It_was*label_1*.*sep+**sent_3*_It_was*label_2*.*sep+**sent_4*_It_was*label_3*.*sep+**sent_5*_It_was*label_4*.*sep+*" \
        --label-map "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "agnews" \
        --template "*cls**mask*_News: *sent_0**sep+*" \
        --template_instruction "*cls*In this task, you . In this task is to classify the article to one of the four topics 'World', 'Sports', 'Business', 'Tech'. *mask*_News: *sent_0**sep+*" \
        --template_incontext "*cls**mask*_News: *sent_0**sep+**label_0*_News: *sent_1*.*sep+**label_1*_News: *sent_2**sep+**label_2*_News: *sent_3**sep+**label_3*_News: *sent_4**sep+*" \
        --label-map "{0:'World',1:'Sports',2:'Business',3:'Tech'}" \
        --seed $seed \
        --truncate_head true \
        --skip_space true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "MRPC" \
        --template "*cls**sent_0**mask*,*+sentl_1**sep+*" \
        --template_instruction "*cls*You are given two sentences. Answer \"Yes\" if these sentences are a paraphrase of one another, otherwise answer \"No\". *sent_0**mask*,*+sentl_1**sep+*" \
        --template_incontext "*cls**sent_0**mask*,*+sentl_1**sep+**sent_2**label_0*,*+sentl_3**sep+**sent_4**label_1*,*+sentl_5**sep+*" \
        --label-map "{'0':'No','1':'Yes'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "RTE" \
        --template "*cls**sent-_0*?*mask*,*+sentl_1**sep+*" \
        --template_instruction "*cls*In this task, you're given a pair of sentences. Your job is to choose whether the two sentences clearly agree (Yes) or disagree (No) with each other. *sent-_0*?*mask*,*+sentl_1**sep+*" \
        --template_incontext "*cls**sent-_0*?*mask*,*+sentl_1**sep+**sent-_2*?*label_0*,*+sentl_3**sep+**sent-_4*?*label_1*,*+sentl_5**sep+*" \
        --label-map "{'not_entailment':'No','entailment':'Yes'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "SNLI" \
        --template "*cls**sent-_0*?*mask*,*+sentl_1**sep+*" \
        --template_instruction "*cls*In this task, you're given a pair of sentences. Your job is to choose whether the two sentences clearly agree (Yes)/disagree (No) with each other, or if this cannot be determined (Maybe). *sent-_0*?*mask*,*+sentl_1**sep+*" \
        --template_incontext "*cls**sent-_0*?*mask*,*+sentl_1**sep+**sent-_2*?*label_0*,*+sentl_3**sep+**sent-_4*?*label_1*,*+sentl_5**sep+**sent-_6*?*label_2*,*+sentl_7**sep+*" \
        --label-map "{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "yelp-2" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --template_instruction "*cls*In this task, you are given Yelp reviews. The task is to classify a review as \"great\" if the overall sentiment of the review is positive or as \"terrible\" if the overall sentiment of the review is negative. *sent_0*_It_was*mask*.*sep+*" \
        --template_incontext "*cls**sent_0*_It_was*mask*.*sep+**sent_1*_It_was*label_0*.*sep+**sent_2*_It_was*label_1*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "yelp-5" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --template_instruction "*cls*In this task, you are given Yelp reviews. Based on the given review, classify it to one of the five classes: (1) terrible, (2) bad, (3) okay, (4) good, and (5) great. *sent_0*_It_was*mask*.*sep+*" \
        --template_incontext "*cls**sent_0*_It_was*mask*.*sep+**sent_1*_It_was*label_0*.*sep+**sent_2*_It_was*label_1*.*sep+**sent_3*_It_was*label_2*.*sep+**sent_4*_It_was*label_3*.*sep+**sent_5*_It_was*label_4*.*sep+*" \
        --label-map "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "yahoo" \
        --template "*cls*Topic *mask*: *sent_0**sep+*" \
        --template_instruction "*cls*You are given a passage. Using the information present in the passage, you need to classify it into one of the 10 topics: 0 - 'Culture', 1 - 'Science', 2 - 'Health', 3 - 'Education', 4 - 'Computers', 5 - 'Sports', 6 - 'Business', 7 - 'Music', 8 - 'Family', 9 - 'Politics'. Topic *mask*: *sent_0**sep+*" \
        --template_incontext "*cls*Topic *mask*: *sent_0**sep+*Topic *label_0*: *sent_1**sep+*Topic *label_1*: *sent_2**sep+*Topic *label_2*: *sent_3**sep+*Topic *label_3*: *sent_4**sep+*Topic *label_4*: *sent_5**sep+*Topic *label_5*: *sent_6**sep+*Topic *label_6*: *sent_7**sep+*Topic *label_7*: *sent_8**sep+*Topic *label_8*: *sent_9**sep+*" \
        --label-map "{0:'culture',1:'science',2:'health',3:'education',4:'computer',5:'sports',6:'business',7:'music',8:'family',9:'politics'}" \
        --seed $seed \
        --truncate_head false \
        --skip_space true \
        --gpu_id $gpu_id \
        --instruction
    python mlm-evaluate.py \
        --task "dbpedia" \
        --template "*cls*[Category:*mask*]*sent_0**sep+*" \
        --template_instruction "*cls*You are given a passage. Using the information present in the passage, you need to classify it into one of the 10 topics: 0 - 'Culture', 1 - 'Science', 2 - 'Health', 3 - 'Education', 4 - 'Computers', 5 - 'Sports', 6 - 'Business', 7 - 'Music', 8 - 'Family', 9 - 'Politics'. [Category:*mask*]*sent_0**sep+*" \
        --template_incontext "*cls*[Category:*mask*]*sent_0**sep+*[Category:*label_0*]*sent_1**sep+*[Category:*label_1*]*sent_2**sep+*[Category:*label_2*]*sent_3**sep+*[Category:*label_3*]*sent_4**sep+*[Category:*label_4*]*sent_5**sep+*[Category:*label_5*]*sent_6**sep+*[Category:*label_6*]*sent_7**sep+*[Category:*label_7*]*sent_8**sep+*[Category:*label_8*]*sent_9**sep+*[Category:*label_9*]*sent_10**sep+*[Category:*label_10*]*sent_11**sep+*[Category:*label_11*]*sent_12**sep+*[Category:*label_12*]*sent_13**sep+*[Category:*label_13*]*sent_14*sep+*" \
        --label-map "{0:'Company',1:'Education',2:'Artist',3:'Sports',4:'Office',5:'Transportation',6:'Building',7:'Natural',8:'Village',9:'Animal',10:'Plant',11:'Album',12:'Film',13:'Written'}" \
        --seed $seed \
        --truncate_head false \
        --gpu_id $gpu_id \
        --instruction
done
