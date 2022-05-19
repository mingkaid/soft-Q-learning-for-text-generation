gpu_id=2
for seed in 13 21 42 87 100
do
    python mlm-pt.py \
        --task "SST-2" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{'0':'terrible','1':'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "mr" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "cr" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "sst-5" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "agnews" \
        --template "*cls**mask*_News: *sent_0**sep+*" \
        --label-map "{0:'World',1:'Sports',2:'Business',3:'Tech'}" \
        --seed $seed \
        --truncate_head true \
        --skip_space true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "MRPC" \
        --template "*cls**sent_0**mask*,*+sentl_1**sep+*" \
        --label-map "{'0':'No','1':'Yes'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "RTE" \
        --template "*cls**sent-_0*?*mask*,*+sentl_1**sep+*" \
        --label-map "{'not_entailment':'No','entailment':'Yes'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "SNLI" \
        --template "*cls**sent-_0*?*mask*,*+sentl_1**sep+*" \
        --label-map "{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "yelp-2" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "yelp-5" \
        --template "*cls**sent_0*_It_was*mask*.*sep+*" \
        --label-map "{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}" \
        --seed $seed \
        --truncate_head true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "yahoo" \
        --template "*cls*Topic *mask*: *sent_0**sep+*" \
        --label-map "{0:'culture',1:'science',2:'health',3:'education',4:'computer',5:'sports',6:'business',7:'music',8:'family',9:'politics'}" \
        --seed $seed \
        --truncate_head false \
        --skip_space true \
        --gpu_id $gpu_id
    python mlm-pt.py \
        --task "dbpedia" \
        --template "*cls*[Category:*mask*]*sent_0**sep+*" \
        --label-map "{0:'Company',1:'Education',2:'Artist',3:'Sports',4:'Office',5:'Transportation',6:'Building',7:'Natural',8:'Village',9:'Animal',10:'Plant',11:'Album',12:'Film',13:'Written'}" \
        --seed $seed \
        --truncate_head false \
        --gpu_id $gpu_id
done
