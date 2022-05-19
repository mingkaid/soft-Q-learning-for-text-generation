python mlm-pt.py \
    --task "SST-2" \
    --template "*cls**sent_0*_It_was*mask*.*sep+*" \
    --label-map "{'0':'terrible','1':'great'}" \
    --seed 13 \
    --truncate_head true \
    --gpu_id 1