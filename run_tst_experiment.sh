python run_experiments.py \
translation.task_name="prompt_tst.yelp_negative" \
translation.training_mode="sql-onpolicy" \
translation.save_dir=/data/yihan.wang/outputs \
translation.num_epochs=501 \
translation.num_batches_per_epoch=100 \
translation.save_frequency=100 \
translation.reward_old_min=60 \
translation.reward_old_max=100 \
translation.reward_shaping_min=-50 \
translation.reward_shaping_max=50 \
translation.top_k=50 \
translation.reward_name="gpt2-sentiment-bleu-no-input"