# Works better than the previous. Best so far in fact
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
translation.task_name="prompt_tst.yelp_gpt2_vocab_negative" \
translation.architecture="gpt2_conditioned_mlp" \
translation.training_mode="sql-onpolicy" \
translation.save_dir=./outputs \
translation.num_epochs=501 \
translation.num_batches_per_epoch=100 \
translation.save_frequency=100 \
translation.reward_old_min=-100 \
translation.reward_old_max=100 \
translation.reward_shaping_min=-60 \
translation.reward_shaping_max=30 \
translation.top_k=50 \
translation.reward_name="gpt2-classifier"
translation.random_seed=2