# Doesn't work
CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
	translation.task_name="prompt_tst.yelp_gpt2_vocab_negative_kshot16" \
	translation.architecture="gpt2_conditioned_mlp" \
	translation.training_mode="sql-onpolicy" \
	translation.save_dir=./outputs \
	translation.num_epochs=701 \
	translation.num_batches_per_epoch=100 \
	translation.save_frequency=100 \
	translation.reward_old_min=0 \
	translation.reward_old_max=1 \
	translation.learning_rate=5e-5\
	translation.reward_shaping_min=0 \
	translation.reward_shaping_max=1\
	translation.top_k=50\
	translation.random_seed=0\
	translation.reward_name=plm-classifier\
	translation.LM_type=gpt2\
	translation.experiment=FSL\
	translation.experiment_seed=0\
	translation.kshot=16
