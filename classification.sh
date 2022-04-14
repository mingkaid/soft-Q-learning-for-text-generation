# CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
# translation.task_name="prompt_tst.yelp_gpt2_vocab_negative" \
# translation.architecture="gpt2_conditioned_mlp" \
# translation.training_mode="sql-onpolicy" \
# translation.save_dir=./outputs \
# translation.num_epochs=501 \
# translation.num_batches_per_epoch=100 \
# translation.save_frequency=100 \
# translation.reward_old_min=-100 \
# translation.reward_old_max=100 \
# translation.reward_shaping_min=-70 \
# translation.reward_shaping_max=30 \
# translation.top_k=50 \
# translation.reward_name="roberta-classifier" \
# translation.random_seed=2 \
# translation.learning_rate=0.0001
# translation.reward_name="gpt2-classifier" \

for seed in 13 21 42 87 100
do
#     TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
#     translation.task_name="prompt_tst.yelp_gpt2_vocab_negative" \
#     translation.architecture="gpt2_conditioned_mlp" \
#     translation.training_mode="sql-onpolicy" \
#     translation.save_dir=./outputs \
#     translation.num_epochs=20 \
#     translation.num_batches_per_epoch=100 \
#     translation.save_frequency=100 \
#     translation.reward_old_min=-100 \
#     translation.reward_old_max=100 \
#     translation.reward_shaping_min=-65 \
#     translation.reward_shaping_max=35 \
#     translation.top_k=50 \
#     translation.reward_name="roberta-glue" \
#     translation.random_seed=$seed \
#     translation.learning_rate=0.0001 \
#     translation.classification.task_name="mpqa" \
#     translation.classification.template="*cls**sent_0*_It_was*mask*.*sep+*" \
#     translation.classification.data_dir="/home/c2hsieh/soft-Q-learning-for-text-generation/tasks/k-shot/mpqa/16-$seed" \
#     translation.classification.mapping="'{0:\'terrible\',1:\'great\'}'"
    
#     TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
#     translation.task_name="prompt_tst.yelp_gpt2_vocab_negative" \
#     translation.architecture="gpt2_conditioned_mlp" \
#     translation.training_mode="sql-onpolicy" \
#     translation.save_dir=./outputs \
#     translation.num_epochs=20 \
#     translation.num_batches_per_epoch=100 \
#     translation.save_frequency=100 \
#     translation.reward_old_min=-100 \
#     translation.reward_old_max=100 \
#     translation.reward_shaping_min=-35 \
#     translation.reward_shaping_max=65 \
#     translation.top_k=50 \
#     translation.reward_name="roberta-glue" \
#     translation.random_seed=$seed \
#     translation.learning_rate=0.0001 \
#     translation.classification.task_name="sst-5" \
#     translation.classification.template="*cls**sent_0*_It_was*mask*.*sep+*" \
#     translation.classification.data_dir="/home/c2hsieh/soft-Q-learning-for-text-generation/tasks/k-shot/sst-5/16-$seed" \
#     translation.classification.mapping="'{0:\'terrible\',1:\'bad\',2:\'okay\',3:\'good\',4:\'great\'}'"

#     TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
#     translation.task_name="prompt_tst.yelp_gpt2_vocab_negative" \
#     translation.architecture="gpt2_conditioned_mlp" \
#     translation.training_mode="sql-onpolicy" \
#     translation.save_dir=./outputs \
#     translation.num_epochs=100 \
#     translation.num_batches_per_epoch=100 \
#     translation.save_frequency=100 \
#     translation.reward_old_min=-100 \
#     translation.reward_old_max=100 \
#     translation.reward_shaping_min=-40 \
#     translation.reward_shaping_max=60 \
#     translation.top_k=50 \
#     translation.reward_name="roberta-glue" \
#     translation.random_seed=$seed \
#     translation.learning_rate=0.0001 \
#     translation.classification.task_name="mnli" \
#     translation.classification.template="*cls**sent-_0*_*mask*\,*+sentl_1**sep+*" \
#     translation.classification.data_dir="/home/c2hsieh/soft-Q-learning-for-text-generation/tasks/k-shot/MNLI/16-$seed" \
#     translation.classification.mapping="'{\'contradiction\':\'No\',\'entailment\':\'Yes\',\'neutral\':\'Maybe\'}'"


#     TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
#     translation.task_name="prompt_tst.yelp_gpt2_vocab_negative" \
#     translation.architecture="gpt2_conditioned_mlp" \
#     translation.training_mode="sql-onpolicy" \
#     translation.save_dir=./outputs \
#     translation.num_epochs=100 \
#     translation.num_batches_per_epoch=100 \
#     translation.save_frequency=100 \
#     translation.reward_old_min=-1 \
#     translation.reward_old_max=0 \
#     translation.reward_shaping_min=-3 \
#     translation.reward_shaping_max=0 \
#     translation.top_k=50 \
#     translation.reward_name="roberta-glue-piece" \
#     translation.random_seed=$seed \
#     translation.learning_rate=0.001 \
#     translation.classification.task_name="mnli" \
#     translation.classification.template="*cls**sent-_0*_*mask*\,*+sentl_1**sep+*" \
#     translation.classification.data_dir="/home/c2hsieh/soft-Q-learning-for-text-generation/tasks/k-shot/MNLI/16-$seed" \
#     translation.classification.mapping="'{\'contradiction\':\'No\',\'entailment\':\'Yes\',\'neutral\':\'Maybe\'}'"

    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
    translation.task_name="prompt_tst.yelp_gpt2_vocab_negative" \
    translation.architecture="gpt2_conditioned_mlp" \
    translation.training_mode="sql-onpolicy" \
    translation.save_dir=./outputs \
    translation.num_epochs=100 \
    translation.num_batches_per_epoch=100 \
    translation.save_frequency=100 \
    translation.reward_old_min=-100 \
    translation.reward_old_max=100 \
    translation.reward_shaping_min=-60 \
    translation.reward_shaping_max=40 \
    translation.top_k=50 \
    translation.reward_name="roberta-glue" \
    translation.random_seed=$seed \
    translation.learning_rate=0.0001 \
    translation.classification.task_name="mrpc" \
    translation.classification.template="*cls**sent_0**mask*\,*+sentl_1**sep+*" \
    translation.classification.data_dir="/home/c2hsieh/soft-Q-learning-for-text-generation/tasks/k-shot/MRPC/16-$seed" \
    translation.classification.mapping="'{\'0\':\'No\',\'1\':\'Yes\'}'"
    
done




# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python run_experiments.py \
# translation.task_name="prompt_tst.yelp_gpt2_vocab_negative" \
# translation.architecture="gpt2_conditioned_mlp" \
# translation.training_mode="sql-onpolicy" \
# translation.save_dir=./outputs \
# translation.num_epochs=501 \
# translation.num_batches_per_epoch=100 \
# translation.save_frequency=100 \
# translation.reward_old_min=-100 \
# translation.reward_old_max=100 \
# translation.reward_shaping_min=-35 \
# translation.reward_shaping_max=65 \
# translation.top_k=50 \
# translation.reward_name="roberta-glue" \
# translation.random_seed=21 \
# translation.learning_rate=0.0001 \
# translation.classification.task_name="sst-5" \
# translation.classification.template="*cls**sent_0*_It_was*mask*.*sep+*" \
# translation.classification.data_dir="/home/c2hsieh/soft-Q-learning-for-text-generation/tasks/k-shot/sst-5/16-21" \
# translation.classification.mapping="'{0:\'terrible\',1:\'bad\',2:\'okay\',3:\'good\',4:\'great\'}'"
