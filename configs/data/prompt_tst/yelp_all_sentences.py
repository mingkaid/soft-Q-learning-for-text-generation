import os


base_path = "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2"
max_source_length = 512
max_decoding_length = 5

source_vocab_file = os.path.join(base_path, "vocab.source")
target_vocab_file = os.path.join(base_path, "vocab.target")

train = {
    "batch_size": 12,
    "allow_smaller_final_batch": False,
    "source_dataset": {
        "files": os.path.join(base_path, "train.source.all_sentences"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "train.target.all_sentences"),
        "vocab_file": target_vocab_file,
    }
}

# No Validation and Test
val = {
    "batch_size": 7,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "validation.source.balanced_10_samples"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "validation.target.balanced_10_samples"),
        "vocab_file": target_vocab_file,
    }
}

# No Validation and Test
test = {
    "batch_size": 7,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "validation.source.balanced_10_samples"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "validation.target.balanced_10_samples"),
        "vocab_file": target_vocab_file,
    }
}
