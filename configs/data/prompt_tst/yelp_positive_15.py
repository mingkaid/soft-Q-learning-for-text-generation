import os


base_path = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only"
max_source_length = 512
max_decoding_length = 15

source_vocab_file = os.path.join(base_path, "vocab.source.positive")
target_vocab_file = os.path.join(base_path, "vocab.target")

train = {
    "batch_size": 12,
    "allow_smaller_final_batch": False,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "train.source.positive"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "train.target.positive_15"),
        "vocab_file": target_vocab_file,
    }
}

# No Validation and Test
val = {
    "batch_size": 10,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "dev.source.positive"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "dev.target.positive_15"),
        "vocab_file": target_vocab_file,
    }
}

# No Validation and Test
test = {
    "batch_size": 10,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "dev.source.positive"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "dev.target.positive_15"),
        "vocab_file": target_vocab_file,
    }
}
