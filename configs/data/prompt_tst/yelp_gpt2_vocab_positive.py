import os


base_path = "/jupyter/prompt-generation/soft-Q-learning-for-text-generation/data/yelp-gpt2-words"
max_source_length = 512
max_decoding_length = 5

source_vocab_file = os.path.join(base_path, "vocab.source.positive")
target_vocab_file = os.path.join(base_path, "vocab.target.gpt2")

train = {
    "batch_size": 8,
    "allow_smaller_final_batch": False,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "train.source.positive"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "train.target.positive"),
        "vocab_file": target_vocab_file,
    }
}

# No Validation and Test
val = {
    "batch_size": 16,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "dev.source.positive"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "dev.target.positive"),
        "vocab_file": target_vocab_file,
    }
}

# No Validation and Test
test = {
    "batch_size": 16,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "test.source.positive"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "test.target.positive"),
        "vocab_file": target_vocab_file,
    }
}