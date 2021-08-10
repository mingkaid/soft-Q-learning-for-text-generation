import os


base_path = "/workspace/soft-Q-learning-for-text-generation/data/yelp-gpt2-control-only"
max_source_length = 512
max_decoding_length = 5

source_vocab_file = os.path.join(base_path, "vocab.source.reconstruct")
target_vocab_file = os.path.join(base_path, "vocab.target")

train = {
    "batch_size": 1,
    "allow_smaller_final_batch": False,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "train.source.reconstruct"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "train.target.reconstruct"),
        "vocab_file": target_vocab_file,
    }
}

# No Validation and Test
val = {
    "batch_size": 1,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "dev.source.reconstruct_1"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "dev.target.reconstruct_1"),
        "vocab_file": target_vocab_file,
    }
}

# No Validation and Test
test = {
    "batch_size": 1,
    "shuffle": False,
    "source_dataset": {
        "files": os.path.join(base_path, "dev.source.reconstruct_1"),
        "vocab_file": source_vocab_file,
    },
    "target_dataset": {
        "files": os.path.join(base_path, "dev.target.reconstruct_1"),
        "vocab_file": target_vocab_file,
    }
}
