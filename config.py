train_positive_data_file = "./data/pos.txt"
train_negative_data_file = "./data/neg.txt"

eval_positive_data_file = "./data/pos-eval.txt"
eval_negative_data_file = "./data/neg-eval.txt"

# Eval Parameters
batch_size = 64
eval_train = False

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

# Data loading params
dev_sample_percentage = .01

# Model Hyperparameters
embedding_dim = 128
filter_sizes = "3,4,5"
num_filters = 128
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0

# Training parameters
num_epochs = 100
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5
