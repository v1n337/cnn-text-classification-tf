positive_data_file = "./data/rt-polaritydata/rt-polarity.pos"
negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"

# Eval Parameters
batch_size = 64
checkpoint_dir = ""
eval_train = False

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

# Data loading params
dev_sample_percentage = .1

# Model Hyperparameters
embedding_dim = 128
filter_sizes = "3,4,5"
num_filters = 128
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0

# Training parameters
num_epochs = 200
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5