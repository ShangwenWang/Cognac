import os

root_dir = os.path.expanduser("~")

# train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "path2train")
eval_data_path = os.path.join(root_dir, "path2validation")
decode_data_path = os.path.join(root_dir, "path2test")
vocab_path = os.path.join(root_dir, "path2vocab")
log_root = os.path.join(root_dir, "path2log")
excluded_type = {}

# Hyperparameters
hidden_dim= 400
emb_dim= 150

stmt_emb_dim = 20
batch_size = 120
max_enc_steps=350
max_dec_steps=6
beam_size=4
min_dec_steps=1
vocab_size=85000
stmt_size = 40
lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0
dropout_prob = 0.25
pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.15
