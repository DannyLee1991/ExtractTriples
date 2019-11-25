import os

DATASET_DIR = "./datasets"

train_data_path = os.path.join(DATASET_DIR, 'train_data.json')
train_data_me_path = os.path.join(DATASET_DIR, 'train_data_me.json')
dev_data_path = os.path.join(DATASET_DIR, 'dev_data.json')
dev_data_me_path = os.path.join(DATASET_DIR, 'dev_data_me.json')

all_50_schemas_path = os.path.join(DATASET_DIR, 'all_50_schemas')
all_50_schemas_me_path = os.path.join(DATASET_DIR, 'all_50_schemas_me.json')
all_chars_me_path = os.path.join(DATASET_DIR, 'all_chars_me.json')

random_order_vote_path = './random_order_vote.json'

# 模型文件路径
model_path = './model/best_model.weights'

# word2vec词向量模型路径
w2v_model_path = "{your_w2v_model_path}"

mode = 0
char_size = 128
maxlen = 512
