#!/usr/bin/env python
# coding: utf-8
import argparse
import math
import os
import pickle
import sys
import random
import copy  
import numpy as np
import tf_slim as slim
from model_row import *
from tqdm import trange

seed_value = 0
random.seed(seed_value)
np.random.seed(randomseed)

"""
python main8.py --exp_id="43" --gpu_id="3" --direction="row" --use_cnn=True --use_rnn=True --"order"="rnn_first" --cnn_name="cnn_1d" --stacked_cnn_layers=4 --kernel_size="3" --strides_conv=1 --dilation_rate="1" --filters="64" --lr_init=0.0002 --lr_decay_rate=0.999999
"""
"""
python main8.py --exp_id="201" --gpu_id="3" --direction="row" --use_cnn=True --use_rnn=True --"order"="rnn_first" --cnn_name="cnn_1d" --stacked_cnn_layers=2 --kernel_size="3" --strides_conv=1 --dilation_rate="1" --filters="128" --lr_init=0.0002 --lr_decay_rate=0.999999
"""

parser = argparse.ArgumentParser(description="Training of cell state recognition for rows")

parser.add_argument("--exp_id", type=str, default="0", help="experiment id start from 1")
parser.add_argument("--gpu_id", type=str, default="0", help="avaliable gpu id: -1, 0, 1, 2, 3 (-1 for cpu)")
parser.add_argument("--steps_check", type=int, default=1000, help="steps to check loss and acc")
parser.add_argument("--epoch", type=int, default=10, help="maximum epoch number")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--direction", default="row", help="sequence direction")
parser.add_argument("--num_steps", type=int, default=100, help="sequence length")
parser.add_argument("--embedding_size", type=int, default=209, help="embedding size")

parser.add_argument("--order", type=str, default="rnn_first", help="cnn first or rnn first")

parser.add_argument("--use_cnn", type=bool, default=True, help="use cnn or not")
parser.add_argument("--cnn_name", type=str, default="cnn_1d", help="avaliable cnn models: cnn_1d, cnn_2d")
parser.add_argument("--stacked_cnn_layers", type=int, default=4, help="num of stacked cnn layers")
parser.add_argument("--filters", type=str, default="64", help="num of filters: |")
parser.add_argument("--kernel_size", type=str, default="3", help="kernel size: |")
parser.add_argument("--strides_conv", type=str, default="1", help="strides of convolutional layer: |")
parser.add_argument("--padding_conv", type=str, default="same", help="avaliable padding types for convolutional layer: same, valid")
parser.add_argument("--dilation_rate", type=str, default="1", help="dilation rate: |")
parser.add_argument("--pooling_name", type=str, default="max_pooling", help="avaliable poolings: max_pooling, avg_pooling")
parser.add_argument("--strides_pool", type=int, default=1, help="strides of pooling layer: 1, 2")
parser.add_argument("--padding_pool", type=str, default="valid", help="avaliable padding types for pooling layer: same, valid")
parser.add_argument("--pool_size", type=int, default=2, help="pooling size")
parser.add_argument("--input_hidden_size", type=int, default=50, help="input hidden size")

parser.add_argument("--use_rnn", type=bool, default=True, help="use rnn or not")
parser.add_argument("--rnn_name", type=str, default="lstm", help="avaliable rnn models: rnn, lstm, gru")
parser.add_argument("--lstm_dim", type=int, default=100, help="lstm dim")

parser.add_argument("--use_attention", type=bool, default=False, help="use attention or not")
parser.add_argument("--attention_size", type=int, default=100, help="attention size")

parser.add_argument("--num_tags", type=int, default=4, help="num tags")
parser.add_argument("--dropout_name", type=str, default="dropout", help="avaliable dropout methods: dropout, alpha_dropout, spatial_dropout_1d, gaussian_dropout")
parser.add_argument("--dropout_rate", type=str, default="0.55", help="keep probability rate = 1 - dropout rate: |")
parser.add_argument("--initializer_name", type=str, default="he_normal", help="avaliable initializers: xavier, he_normal, he_uniform, lecun_uniform, lecun_normal")
parser.add_argument("--activation_name", type=str, default="leaky_relu", help="avaliable activations function: sigmoid, tanh, relu, leaky_relu, elu, selu, crelu, relu6, softplus, softsign")
parser.add_argument("--optimizer_name", type=str, default="adam", help="avaliable optimizers: sgd, momentum, adagrad, adam, rmsprop, adadelta, adam, adagradda, adagradda, prox_adagrad, prox_sgd")
parser.add_argument("--clip", type=int, default=5, help="gradient clip")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--lr_init", type=float, default=0.00002, help="initial learning rate")
parser.add_argument("--lr_decay", type=str, default="exponential_decay", help="avaliable learning rate decay modes: none, polynomial_decay, exponential_decay, natural_exp_decay, cosine_decay, cosine_decay_restarts, linear_cosine_decay, noisy_linear_cosine_decay, inverse_time_decay")
parser.add_argument("--lr_decay_steps", type=int, default=30240, help="learning rate decay steps")
parser.add_argument("--lr_first_decay_steps", type=int, default=30240, help="learning rate first decay steps")
parser.add_argument("--lr_decay_rate", type=float, default=0.999999, help="learning rate decay rate")
parser.add_argument("--lr_staircase", type=bool, default=False, help="learning rate stair case mode")

args = parser.parse_args()
# args = parser.parse_args(args=[])
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


root_path = os.getcwd()
data_path = "./row/data_row/"
if not os.path.exists(os.path.join(root_path, "row_model/")):
    os.mkdir(os.path.join(root_path, "row_model/"))
model_path = "row_model/" + args.exp_id
if not os.path.exists(model_path):
    os.mkdir(model_path)
# log_path = os.path.join(root_path, "log/")
# if not os.path.exists(log_path):
#     os.mkdir(log_path)
# if not os.path.exists(os.path.join(log_path, args.direction)):
#     os.mkdir(os.path.join(log_path, args.direction))

class CellSequenceData(object):
    """row/col sequence
    """
    def __init__(self, file_id=None, sequence_id=None):
        self.row_id = []  # list of row id [seq_len]
        self.col_id = []  # list of col id [seq_len]
        self.label_id = []  # list of 0-4 [seq_len]
        self.valid_index_begin = 0  # closed interval
        self.valid_index_end = 0  # closed interval
        self.input_mask = []  # list of cell mask [seq_len] (this is for crf loss, generated from valid)
        self.cell_embed = []  # list of cell features [seq_len, 11 * 20]
        self.cell_length = 0  # cell length (this is for lstm pack/pad)
        self.file_id = file_id  # sheet file id
        self.sequence_id = sequence_id  # sequence id
    
    def readline(self, line):
        """read line from file
        """
        valid_len = 5 + 20 * 11
        line = line.strip().split(',')
        if len(line) != valid_len:
            raise ValueError(f"line length is not {valid_len}")
        self.row_id.append(int(line[0]))
        self.col_id.append(int(line[1]))
        self.label_id.append(int(line[2]))
        self.valid_index_begin = int(line[3])
        self.valid_index_end = int(line[4])
        features = [float(feat) for feat in line[5:]]
        self.cell_embed.append(features)
    
    def add_cell(self, row_id, col_id, valid_index_begin, valid_index_end, features):
        self.row_id.append(row_id)
        self.col_id.append(col_id)
        self.valid_index_begin = valid_index_begin
        self.valid_index_end = valid_index_end
        self.cell_embed.append(features)
    
    def terminate(self, max_length):
        self.cell_length = len(self.row_id)
        if max_length == -1:
            max_length = self.cell_length
        assert self.cell_length <= max_length, "cell length exceed max length"
        # assert self.cell_length == len(self.col_id) == len(self.label_id) == len(self.cell_embed), "length not match"
        # pad all list to max_length
        self.row_id.extend([-1] * (max_length - len(self.row_id)))
        self.col_id.extend([-1] * (max_length - len(self.col_id)))
        if len(self.label_id) > 0:  # in case of inference mode
            self.label_id.extend([0] * (max_length - len(self.label_id)))  # padding label is the same as outside, this point worth further dicsucssion
        self.cell_embed.extend([[0.] * 11 * 20] * (max_length - len(self.cell_embed)))
        
        # 0 for masked and 1 for valid
        self.input_mask = [0 for _ in range(max_length)]
        for i in range(self.valid_index_begin, self.valid_index_end + 1):
            self.input_mask[i] = 1
            
def load_pickle(data_path):
    cell_sequences_pickle = pickle.load(open(data_path, "rb"))
    batch_inputs_pickle = []
    batch_tags_pickle = []
    batch_lengths_pickle = []
    for cell_sequence_pickle in cell_sequences_pickle:
        if type(cell_sequence_pickle) == type(()):
            batch_inputs_pickle.append(cell_sequence_pickle[0].tolist())
            batch_tags_pickle.append(cell_sequence_pickle[3].tolist())
            batch_lengths_pickle.append(cell_sequence_pickle[2])
            continue
        batch_inputs_pickle.append(cell_sequence_pickle.cell_embed)
        batch_tags_pickle.append(cell_sequence_pickle.label_id)
        batch_lengths_pickle.append(cell_sequence_pickle.cell_length)
    return batch_inputs_pickle, batch_tags_pickle, batch_lengths_pickle

valid_inputs, valid_labels, valid_lengths = load_pickle(data_path + args.direction + "_valid" + ".pkl")
test_inputs, test_labels, test_lengths = load_pickle(data_path + args.direction + "_test" + ".pkl")
test_number = len(test_inputs)
valid_number = len(valid_inputs)
print("test number: ", test_number)
print("valid number:", valid_number)

default_inputs = []
default_inputs_sum = {}

def create_model(sess, args, model_class, model_path):
    model = model_class(args)

    model_id = len(os.listdir(model_path)) - 1
    model_path = model_path + "/" + str(model_id)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())
        
    return model, model_id


def save_model(sess, args, model, model_path, model_id):
    if not os.path.exists(model_path + "/" + str(model_id)):
        os.mkdir(model_path + "/" + str(model_id))
    model_checkpoint_path = os.path.join(model_path + "/" + str(model_id), "exp" + args.exp_id + ".ckpt")
    model.saver.save(sess, model_checkpoint_path)
    print("model {:<3} saved\n".format(model_id))

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
def sample_default_input():
    default_input = copy.deepcopy(random.choice(default_inputs))
    for i in range(209):
        if random.random() < 0.01:
            default_input[i] = 1 - default_input[i]
    if random.random() < 0.7:
        default_input = default_input * 0
    
    return default_input
    
def extract_default_input(batch_inputs, batch_tags, batch_lengths):
    bs = len(batch_inputs)
    for i in range(2000):
        for j in range(len(batch_inputs[i][:batch_lengths[i]])):
            if batch_tags[i][j] == 0:
                if str(sum(batch_inputs[i][j])) not in default_inputs_sum or default_inputs_sum[str(sum(batch_inputs[i][j]))] < 100:
                    default_inputs_sum[str(sum(batch_inputs[i][j]))] = 1 if str(sum(batch_inputs[i][j])) not in default_inputs_sum else default_inputs_sum[str(sum(batch_inputs[i][j]))] + 1
                    default_inputs.append(batch_inputs[i][j])
    return True


def clip(batch_inputs, batch_tags, batch_lengths):
    batch_inputs = copy.deepcopy(batch_inputs)
    batch_tags = copy.deepcopy(batch_tags)
    batch_lengths = copy.deepcopy(batch_lengths)
    bs = len(batch_inputs)
    for i in range(bs):
        useAugfeat = random.random() > 0.5
        useAugBegining = random.random() > 10.5
        default_tag = 0
        default_input = sample_default_input()
        j = len(batch_inputs[i]) - 1
        while j > 0 and batch_tags[i][j] == 0:
            j -= 1
        if useAugfeat: 
            batch_lengths[i] = min(j + int(random.random()*3),len(batch_inputs[i]))
    return batch_inputs, batch_tags, batch_lengths

def augment_padding_start(batch_input, batch_length, batch_tag):
    default_tag = 0
    default_input = sample_default_input()
    if random.random() > 0.7:
        for i in range(int(random.random()*6)):
            batch_input = [default_input] + batch_input
            batch_tag = [default_tag] + batch_tag
            batch_length += 1
    default_input = sample_default_input()  * 0
    if random.random() > 0.5:
        for i in range(4 + int(random.random()*3)):
            batch_input = [default_input] + batch_input
            batch_tag = [default_tag] + batch_tag
            batch_length += 1
    
    return batch_input, batch_length, batch_tag

def augment_padding_end(batch_input, batch_length, batch_tag):
    default_tag = 0
    default_input = sample_default_input()  * 0
    if random.random() > 0.5:
        for i in range(4 + int(random.random()*3)):
            batch_input.append(default_input)
            batch_tag.append(default_tag)
            batch_length += 1
    return batch_input, batch_length, batch_tag
    
def augment_exclude_data(batch_input, batch_length, batch_tag):
    batch_input_result = []
    batch_tag_result = []
    batch_length_result = 0
    if random.random() > 0.7:
        for i in range(len(batch_input)):
            if batch_tag[i] != 3 or random.random()>0.5:
                batch_input_result.append(batch_input[i])
                batch_tag_result.append(batch_tag[i])
                batch_length_result += 1
        return batch_input_result, batch_length_result, batch_tag_result
    return batch_input, batch_length, batch_tag
    
def augment(batch_inputs, batch_tags, batch_lengths):
    batch_inputs = copy.deepcopy(batch_inputs)
    batch_tags = copy.deepcopy(batch_tags)
    batch_lengths = copy.deepcopy(batch_lengths)
    augment_rate = 0.5
    augment_rate_feature = 0.6
    same_augment_rate = 0.3
    insert_range = [0,0,1,1,1,2,2,3,4]
    insert_range_end = [i for i in range(100)]
    bs = len(batch_inputs)
    l = len(batch_inputs[0])
    augment_batch_inputs = []
    augment_batch_lengths = []
    augment_batch_tags = []
    for i in range(bs):
        for k in range(len(batch_inputs[i])):
            for t in range(11):
                batch_inputs[i][k][t*19+17] = batch_inputs[i][k][t*19+17] * 0
    for i in range(bs):
        if random.random() < augment_rate and batch_tags[i][-1] == 0 and batch_lengths[i] > 2 and batch_lengths[i] < l:
            default_tag = 0
            default_input = sample_default_input()
            augment_batch_input = batch_inputs[i][:batch_lengths[i]]
            augment_batch_length = batch_lengths[i]
            augment_batch_tag = batch_tags[i][:batch_lengths[i]]
            augment_batch_input, augment_batch_length, augment_batch_tag = augment_padding_start(augment_batch_input, augment_batch_length, augment_batch_tag)
            j = i
            while augment_batch_length < l and random.random() < augment_rate:
                j = int(random.random()*bs) if random.random() > same_augment_rate else j
                
                for _ in range(random.choice(insert_range)):
                    augment_batch_input.append(sample_default_input())
                    augment_batch_tag.append(default_tag)
                    augment_batch_length += 1
                if augment_batch_length + batch_lengths[j] >= l: break
                augment_batch_input.extend(batch_inputs[j][:batch_lengths[j]])
                augment_batch_tag.extend(batch_tags[j][:batch_lengths[j]])
                augment_batch_length += batch_lengths[j]
            end_interval = random.choice(insert_range_end)
            if random.random()<0.7:end_interval=random.choice(insert_range)
            for _ in range(end_interval):
                augment_batch_input.append(sample_default_input())
                augment_batch_tag.append(default_tag)
                augment_batch_length += 1
            augment_batch_input, augment_batch_length, augment_batch_tag = augment_padding_end(augment_batch_input, augment_batch_length, augment_batch_tag)
            
            augment_batch_input, augment_batch_length, augment_batch_tag = augment_exclude_data(augment_batch_input, augment_batch_length, augment_batch_tag)
            for _ in range(augment_batch_length, l):
                augment_batch_input.append(sample_default_input())
                augment_batch_tag.append(default_tag)
                
            if random.random() < augment_rate_feature:
                for k in range(len(augment_batch_input)):
                    for t in range(11):
                        augment_batch_input[k][t*19+6:t*19+11] = augment_batch_input[k][t*19+6:t*19+11] * 0
                        augment_batch_input[k][t*19+16:t*19+19] = augment_batch_input[k][t*19+16:t*19+19] * 0
            
            augment_batch_inputs.append(augment_batch_input[:l])
            augment_batch_lengths.append(min(augment_batch_length,l))
            augment_batch_tags.append(augment_batch_tag[:l])
            
        else:
            default_tag = 0
            if random.random() < augment_rate_feature:
                for k in range(len(batch_inputs[i])):
                    for t in range(11):
                        batch_inputs[i][k][t*19+6:t*19+11] = batch_inputs[i][k][t*19+6:t*19+11] * 0
                        batch_inputs[i][k][t*19+16:t*19+19] = batch_inputs[i][k][t*19+16:t*19+19] * 0
            augment_batch_input = batch_inputs[i][:batch_lengths[i]]
            augment_batch_length = batch_lengths[i]
            augment_batch_tag = batch_tags[i][:batch_lengths[i]]
            augment_batch_input, augment_batch_length, augment_batch_tag = augment_padding_start(augment_batch_input, augment_batch_length, augment_batch_tag)
            end_interval = random.choice(insert_range_end)
            if random.random()<0.7:end_interval=random.choice(insert_range)
            for _ in range(end_interval):
                augment_batch_input.append(sample_default_input())
                augment_batch_tag.append(default_tag)
                augment_batch_length += 1
            augment_batch_input, augment_batch_length, augment_batch_tag = augment_padding_end(augment_batch_input, augment_batch_length, augment_batch_tag)
            augment_batch_input, augment_batch_length, augment_batch_tag = augment_exclude_data(augment_batch_input, augment_batch_length, augment_batch_tag)
            for _ in range(augment_batch_length, l):
                augment_batch_input.append(sample_default_input())
                augment_batch_tag.append(default_tag)
            
            augment_batch_inputs.append(augment_batch_input[:l])
            augment_batch_lengths.append(min(augment_batch_length,l))
            augment_batch_tags.append(augment_batch_tag[:l])
    return augment_batch_inputs, augment_batch_tags, augment_batch_lengths


train_inputs, train_labels, train_lengths = load_pickle(data_path + args.direction + "_train_" + str(0) + ".pkl")
extract_default_input(train_inputs, train_labels, train_lengths)
valid_inputs_1, valid_labels_1, valid_lengths_1 = clip(valid_inputs, valid_labels, valid_lengths)
valid_inputs_1, valid_labels_1, valid_lengths_1 = augment(valid_inputs_1, valid_labels_1, valid_lengths_1)

valid_inputs_2, valid_labels_2, valid_lengths_2 = clip(valid_inputs, valid_labels, valid_lengths)
valid_inputs_2, valid_labels_2, valid_lengths_2 = augment(valid_inputs_2, valid_labels_2, valid_lengths_2)

valid_inputs_3, valid_labels_3, valid_lengths_3 = clip(valid_inputs, valid_labels, valid_lengths)
valid_inputs_3, valid_labels_3, valid_lengths_3 = augment(valid_inputs_3, valid_labels_3, valid_lengths_3)
print(type(valid_inputs_1), type(valid_labels_1), type(valid_lengths_1))
valid_inputs = valid_inputs_1 + valid_inputs_2 + valid_inputs_3
valid_labels = valid_labels_1+valid_labels_2+valid_labels_3
valid_lengths = valid_lengths_1 + valid_lengths_2 +valid_lengths_3

def train(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model, model_id = create_model(sess, args, BILSTM_CRF, model_path)
        model_summary()
        print("Start training.")
        loss = []
        best_valid_b_f1 = -1
        save_test_b_f1 = -1
        best_test_b_f1 = -1
        n_file = sum([1 if args.direction + "_train" in filename else 0 for filename in os.listdir(data_path)])
        step = -1
        for epoch in range(args.epoch):
            print("=" * 25 + "EPOCH " + str(epoch) + "=" * 25 + "\n")
            for file_id in range(n_file):
                train_inputs, train_labels, train_lengths = load_pickle(data_path + args.direction + "_train_" + str(file_id) + ".pkl")
                #train_inputs, train_labels, train_lengths = load_pickle(open("/spot/hadong/TableSenseMLRevise/data_row/row_train_" + str(0))
                for batch_id in range(math.ceil(len(train_inputs) / args.batch_size)):
                    batch_inputs = train_inputs[batch_id * args.batch_size : (batch_id + 1) * args.batch_size]
                    batch_lengths = train_lengths[batch_id * args.batch_size : (batch_id + 1) * args.batch_size]
                    batch_tags = train_labels[batch_id * args.batch_size : (batch_id + 1) * args.batch_size]
                    batch_inputs, batch_tags, batch_lengths = clip(batch_inputs, batch_tags, batch_lengths)
                    batch_inputs, batch_tags, batch_lengths = augment(batch_inputs, batch_tags, batch_lengths)
                    batch_inputs = np.array(batch_inputs)
                    batch_tags = np.array(batch_tags)
                    _, batch_loss = model.run_step(sess, True, batch_inputs, batch_lengths, batch_tags)
                    loss.append(batch_loss)
                    step += 1
                    if step % args.steps_check == 0:
                        check = step // args.steps_check
                        valid_f1, valid_b_precision, valid_b_recall, valid_b_f1, valid_metric = model.evaluate_set(sess, valid_inputs, valid_lengths, valid_labels)
                        test_f1, test_b_precision, test_b_recall, test_b_f1, test_metric = model.evaluate_set(sess, test_inputs, test_lengths, test_labels)
                        if valid_b_f1 > best_valid_b_f1:
                            best_valid_b_f1 = valid_b_f1
                            save_test_b_f1 = test_b_f1
                            model_id += 1
                            save_model(sess, args, model, model_path, model_id)
                        if test_b_f1 > best_test_b_f1:
                            best_test_b_f1 = test_b_f1
                        print("iter:{:<9} epoch:{:<9} file:{:<9} batch:{:<9} step:{:<9}".format(check, epoch, file_id, batch_id, step))
                        print("train_loss:{:<9.6f} best_valid_border_f1:{:<9.6f} save_test_border_f1:{:<9.6f} best_test_b_f1:{:<9.6f}".format(np.mean(loss), best_valid_b_f1, save_test_b_f1, best_test_b_f1))
                        print("valid f1:{:<9.6f} border_precision:{:<9.6f} border_recall:{:<9.6f} border_f1:{:<9.6f}".format(valid_f1, valid_b_precision, valid_b_recall, valid_b_f1))
                        print("test f1:{:<9.6f} border_precision:{:<9.6f} border_recall:{:<9.6f} border_f1:{:<9.6f}".format(test_f1, test_b_precision, test_b_recall, test_b_f1))
                        print("valid\n" + valid_metric)
                        print("test\n" + test_metric)
                        sys.stdout.flush()
                        loss = []
            valid_f1, valid_b_precision, valid_b_recall, valid_b_f1, valid_metric = model.evaluate_set(sess, valid_inputs, valid_lengths, valid_labels)
            if valid_b_f1 > best_valid_b_f1:
                best_valid_b_f1 = valid_b_f1
                model_id += 1
                save_model(sess, args, model, model_path, model_id)
            test_f1, test_b_precision, test_b_recall, test_b_f1, test_metric = model.evaluate_set(sess, test_inputs, test_lengths, test_labels)
            print("valid f1:{:<9.6f} border_precision:{:<9.6f} border_recall:{:<9.6f} border_f1:{:<9.6f}".format(valid_f1, valid_b_precision, valid_b_recall, valid_b_f1))
            print("test f1:{:<9.6f} border_precision:{:<9.6f} border_recall:{:<9.6f} border_f1:{:<9.6f}".format(test_f1, test_b_precision, test_b_recall, test_b_f1))
            print("valid\n" + valid_metric)
            print("test\n" + test_metric)
            sys.stdout.flush()


if __name__ == "__main__":
    train(args)
