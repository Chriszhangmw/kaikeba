# -*- coding:utf-8 -*-
# Created by LuoJie at 11/29/19

from utils.gpu_utils import config_gpu

import tensorflow as tf

from pgn_tf2.batcher import batcher
from pgn_tf2.pgn_model import PGN
from pgn_tf2.train_helper import train_model
from utils.params_utils import get_params
from utils.wv_loader import Vocab


def train(params):
    # GPU资源配置
    # config_gpu(use_cpu=False, gpu_memory=params['gpu_memory'])
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
        tf.config.experimental.set_memory_growth(gpus[0], enable=True)
    # 读取vocab训练
    print("Building the model ...")
    vocab = Vocab(params["vocab_path"], params["max_vocab_size"])
    params['vocab_size'] = vocab.count

    # 构建模型
    print("Building the model ...")
    # model = Seq2Seq(params)
    model = PGN(params)

    print("Creating the batcher ...")
    dataset = batcher(vocab, params)
    # print('dataset is ', dataset)

    # 获取保存管理者
    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, params['checkpoint_dir'], max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    # 训练模型
    print("Starting the training ...")
    train_model(model, dataset, params, checkpoint_manager)


if __name__ == '__main__':
    # 获得参数m
    params = get_params()
    # 训练模型
    train(params)
