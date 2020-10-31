# -*- coding:utf-8 -*-
# Created by LuoJie at 12/12/19
import tensorflow as tf
from pgn_tf2.batcher import batcher
from pgn_tf2.pgn_model import PGN
from tqdm import tqdm

from pgn_tf2.test_helper import beam_decode
from seq2seq_tf2.test import save_predict_result
from utils.config import checkpoint_dir
from utils.gpu_utils import config_gpu
from utils.wv_loader import Vocab
from utils.params_utils import get_params


def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    # GPU资源配置
    # config_gpu(use_cpu=True)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
        tf.config.experimental.set_memory_growth(gpus[0], enable=True)

    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")
    result = []

    if params['mode'] == 'test':
        for batch in b:
            # yield beam_decode(model, batch, vocab, params)
            result.append(beam_decode(model, batch, vocab, params))
    return result






def predict_result(params, result_save_path):
    # 预测结果
    results = test(params)
    print('prediction results is : ',results)
    # 保存结果
    # save_predict_result(results, result_save_path)


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    params['batch_size'] = 3
    params['beam_size'] = 3
    params['mode'] = 'test'
    test(params)
