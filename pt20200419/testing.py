import tensorflow as tf
from models.seq2seq import SequenceToSequence
from utils.batcher_utils import Vocab, batcher
from utils.test_helper import batch_greedy_decode
from tqdm import tqdm
import  pandas  as pd

def test(params):
    global model, ckpt, checkpoint_dir
    assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    print("Building the model ...")
    if params["model"] == "SequenceToSequence":
        model = SequenceToSequence(params)
    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Creating the checkpoint manager")
    if params["model"] == "SequenceToSequence":
        checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), SequenceToSequence=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")
    for batch in b:
        yield batch_greedy_decode(model, batch, vocab, params)

def test_and_save(params):
    assert params["test_save_dir"], "provide a dir where to save the results"
    gen = test(params)
    results = []
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            trial = list(map(lambda x: x.replace(" ", ""), trial))
            results.append(trial[0])
            pbar.update(1)
    save_predict_result(results)



def save_predict_result(results):
    # 读取结果
    test_df = pd.read_csv('/root/pt20200419/data/test.csv')
    # 填充结果
    test_df['Prediction'] = results
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    test_df.to_csv('/root/pt20200419/data/test_results.csv', index=None, sep=',')

if __name__ == '__main__':
    test('我的帕萨特烧机油怎么办')