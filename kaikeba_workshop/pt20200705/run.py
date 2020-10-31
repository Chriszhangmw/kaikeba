
import os
import tensorflow as tf
from pt20200705.model import Project_model
from pt20200705.processor import process_function
from pt20200705.text_loader import TextLoader
import numpy as np


#超参数
epochs = 10
batch_size = 16
max_len = 64
lr = 5e-6  # 学习率
keep_prob = 0.8
bert_root = './bert_model_chinese'
bert_vocab_file = os.path.join(bert_root, 'vocab.txt')
model_save_path = './model/history.model'

#获取数据
data_path = './data'
train_input,eval_input,predict_input =process_function(data_path,bert_vocab_file,True,True,True,
                                               './temp',max_len,batch_size)
def train():
    model = Project_model(bert_root,data_path,'./temp',model_save_path,batch_size,max_len,lr,keep_prob)
    with tf.Session() as sess:
        # with tf.device('/gpu:0'):
        writer = tf.summary.FileWriter('./tf_log/', sess.graph)
        # saver = tf.train.Saver()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        data_loader = TextLoader(train_input,batch_size)
        for i in range(epochs):
            data_loader.shuff()
            for j in range(data_loader.num_batches):
                x_train,y_train = data_loader.next_batch(j)
                # print(y_train.shape)
                # print(y_train)
                step, loss_= model.run_step(sess,x_train,y_train)

                print('the epoch number is : %d the index of batch is :%d, the loss value is :%f'%(i, j, loss_))




if __name__ == '__main__':
    train()


