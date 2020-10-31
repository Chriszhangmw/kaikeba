# text_gcn

The implementation of Text GCN in our paper:

Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377


## Require

Python 2.7 or 3.6

Tensorflow >= 1.4.0

## Reproducing Results

1. Run `python remove_words.py 20ng`

2. Run `python build_graph.py 20ng`

3. Run `python train.py 20ng`

4. Change `20ng` in above 3 command lines to `R8`, `R52`, `ohsumed` and `mr` when producing results for other datasets.

## Example input data

1. `/data/20ng.txt` indicates document names, training/test split, document labels. Each line is for a document.

2. `/data/corpus/20ng.txt` contains raw text of each document, each line is for the corresponding line in `/data/20ng.txt`

3. `prepare_data.py` is an example for preparing your own data, note that '\n' is removed in your documents or sentences.

## Inductive version

An inductive version of Text GCN is [fast_text_gcn](https://github.com/yao8839836/fast_text_gcn), where test documents are not included in training process.



# 说明

本项目原始git:https://github.com/yao8839836/text_gcn

本zip包只提供了`mr`和`baidu_95`数据集



运行GCN示例模型，使用`mr`数据集

步骤：

1. Run `python remove_words.py mr ` 
2. Run `python build_graph.py mr`
3. Run `python train.py mr`

说明：

`data/{dataset}.txt`是label文件，格式是 `id` `train/text标识` `类别`

`data/corpus/{dataset}.txt` 是原始数据(只有x数据)



运行`baidu_95`数据

1. 单标签文本分类：

   修改`baidu/generate_datafile.py`中

   ``` python
   # split dataset
   # sublabel 先使用单标签分类验证模型效果
   sublabel = 2
   index_split = len(df) * 0.9
   with open(f'../data/{dataset}.txt', 'w') as f:
       for index, row in df.iterrows():
           category = 'train' if index <= index_split else 'test'
           # single lable
           f.write(f'{index}\t{category}\t{row[0].split()[sublabel]}\n')
           # multi_label
           # lables = '\t'.join(row[0].split())
           # f.write(f"{index}\t{category}\t{lables}\n")
   ```

   可查看原始数据文件`baidu_95.csv` 这里的sublabel=1是使用 `生物 历史 地理 政治`作为文本的单标签，=2是使用下一级的知识点作为单标签   =1 =2的单标签效果都非常好

   这一步是生成数据集文件，即`remove_words.py`的效果

   然后使用 `python build_graph baidu_95`创建图特征数据

   然后使用 `python train.py baidu_95`进行单标签训练

2. 多标签文本分类：

   首先修改`baidu/generate_datafile.py`中

   ``` python
   # split dataset
   # sublabel 先使用单标签分类验证模型效果
   # sublabel = 2
   index_split = len(df) * 0.9
   with open(f'../data/{dataset}.txt', 'w') as f:
       for index, row in df.iterrows():
           category = 'train' if index <= index_split else 'test'
           # single lable
           # f.write(f'{index}\t{category}\t{row[0].split()[sublabel]}\n')
           # multi_label
           lables = '\t'.join(row[0].split())
           f.write(f"{index}\t{category}\t{lables}\n")
   ```

   然后 使用`build_graph.py baidu_95`

   然后运行`python train.py baidu_95 1` 第二个1参数表示使用`multi_label`进行训练

   遇到了Loss下降但评估指标一成不变的问题



# 其它

模型的多标签主要是修改了模型的loss和评估，都放在了`metrics.py`文件中，可以查看