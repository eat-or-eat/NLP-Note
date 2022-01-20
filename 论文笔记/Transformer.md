论文地址:[1706.03762.pdf (arxiv.org)](https://arxiv.org/pdf/1706.03762.pdf)

# Transformer

## 一，历史影响

1. 在2014年的翻译任务重出现了attention机制，随后不断地发展attention越来越多的用在NLP模型结构中，同样Transformer中也用了attention机制
2. Transformer沿用Encoder-Decoder结构，主要由self-attention和FFN两种层组合而成
3. 在Transformer出现后，大量预训练模型都使用Transformer的Encoder部分替换掉了RNN、GRU、LSTM等结构。比如Bert，GPT等各种预训练模型

## 二，模型结构

### 1.输入部分

1. 将单个字做Embedding
2. 通过相对位置编码获取字的位置信息
3. 将上面两个向量相加作为输入

### 2.编码与解码部分

通过Multi-head串联Feed Forward结构组成基本单元来堆叠N次形成

一些可以关注的点

1. 三种Multi head attention
2. self attention与Scaled Dot-Product Attention
3. Add&Norm层(残差机制)
4. 三种Regularization，其中两种是dorpout，一种是label smoothing

### 3.预测输出部分

通过一个线性层将输出转换成待生成字的概率

## 三，细节记录

pass

## 四，总结

优点:

1. Transformer最大的创新点在于完全基于attention机制实现Encoder-Decoder结构
2. 获取句子之间的相关性可以在常数时间内完成，因为Self-attention计算两个位置之间的关联所需的操作次数是与距离无关的

缺点:

1. 结构本身无法获取字的位置信息，需要添加Position Embedding来获取
