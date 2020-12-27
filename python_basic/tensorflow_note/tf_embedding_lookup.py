"""
tf.nn.embedding_lookup(
               params,
               ids,
               partition_strategy='mod',
               name=None,
               validate_indices=True,
               max_norm=None
)
参数说明：
params: 表示完整的嵌入张量，或者除了第一维度之外具有相同形状的P个张量的列表，表示经分割的嵌入张量
ids: 一个类型为int32或int64的Tensor，包含要在params中查找的id
partition_strategy: 指定分区策略的字符串，如果len（params）> 1，则相关。当前支持“div”和“mod”。 默认为“mod”
name: 操作名称（可选）
validate_indices:  是否验证收集索引
max_norm: 如果不是None，嵌入值将被l2归一化为max_norm的值

tf.nn.embedding_lookup()函数的用法主要是选取一个张量里面索引对应的元素
tf.nn.embedding_lookup(tensor,id)：即tensor就是输入的张量，id 就是张量对应的索引
tf.nn.embedding_lookup()就是根据input_ids中的id，寻找embeddings中的第id行。比如input_ids=[1,3,5]，则找出embeddings中第1，3，5行，组成一个tensor返回
embedding_lookup不是简单的查表，id对应的向量是可以训练的，训练参数个数应该是 category num*embedding size，也就是说lookup是一种全连接层

一般做自然语言相关的。需要把每个词都映射成向量，这个向量可以是word2vec预训练好的，也可以是在网络里训练的，在网络里需要先把词的id转换成对应的向量，这个函数就是做这件事的
在基于深度学习的实体识别中，字向量会提前训练好，这个就可以理解成上面的tensor，而在实际的句子中每一个字所对应的字向量是通过id进行关联上的
"""
# coding:utf-8

import tensorflow as tf
import numpy as np


c = np.random.random([5, 1])  # 随机生成一个5*1的数组
b = tf.nn.embedding_lookup(c, [1, 3])  # 查找数组中的序号为1和3的
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(b))
    print(c)


"""
[[0.66793107]
 [0.87975865]]
[[0.0374401 ]
 [0.66793107]
 [0.95599935]
 [0.87975865]
 [0.66317383]]
"""