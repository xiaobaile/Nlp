# coding:utf-8
"""
神经网络的参数：
    是指神经元的权重w和偏置b，用变量表示，一般会先随机生成这些参数。生成参数的方法是让w等于tf.Variable，把生成的方式写在括号里。
    神经网络中常用的生成随机数/数组的函数有：
        tf.random_normal()      生成正态分布随机数
        tf.truncated_normal()   生成去掉过大偏离点的正态分布随机数
        tf.random_uniform()     生成均匀分布随机数
        tf.zeros（）              表示生成全 0 数组
        tf.ones（）               表示生成全 1 数组
        tf.fill（）               表示生成全定值数组
        tf.constant（）           表示生成直接给定值的数组
        例如：
            1 w=tf.Variable(tf.random_normal([2,3],stddev=2, mean=0, seed=1))，
            表 示生成正态分布随机数，形状两行三列，标准差是 2，均值是 0，随机种子是 1。
            2 w=tf.Variable(tf.Truncated_normal([2,3],stddev=2, mean=0, seed=1))，
            表示去掉偏离过大的正态分布，也就是如果随机出来的数据偏离平均值超过两个 标准差，这个数据将重新生成。
            3 w=random_uniform(shape=7,minval=0,maxval=1,dtype=tf.int32，seed=1),
            表示从一个均匀分布[minval maxval)中随机采样，注意定义域是左闭右开，即 包含 minval，不包含 maxval。
            4 除了生成随机数，还可以生成常量。tf.zeros([3,2],int32)
            表示生成 [[0,0],[0,0],[0,0]];
            5 tf.ones([3,2],int32)表示生成[[1,1],[1,1],[1,1];
            6 tf.fill([3,2],6)表示生成[[6,6],[6,6],[6,6]];
            7 tf.constant([3,2,1])表示 生成[3,2,1]。
            注意：
                - 随机种子如果去掉每次生成的随机数将不一致。
                - 如果没有特殊要求标准差，均值，随机种子是可以不写的。

神经网络的搭建：
    神经网络的实现过程：
        1、准备数据集，提取特征，作为输入喂给神经网络(Neural Network，NN)
        2、搭建 NN 结构，从输入到输出(先搭建计算图，再用会话执行)
            ( NN 前向传播算法======>计算输出)
        3、大量特征数据喂给 NN，迭代优化 NN 参数
            ( NN 反向传播算法======>优化参数训练模型)
        4、使用训练好的模型预测和分类
    基于神经网络的机器学习主要分为两个过程：训练过程和使用过程。前三步是循环迭代的训练过程，第四步是使用过程，
    一旦参数优化完成就可以固定这些参数，实现特定应用了。
    在实际中，会先使用现有的成熟的网络结构，喂入新的数据，训练相应模型，判断是否能对喂入的从未见过的新数据作出正确响应，
    再适当更改网络结构，反复迭代，让机器自动训练参数找出最优结构和参数，以固定专用模型。
向前传播
    就是搭建模型的计算过程，让模型具有推理能力，可以针对一组输入给出相应的输出。
    栗子：
        假如生产一批零件，体积为 x1，重量为 x2，体积和重量就是我们选择的特征， 把它们喂入神经网络，
        当体积和重量这组数据走过神经网络后会得到一个输出。假如输入的特征值是:体积 0.7 重量 0.5。
        推导：
            第一层：X是输入，1*2的矩阵
                X：表示输入，是一个1行2列的矩阵，表示一次输入一组特征，这组特征包含了体积和重量两个元素。
                W：下标为（前节点编号，后节点编号），上标为（层数），为待优化的参数。
                    对于第一层的w，前面有两个节点（重量和体积），后面有三个节点（隐藏层数目），因此w是一个两行三列的矩阵。
                神经网络共有几层，都是指的计算层，输入不是计算层，所以a为第一层网络，a是一个一行三列矩阵。
                    a = x*w
            第二层：参数要满足前面三个节点，后面一个节点，所以w是三行一列矩阵。
                我们把每层输入乘以权重w，就可以计算出输出y了。
                    y = a*w
                由于需要计算结果，就要用with结构实现，所有变量初始化过程，计算过程都放到sess.run函数中。
                对于变量初始化，我们在 sess.run 中写入 tf.global_variables_initializer 实现对所有变量初始化，也就是赋初值。
                对于计算图中的运算，我们直接把运算节点填入 sess.run 即可，比如要计算输出 y，直接写 sess.run(y) 即可。

                在实际应用中，我们可以一次喂入一组或多组输入，让神经网络计算输出 y，可以先用 tf.placeholder 给输入占位。
                如果一次喂一组数据 shape 的第一维位置写 1，第二维位置看有几个输入特征;
                如果一次想喂多组数据，shape 的第一维位置可以写 None 表示先空着，第二维位置写有几个输入特征。
                这样在 feed_dict 中可以喂入若干组体积重量了。

            向前传播：
                变量初始化，计算图节点运算都要用会话实现。
                变量初始化：在 sess.run 函数中用 tf.global_variables_initializer()汇总所有待优化变量。
                计算图节点运算：在 sess.run 函数中写入待运算的节点。
                喂入数据：用 tf.placeholder 占位，在 sess.run 函数中用 feed_dict 喂数据。
                    喂一组数据:
                        x = tf.placeholder(tf.float32, shape=(1, 2))
                        sess.run(y, feed_dict={x: [[0.5,0.6]]})
                    喂多组数据:
                        x = tf.placeholder(tf.float32, shape=(None, 2))
                        sess.run(y, feed_dict={x: [[0.1,0.2],[0.2,0.3],[0.3,0.4],[0.4,0.5]]})

"""


import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 1.定义输入和参数
# x = tf.placeholder(tf.float32, shape=(1, 2))
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 2.定义向前传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 3.用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # print("y is:\n", sess.run(y, feed_dict={x: [[0.7, 0.5]]}))
    print("y is:\n", sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))
