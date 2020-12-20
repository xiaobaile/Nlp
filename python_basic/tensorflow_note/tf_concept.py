"""
基本概念：
    基于tensorflow的神经网络：
        用张量表示数据，用计算图搭建神经网络，用会话执行计算图，优化线上的权重，得到模型。
    张量：
        张量就是多维数组（列表），用阶表示张量的维度。
        0阶张量：就是标量，表示一个单独的数。
        1阶张量：就是向量，表示一个一维数组。
        2阶张量：就是矩阵，表示为一个二维数组。
    判断方法：
        判断张量是几阶的，就通过张量右边的方括号数。
    数据类型：
        tensorflow的数据类型有tf.float32, tf.int32等等.
    计算图：
        搭建神经网络的计算过程，是承载一个或多个计算节点的一张图，只搭建网络，不运算。
    会话：
        执行计算图中的节点运算。
        通常用with结构实现，语法如下：
        with tf.Session() as sess:
            print(sess.run(y)

"""


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
在运行 Session()会话时，有时会出现“提示 warning”，是因为有的电脑可以支持加速指令，但是运行代码时并没有启动这些指令。
可以把 这些“提示 warning”暂时屏蔽掉。屏蔽方法为进入主目录下的 bashrc 文件，在 bashrc 文件中加入这样一句:
except TF_CPP_MIN_LOG_LEVEL=2，从而把“提示 warning”等级降低。
"""

a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
result = a + b
print(result)
# 输出结果为：Tensor("add:0", shape=(2,), dtype=float32)
# 意思为 result 是一个名称为 add:0 的张量，shape=(2,)表示一维数组长度为 2， dtype=float32 表示数据类型为浮点型。

# 搭建神经元计算图
x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])
y = tf.matmul(x, w)
print(y)
# 输出结果为：Tensor("MatMul:0", shape=(1, 1), dtype=float32)
# print 的结果显示 y 是一个张量，只搭建承载计算过程的 计算图，并没有运算，如果我们想得到运算结果就要用到“会话 Session()”了。

with tf.Session() as sess:
    print(sess.run(y))
# 输出结果为：[[11.]]
