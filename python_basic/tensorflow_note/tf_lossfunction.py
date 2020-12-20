"""
神经网络优化：
    神经元模型:用数学公式表示为:𝐟(∑𝒊𝒙𝒊𝒘𝒊 + 𝐛)，f 为激活函数。神经网络是以神经元为基本单元构成的。
    激活函数:引入非线性激活因素，提高模型的表达力。 常用的激活函数有 relu、sigmoid、tanh 等。
        1 激活函数 relu: 在 Tensorflow 中，用 tf.nn.relu()表示。
        2 激活函数 sigmoid:在 Tensorflow 中，用 tf.nn.sigmoid()表示。
        3 激活函数 tanh:在 Tensorflow 中，用 tf.nn.tanh()表示。
    神经网络的复杂度:
        可用神经网络的层数和神经网络中待优化参数个数表示
    神经网路的层数:
        一般不计入输入层，层数 = n 个隐藏层 + 1 个输出层
    神经网路待优化的参数:
        神经网络中所有参数 w 的个数 + 所有参数 b 的个数
    损失函数(loss):
        用来表示预测值(y)与已知答案(y_)的差距。在训练神经网络时，通过不断改变神经网络中所有参数，使损失函数不断减小，
        从而训练出更高准确率的神经网络模型。
        常用的损失函数有均方误差、自定义和交叉熵等。
    均方误差 mse:
        n 个样本的预测值 y 与已知答案 y_之差的平方和，再求平均值。
        在 Tensorflow 中用 loss_mse = tf.reduce_mean(tf.square(y_ - y))
    自定义损失函数:根据问题的实际情况，定制合理的损失函数。
        对于预测酸奶日销量问题，如果预测销量大于实际销量则会损失成本;
        如果预测销量小于实际销量则 会损失利润。
        在实际生活中，往往制造一盒酸奶的成本和销售一盒酸奶的利润是不等价的。
        因此，需要使用符合该问题的自定义损失函数。
            自定义损失函数为:loss = ∑𝑛𝑓(y_, y) 其中，损失定义成分段函数:
            f(y_,y) = 𝑃𝑅𝑂𝐹𝐼𝑇∗(𝑦_−𝑦) 𝑦<𝑦_
                        𝐶𝑂𝑆𝑇∗(𝑦−𝑦_) 𝑦>=𝑦_
            损失函数表示，若预测结果 y 小于标准答案 y_，损失函数为利润乘以预测结果 y 与标准答案 y_之差;
            若预测结果 y 大于标准答案 y_，损失函数为成本乘以预测结果 y 与标准答案 y_之差。
            用 Tensorflow 函数表示为:
                loss = tf.reduce_sum(tf.where(tf.greater(y,y_),COST(y-y_),PROFIT(y_-y)))
    栗子：
        预测酸奶日销量 y，x1 和 x2 是影响日销量的两个因素。

"""


import tensorflow as tf
import numpy as np


BATCH_SIZE = 8
SEED = 23455
COST = 1
PROFIT = 9


rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)
Y_ = [[x1 + x2 + (rdm.rand()/10.0 - 0.05)] for x1, x2 in X]

# 定义神经网络的输入，参数和输出，定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义损失函数及反向传播方法
# 定义损失函数为MSE，反向传播方法为梯度下降
# loss_mse = tf.reduce_mean(tf.square(y_ - y))
# 根据实际情况，重新构造损失函数。
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), COST*(y - y_), PROFIT*(y_ - y)))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 20000
    for i in range(STEPS):
        start = i*BATCH_SIZE % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y_[start: end]})
        if i % 500 == 0:
            print("After %d training steps,w1 is: " % i)
            print(sess.run(w1), "\n")
    print("Finally w1 is : \n", sess.run(w1))
