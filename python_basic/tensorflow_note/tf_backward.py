"""
反向传播：
        反向传播：训练模型参数，在所有参数上用梯度下降，使神经网络模型在训练数据上的损失函数最小。
        损失函数：计算得到的预测值y与已知答案y_的差距。
            损失函数的计算方法有很多，均方误差MSE是比较常用的方法之一。
        均方误差：求前向传播计算结果与已知答案之差的平方再求平均。
            用tensorflow函数表示为：
            loss_mse = tf.reduce_mean(tf.square(y_ - y))
        反向传播训练方法：以减小loss值为优化目标，有梯度下降，momentum优化器，adam优化器等优化方法。
            这三种优化方法用 tensorflow 的函数可以表示为:
            train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            train_step=tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
            train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)
            三种优化方法区别如下:
            1 tf.train.GradientDescentOptimizer()使用随机梯度下降算法，使参数沿着梯度的反方向，
            即总损失减小的方向移动，实现更新参数。
            2 tf.train.MomentumOptimizer()在更新参数时，利用了超参数，参数更新公式是
                𝑑𝑖 = 𝛽𝑑𝑖−1 + 𝑔(𝜃𝑖−1) 𝜃𝑖 = 𝜃𝑖−1 − 𝛼𝑑𝑖
            其中，𝛼为学习率，超参数为𝛽，𝜃为参数，𝑔(𝜃𝑖−1)为损失函数的梯度。
            3 tf.train.AdamOptimizer()是利用自适应学习率的优化算法，Adam 算法和随机梯度下降算法不同。
            随机梯度下降算法保持单一的学习率更新所有的参数，学习率在训练过程中并不会改变。
            而 Adam 算法通过计算梯度的一阶矩估计和二 阶矩估计而为不同的参数设计独立的自适应性学习率。
        学习率：
            优化器中都需要一个叫做学习率的参数，使用时，如果学习率选择过大会出现震荡不收敛的情况，
            如果学习率选择过小，会出现收敛速度慢的情况。我们可以选个比较小的值填入，比如 0.01、0.001。

步骤：
    1.导入模块，生成模拟数据集
        import
        常量定义
        生成数据集
    2.向前传播：定义输入，参数和输出
        x =     y_ =    w1 =    w2 =    a =     y =
    3.反向传播：定义损失函数，反向传播方法
        loss =      train_step =
    4.生成会话，训练STEPS轮
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            STEPS = 3000
            for i in range(STEPS):
                start =
                end =
                sess.run(train_step, feed_dict:{})

举例
    随机产生 32 组生产出的零件的体积和重量，训练 3000 轮，每 500 轮输出一次损失函数。
"""

# 1。导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455


rng = np.random.RandomState(seed)
X = rng.rand(32, 2)
Y = [[int(x0 + x1 < 1)] for x0, x1 in X]
print("X:\n", X)
print("Y:\n", Y)

# 2。向前传播：定义输入，参数和输出
x = tf.placeholder(tf.float32, shape=(None, 2))
y = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
a = tf.matmul(x, w1)
y_ = tf.matmul(a, w2)

# 3。反向传播：定义损失函数，反向传播方法
loss = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 4。生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")
    STEPS = 3000
    for i in range(STEPS):
        start = i*BATCH_SIZE % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start: end], y: Y[start: end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y: Y})
            print("After %d training step(s), loss on all data is %g..." % (i, total_loss))
    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
