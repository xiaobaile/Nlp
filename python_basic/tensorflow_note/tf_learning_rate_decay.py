"""
指数衰减学习率：
    学习率随着训练轮数变化而动态更新。
    学习率计算公式如下:
        Learning_rate = LEARNING_RATE_BASE * LEARNING_RATE_DECAY *（𝒈𝒍𝒐𝒃𝒂𝒍_𝒔𝒕𝒆𝒑 /𝑳𝑬𝑨𝑹𝑵𝑰𝑵𝑮_𝑹𝑨𝑻𝑬_𝑩𝑨𝑻𝑪𝑯_𝑺𝑰𝒁𝑬）
    用 Tensorflow 的函数表示为:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
                                                LEARNING_RATE_BASE,
                                                global_step,
                                                LEARNING_RATE_STEP, LEARNING_RATE_DECAY,
                                                staircase=True/False)
        其中，LEARNING_RATE_BASE 为学习率初始值，LEARNING_RATE_DECAY 为学习率衰减率,
        global_step 记录了当前训练轮数，为不可训练型参数。
        学习率 learning_rate 更新频率为输入数据集总样本数除以每次喂入样本数。
        若 staircase 设置为 True 时，表示 global_step/learning rate step 取整数，学习率阶梯型衰减;
        若 staircase 设置为 false 时，学习率会是一条平滑下降的曲线。
"""


import tensorflow as tf


""" 损失函数：loss = (w + 1)^2，令w初值值是常数10。反向传播就是求最优w，即求最小loss对应的w值
使用指数衰减学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得更有收敛性。
"""
# 初始学习率
LEARNING_RATE_BASE = 1.0
# 学习率衰减率
LEARNING_RATE_DECAY = 0.99
# 喂入多少轮BATCH_SIZE后，更新一次学习率，一般设为：总样本数/BATCH_SIZE
LEARNING_RATE_STEP = 1

# 运行了几轮BATCH_SIZE的计数器，初始值为0， 设置为不被训练。
global_step = tf.Variable(0, trainable=False)
# 定义学习率为指数衰减学习率
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    LEARNING_RATE_STEP,
    LEARNING_RATE_DECAY,
    staircase=True
)
# 定义待优化参数，初始值为10
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# 定义损失函数。
loss = tf.square(w+1)
# 定义反向传播方法。
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
# 生成会话，训练40轮。
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps: global_step is %f, w is %f, learning rate is %f, loss is %f..."
              % (i, global_step_val, w_val, learning_rate_val, loss_val))
