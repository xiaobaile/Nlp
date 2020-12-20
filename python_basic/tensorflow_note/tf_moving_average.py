"""
滑动平均:
        记录了一段时间内模型中所有参数 w 和 b 各自的平均值。利用滑动平均值可以增强模型的泛化能力。
        滑动平均值(影子)计算公式:
            影子 = 衰减率 * 影子 + (1 - 衰减率) * 参数
            其中，衰减率 = 𝐦𝐢𝐧 {𝑴𝑶𝑽𝑰𝑵𝑮𝑨𝑽𝑬𝑹𝑨𝑮𝑬𝑫𝑬𝑪𝑨𝒀 , (𝟏+轮数)/(𝟏𝟎+轮数)}，影子初值=参数初值
        用 Tensorflow 函数表示为:
            ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY，global_step)
            其中，MOVING_AVERAGE_DECAY 表示滑动平均衰减率，一般会赋接近 1 的值，global_step 表示当前训练了多少轮。
            ema_op = ema.apply(tf.trainable_variables())
            其中，ema.apply()函数实现对括号内参数求滑动平均，tf.trainable_variables()函数实现把所有待训练参数汇总为列表。
            with tf.control_dependencies([train_step, ema_op]):
                train_op = tf.no_op(name='train')
            其中，该函数实现将滑动平均和训练过程同步运行。
            查看模型中参数的平均值，可以用 ema.average()函数。
"""


import tensorflow as tf


# 1.定义变量及滑动平均类
# 定义一个32位浮点变量，初始值为0，这个代码就是不断更新w1参数，优化w1参数，滑动平均做了一个w1的影子
w1 = tf.Variable(0, dtype=tf.float32)
# 定义num_updates(NN的迭代轮数)，初始值为0，不可被优化（训练），这个参数不训练
global_step = tf.Variable(0, trainable=False)
# 实例化滑动平均类，给删减率为0.99,当前轮数为global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(
    MOVING_AVERAGE_DECAY, global_step
)
# ema.apply后的括号里是更新列表，每次运行sess.run(ema_op)时，对更新列表中的元素求滑动平均值。
# 在实际应用中会使用tf.trainable_variables()自动将所有待训练的参数汇总为列表
# ema_op = ema.apply([w1])
ema_op = ema.apply(tf.trainable_variables())

# 2.查看不同迭代中变量取值的变化。
with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 用ema.average(w1)获取w1滑动平均值（要运行多个节点，作为列表中的元素列出，写在sess.run中）
    # 打印出当前参数w1和w1滑动平均值
    print(sess.run([w1, ema.average(w1)]))
    # 参数w1的值赋为1
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    # 更新step和w1的值，模拟出100轮迭代后，参数w1变为10
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
    # 每次sess run会更新一次w1的滑动平均值
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))
