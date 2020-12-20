"""
步骤流程：
        前向传播:由输入到输出，搭建完整的网络结构
        描述前向传播的过程需要定义三个函数:
        def forward(x, regularizer):
            w=
            b=
            y=
            return y
        第一个函数 forward()完成网络结构的设计，从输入到输出搭建完整的网络结构，实现前向传播过程。
        该函数中，参数 x 为输入，regularizer 为正则化权重，返回值为预测或分类结果 y。
        def get_weight(shape, regularizer):
            w = tf.Variable( )
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
            return w
        第二个函数 get_weight()对参数 w 设定。该函数中，参数 shape 表示参数 w 的形状，regularizer 表示正则化权重，返回值为参数 w。
        其中，tf.variable()给 w 赋初值，tf.add_to_collection()表 示将参数 w 正则化损失加到总损失 losses 中。
        def get_bias(shape):
            b = tf.Variable( )
            return b
        第三个函数 get_bias()对参数 b 进行设定。该函数中，参数 shape 表示参数 b 的形状,返回值为参数 b。
        其中，tf.variable()表示给 w 赋初值。

        反向传播:训练网络，优化网络参数，提高模型准确性。
        def backward( ):
            x = tf.placeholder( )
            y_ = tf.placeholder(
            y = forward.forward(x, REGULARIZER)
            global_step = tf.Variable(0, trainable=False)
            loss =
        函数 backward()中，placeholder()实现对数据集 x 和标准答案 y_占位，forward.forward()实现前向传播的网络结构，
        参数 global_step 表示训练轮数，设置为不可训练型参数。
        在训练网络模型时，常将正则化、指数衰减学习率和滑动平均这三个方法作为模型优化方法。
        在 Tensorflow 中，正则化表示为:
            首先，计算预测结果与标准答案的损失值
                1 MSE: y 与 y_的差距(loss_mse) = tf.reduce_mean(tf.square(y-y_))
                2 交叉熵:ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
                    y 与 y_的差距(cem) = tf.reduce_mean(ce)
                3 自定义:y 与 y_的差距
                其次，总损失值为预测结果与标准答案的损失值加上正则化项
                loss = y 与 y_的差距 + tf.add_n(tf.get_collection('losses'))
            在 Tensorflow 中，指数衰减学习率表示为:
            learning_rate = tf.train.exponential_decay(
                                                        LEARNING_RATE_BASE, global_step,
                                                        数据集总样本数 / BATCH_SIZE,
                                                        LEARNING_RATE_DECAY,
                                                        staircase=True)
            train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            在 Tensorflow 中，滑动平均表示为:
            ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            ema_op = ema.apply(tf.trainable_variables())
            with tf.control_dependencies([train_step, ema_op]):
                train_op = tf.no_op(name='train')
            其中，滑动平均和指数衰减学习率中的 global_step 为同一个参数。
            用 with 结构初始化所有参数
            with tf.Session() as sess:
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                for i in range(STEPS):
                    sess.run(train_step, feed_dict={x: , y: }
                    if i % 轮数 == 0:
                        print
                其中，with 结构用于初始化所有参数信息以及实现调用训练过程，并打印出 loss 值。
            判断 python 运行文件是否为主文件
                if __name__=='__main__':
                    backward()
                该部分用来判断 python 运行的文件是否为主文件。若是主文件，则执行 backword()函数。
例如:
用 300 个符合正态分布的点 X[x0, x1]作为数据集，根据点 X[x0, x1]的不同进行标注 Y_，将数据集标注为红色和蓝色。
标注规则为:当 x0 + x1 < 2 时，y_=1，点 X 标注为红色;当 x0 + x1 ≥2 时， y_=0，点 X 标注为蓝色。
我们加入指数衰减学习率优化效率，加入正则化提高泛化性，并使用模块化 设计方法，把红色点和蓝色点分开。
代码总共分为三个模块:生成数据集(generateds.py)、前向传播(forward.py)、反向传播 (backward.py)。
"""