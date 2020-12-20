"""
交叉熵(Cross Entropy):
        表示两个概率分布之间的距离。交叉熵越大，两个概率分布距离越远，两个概率分布越相异;
        交叉熵越小，两个概率分布距离越近，两个概率分布越相似。
        交叉熵计算公式:
            𝐇(𝐲_ , 𝐲) = −∑𝐲_ ∗ 𝒍𝒐𝒈 𝒚
        用 Tensorflow 函数表示为:
            ce= -tf.reduce_mean(y_* tf.log(tf.clip_by_value(y, 1e-12, 1.0)))
softmax 函数:
        将 n 分类的 n 个输出(y1,y2...yn)变为满足以下概率分布要求的函数。
        ∀𝐱 𝐏(𝐗=𝐱)∈[𝟎,𝟏] 且∑𝑷 (𝑿=𝒙)=𝟏
        softmax 函数应用:在 n 分类中，模型会有 n 个输出，即 y1,y2...yn，其中 yi 表示第 i 种情况出现的可能性大小。
        将n个输出经过 softmax 函数，可得到符合概率分布的分类结果。
        在 Tensorflow 中，一般让模型的输出经过 softmax 函数，以获得输出分类的概率分布，再与标准答案对比，求出交叉熵，得到损失函数，
        用如下函数实现:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
            cem = tf.reduce_mean(ce)
学习率 learning_rate:
        表示了每次参数更新的幅度大小。学习率过大，会导致待优化的参数在最小值附近波动，不收敛;
        学习率过小，会导致待优化的参数收敛缓慢。 在训练过程中，参数的更新向着损失函数梯度下降的方向。
        参数的更新公式为:
            𝒘𝒏+𝟏 = 𝒘𝒏 − 𝒍𝒆𝒂𝒓𝒏𝒊𝒏𝒈_𝒓𝒂𝒕𝒆𝛁
        学习率的设置：
            学习率过大，会导致待优化的参数在最小值附近波动，不收敛;学习率过小，会导致待优化的参数收敛缓慢。
"""


import tensorflow as tf


""" 设损失函数loss = (w+1)^2, 令w的初始值为5，反向传播就是优化w，即求最小loss对应的w值。
定义待优化的参数w，并赋予初始值为5.以前我们给变量赋值的时候，都是通过随机分布来复制的，
这次通过一个常数来赋值，并且在后面的运算中，这个变量的值是随着训练的轮数增加而改变的。
"""
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# 定义损失函数loss。
loss = tf.square(w + 1)
# 定义反向传播方法。
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
# train_step = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
# 生成会话，训练40轮。
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps: w is %f, loss is %f..." % (i, w_val, loss_val))

