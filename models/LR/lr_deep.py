import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.random(*train_X.shape) * 0.3
plt.plot(train_X, train_Y, "ro", label="Original data")
plt.legend()
plt.show()

""" 创建模型 """
# 1。占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
# 2。模型训练参数
w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
# 3。前向传递,注意这里不是matmul，这个是矩阵乘法。
z = tf.multiply(X, w) + b
# 4。计算代价函数，或者损失函数
cost = tf.reduce_mean(tf.square(Y - z))
# 5。反向传播，优化参数，定义学习率
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# 统计loss平均值
def moving_average(a, ww=10):
    if len(a) < ww:
        return a[:]
    # 注意列表生成式中存在if判断的时候该如何书写。
    return [val if idx < ww else sum(a[(idx-ww):idx])/ww for idx, val in enumerate(a)]


# 定义参数
training_epochs = 20
display_step = 2

# 6。开启会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plot_data = {"batch_size": [], "loss": []}
    for epoch in range(training_epochs):
        for x, y in zip(train_X, train_Y):
            # 每喂入一组数据，进行一次优化。
            sess.run(optimizer, feed_dict={X: x, Y: y})

            if epoch % display_step == 0:
                # 每轮的损失。
                loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(w), "b=", sess.run(b))
                if not (loss == "NA"):
                    plot_data["batch_size"].append(epoch)
                    plot_data["loss"].append(loss)
    print("Finished")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(w), "b=", sess.run(b))

    # 显示拟合曲线
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Fitted')
    plt.legend()
    plt.show()

    # 显示loss曲线
    plot_data["avg_loss"] = moving_average(plot_data["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_data["batch_size"], plot_data["avg_loss"], 'b--')
    plt.xlabel('mini_batch number')
    plt.ylabel('loss')
    plt.title('')
    plt.show()
