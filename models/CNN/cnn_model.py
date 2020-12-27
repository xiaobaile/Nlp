import tensorflow as tf


class TCNNConfig(object):
    """ CNN 配置参数"""
    embedding_dim = 64  # 词向量的长度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数目
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 每个卷积和的宽度是5，正常情况下，卷积和的大小是二维数据，由于只能从上向下扫描，因此默认长度与词向量维相同为64。
    vocab_size = 5000  # 词汇表的大小
    hidden_dim = 128  # 全连接层神经元
    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3
    batch_size = 64  # 每批喂入数据的数量
    num_epochs = 10  # 迭代次数
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮保存一次数据


class TextCNN(object):
    """ 文本分类的CNN模型"""
    def __init__(self, config):
        self.config = config
        # input_x dim 64@600,每次输入64句话，每句话的长度是600。
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_y")
        # If the shape is not specified, you can feed a tensor of any shape.
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.cnn()

    def cnn(self):
        with tf.device("/cpu:0"):
            # 创建新的tensorflow变量，变量的形状是5000行64列。
            # 每次输入64个文本，每个文本600个词，每个词用64维的向量表示。
            embedding = tf.get_variable("embedding", [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # 一维卷积一般用于处理文本数据，常用语自然语言处理中，输入一般是文本经过embedding的二维数据。
            # 64@600@64   256个卷积核，卷积核宽度是5，--------64@596@256
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name="conv")
            # 找到行上的最大值。
            gmp = tf.reduce_max(conv, reduction_indices=[1], name="gmp")

        with tf.name_scope("score"):
            # 全连接层。
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name="fc1")
            # 随机的拿掉网络中的部分神经元，从而减小对W权重的依赖，以达到减小过拟合的效果。
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            # 输入小于0的值，幅值为0，输入大于0的值则不变。
            fc = tf.nn.relu(fc)
            self.logits = tf.layers.dense(fc, self.config.num_classes, name="fc2")
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimizer"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
