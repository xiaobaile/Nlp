import tensorflow as tf


"""
Tensorflow有tf.Graph类，用于存储计算图。
而计算图其实就是由节点和有向边组成，每个点包括操作Op、数值value、类型dtype、形状shape等属性。
探索诸如tf.Variable()等函数的内部机制的过程中，就需要查看计算图的变化情况，包括新建了哪些节点，输入是什么等等。
"""
a = tf.constant(1)
b = tf.Variable(a)
print(tf.get_default_graph().as_graph_def())
