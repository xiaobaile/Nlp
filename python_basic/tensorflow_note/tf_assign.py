import tensorflow as tf

""" tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number."""
#
# W = tf.Variable(10)
# W.assign(100)
# with tf.Session() as sess:
#     sess.run(W.initializer)
#     print(sess.run(W))

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    # sess.run(assign_op)
    sess.run(W.initializer)
    print(W.eval())
    sess.run(assign_op)
    print(W.eval())
