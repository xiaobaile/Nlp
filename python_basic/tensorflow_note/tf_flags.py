import os
import tensorflow as tf

"""
1、flags可以帮助我们通过命令行来动态的更改代码中的参数。Tensorflow 使用flags定义命令行参数的方法。
    ML的模型中有大量需要tuning的超参数，所以此方法，迎合了需要一种灵活的方式对代码某些参数进行调整的需求。
(1)、比如，在这个py文件中，首先定义了一些参数，然后将参数统一保存到变量FLAGS中，相当于赋值，后边调用这些参数的时候直接使用FLAGS参数即可。
(2)、基本参数类型有三种flags.DEFINE_integer、flags.DEFINE_float、flags.DEFINE_boolean。
(3)、第一个是参数名称，第二个参数是默认值，第三个是参数描述.

2、使用过程
# 第一步，调用flags = tf.flags，进行定义参数名称，并可给定初值、参数说明
# 第二步，flags参数直接赋值
# 第三步，运行tf.run()
"""
flags = tf.flags
flags.DEFINE_string("path_data", "./data", "tfRecord sava path.")
flags.DEFINE_string("path_style", "./style_imgs", "Row style images path")
flags.DEFINE_string("path_content", "./MSCOCO", "Row style images path")
flags.DEFINE_string("record_style_name", "styles.tfrecords", "Style tfrecord name")
flags.DEFINE_string("record_dataset_name", "coco_train.tfrecords", "Data set tfrecord name")
FLAGS = flags.FLAGS


def main():
    if not os.path.exists(FLAGS.path_data):
        os.mkdir(FLAGS.path_data)
        print("the directory was created successful!")
    else:
        print("directory already exists")


# 在终端运行： python tf1_flags.py --path_data="./data",默认参数为None，如果不给参数会提示错误，
# 因为指定该参数的数据类型是string。
if __name__ == '__main__':
    main()
