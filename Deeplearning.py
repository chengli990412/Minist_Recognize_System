import numpy as np
import os
import gzip
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, datasets
import tensorboard as tb

# #深度学习类
class DeepLearningClass:
    def load_data(data_folder):
        files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz']
        paths = []
        for fname in files:
            paths.append(os.path.join(data_folder, fname))
        if len(paths) == 0:
            return "NG"

        # 打开文件
        with gzip.open(paths[0], 'rb') as lbpath:
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(paths[1], 'rb') as imgpath:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        with gzip.open(paths[2], 'rb') as lbpath:
            y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(paths[3], 'rb') as imgpath:
            x_test = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

        return (x_train, y_train), (x_test, y_test)


#训练
#
#
# def Dl_Run(train_images,train_labels):
#     x = tf.convert_to_tensor(train_images, dtype=tf.float32) / 255
#     db = tf.data.Dataset.from_tensor_slices((x, train_labels))
#     db = db.batch(100).repeat(20)
#
#     model = keras.Sequential([layers.Dense(256, activation='relu'),
#                               layers.Dense(128, activation='relu'),
#                               layers.Dense(10)])
#
#     model.build(input_shape=(None, 28 * 28))
#     model.summary()
#     optimizer = optimizers.SGD(lr=0.02)
#     acc_meter = keras.metrics.Accuracy()
#     summary_writer = tf.summary.create_file_writer('tf_log')
#     text = ''
#
#     for step, (xx, yy) in enumerate(db):
#
#         with tf.GradientTape() as tape:
#             # 图像样本大小重置(-1, 28*28)
#             xx = tf.reshape(xx, (-1, 28 * 28))
#             # 获取输出
#             out = model(xx)
#             # 实际标签转为onehot编码
#             y_onehot = tf.one_hot(yy, depth=10)
#             # 计算误差
#             loss = tf.square(out - y_onehot)
#             loss = tf.reduce_sum(loss / xx.shape[0])
#         # 更新准备率
#         acc_meter.update_state(tf.argmax(out, axis=1), yy)
#         # 更新梯度参数
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         # 参数存储，便于查看曲线图
#         with summary_writer.as_default():
#             tf.summary.scalar('train-loss', float(loss), step=step)
#             tf.summary.scalar('test-acc', acc_meter.result().numpy(), step=step)
#             # tf.summary.image('Training data', xx,step=step)
#
#         if step % 1000 == 0:
#             text = acc_meter.result().numpy()
#             print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
#             acc_meter.reset_states()
#
#     return text





