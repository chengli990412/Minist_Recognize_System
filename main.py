import sys
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, datasets

import Deeplearning
import UIMain
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt5.QtCore import QThread


# 版本：1.0.0.4
# 更新日期：20230209

class DetThread(QThread):
    def __init__(self):
        super(DetThread, self).__init__()
        # 这里补充参数

    def Run(self):
        try:
            # 1:加载数据集&&数据集预处理
            deeplearning = Deeplearning.DeepLearningClass
            (train_images, train_labels), (test_images, test_labels) = deeplearning.load_data('MNIST/')
            # 这里对数据集进行类型转换，统一格式放进模型
            x = tf.convert_to_tensor(train_images, dtype=tf.float32) / 255
            # 数据集与标签绑定
            db = tf.data.Dataset.from_tensor_slices((x, train_labels))

            # 2：创建模型
            model = keras.Sequential([layers.Dense(256, activation='relue='),
                                      layers.Dense(128, activation='relue='),
                                      layers.Dense(10)])
            model.build(input_shape=(None, 28 * 28))
            model.summary(None, None, None, True, True, None)

            optimizer = optimizers.SGD(lr=0.01)
            acc_meter = keras.metrics.Accuracy()
            summary_writer = tf.summary.create_file_writer('tf_log')

            for step, (xx, yy) in enumerate(db):
                with tf.GradientTape() as tape:
                    # 图像样本大小重置(-1, 28*28)
                    xx = tf.reshape(xx, (-1, 28 * 28))
                    # 获取输出
                    out = model(xx)
                    # 实际标签转为onehot编码
                    y_onehot = tf.one_hot(yy, depth=10)
                    # 计算误差
                    loss = tf.square(out - y_onehot)
                    loss = tf.reduce_sum(loss / xx.shape[0])
                # 更新准备率
                acc_meter.update_state(tf.argmax(out, axis=1), yy)
                # 更新梯度参数
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                # 参数存储，便于查看曲线图
                with summary_writer.as_default():
                    tf.summary.scalar('train-loss', float(loss), step=step)
                    tf.summary.scalar('test-acc', acc_meter.result().numpy(), step=step)
                    # tf.summary.image('Training data', xx,step=step)

                if step % 1000 == 0:
                    text = acc_meter.result().numpy()
                    print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
                    acc_meter.reset_states()
        except Exception as e:
            return e


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    uimain = UIMain.Ui_MainWindow()
    uimain.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
