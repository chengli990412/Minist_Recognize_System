import sys
import time
import MyGlobal as gl

from PyQt5 import QtCore
from PyQt5.QtCore import QObject

import Deeplearning
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from UIMain import Ui_MainWindow


# 版本：1.0.0.6
# 更新日期：20230214

class Myclass(QWidget, Ui_MainWindow):
    textWritten = QtCore.pyqtSignal(str)

    def __init__(self):
        super(Myclass, self).__init__()
        self.init()

    # 初始化
    def init(self):
        self.setupUi(self)
        self.textEdit.append('软件初始化成功')
        self.setWindowTitle('基于特征识别的手写数字识别')
        self.Button_LoadData.clicked.connect(self.loaddata)
        self.Button_SaveModel.clicked.connect(self.SaveModel)
        self.Button_LoadModel.clicked.connect(self.Load_model)
        self.Button_StopTrain.clicked.connect(self.CreateModel)

    # 数据加载
    def loaddata(self):
        Deeplearning.load_data()
        self.Writelog('数据集加载成功')

    # 写log
    def Writelog(self, str):
        mes = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ':' + str
        self.textEdit.append(mes)

    # 保存模型
    def SaveModel(self):
        model = Deeplearning.CreateModel()
        Deeplearning.SaveModel(model)
        self.Writelog('模型保存成功')

    def CreateModel(self):
        model = Deeplearning.CreateModel()
        self.Writelog('模型创建成功')

    def Load_model(self):
        directory = QFileDialog.getExistingDirectory(self, '请选择文件路径')
        gl.gl_str_i2 = directory
        self.Writelog('加载模型成功', '模型所在位置：%s' % gl.gl_str_i2)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):  # real signature unknown; restored from __doc__
        """ flush(self) """
        pass




if __name__ == '__main__':
    app = QApplication(sys.argv)
    Mc = Myclass()
    Mc.show()
    sys.exit(app.exec_())
