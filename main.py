import sys
import time
import Deeplearning
from PyQt5.QtWidgets import QApplication, QWidget
from UIMain import Ui_MainWindow


# 版本：1.0.0.5
# 更新日期：20230212

class Myclass(QWidget, Ui_MainWindow):
    def __init__(self):
        super(Myclass, self).__init__()
        self.init()

    # 初始化
    def init(self):
        self.setupUi(self)
        self.textEdit.append('软件初始化成功')
        self.setWindowTitle('基于特征识别的手写数字识别')
        self.Button_LoadData.clicked.connect(self.loaddata)

    # 数据加载
    def loaddata(self):
        Deeplearning.load_data()
        self.Writelog('数据集加载成功')

    # 写log
    def Writelog(self, str):
        mes = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ':' + str
        self.textEdit.append(mes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Mc = Myclass()
    Mc.show()
    sys.exit(app.exec_())
