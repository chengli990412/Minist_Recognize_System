from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

import Deeplearning


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(605, 385)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(330, 10, 54, 20))
        self.label.setObjectName("label")
        self.Button_Load = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Load.setGeometry(QtCore.QRect(490, 30, 100, 50))
        self.Button_Load.setObjectName("Button_Load")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(240, 30, 241, 151))
        self.tableView.setObjectName("tableView")
        self.Button_Run = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Run.setGeometry(QtCore.QRect(490, 80, 100, 50))
        self.Button_Run.setObjectName("Button_Run")
        self.Button_Load_3 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Load_3.setGeometry(QtCore.QRect(490, 140, 100, 50))
        self.Button_Load_3.setObjectName("Button_Load_3")
        self.Button_Load_4 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Load_4.setGeometry(QtCore.QRect(490, 182, 100, 50))
        self.Button_Load_4.setObjectName("Button_Load_4")
        self.Button_Load_5 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Load_5.setGeometry(QtCore.QRect(489, 232, 100, 50))
        self.Button_Load_5.setObjectName("Button_Load_5")
        self.Button_Load_6 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Load_6.setGeometry(QtCore.QRect(489, 282, 100, 50))
        self.Button_Load_6.setObjectName("Button_Load_6")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(240, 240, 251, 20))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(330, 220, 54, 20))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 605, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于特征处理的手写数字识别"))
        self.label.setText(_translate("MainWindow", "训练参数"))
        self.Button_Load.setText(_translate("MainWindow", "导入数据集"))
        self.Button_Run.setText(_translate("MainWindow", "神经网络训练"))
        self.Button_Load_3.setText(_translate("MainWindow", "训练暂停"))
        self.Button_Load_4.setText(_translate("MainWindow", "网络模型导入"))
        self.Button_Load_5.setText(_translate("MainWindow", "网络模型保存"))
        self.Button_Load_6.setText(_translate("MainWindow", "识别"))
        self.label_2.setText(_translate("MainWindow", "训练进度"))

    def Button_Click_LoadData(self):
        dl = Deeplearning.DLearning
        # 拿数据
        (train_images, train_labels), (test_images, test_labels) = dl.load_data('MNIST/')
        msg=QMessageBox()
        msg.setWindowTitle("数据集")
        msg.setText('数据集加载成功')
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ignore)
        x = msg.exec_()

        text = dl.Dl_Run(train_images , train_labels)
        msg1 = QMessageBox()
        msg1.setWindowTitle("模型训练")
        msg1.setText(str(text))
        msg1.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ignore)
        y = msg1.exec_()
