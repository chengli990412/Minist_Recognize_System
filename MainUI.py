
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

import Deeplearning
import time


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(605, 385)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(110, 30, 54, 20))
        self.label.setObjectName("label")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(20, 50, 241, 151))
        self.tableView.setObjectName("tableView")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(20, 260, 251, 20))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 240, 54, 20))
        self.label_2.setObjectName("label_2")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(280, 20, 161, 301))
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.Button_LoadData = QtWidgets.QPushButton(self.splitter)
        self.Button_LoadData.setObjectName("Button_LoadData")
        self.Button_LoadData.clicked.connect(self.Button_Click_LoadData)
        self.Button_Train = QtWidgets.QPushButton(self.splitter)
        self.Button_Train.setObjectName("Button_Train")
        self.Button_Stop = QtWidgets.QPushButton(self.splitter)
        self.Button_Stop.setObjectName("Button_Stop")
        self.Button_LoadModel = QtWidgets.QPushButton(self.splitter)
        self.Button_LoadModel.setObjectName("Button_LoadModel")
        self.Button_SaveModel = QtWidgets.QPushButton(self.splitter)
        self.Button_SaveModel.setObjectName("Button_SaveModel")
        self.Button_Recognize = QtWidgets.QPushButton(self.splitter)
        self.Button_Recognize.setObjectName("Button_Recognize")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(500, 0, 150, 381))
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于特征处理的手写数字识别"))
        self.label.setText(_translate("MainWindow", "训练参数"))
        self.label_2.setText(_translate("MainWindow", "训练进度"))
        self.Button_LoadData.setText(_translate("MainWindow", "导入数据集"))
        self.Button_Train.setText(_translate("MainWindow", "神经网络训练"))
        self.Button_Stop.setText(_translate("MainWindow", "训练暂停"))
        self.Button_LoadModel.setText(_translate("MainWindow", "网络模型导入"))
        self.Button_SaveModel.setText(_translate("MainWindow", "网络模型保存"))
        self.Button_Recognize.setText(_translate("MainWindow", "识别"))

        # 数据集导入
    def Button_Click_LoadData(self):
        dl = Deeplearning.DLearning
        (Ui_MainWindow.train_images, train_labels), (Ui_MainWindow.test_images, Ui_MainWindow.test_labels) = dl.load_data('MNIST/')
        self.Write_Log('数据集加载成功')

    def Write_Log(self , mes):
        text = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(mes)
        self.textEdit.append(text)
