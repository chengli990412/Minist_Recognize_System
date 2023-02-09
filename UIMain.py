# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UIMain.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import time

import Deeplearning


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1256, 793)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(1020, 0, 231, 781))
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1021, 771))
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.Button_LoadData = QtWidgets.QPushButton(self.tab)
        self.Button_LoadData.setGeometry(QtCore.QRect(860, 90, 100, 50))
        self.Button_LoadData.setObjectName("Button_LoadData")
        self.Button_LoadData.clicked.connect(self.Button_Click_LoadData)
        self.Button_Recognize = QtWidgets.QPushButton(self.tab)
        self.Button_Recognize.setGeometry(QtCore.QRect(860, 460, 100, 50))
        self.Button_Recognize.setObjectName("Button_Recognize")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.Button_Train_3 = QtWidgets.QPushButton(self.tab_2)
        self.Button_Train_3.setGeometry(QtCore.QRect(880, 236, 100, 50))
        self.Button_Train_3.setObjectName("Button_Train_3")
        self.Button_Train = QtWidgets.QPushButton(self.tab_2)
        self.Button_Train.setGeometry(QtCore.QRect(880, 132, 100, 50))
        self.Button_Train.setObjectName("Button_Train")
        # self.Button_Train.clicked.connect(self.Button_Click_Dltrain)
        self.Button_Train_2 = QtWidgets.QPushButton(self.tab_2)
        self.Button_Train_2.setGeometry(QtCore.QRect(880, 184, 100, 50))
        self.Button_Train_2.setObjectName("Button_Train_2")
        self.tableView = QtWidgets.QTableView(self.tab_2)
        self.tableView.setGeometry(QtCore.QRect(440, 280, 371, 261))
        self.tableView.setObjectName("tableView")
        self.label = QtWidgets.QLabel(self.tab_2)
        self.label.setGeometry(QtCore.QRect(450, 260, 54, 20))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(460, 170, 54, 20))
        self.label_2.setObjectName("label_2")
        self.progressBar = QtWidgets.QProgressBar(self.tab_2)
        self.progressBar.setGeometry(QtCore.QRect(520, 170, 251, 20))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.Button_Train_4 = QtWidgets.QPushButton(self.tab_2)
        self.Button_Train_4.setGeometry(QtCore.QRect(880, 288, 100, 50))
        self.Button_Train_4.setObjectName("Button_Train_4")
        self.Button_Train_5 = QtWidgets.QPushButton(self.tab_2)
        self.Button_Train_5.setGeometry(QtCore.QRect(880, 340, 100, 50))
        self.Button_Train_5.setObjectName("Button_Train_5")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于特征处理的手写数字识别"))
        self.Button_LoadData.setText(_translate("MainWindow", "导入数据集"))
        self.Button_Recognize.setText(_translate("MainWindow", "识别"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "识别"))
        self.Button_Train_3.setText(_translate("MainWindow", "训练强制停止"))
        self.Button_Train.setText(_translate("MainWindow", "神经网络训练"))
        self.Button_Train_2.setText(_translate("MainWindow", "网络模型导入"))
        self.label.setText(_translate("MainWindow", "训练参数"))
        self.label_2.setText(_translate("MainWindow", "训练进度"))
        self.Button_Train_4.setText(_translate("MainWindow", "模型保存"))
        self.Button_Train_5.setText(_translate("MainWindow", "模型读取"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "神经网络"))

        # 数据集导入
    def Button_Click_LoadData(self):
        L= Deeplearning.DeepLearningClass
        L.Load_data ()
        # (Ui_MainWindow.train_images, train_labels), (Ui_MainWindow.test_images, Ui_MainWindow.test_labels) = Deeplearning.load_data('MNIST/')
        self.Write_Log('数据集加载成功')

    # def Button_Click_Dltrain(self):
    #     Deeplearning.Dl_Run()
    #     self.Write_Log('模型训练完成！')

    def Write_Log(self , mes):
        text = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+str(mes)
        self.textEdit.append(text)