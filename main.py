import sys
import MainUI
from PyQt5.QtWidgets import QApplication,QMainWindow,QPushButton

#版本：1.0.0.1
#更新日期：20230204

if __name__ == '__main__':
    app = QApplication(sys.argv)   #声明应用程序
    MainWindow = QMainWindow()     #声明窗口
    ui = MainUI.Ui_MainWindow()          #声明UI对话框，即对应的UI的python文件
    ui.setupUi(MainWindow)      #将UI对话框对应声明的窗口
    MainWindow.show()
    sys.exit(app.exec_())    #当点击窗口的x时，退出程序