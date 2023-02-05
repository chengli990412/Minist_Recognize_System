import sys

import UIMain
from PyQt5.QtWidgets import QApplication,QMainWindow,QPushButton

#版本：1.0.0.3
#更新日期：20230205

if __name__ == '__main__':
    app = QApplication(sys.argv)

    MainWindow = QMainWindow()

    uimain = UIMain.Ui_MainWindow()

    uimain.setupUi(MainWindow)

    MainWindow.show()
    sys.exit(app.exec_())