# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ui\test.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_zoom_in = QtWidgets.QPushButton(self.centralwidget)
        self.btn_zoom_in.setGeometry(QtCore.QRect(180, 600, 80, 40))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(10)
        self.btn_zoom_in.setFont(font)
        self.btn_zoom_in.setObjectName("btn_zoom_in")
        self.btn_zoom_out = QtWidgets.QPushButton(self.centralwidget)
        self.btn_zoom_out.setGeometry(QtCore.QRect(690, 600, 80, 40))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(10)
        self.btn_zoom_out.setFont(font)
        self.btn_zoom_out.setObjectName("btn_zoom_out")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(80, 10, 1121, 561))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.scrollArea = QtWidgets.QScrollArea(self.verticalLayoutWidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.label_img = QtWidgets.QLabel()
        self.label_img.setGeometry(QtCore.QRect(0, 0, 941, 521))
        self.label_img.setObjectName("label_img")
        self.scrollArea.setWidget(self.label_img)
        self.verticalLayout.addWidget(self.scrollArea)
        self.label_img_shape = QtWidgets.QLabel(self.centralwidget)
        self.label_img_shape.setGeometry(QtCore.QRect(779, 650, 421, 25))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(10)
        self.label_img_shape.setFont(font)
        self.label_img_shape.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_img_shape.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img_shape.setObjectName("label_img_shape")
        self.label_ratio = QtWidgets.QLabel(self.centralwidget)
        self.label_ratio.setGeometry(QtCore.QRect(780, 610, 100, 25))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        self.label_ratio.setFont(font)
        self.label_ratio.setObjectName("label_ratio")
        self.btn_open_file = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open_file.setGeometry(QtCore.QRect(80, 600, 80, 40))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(10)
        self.btn_open_file.setFont(font)
        self.btn_open_file.setObjectName("btn_open_file")
        self.label_file_name = QtWidgets.QLabel(self.centralwidget)
        self.label_file_name.setGeometry(QtCore.QRect(80, 650, 671, 25))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        self.label_file_name.setFont(font)
        self.label_file_name.setObjectName("label_file_name")
        self.slider_zoom = QtWidgets.QSlider(self.centralwidget)
        self.slider_zoom.setGeometry(QtCore.QRect(280, 610, 391, 22))
        self.slider_zoom.setOrientation(QtCore.Qt.Horizontal)
        self.slider_zoom.setObjectName("slider_zoom")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_zoom_in.setText(_translate("MainWindow", "zoom in"))
        self.btn_zoom_out.setText(_translate("MainWindow", "zoom out"))
        self.label_img.setText(_translate("MainWindow", "TextLabel"))
        self.label_img_shape.setText(_translate("MainWindow", "Current image shape：(0, 0), Origin image shape：(0, 0)"))
        self.label_ratio.setText(_translate("MainWindow", "ratio:100%"))
        self.btn_open_file.setText(_translate("MainWindow", "Open file"))
        self.label_file_name.setText(_translate("MainWindow", "file name："))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
