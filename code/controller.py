from PyQt5 import QtCore 
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from UI import Ui_MainWindow
from img_controller import img_controller

class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.file_path = ''
        self.img_controller = img_controller(img_path=self.file_path,
                                             label_img=self.ui.label_img,
                                             label_file_path=self.ui.label_file_name)
                                            
        self.ui.btn_open_file.clicked.connect(self.open_file)
        self.ui.btn_knn_show.clicked.connect(self.img_controller.knn_show)
        self.ui.btn_ans_show.clicked.connect(self.img_controller.ans_show)

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.init_new_picture(filename)

    def init_new_picture(self, filename):
        self.img_controller.set_path(filename)        
