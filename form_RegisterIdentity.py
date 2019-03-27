# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form_RegisterIdentity.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(512, 214)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Logo/src/icon/Altech Logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayout_2 = QtWidgets.QFormLayout(self.centralwidget)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.txt_fname = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_fname.setObjectName("txt_fname")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.txt_fname)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.txt_lname = QtWidgets.QLineEdit(self.centralwidget)
        self.txt_lname.setObjectName("txt_lname")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.txt_lname)
        self.cmb_CameraList = QtWidgets.QComboBox(self.centralwidget)
        self.cmb_CameraList.setObjectName("cmb_CameraList")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.cmb_CameraList)
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(self.frame_3)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setContentsMargins(-1, 0, -1, -1)
        self.gridLayout.setObjectName("gridLayout")
        self.btn_startRegistration = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_startRegistration.sizePolicy().hasHeightForWidth())
        self.btn_startRegistration.setSizePolicy(sizePolicy)
        self.btn_startRegistration.setObjectName("btn_startRegistration")
        self.gridLayout.addWidget(self.btn_startRegistration, 0, 0, 1, 1)
        self.btn_stopRegistration = QtWidgets.QPushButton(self.frame)
        self.btn_stopRegistration.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_stopRegistration.sizePolicy().hasHeightForWidth())
        self.btn_stopRegistration.setSizePolicy(sizePolicy)
        self.btn_stopRegistration.setObjectName("btn_stopRegistration")
        self.gridLayout.addWidget(self.btn_stopRegistration, 0, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame, 1, 1, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.frame_3)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.formLayout = QtWidgets.QFormLayout(self.frame_2)
        self.formLayout.setContentsMargins(-1, 0, -1, -1)
        self.formLayout.setObjectName("formLayout")
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.label_4 = QtWidgets.QLabel(self.frame_2)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.txt_directory = QtWidgets.QLineEdit(self.frame_2)
        self.txt_directory.setReadOnly(True)
        self.txt_directory.setObjectName("txt_directory")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.txt_directory)
        self.txt_process = QtWidgets.QLineEdit(self.frame_2)
        self.txt_process.setReadOnly(True)
        self.txt_process.setObjectName("txt_process")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.txt_process)
        self.gridLayout_2.addWidget(self.frame_2, 1, 0, 1, 1)
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.SpanningRole, self.frame_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 512, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.btn_startRegistration.clicked['bool'].connect(self.txt_fname.setEnabled)
        self.btn_startRegistration.clicked['bool'].connect(self.txt_lname.setEnabled)
        self.btn_startRegistration.clicked['bool'].connect(self.btn_stopRegistration.setDisabled)
        self.btn_stopRegistration.clicked['bool'].connect(self.btn_stopRegistration.setEnabled)
        self.btn_stopRegistration.clicked['bool'].connect(self.txt_fname.setDisabled)
        self.btn_stopRegistration.clicked['bool'].connect(self.txt_lname.setDisabled)
        self.btn_stopRegistration.clicked['bool'].connect(self.cmb_CameraList.setDisabled)
        self.btn_stopRegistration.clicked['bool'].connect(self.btn_startRegistration.setDisabled)
        self.btn_startRegistration.clicked['bool'].connect(self.btn_startRegistration.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.txt_fname, self.txt_lname)
        MainWindow.setTabOrder(self.txt_lname, self.cmb_CameraList)
        MainWindow.setTabOrder(self.cmb_CameraList, self.btn_startRegistration)
        MainWindow.setTabOrder(self.btn_startRegistration, self.btn_stopRegistration)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CDS-1 | Register Identity"))
        self.label.setText(_translate("MainWindow", "First Name"))
        self.txt_fname.setPlaceholderText(_translate("MainWindow", "First Name"))
        self.label_2.setText(_translate("MainWindow", "Last Name"))
        self.txt_lname.setPlaceholderText(_translate("MainWindow", "Last Name"))
        self.btn_startRegistration.setText(_translate("MainWindow", "Start"))
        self.btn_stopRegistration.setText(_translate("MainWindow", "Stop"))
        self.label_3.setText(_translate("MainWindow", "Process:"))
        self.label_4.setText(_translate("MainWindow", "Directory:"))


import resources_rc
