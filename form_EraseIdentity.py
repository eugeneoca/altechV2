# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form_EraseIdentity.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(382, 127)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Logo/src/icon/Altech Logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayout = QtWidgets.QFormLayout(self.centralwidget)
        self.formLayout.setObjectName("formLayout")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.label_2)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.cmb_IdentityList = QtWidgets.QComboBox(self.centralwidget)
        self.cmb_IdentityList.setObjectName("cmb_IdentityList")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.cmb_IdentityList)
        self.btn_Erase = QtWidgets.QPushButton(self.centralwidget)
        self.btn_Erase.setObjectName("btn_Erase")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.btn_Erase)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 382, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CDS-1 | Erase Identity"))
        self.label_2.setText(_translate("MainWindow", "Note: The following task cannot be undone."))
        self.label.setText(_translate("MainWindow", "Select identity:"))
        self.btn_Erase.setText(_translate("MainWindow", "Erase"))

import resources_rc
