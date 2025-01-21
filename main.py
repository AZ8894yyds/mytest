#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 当前的项目名：FatigueLife.py
# 当前编辑文件名：main
# 当前用户的登录名：AZ
# 当前系统日期：2025/1/11
# 当前系统时间：17:53
# 用于创建文件的IDE的名称: PyCharm

import os
import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QGraphicsScene, QTableWidgetItem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ui_composite_FS import Ui_MainWindow
from functions import *


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.default_table_data()
        self.textBrowser.setText('初始化完毕')
        self._plot()

        self.pushButton_2.clicked.connect(self.textBrowser.clear)  # 文本浏览器刷新按钮
        # self.pushButton_4.clicked.connect(self.refresh_figure)
        # self.pushButton_5.clicked.connect(self.refresh_figure)
        self.pushButton_8.clicked.connect(self.rf_process)
        self.pushButton_11.clicked.connect(self.deep_learning_process)
        self.pushButton_13.clicked.connect(self.svm_process)  # SVM 参数确认按钮
        self.pushButton_14.clicked.connect(self.load_csv)
        self.pushButton_15.clicked.connect(self.set_frame3_checked_state)
        self.comboBox.activated.connect(self.set_groupbox2_checked_state)

    def set_groupbox2_checked_state(self):
        # 设置初始的选择框状态
        if self.comboBox.currentIndex() == 0:
            self.groupBox_5.setChecked(True)
            self.groupBox_4.setChecked(False)
            self.groupBox_3.setChecked(False)
        elif (self.comboBox.currentIndex() == 1) or (self.comboBox.currentIndex() == 2):
            self.groupBox_5.setChecked(False)
            self.groupBox_4.setChecked(True)
            self.groupBox_3.setChecked(False)
        elif self.comboBox.currentIndex() == 3:
            self.groupBox_5.setChecked(False)
            self.groupBox_4.setChecked(False)
            self.groupBox_3.setChecked(True)

    def set_frame3_checked_state(self):
        self.checkBox_7.setChecked(True)
        self.checkBox_8.setChecked(True)
        self.checkBox_9.setChecked(True)
        self.checkBox_10.setChecked(True)

    def openFile(self):
        sender = self.sender()
        self.textBrowser.append(sender.text() + '按键被点击')
        self.load_csv()

    def click_sender_name(self):
        sender = self.sender()
        self.textBrowser.append(sender.objectName() + '按键被点击')

    def evaluation_of_ml(self, y_true, y_pred):
        # 拟合系数 R² (R-squared)
        r2 = r2_score(y_true, y_pred)
        if self.checkBox.isChecked():
            self.textBrowser.append(f'R² = {r2}')

        # 相对绝对误差 (Mean Absolute Error)
        mae = mean_absolute_error(y_true, y_pred)
        if self.checkBox_6.isChecked():
            self.textBrowser.append(f'Mean Absolute Error = {mae}')

        # 平均偏差误差 (Mean Bias Error) = 平均预测值 - 平均真实值
        mbe = np.mean(y_pred - y_true)
        if self.checkBox_5.isChecked():
            self.textBrowser.append(f'Mean Bias Error = {mbe}')

        # 平均绝对误差 (Mean Absolute Error) = MAE, 其实和上面的 MAE 是一样的
        mae_custom = np.mean(np.abs(y_pred - y_true))
        if self.checkBox_6.isChecked():
            self.textBrowser.append(f'Mean Bias Error = {mae_custom}')

        # 均方误差 (Mean Square Error)
        mse = mean_squared_error(y_true, y_pred)
        if self.checkBox_2.isChecked():
            self.textBrowser.append(f'Mean Square Error = {mse}')

        # 均方根误差 (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        if self.checkBox_3.isChecked():
            self.textBrowser.append(f'Root Mean Squared Error = {rmse}')

    def rf_process(self):
        self.click_sender_name()
        initialize()
        X_train, X_test, y_train, y_test = feature_engineer(self.lineEdit.text(), self.doubleSpinBox.value())
        model = rf_train(X_train, y_train, self.spinBox_5.value(), self.spinBox_6.value())
        self.test_plot(model, X_test, y_test, MODEL[self.comboBox.currentIndex()])
        self.textBrowser.append('训练完毕 (RF)')

    def svm_process(self):
        self.click_sender_name()
        X_train, X_test, y_train, y_test = feature_engineer(self.lineEdit.text(), self.doubleSpinBox.value())
        model = svm_train(X_train, y_train, self.comboBox_2.currentText(), self.comboBox_3.currentText(),
                          self.spinBox_3.value(), self.doubleSpinBox_2.value())
        Fig = self.test_plot(model, X_test, y_test, MODEL[self.comboBox.currentIndex()])
        self.textBrowser.append('训练完毕 (SVM)')
        return Fig

    def deep_learning_process(self):
        self.click_sender_name()
        X_train, X_test, y_train, y_test = feature_engineer(self.lineEdit.text(), self.doubleSpinBox.value())
        y_train = y_train.values
        X_test, loader = quantization(X_train, y_train.reshape(-1, 1), X_test)

        if self.comboBox.currentIndex() == 1:
            model, epochs, losses = back_propagation_neural_network(loader, num_epochs=int(self.spinBox_4.value()))
            self.textBrowser.append('训练完毕 (BPNN)')
        else:
            model, epochs, losses = bayesian_neural_network(loader, num_epochs=int(self.spinBox_4.value()))
            self.textBrowser.append('训练完毕 (BNN)')

        self.test_plot(model, X_test, y_test, MODEL[self.comboBox.currentIndex()])

    def load_csv(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV", os.path.abspath(__file__),
                                                             "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
                                                             options=options)
        if file_name:
            self.lineEdit.setText(f'{file_name}')
            df = pd.read_csv(file_name, header=[0], encoding='ANSI')  # 设置好表头和编码
            self.tableView.setRowCount(df.shape[0])  # 设置表格行数为 CSV 数据的行数
            # print(df.shape[0])
            self.tableView.setColumnCount(df.shape[1])
            self.tableView.setHorizontalHeaderLabels(df.columns)
            for i, songInfo in enumerate(df.values):  # df.values 返回的是二维 ndarray
                for j in range(df.shape[1]):  # 遍历每一列
                    self.tableView.setItem(i, j, QTableWidgetItem(str(songInfo[j])))

    def default_table_data(self):
        self.tableView.setRowCount(30)
        self.tableView.setColumnCount(5)
        songInfos = [
            ['[0,90,90,0]', '50%', '0J', '0.95', '5764'],
            ['[0,90,90,0]', '50%', '12J', '0.95', '4467'],
            ['[0,90,90,0]', '50%', '16J', '0.95', '937'],
            ['[0,90,90,0]', '50%', '16J', '0.85', '9922'],
            ['[0,90,0,90]', '60%', '0J', '0.95', '521'],
            ['[0,90,0,90]', '60%', '12J', '0.85', '13447'],
            ['[0,90,0,90]', '60%', '8J', '0.95', '500'],
            ['[0,90,0,90]', '60%', '16J', '0.85', '854'],
            ['[0,90,0,90]', '70%', '0J', '0.95', '316'],
            ['[45,90,-45,0]', '50%', '0J', '0.95', '435'],
            ['[45,90,-45,0]', '50%', '12J', '0.75', '602'],
            ['[45,90,-45,0]', '50%', '16J', '0.65', '441'],
            ['[45,90,-45,0]', '60%', '0J', '0.85', '450'],
            ['[45,90,-45,0]', '60%', '16J', '0.95', '418'],
            ['[45,90,-45,0]', '70%', '0J', '0.65', '9012'],
            ['[45,0,-45,90]', '60%', '0J', '0.95', '354'],
            ['[45,0,-45,90]', '60%', '12J', '0.95', '1014'],
            ['[45,0,-45,90]', '50%', '16J', '0.85', '1045'],
            ['[45,0,-45,90]', '50%', '0J', '0.75', '112798'],
            ['[45,0,-45,90]', '70%', '0J', '0.65', '4024']
        ]
        for i, songInfo in enumerate(songInfos):
            for j in range(5):
                self.tableView.setItem(i, j, QTableWidgetItem(songInfo[j]))
        self.tableView.verticalHeader().hide()
        self.tableView.setHorizontalHeaderLabels(['Layer', 'Fiber', 'Impact', 'Stress', 'Fatigue'])
        self.tableView.resizeColumnsToContents()
        self.spinBox.setValue(4)

    def test_plot(self, model, X_test, y_test, model_name):
        initialize()
        Fig = MyFigure()
        Fig.axes = Fig.fig.add_subplot(111)
        set_style('Experimental (Cycle)', 'Predicted (Cycle)', Fig.axes)

        if (self.comboBox.currentIndex() == 1) or (self.comboBox.currentIndex() == 2):
            model.eval()
            with torch.no_grad():
                y_predict = model(X_test).detach().numpy()
        else:
            y_predict = model.predict(X_test)
        y_predict = 10 ** y_predict
        y_test = 10 ** y_test
        Fig.axes.scatter(y_test, y_predict, c='white', edgecolors='tab:red', label=model_name, s=64, zorder=4)
        self.evaluation_of_ml(y_test, y_predict)

        a = np.linspace(20, 50e4, 100)  # 注意从0.1开始以避免对数坐标的负无穷问题
        b = a
        Fig.axes.set_xlim(500, 50e4)
        Fig.axes.set_ylim(500, 50e4)

        if self.checkBox_10.isChecked():
            Fig.axes.loglog(a, b)

        if self.checkBox_9.isChecked():
            Fig.axes.plot(a, b, c='tab:blue', lw=1.5, zorder=1, label='Perfect predict line')

        if self.checkBox_7.isChecked():
            b_upper, b_lower = calculate_dispersion_band(a, factor=1.5)
            Fig.axes.plot(a, b_lower, color='grey', label='Scatter band of 1.5 times', zorder=2, lw=1,
                          linestyle='-')
            Fig.axes.plot(a, b_upper, color='grey', zorder=2, lw=1, linestyle='-')

        if self.checkBox_8.isChecked():
            b_upper, b_lower = calculate_dispersion_band(a, factor=2.0)
            Fig.axes.plot(a, b_lower, color='grey', label='Scatter band of 2.0 times', zorder=3, lw=1,
                          linestyle='-.')
            Fig.axes.plot(a, b_upper, color='grey', zorder=3, lw=1, linestyle='-.')

        Fig.axes.legend(fontsize=12, facecolor='w', edgecolor='#DCDCDC', fancybox=True)
        Fig.fig.tight_layout()
        width, height = self.graphicsView.width(), self.graphicsView.height()
        Fig.resize(width, height)
        self.scene.clear()
        self.scene.addWidget(Fig)
        self.graphicsView.setScene(self.scene)
        return Fig

    def _plot(self):
        initialize()
        # width, height = self.graphicsView.width(), self.graphicsView.height()
        Fig = MyFigure(6, 5.2)
        Fig.axes = Fig.fig.add_subplot(111)
        set_style('X', 'Y', Fig.axes)
        a = np.linspace(20, 50e4, 100)  # 注意从0.1开始以避免对数坐标的负无穷问题
        b = a
        Fig.axes.set_xlim(500, 50e4)
        Fig.axes.set_ylim(500, 50e4)

        Fig.axes.loglog(a, b)

        Fig.axes.plot(a, b, c='tab:blue', lw=1.5, zorder=1, label='Perfect predict line')
        b_upper, b_lower = calculate_dispersion_band(a, factor=1.5)
        Fig.axes.plot(a, b_lower, color='grey', label='Scatter band of 1.5 times', zorder=2, lw=1, linestyle='-')
        Fig.axes.plot(a, b_upper, color='grey', zorder=2, lw=1, linestyle='-')
        b_upper, b_lower = calculate_dispersion_band(a, factor=2.0)
        Fig.axes.plot(a, b_lower, color='grey', label='Scatter band of 2.0 times', zorder=3, lw=1, linestyle='-.')
        Fig.axes.plot(a, b_upper, color='grey', zorder=3, lw=1, linestyle='-.')

        Fig.axes.legend(fontsize=12, facecolor='w', edgecolor='#DCDCDC', fancybox=True)
        Fig.fig.tight_layout()
        self.scene = QGraphicsScene()
        self.scene.addWidget(Fig)
        self.graphicsView.setScene(self.scene)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
