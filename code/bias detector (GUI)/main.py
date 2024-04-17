import sys
import collections
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QIcon
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree
import bias_detector
from window2 import Ui_SecondWindow

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from QCandyUi import CandyWindow

pd.set_option('display.max_columns', 1000)

pd.set_option('display.width', 1000)

pd.set_option('display.max_colwidth', 1000)

def read_dataset(filename,target,predicted):
    df = pd.read_csv(filename)
    X = df.copy()
    p = X.pop(predicted)
    y = X.pop(target)
    for i in range(len(X)):
        y.iloc[i]= abs(y.iloc[i]-p.iloc[i])
        # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    return  X, y
        # return x_train, y_train

###################### the detector
def bias_detector_meta_dtr(x_train, y_train, M_dep,M_leaf,Min_sample,target,min_bias):                           # users can input the train and test dataset, and the depth and the leaf of the decision regression tree meta model to find biases
    dtr=DecisionTreeRegressor(max_depth=M_dep, max_leaf_nodes=M_leaf, min_samples_split=Min_sample, random_state=42)
    dtr.fit(x_train,y_train)
    fig = plt.figure(figsize=(8, 6))
    _ = tree.plot_tree(dtr, feature_names=x_train.columns.values, filled=True)

    fig.savefig("dtr.png")
    df = x_train.join(y_train)

    #reference: https://stackoverflow.com/questions/45398737/is-there-any-way-to-get-samples-under-each-leaf-of-a-decision-tree
    def get_rules(tree, feature_names, class_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths_2 = []   # record all the paths from the root to the leaf
        path_2 = []
        def recurse(node,path_2,paths_2):

            if tree_.feature[node] != _tree.TREE_UNDEFINED: #check whether the current node is null
                p3, p4 = list(path_2),list(path_2)
                p3 += [node]
                # recurse the left child of the tree
                recurse(tree_.children_left[node],p3,paths_2)
                p4 += [node]
                # recurse the right child of the tree
                recurse(tree_.children_right[node],p4,paths_2)
            else:
                path_2 += [node]
                paths_2 += [path_2]

        recurse(0, path_2,paths_2)

        # to get all the decision rules of each nodes
        route_all_nodes= []
        route_single_node = []
        for i in range(0,tree_.node_count):
            for item in paths_2:
                if i in item:
                    for j in item:
                        if j==i:
                            route_single_node.append(j)
                            break
                        else:
                            route_single_node.append(j)
                    break
            route_all_nodes.append(route_single_node)
            route_single_node = []

        paths = []

        node_number = []
        cur_node = 0
        for i in route_all_nodes:
            path = []
            iter_node = 0
            for j in range(0,len(i)):
                if iter_node != cur_node:
                    name = feature_name[iter_node]
                    threshold = tree_.threshold[iter_node]
                    if tree_.children_left[iter_node] == i[j+1]:
                        path += [f"({name} <= {np.round(threshold, 3)})"]
                        iter_node = tree_.children_left[iter_node]
                    else:
                        path += [f"({name} > {np.round(threshold, 3)})"]
                        iter_node = tree_.children_right[iter_node]
                else:
                    if (tree_.n_node_samples[iter_node] > 30) & (cur_node!=0):
                        path += [(tree_.value[iter_node], tree_.n_node_samples[iter_node])]
                        paths += [path]
                        node_number.append(cur_node)
            cur_node+=1

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))

        paths = [paths[i] for i in reversed(ii)]
        node_number = [node_number[i] for i in reversed(ii)]
        samples_count = [p[-1][1] for p in paths]

        rules = []
        mean_sample_value = []
        sample_size = []
        for path in paths:
            # rule = "if "
            rule = ""
            for p in path[:-1]:
                if rule != "":
                    rule += " and "
                rule += str(p)
            rules += [rule]
            # rule += " then "
            if class_names is None:
                mean_sample_value += [str(np.round(path[-1][0][0][0], 3))]
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
            # rule += f" | based on {path[-1][1]:,} samples"
            sample_size += [path[-1][1]]
            # rules += [rule]

        return rules, mean_sample_value, sample_size, node_number

    rules, mean_sample_value, sample_size, node_number = get_rules(dtr, x_train.columns.values, None)

    samples = collections.defaultdict(list)
    dec_paths = dtr.decision_path(x_train)

    for d, dec in enumerate(dec_paths):
        for i in range(dtr.tree_.node_count):
            if dec.toarray()[0][i] == 1:
                samples[i].append(d)

    bia = 0
    bias =[]
    confidence_interval= []
    Overal_mean = dtr.tree_.value[0][0][0]


    #to calculate and show the results
    for item in node_number:
        df_empty = pd.DataFrame(columns=df.columns.values)
        for i in samples[item]:
            df_empty = pd.concat([df_empty, df.iloc[[i]]])
        std = df_empty[target].std(ddof=0)
        mean = df_empty[target].mean()
        sqr_number = np.sqrt(dtr.tree_.n_node_samples[item])
        std_new = std / sqr_number # The standard deviation of the sampling distribution
        L_limit = mean - 2 * std_new  # the lower limit of the confidence interval
        U_limit = mean + 2 * std_new  # the upper limit of the confidence interval
        confidence_interval.append("(" + str(np.round(L_limit, 5)) + "," + str(np.round(U_limit, 5)) + ")")
        #The calculation of bias score
        if (L_limit <= Overal_mean) & (Overal_mean <= U_limit):
            bia = 0
        elif Overal_mean < L_limit:
            bia = L_limit - Overal_mean
        else:
            bia = U_limit - Overal_mean
        bias += [np.round(bia, 5)]  # the list of bias
    bias_2=bias.copy()
    for i in range(0, len(bias_2)):
        bias_2[i] = abs(bias_2[i])
    ii = list(np.argsort(bias_2))  # ascending order

    rules = [rules[i] for i in reversed(ii)]  # the list of decision rules
    mean_sample_value = [mean_sample_value[i] for i in reversed(ii)]
    sample_size = [sample_size[i] for i in reversed(ii)]
    confidence_interval = [confidence_interval[i] for i in reversed(ii)]  # the list of confidence rules
    bias = [bias[i] for i in reversed(ii)]  # the list of bias
    for i in range(0,len(bias)):
        if abs(bias[i]) < min_bias:
            del bias[i:(len(bias)+1)]
            print(bias)
            del rules[i:(len(bias) + 1)]
            del mean_sample_value[i:(len(bias) + 1)]
            del sample_size[i:(len(bias) + 1)]
            del confidence_interval[i:(len(bias) + 1)]
            break
    print("4")
    node_number = [node_number[i] for i in reversed(ii)]
    output = str(df[target].mean())
    print("5")
    return output,rules,mean_sample_value,sample_size,confidence_interval,bias
# 中间类Window的写法
class Window(bias_detector.Ui_MainWindow, QtWidgets.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        print("here")
        self.Select.clicked.connect(self.msg)
        self.Start.clicked.connect(self.start)
        self.dtr.clicked.connect(self.openWindow)
        self.Reset.clicked.connect(self.reset)
        self.export_excel.clicked.connect(self.toexcel)
        self.export_latex.clicked.connect(self.tolatex)
        self.tableWidget.setColumnWidth(0,300)
        # self.tableWidget.setColumnWidth(1,150)
        self.tableWidget.setColumnWidth(3,200)
        self.dataset_question.clicked.connect(self.show_popup_dataset)
        self.target_question.clicked.connect(self.show_popup_target)
        self.prediction_question.clicked.connect(self.show_popup_prediction)
        self.depth_question.clicked.connect(self.show_popup_depth)
        self.leaf_question.clicked.connect(self.show_popup_leaf)
        self.mim_sample_question.clicked.connect(self.show_popup_mim_sample)
        self.min_bias_question.clicked.connect(self.show_popup_mim_bias)
        self.whole_sample_question.clicked.connect(self.show_popup_whole_sample)
        self.overall_mean_question.clicked.connect(self.show_popup_overall_mean)
        self.graph_question.clicked.connect(self.show_popup_show_graph)
        self.csv_question.clicked.connect(self.show_popup_csv)
        self.reset_question.clicked.connect(self.show_popup_reset)
        self.latex_question.clicked.connect(self.show_popup_latex)
    def msg(self):
        filePath, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "Please choose the file", "./",
                                                                   "CSV files (*.csv)")
        self.textBrowser_dataset.setText(filePath)
        print(self.textBrowser_dataset.toPlainText())

    def start(self):
        if self.textBrowser_dataset.toPlainText() == "":
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("You must upload the dataset in the csv type before starting bias detection.")
            msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
            x = msg.exec_()
            return

        df = pd.read_csv(self.textBrowser_dataset.toPlainText())

        if self.lineEdit_target.text() not in df.columns:
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("The target variable name you entered is not one of the column in the dataset.")
            msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
            x = msg.exec_()
            return

        if self.lineEdit_predicted.text() not in df.columns:
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("The predicted variable name you entered is not one of the column in the dataset.")
            msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
            x = msg.exec_()
            return

        x_train, y_train = read_dataset(self.textBrowser_dataset.toPlainText(),self.lineEdit_target.text(),self.lineEdit_predicted.text())
        print("1")
        # print(len(x_train))
        self.textBrowser_sample_size.setText(str(len(x_train)))
        output, rules, mean_sample_value, sample_size, confidence_interval, bias = bias_detector_meta_dtr(x_train, y_train,self.spinBox_depth.value(),self.spinBox_leaf.value(),self.spinBox_min_number.value(),self.lineEdit_target.text(),self.doubleSpinBox.value())
        print("2")
        self.textBrowser_results.setText(output)
        print("3")
        row = 0
        self.tableWidget.setRowCount(len(bias))
        for i in range(0, len(bias)):
            self.tableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(rules[i]))
            self.tableWidget.setItem(row, 1, QtWidgets.QTableWidgetItem(mean_sample_value[i]))
            self.tableWidget.setItem(row, 2, QtWidgets.QTableWidgetItem(str(sample_size[i])))
            self.tableWidget.setItem(row, 3, QtWidgets.QTableWidgetItem(confidence_interval[i]))
            self.tableWidget.setItem(row, 4, QtWidgets.QTableWidgetItem(str(bias[i])))
            row=row+1

    def openWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_SecondWindow()
        self.ui.setupUi(self.window)
        self.ui.image.setPixmap(QPixmap('dtr.png'))
        self.window.show()

    def reset(self):
        self.textBrowser_dataset.setText("")
        self.lineEdit_target.setText("")
        self.lineEdit_predicted.setText("")
        self.textBrowser_sample_size.setText("")
        self.textBrowser_results.setText("")
        self.spinBox_depth.setProperty("value", 0)
        self.spinBox_leaf.setProperty("value", 0)
        self.spinBox_min_number.setProperty("value", 2)
        self.doubleSpinBox.setProperty("value", 0.0)
        self.tableWidget.setRowCount(0)

    def toexcel(self):
        columnHeaders = []

        for j in range(self.tableWidget.model().columnCount()):
            columnHeaders.append(self.tableWidget.horizontalHeaderItem(j).text())

        df = pd.DataFrame(columns=columnHeaders)

        for row in range(self.tableWidget.rowCount()):
            for col in range(self.tableWidget.columnCount()):
                df.at[row, columnHeaders[col]]= self.tableWidget.item(row,col).text()

        filepath, type = QFileDialog.getSaveFileName(self, "save as csv", "/bias_score", 'csv(*.csv)')
        if filepath == '':
            pass
        else:
            df.to_csv(filepath,index=False)

    def tolatex(self):
        columnHeaders = []

        for j in range(self.tableWidget.model().columnCount()):
            columnHeaders.append(self.tableWidget.horizontalHeaderItem(j).text())

        df = pd.DataFrame(columns=columnHeaders)

        for row in range(self.tableWidget.rowCount()):
            for col in range(self.tableWidget.columnCount()):
                df.at[row, columnHeaders[col]]= self.tableWidget.item(row,col).text()
        print(df)

        filepath, type = QFileDialog.getSaveFileName(self, "save as tex", "/bias_score", 'tex(*.tex)')

        if filepath == '':
            pass
        else:
            df.to_latex(filepath)

            texdoc = []  # a list of string representing the latex document in python

            # read the .tex file, and modify the lines
            with open(filepath) as fin:
                for line in fin:
                    line = line.replace(r'>', r'$\textgreater$')
                    line = line.replace('<=', '$\leq$')
                    texdoc.append(line)

            # write back the new document
            with open(filepath, 'w') as fout:
                for i in range(len(texdoc)):
                    fout.write(texdoc[i])

    def show_popup_dataset(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("You should input a csv file including all the samples with the features, target variable and the predicted variable.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_target(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("Please input the name of your target variable.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_prediction(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("Please input the name of your prediction variable.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_depth(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("Please enter the maximum depth of the decision regression tree.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_leaf(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("Please enter the maximum number of leaves in the decision regression tree.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_mim_sample(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("Please enter the minimum number of samples required to split an internal node.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_mim_bias(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("Please enter the minimum bias score magnitude you are interested in.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_whole_sample(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("This field shows the sample size of the whole dataset.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_overall_mean(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("This field shows the mean error of the whole dataset.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_show_graph(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("Click this to view the graph of the decision tree.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_csv(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("Click this to export the output table to csv.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_latex(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("Click this to export the output table to latex format(.tex).")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()

    def show_popup_reset(self):
        msg = QMessageBox()
        msg.setWindowTitle("help")
        msg.setText("Click this to reset the GUI.")
        msg.setStyleSheet('QMessageBox{background-color: white; border: 1px solid black; font: bold 24px;}')
        x = msg.exec_()



if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    mywindow = Window()
    # print(0)
    mywindow = CandyWindow.createWindow(mywindow, 'blueDeep')
    # print(1)
    mywindow.setWindowTitle("Bias detector")
    # print(2)
    mywindow.show()
    # print(3)
    sys.exit(app.exec_())
