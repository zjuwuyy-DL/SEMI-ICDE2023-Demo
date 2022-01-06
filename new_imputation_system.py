import sys
sys.path.append('Model')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from importlib import import_module
from PySide2.QtWidgets import *
from PySide2.QtGui import QBrush, QColor, QIcon, QFont, QImage, QPixmap, QIntValidator, QDoubleValidator
from PySide2.QtCore import QThread, Qt, QSize, Signal, QTimer
import pandas as pd
import numpy as np
import subprocess
import merge
import Regression

'''
StepWidget:
    Widget for the five buttons on the top of the screen.
'''
class StepWidget(QWidget):
    def __init__(self):
        super(StepWidget, self).__init__()
        self.initUI()
    def initUI(self):    
        self.layout = QHBoxLayout()
        self.SCIS = QLabel()
        SCIS_pic = QImage('fig/semi.png')
        scale_SCIS_pic = SCIS_pic.scaled(66, 24, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.SCIS.setPixmap(QPixmap.fromImage(scale_SCIS_pic))
        self.app_pic = QLabel() 
        pic = QImage('fig/zju2.png')
        scale_pic = pic.scaled(35, 28, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.app_pic.setPixmap(QPixmap.fromImage(scale_pic))
        self.SCIS.setMinimumHeight(55)
        self.SCIS.setMaximumHeight(55)
        self.app_pic.setMaximumHeight(55)
        self.app_pic.setMinimumHeight(55)
        self.step_layout_button1 = QPushButton("Data Preprocessing")
        # self.step_layout_button2 = QPushButton("File Preview")
        # self.step_layout_button3 = QPushButton("Data Preprocessing")
        self.step_layout_button4 = QPushButton("Data Imputation")
        # self.step_layout_button5 = QPushButton("Data Imputation")
        self.step_layout_button6 = QPushButton("Post-imputation Prediction")
        self.placeholder = QLabel()
        self.placeholder.setMinimumHeight(55)
        self.placeholder.setMaximumHeight(55)
        self.layout.addWidget(self.SCIS, 0)
        self.layout.addWidget(self.app_pic, 0)
        self.layout.addWidget(self.step_layout_button1, 0, Qt.AlignCenter)
        #self.layout.addWidget(self.step_layout_button2, 0, Qt.AlignCenter)
        # self.layout.addWidget(self.step_layout_button3, 0, Qt.AlignCenter)
        self.layout.addWidget(self.step_layout_button4, 0, Qt.AlignCenter)
        # self.layout.addWidget(self.step_layout_button5, 0, Qt.AlignCenter)
        self.layout.addWidget(self.step_layout_button6, 0, Qt.AlignCenter)
        self.layout.addWidget(self.placeholder, 1)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        self.setStyleSheet('''
            QPushButton{
                border: none;
                height: 55px;
                font-size:21px;
                font: large "Microsoft YaHei UI";
                qproperty-icon:url(fig/go_right.png);
                qproperty-iconSize:18px 18px;
                text-align: left
            }
            QPushButton:hover{
                height: 55px;
                font-size:21px;
                font: bold large "Microsoft YaHei UI";
                background-color: #0e83c2;
            }
            *{
                background-color: #0c6ca1;
                color: #F7F9FF;
            }
        ''')
        self.step_layout_button1.setMaximumWidth(227)
        self.step_layout_button1.setMinimumWidth(227)
        # self.step_layout_button2.setMaximumWidth(273)
        # self.step_layout_button2.setMinimumWidth(273)
        # self.step_layout_button3.setMaximumWidth(227)
        # self.step_layout_button3.setMinimumWidth(227)
        self.step_layout_button4.setMaximumWidth(195)
        self.step_layout_button4.setMinimumWidth(195)
        # self.step_layout_button5.setMaximumWidth(195)
        # self.step_layout_button5.setMinimumWidth(195)
        self.step_layout_button6.setMaximumWidth(307)
        self.step_layout_button6.setMinimumWidth(307)
        
'''
Page1_Widget:
    Widget for the first interface.
'''        
class Page1_Widget(QWidget):
    def __init__(self):
        super(Page1_Widget, self).__init__()
        self.initUI()
    def initUI(self):    
        # ===================================BUTTON================================
        button_layout = QHBoxLayout()
        self.select_button = QPushButton(" Select File(s)")
        self.select_button.setProperty('name', 'file_select_button')
        self.setStyleSheet('''
            QPushButton[name='file_select_button']{
                background-color: #326AA9;
                font: 21px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 14px;
                height: 60px;
                width: 190px;
                qproperty-icon:url(fig/folder.png);
                qproperty-iconSize:35px 35px;
                text-align: center;
                margin-right: 10px;
            }
            QPushButton:hover[name='file_select_button']{
                background-color: #457FBF;
                font: bold 21px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 14px;
                height: 60px;
                width: 190;
            }
        '''
        )
        button_layout.addWidget(QLabel(),1)
        button_layout.addWidget(self.select_button,0)
        button_layout.addWidget(QLabel(),1)
        # ================================FILE_DISPLAYER=============================
        filedisplay_layout = QVBoxLayout()
        # (1) file_name sub-layout
        file_name_layout = QHBoxLayout()
        file_name = QLabel("File Name")
        file_name.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
                color: #666666;
            }
        ''')
        file_name_layout.addWidget(file_name, 0)
        file_name_layout.addWidget(QLabel(), 1)
        filedisplay_layout.addLayout(file_name_layout)
        # (2) Horizontal Line
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setLineWidth(1)
        hline.setStyleSheet('''
            color: #666666;
        ''')
        # filedisplay_layout.addWidget(hline)
        # (3) Display and Delete Table
        self.file_table = QTableWidget()
        self.file_table.setStyleSheet('''
            border: none;
        ''')
        self.file_table.setColumnCount(2)
        self.file_table.setRowCount(0)
        self.file_table.horizontalHeader().setStretchLastSection(True)     
        self.file_table.horizontalHeader().hide()
        self.file_table.verticalHeader().hide() 
        self.file_table.setShowGrid(False)
        self.file_table.setStyleSheet('''
            font-size: 19px;
        ''')
        self.file_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.file_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        filedisplay_layout.addWidget(self.file_table)

        FileUploadLayout = QHBoxLayout()
        FileUploadLayout.addWidget(self.select_button, 0)
        FileUploadLayout.addLayout(filedisplay_layout, 1)

        # ===================================NEXT_BUTTON===========================
        nextbutton_layout = QHBoxLayout()
        self.next_button = QPushButton("Next")
        self.next_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 20px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 20px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
        ''')
        nextbutton_layout.addWidget(QLabel(), 1)
        nextbutton_layout.addWidget(self.next_button, 0)

        # =================================ALL LAYOUTS=================================
        # placeholder = QLabel()
        # placeholder.setMinimumHeight(20)
        # placeholder.setMaximumHeight(20)
        # layout = QVBoxLayout()
        # layout.addWidget(placeholder)
        # layout.addLayout(FileUploadLayout)
        # layout.addWidget(placeholder)
        # layout.addLayout(nextbutton_layout)
        self.setLayout(FileUploadLayout)

'''
FileDeleteButton:
    "Delete" button in Page1 and Page2.
'''
class FileDeleteButton(QWidget):
    def __init__(self, idx):
        super(FileDeleteButton, self).__init__()
        self.initUI(idx)
    def initUI(self, idx):    
        self.delete_button = QPushButton("Delete")
        self.delete_button.setProperty('idx', idx)
        self.delete_button.setStyleSheet('''
            QPushButton{
                font: 18px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/delete.png);
                qproperty-iconSize: 18px 18px;
                margin: 0px;
            }
            QPushButton:hover{
                font: bold 18px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/delete.png);
                margin: 0px;
            }
        ''')
        self.delete_button.setMaximumWidth(230)
        self.delete_button.setMinimumWidth(230)
        layout = QHBoxLayout()
        layout.addWidget(QLabel(),1)
        layout.addWidget(self.delete_button,0)
        self.setLayout(layout)

'''
HintButton:
    "See template file" button in Page4.
'''
class HintButton(QWidget):
    def __init__(self):
        super(HintButton, self).__init__()
        self.initUI()
    def initUI(self):    
        self.hint_button = QPushButton("See Template File")
        self.hint_button.setStyleSheet('''
            QPushButton{
                font: 18px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/hint.png);
                qproperty-iconSize: 25px 25px;
                margin: 0px;
                height: 22px;
            }
            QPushButton:hover{
                font: bold 18px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/hint.png);
                margin: 0px;
                height: 22px;
            }
        ''')
        self.hint_button.setMaximumWidth(230)
        self.hint_button.setMinimumWidth(230)
        layout = QHBoxLayout()
        layout.addWidget(QLabel(),1)
        layout.addWidget(self.hint_button,0)
        layout.addWidget(QLabel(),1)
        self.setLayout(layout)

'''
DataPreviewWidget:
    The tab which contains the data-preview table.
'''
class DataPreviewWidget(QWidget):
    def __init__(self, filename, idx, DEMO_MODE=False):
        super(DataPreviewWidget, self).__init__()
        self.DEMO_MODE = DEMO_MODE
        self.initUI(filename, idx)
    def initUI(self, filename, idx):
        # Load the file into DataFrame first
        df = merge.FilePreprocess(filename, True, DEMO_MODE=self.DEMO_MODE)
        # ============================DATA PREVIEW TABLE===========================
        self.datapreview_table = QTableWidget()
        num_row, num_col = df.shape
        num_row = min(100, num_row)
        all_num_row = max(num_row, 30)
        all_num_col = max(num_col, 3)
        self.datapreview_table.setColumnCount(all_num_col)
        self.datapreview_table.setRowCount(all_num_row)
        self.datapreview_table.setHorizontalHeaderLabels(list(df.columns))
        self.datapreview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i in range(num_row):
            for j in range(num_col):
                cur_item = str(df.iat[i,j])
                if cur_item == 'nan':
                    empty_item = QTableWidgetItem()
                    empty_item.setBackground(QBrush(QColor(201,252,255)))
                    self.datapreview_table.setItem(i, j, empty_item)
                    continue
                self.datapreview_table.setItem(i, j, QTableWidgetItem(cur_item))    
        tablefont = QFont()
        tablefont.setPointSize(11)
        headerfont = QFont()
        headerfont.setPointSize(10)
        self.datapreview_table.setFont(tablefont)
        self.datapreview_table.horizontalHeader().setFont(headerfont)
        self.datapreview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.datapreview_table.setAlternatingRowColors(True)
        self.datapreview_table.setStyleSheet('''
            QTableWidget{
                alternate-background-color:#f9f9f9;
            }
        ''')
        # =============================DELETE BUTTON===============================
        self.filedelete_button = FileDeleteButton(idx)
        note = QLabel("Note: Preview mode only displays the first 100 lines for large files.")
        note.setStyleSheet('''
            *{
                font: 18px "Microsoft YaHei UI";
                color: #666666;
                height: 30px;
            }
        ''')
        note_and_delete_layout = QHBoxLayout()
        note_and_delete_layout.addWidget(note, 0)
        note_and_delete_layout.addWidget(QLabel(), 1)
        # note_and_delete_layout.addWidget(self.filedelete_button, 0)

        layout = QVBoxLayout()
        layout.addWidget(self.datapreview_table, 1)
        layout.addLayout(note_and_delete_layout, 0)
        self.setLayout(layout)

'''
Page2_Widget:
    Widget for the second interface.
'''
class Page2_Widget(QWidget):
    def __init__(self, filenames, file_validflag, DEMO_MODE=False):
        super(Page2_Widget, self).__init__()
        self.DEMO_MODE = DEMO_MODE
        self.initUI(filenames, file_validflag)
    def initUI(self, filenames, file_validflag):
        # ============================DATA PREVIEW==================================
        self.datadisplay_tab = QTabWidget()
        self.datadisplay_tab.setStyleSheet('''
            *{
                height: 350px;
            }
        ''')
        self.alltabwidgets = []
        for i in range(len(filenames)):
            if file_validflag[i]:
                file_name_idx = max(filenames[i].rfind('/'), filenames[i].rfind('\\'))
                filename_abbr = filenames[i][file_name_idx+1:]
                self.alltabwidgets.append(DataPreviewWidget(filenames[i], i, self.DEMO_MODE))
                excel_icon = QIcon(QPixmap('fig/excel.png'))
                self.datadisplay_tab.addTab(self.alltabwidgets[-1],excel_icon, filename_abbr)
                tabbar_font = QFont()
                tabbar_font.setFamily("Microsoft YaHei UI")
                tabbar_font.setPointSize(10)
                self.datadisplay_tab.tabBar().setFont(tabbar_font)   
                self.datadisplay_tab.tabBar().setIconSize(QSize(21,21))   
        # ===================================NEXT_BUTTON===========================
        nextbutton_layout = QHBoxLayout()
        self.NotingInfo = QLabel("Start loading the uploaded files")
        self.NotingInfo.setStyleSheet('''
            *{
                font: 23px "Microsoft YaHei UI";
            }
        ''')
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 20px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 20px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
        ''')        
        self.next_button = QPushButton("Merge")
        self.next_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 20px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 20px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
        ''')
        nextbutton_layout.addWidget(self.NotingInfo, 0)
        nextbutton_layout.addWidget(QLabel(), 1)
        # nextbutton_layout.addWidget(self.back_button, 0)
        nextbutton_layout.addWidget(self.next_button, 0)
        self.NotingInfo.hide()      
        
        placeholder = QLabel()
        placeholder.setMinimumHeight(10)
        placeholder.setMaximumHeight(10)

        layout = QVBoxLayout()
        layout.addWidget(self.datadisplay_tab)
        # layout.addWidget(placeholder)
        layout.addLayout(nextbutton_layout)
        self.setLayout(layout)

'''
MergeFileThread:
    It is the new thread which loads the uploaded file. If more than one files are uploaded, 
    they will also be merged.
Args:
    - filenames: list of all uploaded files
    - file_validflag: list of indicators, which indicates whether each file is deleted or not
    - previous_pagenum: previous page number, which will be sent back to the main thread
    - DEMO_MODE: Indicates whether in DEMO mode.
'''
class MergeFileThread(QThread):
    MergeFileFinish = Signal(pd.DataFrame, int)
    def __init__(self, filenames, file_validflag, previous_pagenum, DEMO_MODE=False):
        super(MergeFileThread, self).__init__()    
        self.filenames = filenames
        self.file_validflag = file_validflag
        self.previous_pagenum = previous_pagenum
        self.DEMO_MODE = DEMO_MODE
    def run(self):
        valid_files = []
        for i in range(len(self.filenames)):
            if self.file_validflag[i]:
                valid_files.append(self.filenames[i])
        if len(valid_files) == 1:
            self.df = merge.FilePreprocess(valid_files[0], DEMO_MODE=self.DEMO_MODE)
        else:
            self.df, RealCol = merge.MergeTwoFile(valid_files[0], valid_files[1], None, DEMO_MODE=self.DEMO_MODE)
            for i in range(2, len(valid_files)):
                self.df, RealCol = merge.MergeTwoFile(self.df, valid_files[i], RealCol, DEMO_MODE=self.DEMO_MODE)
        # ErrorColIdx, ErrorRowIdx = error_detection(self.df)
        self.MergeFileFinish.emit(self.df, self.previous_pagenum)

'''
TransformThread:
    Thread to perform the transform operation.
'''
class TransformThread(QThread):
    TenPercent = Signal(int)
    TransformFinish = Signal(dict, int)
    def __init__(self, features, col_idx):
        super(TransformThread, self).__init__()    
        self.features = features
        self.col_idx = col_idx
    def run(self):
        TotalLength = len(self.features)
        TenPercentCounter = 1
        TransformDict = {}
        ClassType = 0
        self.TenPercent.emit(0)
        for i in range(TotalLength):
            if str(self.features[i]) == 'nan':
                pass
            elif not self.features[i] in TransformDict:
                TransformDict[self.features[i]] = ClassType
                ClassType += 1
            if (i+5) == int(TotalLength * TenPercentCounter / 10):
                self.TenPercent.emit(TenPercentCounter)
                TenPercentCounter += 1
        self.TransformFinish.emit(TransformDict, self.col_idx)

'''
SelectionThread:
    Thread to perform the select operation.
'''
class SelectionThread(QThread):
    TenPercent = Signal(int)
    SelectionFinish = Signal(np.ndarray, np.ndarray, int)
    def __init__(self, features, SelectionMethod, SelectionRangeDown, SelectionRangeUp, SelectionCondition, SelectionValue, SelectionIndex, col_idx):
        super(SelectionThread, self).__init__()
        self.features = features
        self.SelectionMethod = SelectionMethod
        self.SelectionRangeDown = SelectionRangeDown
        self.SelectionRangeUp = SelectionRangeUp
        self.SelectionCondition = SelectionCondition
        self.SelectionValue = SelectionValue
        self.SelectionIndex = SelectionIndex
        self.col_idx = col_idx
    def run(self):
        TotalLength = len(self.features)
        TenPercentCounter = 1
        SelectedIndicator = np.array([False]*TotalLength)
        self.TenPercent.emit(0)
        if self.SelectionMethod == 'Range':
            if self.SelectionRangeDown == None:
                self.SelectionRangeDown = -float('inf')
            if self.SelectionRangeUp == None:
                self.SelectionRangeUp = float('inf')
            for i in range(TotalLength):
                try:
                    if str(self.features[i]) == 'nan':
                        SelectedIndicator[i] = True
                    else:
                        if float(self.features[i]) >= self.SelectionRangeDown and float(self.features[i]) <= self.SelectionRangeUp:
                            SelectedIndicator[i] = True
                        else:
                            SelectedIndicator[i] = False
                except:
                    SelectedIndicator[i] = False
                if (i+5) == int(TotalLength*TenPercentCounter/9):
                    self.TenPercent.emit(TenPercentCounter)
                    TenPercentCounter += 1
        elif self.SelectionMethod == 'Condition':
            if self.SelectionCondition == 0: # Equal
                for i in range(TotalLength):
                    try:
                        if float(self.features[i]) == float(self.SelectionValue) or str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = (str(self.SelectionValue) == str(self.features[i]) or str(self.features[i]) == 'nan')
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 1: # Not Equal
                for i in range(TotalLength):
                    try:
                        if float(self.features[i]) != float(self.SelectionValue) or str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = (str(self.SelectionValue) != str(self.features[i]) or str(self.features[i]) == 'nan')
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 2: # Greater Than
                for i in range(TotalLength):
                    try:
                        if str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            if float(self.features[i]) > self.SelectionValue:
                                SelectedIndicator[i] = True
                            else:
                                SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = False
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 3: # Greater/Equal
                for i in range(TotalLength):
                    try:
                        if str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            if float(self.features[i]) >= self.SelectionValue:
                                SelectedIndicator[i] = True
                            else:
                                SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = False
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 4: # Smaller Than
                for i in range(TotalLength):
                    try:
                        if str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            if float(self.features[i]) < self.SelectionValue:
                                SelectedIndicator[i] = True
                            else:
                                SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = False
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 5: # Smaller/Equal
                for i in range(TotalLength):
                    try:
                        if str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            if float(self.features[i]) <= self.SelectionValue:
                                SelectedIndicator[i] = True
                            else:
                                SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = False
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 6: # Start with
                for i in range(TotalLength):
                    if str(self.features[i]) == 'nan':
                        SelectedIndicator[i] = True
                    else: 
                        if len(str(self.SelectionValue)) > len(str(self.features[i])):
                            SelectedIndicator[i] = False
                        else:
                            SelectedIndicator[i] = str(self.features[i])[:len(str(self.SelectionValue))] == str(self.SelectionValue)
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 7: # Not Start with
                for i in range(TotalLength):
                    if str(self.features[i]) == 'nan':
                        SelectedIndicator[i] = True
                    else: 
                        if len(str(self.SelectionValue)) > len(str(self.features[i])):
                            SelectedIndicator[i] = True
                        else:
                            SelectedIndicator[i] = str(self.features[i])[:len(str(self.SelectionValue))] != str(self.SelectionValue)
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 8: # End with
                for i in range(TotalLength):
                    if str(self.features[i]) == 'nan':
                        SelectedIndicator[i] = True
                    else:
                        if len(str(self.SelectionValue)) > len(str(self.features[i])):
                            SelectedIndicator[i] = False
                        else:
                            SelectedIndicator[i] = str(self.features[i])[len(str(self.features[i]))-len(str(self.SelectionValue)):] == str(self.SelectionValue)
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 9: # Not End with
                for i in range(TotalLength):
                    if str(self.features[i]) == 'nan':
                        SelectedIndicator[i] = True
                    else:
                        if len(str(self.SelectionValue)) > len(str(self.features[i])):
                            SelectedIndicator[i] = True
                        else:
                            SelectedIndicator[i] = str(self.features[i])[len(str(self.features[i]))-len(str(self.SelectionValue)):] != str(self.SelectionValue)
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 10: # Contain
                for i in range(TotalLength):
                    SelectedIndicator[i] = (str(self.SelectionValue) in str(self.features[i]) or str(self.features[i]) == 'nan')
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 11: # Not contain
                for i in range(TotalLength):
                    SelectedIndicator[i] = (not (str(self.SelectionValue) in str(self.features[i]))) or str(self.features[i]) == 'nan'
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
        NewSelectionIndex = np.argwhere(SelectedIndicator)
        NewSelectionIndex = np.intersect1d(NewSelectionIndex, self.SelectionIndex, assume_unique=True)
        self.TenPercent.emit(10)
        self.SelectionFinish.emit(SelectedIndicator, NewSelectionIndex, self.col_idx)

'''
DownloadThread:
    Thread to download the preprocessed file.
'''
class DownloadThread(QThread):
    Progress = Signal(int)
    DownloadFinish = Signal(bool)
    def __init__(self, df, FeatureDeleteFlag, CategoricalFlag, CategoricalTransformDict, SelectionIndex, savefile):
        super(DownloadThread, self).__init__()    
        self.df = df.copy()
        self.FeatureDeleteFlag = FeatureDeleteFlag
        self.CategoricalFlag = CategoricalFlag
        self.CategoricalTransformDict = CategoricalTransformDict
        self.SelectionIndex = SelectionIndex
        self.savefile = savefile
    def run(self):
        # First: Transform
        self.Progress.emit(0)
        for i in range(len(self.FeatureDeleteFlag)):
            if self.CategoricalFlag[i]:
                self.df.iloc[:,i] = self.df.iloc[:,i].map(lambda x:self.TransformToCategorical(x, i))
        self.Progress.emit(1)
        # Second: Delete Columns/Features
        DeleteCols = np.argwhere(np.array(self.FeatureDeleteFlag)).reshape(-1,)
        self.df = self.df.drop(columns=self.df.columns[DeleteCols], axis=1)
        self.Progress.emit(2)
        # Third: Delete Rows/Samples
        self.df = self.df.loc[self.SelectionIndex,:]
        self.Progress.emit(3)
        try:
            self.df.to_csv(self.savefile, index=False)
            self.DownloadFinish.emit(True)      
        except PermissionError:
            self.DownloadFinish.emit(False)
        self.Progress.emit(6)

    def TransformToCategorical(self, x, i):
        if str(x) == 'nan':
            return np.nan
        else:
            return self.CategoricalTransformDict[i][x]

'''
Page5DownloadThread:
    Thread to download the imputed file.
'''
class Page5DownloadThread(QThread):
    DownloadFinish = Signal(bool)
    def __init__(self, df, savefile):
        super(Page5DownloadThread, self).__init__()  
        self.df = df
        self.savefile = savefile
    def run(self):
        try:
            self.df.to_csv(self.savefile, index=False)
            self.DownloadFinish.emit(True)
        except PermissionError:
            self.DownloadFinish.emit(False)

'''
ImputationThread:
    Thread which performs the imputation process.
Args:
    - df: The raw input and merged dataframe
    - FeatureDeleteFlag: list of indicators which indicate whether each feature is deleted or not
    - CategoricalFlag: list of indicators wihch indicate whether each feature is transformed to categorical form
    - CategoricalTransformDict: list of dictionary 
    - SelectionIndex: index of the selected rows
    - AlgorithmFile: name of the uploaded algorithm file
    - ParameterFile: name of the uploaded parameter file
    - ThreadIndex: 1 or 2, indicates which imputation thread it is
'''
class ImputationThread(QThread):
    ImputationFinish = Signal(pd.DataFrame, pd.DataFrame, bool, int, int)
    def __init__(self, df, FeatureDeleteFlag, CategoricalFlag, CategoricalTransformDict, SelectionIndex, AlgorithmFile, ParameterFile, ThreadIndex, SCIS_FLAG, SampleSize, ErrorBound):
        super(ImputationThread, self).__init__()    
        self.df = df.copy()
        self.FeatureDeleteFlag = FeatureDeleteFlag
        self.CategoricalFlag = CategoricalFlag
        self.CategoricalTransformDict = CategoricalTransformDict
        self.SelectionIndex = SelectionIndex
        self.AlgorithmFile = AlgorithmFile
        self.ParameterFile = ParameterFile    
        self.ThreadIndex = ThreadIndex
        self.SCIS_FLAG = SCIS_FLAG
        self.SampleSize = SampleSize
        self.ErrorBound = ErrorBound
    def run(self):
        # ====================OBTAIN THE PREPROCESSED DATA=====================
        # First: Transform
        for i in range(len(self.FeatureDeleteFlag)):
            if self.CategoricalFlag[i]:
                self.df.iloc[:,i] = self.df.iloc[:,i].map(lambda x:self.TransformToCategorical(x, i))
        # Second: Delete Columns/Features
        DeleteCols = np.argwhere(np.array(self.FeatureDeleteFlag)).reshape(-1,)
        self.df = self.df.drop(columns=self.df.columns[DeleteCols], axis=1)
        # Third: Delete Rows/Samples
        self.df = self.df.loc[self.SelectionIndex,:]
        self.dfisnull = self.df.isnull()
        if self.SCIS_FLAG:
            AlgFilePy = 'GAIN'
            ParFilePy = 'GAIN_Parameter'
            sys.path.append('GAIN')
            AlgModule = import_module(AlgFilePy)
            ParModule = import_module(ParFilePy)
        else:
            # ======================LOAD THE ALGORITHM FILE=========================
            AlgFileSplit = self.AlgorithmFile.split('/')
            AlgFileDir = '/'.join(AlgFileSplit[:-1])            # directory of the algorithm file
            AlgFilePy = AlgFileSplit[-1][:-3]                   # name of the algorithm file (name of module)
            try:
                sys.path.append(AlgFileDir)
                AlgModule = import_module(AlgFilePy)
            except ImportError:
                self.ImputationFinish.emit(pd.DataFrame(), pd.DataFrame(), False, 0, self.ThreadIndex)
                return
            # ======================LOAD THE PARAMETER FILE=========================
            ParFileSplit = self.ParameterFile.split('/')
            ParFileDir = '/'.join(ParFileSplit[:-1])            # directory of the parameter file
            ParFilePy = ParFileSplit[-1][:-3]                   # name of the parameter file (name of module)
            try:
                if ParFileDir != AlgFileDir:
                    sys.path.append(ParFileDir)
                ParModule = import_module(ParFilePy)
            except ImportError:
                self.ImputationFinish.emit(pd.DataFrame(), pd.DataFrame(), False, 1, self.ThreadIndex)
                return
        # =========================START IMPUTATION=============================
        # Read the parameters from the parameter file
        try:
            args = ParModule.parameter()
            if self.SCIS_FLAG:
                args.thre_value = self.ErrorBound
                args.initial_value = int(self.SampleSize)
        except AttributeError:
            self.ImputationFinish.emit(pd.DataFrame(), pd.DataFrame(), False, 2, self.ThreadIndex)
            return
        # pass the dataframe together with the parameter to the main() function of the algorithm file
        try:
            imputed_df = AlgModule.main(self.df, args)
        except AttributeError:
            self.ImputationFinish.emit(pd.DataFrame(), pd.DataFrame(), False, 3, self.ThreadIndex)
            return
        except:
            self.ImputationFinish.emit(pd.DataFrame(), pd.DataFrame(), False, 4, self.ThreadIndex)
            return
        self.ImputationFinish.emit(pd.DataFrame(imputed_df), self.dfisnull, True, None, self.ThreadIndex)
    
    def TransformToCategorical(self, x, i):
        if str(x) == 'nan':
            return np.nan
        else:
            return self.CategoricalTransformDict[i][x]

class PredictionThread(QThread):
    PredictionFinished = Signal(list, list, bool, int)
    def __init__(self, data_x_df1, data_x_df2, data_y_filepath, SCIS_Enable, USER_Enable):
        super(PredictionThread, self).__init__()   
        self.data_x_df1 = data_x_df1
        self.data_x_df2 = data_x_df2
        self.data_y_filepath = data_y_filepath
        self.SCIS_Enable = SCIS_Enable
        self.USER_Enable = USER_Enable
    def run(self):
        if self.USER_Enable:
            num_sample, _ = self.data_x_df1.shape
        else:
            num_sample, _ = self.data_x_df2.shape
        data_y_df = pd.read_csv(self.data_y_filepath, nrows=num_sample)
        # Check the validity of the input file
        _, num_label = data_y_df.shape
        if num_label != 1:
            self.PredictionFinished.emit([0], [0], False, 0)
            return
        # Due to some unknow issues of imported module, somtimes it needs to run for several times
        if self.USER_Enable:
            for try_counter in range(5):
                try:
                    avg_MAE1, max_MAE1, min_MAE1, avg_MRE1, max_MRE1, min_MRE1 = Regression.Regression(self.data_x_df1, data_y_df)
                    break
                except:
                    if try_counter == 4:
                        self.PredictionFinished.emit([0], [0], False, 1)
                        return
        else:
            avg_MAE1, max_MAE1, min_MAE1, avg_MRE1, max_MRE1, min_MRE1 = (0, 0, 0, 0, 0, 0)
        self.sleep(2)
        if self.SCIS_Enable:
            for try_counter in range(5):
                try:
                    avg_MAE2, max_MAE2, min_MAE2, avg_MRE2, max_MRE2, min_MRE2 = Regression.Regression(self.data_x_df2, data_y_df)
                    break
                except:
                    if try_counter == 4:
                        self.PredictionFinished.emit([0], [0], False, 1)
                        return
        else:
            avg_MAE2, max_MAE2, min_MAE2, avg_MRE2, max_MRE2, min_MRE2 = (0, 0, 0, 0, 0, 0)
        self.PredictionFinished.emit([[avg_MAE1, max_MAE1, min_MAE1], [avg_MAE2, max_MAE2, min_MAE2]], [[avg_MRE1, max_MRE1, min_MRE1], [avg_MRE2, max_MRE2, min_MRE2]], True, 0)

class PredictionThreadUser(QThread):
    PredictionFinished = Signal(list, list, bool, int)
    def __init__(self, data_x_df1, data_x_df2, data_y_filepath, alg_filepath, SCIS_Enable, USER_Enable):
        super(PredictionThreadUser, self).__init__()   
        self.data_x_df1 = data_x_df1
        self.data_x_df2 = data_x_df2
        self.data_y_filepath = data_y_filepath
        self.alg_filepath = alg_filepath
        self.SCIS_Enable = SCIS_Enable
        self.USER_Enable = USER_Enable
    def run(self):
        if self.USER_Enable:
            num_sample, _ = self.data_x_df1.shape
        else:
            num_sample, _ = self.data_x_df2.shape
        data_y_df = pd.read_csv(self.data_y_filepath, nrows=num_sample)
        # ================Check the validity of the input file================
        _, num_label = data_y_df.shape
        if num_label != 1:
            self.PredictionFinished.emit([0], [0], False, 0)
            return
        # =================Load the prediction algorithm file=================
        AlgFileSplit = self.alg_filepath.split('/')
        AlgFileDir = '/'.join(AlgFileSplit[:-1])
        AlgFilePy = AlgFileSplit[-1][:-3]
        try:
            sys.path.append(AlgFileDir)
            AlgModule = import_module(AlgFilePy)
        except:
            self.PredictionFinished.emit([0], [0], False, 2)
            return
        # ====================Start the prediction process====================
        # Due to some unknow issues of imported module, somtimes it needs to run for several times
        if self.USER_Enable:
            for try_counter in range(5):
                try:
                    avg_MAE1, max_MAE1, min_MAE1, avg_MRE1, max_MRE1, min_MRE1 = AlgModule.main(self.data_x_df1, data_y_df)
                    break
                except:
                    if try_counter == 4:
                        self.PredictionFinished.emit([0], [0], False, 3)
                        return
        else:
            avg_MAE1, max_MAE1, min_MAE1, avg_MRE1, max_MRE1, min_MRE1 = (0, 0, 0, 0, 0, 0)
        self.sleep(3)
        if self.SCIS_Enable:
            for try_counter in range(5):
                try:
                    avg_MAE2, max_MAE2, min_MAE2, avg_MRE2, max_MRE2, min_MRE2 = AlgModule.main(self.data_x_df2, data_y_df)
                    break
                except:
                    if try_counter == 4:
                        self.PredictionFinished.emit([0], [0], False, 3)
                        return
        else:
            avg_MAE2, max_MAE2, min_MAE2, avg_MRE2, max_MRE2, min_MRE2 = (0, 0, 0, 0, 0, 0)
        self.PredictionFinished.emit([[avg_MAE1, max_MAE1, min_MAE1], [avg_MAE2, max_MAE2, min_MAE2]], [[avg_MRE1, max_MRE1, min_MRE1], [avg_MRE2, max_MRE2, min_MRE2]], True, 0)

'''
Page3_Widget:
    Widget for the third interface.
'''
class Page3_Widget(QWidget):
    def __init__(self, df):
        super(Page3_Widget, self).__init__()
        self.initUI(df)
    def initUI(self, df):
        num_row, num_col = df.shape
        self.num_col = num_col
        if num_row == 0:
            self.MissingRate = 0
        else:
            self.MissingRate = 100 * df.isnull().sum().sum() / (num_row * num_col)
        # ==================================TOOL MENU===============================
        self.MissingRateLE = QLabel("Missing Rate: %.2f%%" %self.MissingRate)
        self.MissingRateLE.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
            }
        ''')
        self.FeatureNumLE = QLabel("Feature Number: %i" %num_col)
        self.FeatureNumLE.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
                margin-left: 25px;
            }
        ''')        
        self.SampleNumLE = QLabel("Sample Number: %i" %num_row)
        self.SampleNumLE.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
                margin-left: 25px;
            }
        ''')  

        GoToLineLE = QLabel("Go To Page")
        GoToLineLE.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
            }
        ''')
        self.InputLineNumber = QLineEdit()
        intval = QIntValidator()
        self.InputLineNumber.setValidator(intval)
        self.InputLineNumber.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
                margin: 7px;
                border: 1px solid #A0A0A0;
                border-radius: 6px;
                width: 53px;
            }
        ''')
        self.TotalNumLine = QLabel("/%i" %(int(num_row/100)+1))
        self.TotalNumLine.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
            }
        ''')
        self.UndoButton = QPushButton(" Undo")
        self.UndoButton.setStyleSheet('''
            QPushButton{
                font: 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #bfbfbf;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:20px 20px;
                margin-left: 24px;
                width: 80px;
            }
            QPushButton:hover{
                font: bold 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #bfbfbf;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:20px 20px;
                margin-left: 24px;
            }
        ''')
        self.UndoButton.setEnabled(False)
        self.DownloadButton = QPushButton(" Download")
        self.DownloadButton.setStyleSheet('''
            QPushButton{
                font: 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/download.png);
                qproperty-iconSize:20px 20px;
                width: 145px;
            }
            QPushButton:hover{
                font: bold 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/download.png);
                qproperty-iconSize:20px 20px;
            }
        ''')     

        layout = QHBoxLayout()
        layout.addWidget(self.MissingRateLE, 0)
        layout.addWidget(self.SampleNumLE, 0)
        layout.addWidget(self.FeatureNumLE, 0)
        layout.addWidget(QLabel(), 1)
        layout.addWidget(GoToLineLE, 0)
        layout.addWidget(self.InputLineNumber, 0)
        layout.addWidget(self.TotalNumLine, 0)
        layout.addWidget(self.UndoButton, 0)
        layout.addWidget(self.DownloadButton,0)
        layout.setSpacing(0)

        # ===============================MAIN WINDOW================================
        self.MainWindow = QTableWidget()
        display_num_row = min(100, num_row)
        self.MainWindow.setRowCount(max(30, display_num_row))
        self.MainWindow.setColumnCount(max(3, num_col))
        self.MainWindow.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.MainWindow.setHorizontalHeaderLabels(list(df.columns))
        for i in range(display_num_row):
            for j in range(num_col):
                CurItem = str(df.iat[i, j])
                if CurItem == 'nan':
                    EmptyTableItem = QTableWidgetItem()
                    EmptyTableItem.setBackground(QBrush(QColor(201,252,255)))
                    self.MainWindow.setItem(i, j, EmptyTableItem)
                    continue
                self.MainWindow.setItem(i, j, QTableWidgetItem(CurItem))
        tablefont = QFont()
        tablefont.setPointSize(11)
        headerfont = QFont()
        headerfont.setPointSize(10)
        self.MainWindow.setFont(tablefont)
        self.MainWindow.horizontalHeader().setFont(headerfont)
        self.MainWindow.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.MainWindow.setContextMenuPolicy(Qt.CustomContextMenu)
        self.MainWindow.setAlternatingRowColors(True)
        self.MainWindow.setStyleSheet('''
            QTableWidget{
                alternate-background-color:#f9f9f9;
            }
        ''')
        # =============================BACK-NEXT BUTTON===============================
        nextbutton_layout = QHBoxLayout()
        self.PreprocessProgressLE = QLabel("Preprocess: ")
        self.PreprocessProgressLE.setStyleSheet('*{font-size:20px;}')
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(10)
        self.progressbar.setMinimumWidth(600)
        self.progressbar.setMaximumWidth(600)
        self.progressbar.setStyleSheet('''
            QProgressBar{
                background-color: #cbebe4;
                color: #3F4756;
                border-radius: 10px;
                border-color: transparent;
                text-align: center;
            }
            QProgressBar::chunk{
                background-color: #00B1AE;
                border-radius: 10px;
            }
        ''')
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 20px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 20px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
        ''')        
        self.next_button = QPushButton("Start to Impute")
        self.next_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 20px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 40px;
                width: 200px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 20px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 40px;
                width: 200px;
            }
        ''')
        nextbutton_layout.addWidget(self.PreprocessProgressLE, 0)
        nextbutton_layout.addWidget(self.progressbar, 0)
        nextbutton_layout.addWidget(QLabel(), 1)
        nextbutton_layout.addWidget(self.back_button, 0)
        nextbutton_layout.addWidget(self.next_button, 0)  
        self.PreprocessProgressLE.hide()
        self.progressbar.hide()
        # ==================================OVERALL LAYOUT================================
        OverallLayout = QVBoxLayout()
        OverallLayout.addLayout(layout)
        OverallLayout.addWidget(self.MainWindow)
        OverallLayout.addLayout(nextbutton_layout)   
        self.setLayout(OverallLayout) 

'''
SelectionDialog:
    The dialog for selection operation.
'''
class SelectionDialog(QDialog):
    def __init__(self):
        super(SelectionDialog, self).__init__()
        self.initUI()

    def initUI(self):  
        self.setWindowTitle("Selection Methods")
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        layout = QVBoxLayout()
        placeholder = QLabel()
        # ===============LAYOUT 1===================
        layout1 = QHBoxLayout()
        SelectionMethodLE = QLabel("Selection Methods")
        SelectionMethodLE.setProperty('name','in')
        layout1.addWidget(QLabel(),1)
        layout1.addWidget(SelectionMethodLE,0)
        layout1.addWidget(QLabel(),1)

        # ===============LAYOUT 2===================
        layout2 = QHBoxLayout()
        self.SelectButton1 = QRadioButton("Range Selection")
        self.SelectButton1.setProperty('name','in')
        layout2.addWidget(self.SelectButton1, 0)
        layout2.addWidget(placeholder, 0)

        # ===============LAYOUT 3===================
        layout3 = QHBoxLayout()
        self.RangeLE1 = QLineEdit()
        ToLE = QLabel(" to ")
        ToLE.setProperty('name','in')
        self.RangeLE2 = QLineEdit()
        self.RangeLE1.setEnabled(False)
        self.RangeLE2.setEnabled(False)
        self.RangeLE1.setProperty('name','in')
        self.RangeLE2.setProperty('name','in')
        doubleval = QDoubleValidator()
        self.RangeLE1.setValidator(doubleval)
        self.RangeLE2.setValidator(doubleval)
        layout3.addWidget(self.RangeLE1, 0)
        layout3.addWidget(ToLE, 0)
        layout3.addWidget(self.RangeLE2, 0)
        layout3.addWidget(placeholder, 1)

        # ===============LAYOUT 4===================
        layout4 = QHBoxLayout()
        self.SelectButton2 = QRadioButton("Conditional Selection")
        self.SelectButton2.setProperty('name','in')
        layout4.addWidget(self.SelectButton2, 0)
        layout4.addWidget(placeholder, 1)

        # ===============LAYOUT 5===================
        layout5 = QHBoxLayout()
        self.SelectionMethodComboBox = QComboBox()
        self.SelectionMethodComboBox.addItems(['Equal', 'Not Equal', 'Greater than', 'Greater/Equal', 'Smaller than', 'Smaller/Equal', 'Start with', 'Not Start With', 'End with', 'Not End with', 'Contain', 'Not Contain'])
        self.ValueLE = QLineEdit()
        self.SelectionMethodComboBox.setEnabled(False)
        self.ValueLE.setEnabled(False)
        self.SelectionMethodComboBox.setProperty('name','in')
        self.ValueLE.setProperty('name','in')
        layout5.addWidget(self.SelectionMethodComboBox)
        layout5.addWidget(self.ValueLE)
        layout5.addWidget(placeholder, 1)

        # ===============LAYOUT 6===================
        layout6 = QHBoxLayout()
        self.OKButton = QPushButton("OK")
        self.CancelButton = QPushButton("Cancel")
        self.OKButton.setProperty('name','confirm')
        self.CancelButton.setProperty('name','confirm')
        layout6.addWidget(QLabel(), 1)
        layout6.addWidget(self.OKButton, 0)
        layout6.addWidget(self.CancelButton, 0)
        layout6.addWidget(QLabel(), 1)

        # ================LAYOUT ALL==================
        layout.addLayout(layout1)
        layout.addLayout(layout2)
        layout.addLayout(layout3)
        layout.addLayout(layout4)
        layout.addLayout(layout5)
        layout.addLayout(layout6)
        self.setStyleSheet('''
            *[name='in']{
                font: 17px "Microsoft YaHei UI";
            }
            QPushButton[name='confirm']{
                background-color: #326aa9;
                font: 17px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 33px;
                width: 83px;
                margin: 10px;
            }
            QPushButton:hover[name='confirm']{
                background-color: #457fbf;
                font: bold 17px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 33px;
                width: 83px;
            }
            QLineEdit{
                border: 1px solid #A0A0A0;
                border-radius: 6px;
                width: 160px;
            }
            QComboBox{ 
                border: 1px solid #A0A0A0;
                border-radius: 6px;
                height: 27px;
                width: 143px;
            }
        ''')
        SelectionMethodLE.setStyleSheet('''
        *{
            font-size: 19px;
        }
        ''')
        self.ValueLE.setStyleSheet('''
            QLineEdit{
                border: 1px solid #A0A0A0;
                border-radius: 6px;
                width: 160px;
                margin-left: 30px;
            }
        ''')
        self.setLayout(layout)

# class InfoCanvas1(FigureCanvas):
#     def __init__(self, width=20, height=6, dpi=100):
#         fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes1 = fig.add_subplot(121)
#         self.axes2 = fig.add_subplot(122)
#         super(InfoCanvas1, self).__init__(fig)
#         x = np.arange(8)
#         width = 0.37 
#         labels = ["MissF", "MICE", "RRSI", "MIDAE", "VAEI", "HIVAE", "GINN", "GAIN"]
#         # ==================FIRST PLOT===================
#         RMSE_original = [0.171, 0.157, 0.125, 0.211, 0.256, 0.132, 0.202, 0.129]
#         RMSE_scis = [0.174, 0.155, 0.124, 0.208, 0.232, 0.130, 0.192, 0.122]
#         rects11 = self.axes1.bar(x - width/2, RMSE_original, width, label='Original Algorithm')
#         rects12 = self.axes1.bar(x + width/2, RMSE_scis, width, label='SCIS Algorithm')
#         for rect in rects11:
#             self.axes1.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='left', va='bottom', fontsize='x-small')
#         for rect in rects12:
#             self.axes1.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='right', va='top', fontsize='x-small')
#         ylim_max = 0.3
#         self.axes1.set_ylabel('RMSE(Bias)')
#         self.axes1.set_xlabel("Algorithm")
#         self.axes1.set_title('Root Mean Square Error over Dataset $Trail$')
#         self.axes1.set_xticks(x)
#         self.axes1.set_xticklabels(labels)
#         self.axes1.set_ylim(0, ylim_max)
#         self.axes1.legend()
#         # =================SECOND PLOT====================
#         Time_original = [320, 470, 403, 2321, 240, 380, 664, 340]
#         Time_scis = [282, 350, 306, 1502, 181, 283, 510, 260]
#         rects21 = self.axes2.bar(x - width/2, Time_original, width, label='Original Algorithm')
#         rects22 = self.axes2.bar(x + width/2, Time_scis, width, label='SCIS Algorithm')
#         for rect in rects21:
#             self.axes2.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom', fontsize='x-small')
#         for rect in rects22:
#             self.axes2.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom', fontsize='x-small')
#         ylim_max = 2600
#         self.axes2.set_ylabel('Time(s)')
#         self.axes2.set_xlabel("Algorithm")
#         self.axes2.set_title('Running Time over Dataset $Trail$')
#         self.axes2.set_xticks(x)
#         self.axes2.set_xticklabels(labels)
#         self.axes2.set_ylim(0, ylim_max)
#         self.axes2.legend()

class InfoCanvas1(FigureCanvas):
    def __init__(self, width=20, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = fig.add_subplot(221)
        self.axes2 = fig.add_subplot(222)
        self.axes3 = fig.add_subplot(223)
        self.axes4 = fig.add_subplot(224)
        super(InfoCanvas1, self).__init__(fig)
        x = np.arange(10, 100, 10)
        width = 3.5
        labels = [10*int(i) for i in range(1, 10)]
        # ==================FIRST PLOT===================
        RMSE_original1 = [0.127, 0.129, 0.151, 0.175, 0.195, 0.235, 0.276, 0.312, 0.369]
        RMSE_scis1     = [0.120, 0.122, 0.153, 0.175, 0.196, 0.233, 0.276, 0.314, 0.365]
        self.axes1.plot(labels, RMSE_original1, label="GAIN")
        self.axes1.plot(labels, RMSE_scis1, label="SCIS")
        self.axes1.set_ylabel('RMSE(Bias)')
        self.axes1.set_title('Performace of SCIS with GAIN over $Trail$ v.s. $R_m$')
        self.axes1.set_xticks(x)
        self.axes1.set_xticklabels(labels)
        self.axes1.set_ylim(0.1, 0.38)
        self.axes1.set_xlim(5, 95)
        self.axes1.legend()
        # =================THIRD PLOT====================
        Time_original1 = [338, 340, 335, 341, 345, 349, 350, 346, 344]
        Time_scis1     = [265, 260, 253, 245, 233, 217, 202, 185, 152]
        self.axes3.bar(x - width/2, Time_original1, width, label='GAIN')
        self.axes3.bar(x + width/2, Time_scis1, width, label='SCIS')
        self.axes3.set_ylabel('Time(s)')
        self.axes3.set_xlabel("Missing Rate $R_m$ (%)")
        self.axes3.set_xticks(x)
        self.axes3.set_xticklabels(labels)
        self.axes3.set_ylim(0, 490)
        self.axes3.set_xlim(5, 95)
        self.axes3.legend(loc='upper left')
        # ==================SECOND PLOT===================
        RMSE_original2 = [0.149, 0.165, 0.177, 0.204, 0.236, 0.318, 0.382, 0.463, 0.540]
        RMSE_scis2     = [0.148, 0.163, 0.179, 0.204, 0.236, 0.305, 0.381, 0.463, 0.538]
        self.axes2.plot(labels, RMSE_original2, label="GAIN")
        self.axes2.plot(labels, RMSE_scis2, label="SCIS")
        self.axes2.set_ylabel('RMSE(Bias)')
        self.axes2.set_title('Performace of SCIS with GAIN over $Weather$ v.s. $R_m$')
        self.axes2.set_xticks(x)
        self.axes2.set_xticklabels(labels)
        self.axes2.set_ylim(0.1, 0.55)
        self.axes2.set_xlim(5, 95)
        self.axes2.legend()
        # =================FORTH PLOT====================
        Time_original2 = [9498, 9503, 9484, 9506, 9517, 9541, 9512, 9499, 9500]
        Time_scis2     = [1880, 1898, 1712, 1690, 1523, 1487, 1328, 1169, 1004]
        self.axes4.bar(x - width/2, Time_original2, width, label='GAIN')
        self.axes4.bar(x + width/2, Time_scis2, width, label='SCIS')
        self.axes4.set_ylabel('Time(s)')
        self.axes4.set_xlabel("Missing Rate $R_m$ (%)")
        self.axes4.set_xticks(x)
        self.axes4.set_xticklabels(labels)
        self.axes4.set_ylim(0, 11000)
        self.axes4.set_xlim(5, 95)
        self.axes4.legend(loc='upper left')
    
# class InfoCanvas4(FigureCanvas):
#     def __init__(self, width=20, height=6, dpi=100):
#         fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes1 = fig.add_subplot(221)
#         self.axes2 = fig.add_subplot(222)
#         self.axes3 = fig.add_subplot(223)
#         self.axes4 = fig.add_subplot(224)
#         super(InfoCanvas4, self).__init__(fig)
#         x = np.arange(10, 100, 10)
#         width = 3.5
#         labels = [10*int(i) for i in range(1, 10)]
#         # ==================FIRST PLOT===================
#         RMSE_original1 = [0.138, 0.145, 0.172, 0.206, 0.256, 0.337, 0.387, 0.447, 0.522]
#         RMSE_scis1     = [0.138, 0.146, 0.171, 0.205, 0.255, 0.335, 0.385, 0.445, 0.518]
#         self.axes1.plot(labels, RMSE_original1, label="HIVAE")
#         self.axes1.plot(labels, RMSE_scis1, label="SCIS")
#         self.axes1.set_ylabel('RMSE(Bias)')
#         self.axes1.set_title('Performace of SCIS with HIVAE over $Weather$ v.s. $R_m$')
#         self.axes1.set_xticks(x)
#         self.axes1.set_xticklabels(labels)
#         self.axes1.set_ylim(0.1, 0.55)
#         self.axes1.set_xlim(5, 95)
#         self.axes1.legend()
#         # =================THIRD PLOT====================
#         Time_original1 = [14652, 14612, 14589, 14602, 14655, 14660, 14694, 14671, 14644]
#         Time_scis1     = [2412,  2204,  2004,  1792,  1585,  1413,  1252,  1360,  672]
#         self.axes3.bar(x - width/2, Time_original1, width, label='HIVAE')
#         self.axes3.bar(x + width/2, Time_scis1, width, label='SCIS')
#         self.axes3.set_ylabel('Time(s)')
#         self.axes3.set_xlabel("Missing Rate $R_m$ (%)")
#         self.axes3.set_xticks(x)
#         self.axes3.set_xticklabels(labels)
#         self.axes3.set_ylim(0, 16000)
#         self.axes3.set_xlim(5, 95)
#         self.axes3.legend(loc='upper left')
#         # ==================SECOND PLOT===================
#         RMSE_original2 = [0.149, 0.165, 0.177, 0.204, 0.236, 0.318, 0.382, 0.463, 0.540]
#         RMSE_scis2     = [0.148, 0.163, 0.179, 0.204, 0.236, 0.305, 0.381, 0.463, 0.538]
#         self.axes2.plot(labels, RMSE_original2, label="GAIN")
#         self.axes2.plot(labels, RMSE_scis2, label="SCIS")
#         self.axes2.set_ylabel('RMSE(Bias)')
#         self.axes2.set_title('Performace of SCIS with GAIN over $Weather$ v.s. $R_m$')
#         self.axes2.set_xticks(x)
#         self.axes2.set_xticklabels(labels)
#         self.axes2.set_ylim(0.1, 0.55)
#         self.axes2.set_xlim(5, 95)
#         self.axes2.legend()
#         # =================FORTH PLOT====================
#         Time_original2 = [9498, 9503, 9484, 9506, 9517, 9541, 9512, 9499, 9500]
#         Time_scis2     = [1880, 1898, 1712, 1690, 1523, 1487, 1328, 1169, 1004]
#         self.axes4.bar(x - width/2, Time_original2, width, label='GAIN')
#         self.axes4.bar(x + width/2, Time_scis2, width, label='SCIS')
#         self.axes4.set_ylabel('Time(s)')
#         self.axes4.set_xlabel("Missing Rate $R_m$ (%)")
#         self.axes4.set_xticks(x)
#         self.axes4.set_xticklabels(labels)
#         self.axes4.set_ylim(0, 11000)
#         self.axes4.set_xlim(5, 95)
#         self.axes4.legend(loc='upper left')

class InfoCanvas2(FigureCanvas):
    def __init__(self, width=20, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = fig.add_subplot(221)
        self.axes2 = fig.add_subplot(222)
        self.axes3 = fig.add_subplot(223)
        self.axes4 = fig.add_subplot(224)
        super(InfoCanvas2, self).__init__(fig)
        x = np.arange(10, 100, 10)
        width = 3.5
        labels = [10*int(i) for i in range(1, 10)]
        # ==================FIRST PLOT===================
        RMSE_original1 = [0.447, 0.450, 0.456, 0.464, 0.484, 0.501, 0.523, 0.536, 0.552]
        RMSE_scis1     = [0.445, 0.448, 0.454, 0.459, 0.478, 0.495, 0.522, 0.536, 0.550]
        self.axes1.plot(labels, RMSE_original1, label="GAIN")
        self.axes1.plot(labels, RMSE_scis1, label="SCIS")
        self.axes1.set_ylabel('RMSE(Bias)')
        self.axes1.set_title('Performace of SCIS with GAIN over $Surveil$ v.s. $R_m$')
        self.axes1.set_xticks(x)
        self.axes1.set_xticklabels(labels)
        self.axes1.set_ylim(0.4, 0.6)
        self.axes1.set_xlim(5, 95)
        self.axes1.legend()
        # =================THIRD PLOT====================
        Time_original1 = [28965, 28998, 29030, 29005, 28912, 29054, 29002, 28990, 29014]
        Time_scis1     = [2278,  2160,  2230,  2189,  2004,  1829,  1679,  1420,  1120]
        self.axes3.bar(x - width/2, Time_original1, width, label='GAIN')
        self.axes3.bar(x + width/2, Time_scis1, width, label='SCIS')
        self.axes3.set_ylabel('Time(s)')
        self.axes3.set_xlabel("Missing Rate $R_m$ (%)")
        self.axes3.set_xticks(x)
        self.axes3.set_xticklabels(labels)
        self.axes3.set_ylim(0, 31000)
        self.axes3.set_xlim(5, 95)
        self.axes3.legend()
        # ==================SECOND PLOT===================
        RMSE_original2 = [0.219, 0.252, 0.298, 0.309, 0.378, 0.412, 0.455, 0.497, 0.546]
        RMSE_scis2     = [0.215, 0.250, 0.302, 0.309, 0.378, 0.395, 0.451, 0.497, 0.544]
        self.axes2.plot(labels, RMSE_original2, label="GAIN")
        self.axes2.plot(labels, RMSE_scis2, label="SCIS")
        self.axes2.set_ylabel('RMSE(Bias)')
        self.axes2.set_title('Performace of SCIS with GAIN over $Search$ v.s. $R_m$')
        self.axes2.set_xticks(x)
        self.axes2.set_xticklabels(labels)
        self.axes2.set_ylim(0.2, 0.6)
        self.axes2.set_xlim(5, 95)
        self.axes2.legend()
        # =================FORTH PLOT====================
        Time_original2 = [52612, 52686, 52664, 52636, 52712, 52741, 52654, 52701, 52612]
        Time_scis2     = [7204,  7210,  6921,  7190,  6798,  6720,  6481,  5612,  4701]
        self.axes4.bar(x - width/2, Time_original2, width, label='GAIN')
        self.axes4.bar(x + width/2, Time_scis2, width, label='SCIS')
        self.axes4.set_ylabel('Time(s)')
        self.axes4.set_xlabel("Missing Rate $R_m$ (%)")
        self.axes4.set_xticks(x)
        self.axes4.set_xticklabels(labels)
        self.axes4.set_ylim(0, 60000)
        self.axes4.set_xlim(5, 95)
        self.axes4.legend()

# class InfoCanvas6(FigureCanvas):
#     def __init__(self, width=20, height=6, dpi=100):
#         fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes1 = fig.add_subplot(221)
#         self.axes2 = fig.add_subplot(222)
#         self.axes3 = fig.add_subplot(223)
#         self.axes4 = fig.add_subplot(224)
#         super(InfoCanvas6, self).__init__(fig)
#         x = np.arange(10, 100, 10)
#         width = 3.5
#         labels = [10*int(i) for i in range(1, 10)]
#         # ==================FIRST PLOT===================
#         RMSE_scis1     = [0.21,  0.237, 0.262, 0.269, 0.295, 0.331, 0.386, 0.469, 0.538]
#         self.axes1.plot(labels, RMSE_scis1, label="SCIS")
#         self.axes1.set_ylabel('RMSE(Bias)')
#         self.axes1.set_title('Performace of SCIS with HIVAE over $Search$ v.s. $R_m$')
#         self.axes1.set_xticks(x)
#         self.axes1.set_xticklabels(labels)
#         self.axes1.set_ylim(0.15, 0.6)
#         self.axes1.set_xlim(5, 95)
#         self.axes1.legend()
#         # =================THIRD PLOT====================
#         Time_scis1     = [7985, 8020, 7793, 7784, 7402, 7381, 6821, 6290, 5745]
#         self.axes3.bar(x, Time_scis1, width, label='SCIS')
#         self.axes3.set_ylabel('Time(s)')
#         self.axes3.set_xlabel("Missing Rate $R_m$ (%)")
#         self.axes3.set_xticks(x)
#         self.axes3.set_xticklabels(labels)
#         self.axes3.set_ylim(0, 12000)
#         self.axes3.set_xlim(5, 95)
#         self.axes3.legend(loc='upper left')
#         # ==================SECOND PLOT===================
#         RMSE_original2 = [0.219, 0.252, 0.298, 0.309, 0.378, 0.412, 0.455, 0.497, 0.546]
#         RMSE_scis2     = [0.215, 0.250, 0.302, 0.309, 0.378, 0.395, 0.451, 0.497, 0.544]
#         self.axes2.plot(labels, RMSE_original2, label="GAIN")
#         self.axes2.plot(labels, RMSE_scis2, label="SCIS")
#         self.axes2.set_ylabel('RMSE(Bias)')
#         self.axes2.set_title('Performace of SCIS with GAIN over $Search$ v.s. $R_m$')
#         self.axes2.set_xticks(x)
#         self.axes2.set_xticklabels(labels)
#         self.axes2.set_ylim(0.2, 0.6)
#         self.axes2.set_xlim(5, 95)
#         self.axes2.legend()
#         # =================FORTH PLOT====================
#         Time_original2 = [52612, 52686, 52664, 52636, 52712, 52741, 52654, 52701, 52612]
#         Time_scis2     = [7204,  7210,  6921,  7190,  6798,  6720,  6481,  5612,  4701]
#         self.axes4.bar(x - width/2, Time_original2, width, label='GAIN')
#         self.axes4.bar(x + width/2, Time_scis2, width, label='SCIS')
#         self.axes4.set_ylabel('Time(s)')
#         self.axes4.set_xlabel("Missing Rate $R_m$ (%)")
#         self.axes4.set_xticks(x)
#         self.axes4.set_xticklabels(labels)
#         self.axes4.set_ylim(0, 60000)
#         self.axes4.set_xlim(5, 95)
#         self.axes4.legend()

class SCISInfoDialog(QDialog):
    def __init__(self):
        super(SCISInfoDialog, self).__init__()
        self.initUI()

    def initUI(self):  
        self.setWindowTitle("Introduction to SCIS Algorithm")
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.GoLeft = QPushButton()
        self.GoLeft.setStyleSheet('''
            QPushButton{
                border: none;
                background-color: none;
                qproperty-icon:url(fig/go_left.png);
                qproperty-iconSize: 45px 45px;
                margin-right: 0px;
            }
            QPushButton:hover{
                border: none;
                background-color: none;
                margin-right: 0px;
            }
        ''')
        self.GoRight = QPushButton()
        self.GoRight.setStyleSheet('''
            QPushButton{
                border: none;
                background-color: none;
                qproperty-icon:url(fig/go_right.png);
                qproperty-iconSize: 55px 45px;
                margin-right: 0px;
            }
            QPushButton:hover{
                border: none;
                background-color: none;
                margin-right: 0px;
            }
        ''')
        # =============Page 0: Basic Introduction================
        self.Page0Layout = QVBoxLayout()
        self.Page0Title = QLabel("SCalable Imputation System (SCIS)")
        self.Page0Title.setStyleSheet('*{font: 28px "Microsoft YaHei UI";}')
        self.Page0Contents = QLabel("           We propose a novel scalable imputation system named SCIS, with the differentiable adversarial imputation\nmodels. It is able to train the imputation models constructed by the generative adversarial network (GAN) via using\nan appropriate size of samples under accuracy-guarantees for large-scale incomplete data. SCIS consists of two\nmodules,differentiable imputation modeling (DIM) and sample size estimation (SSE). DIM leverages a new masking\nSinkhorn divergence function to make an arbitrary GAN-based imputation model differentiable,while for a differen-\n-tiable imputation model, SSE can estimate an appropriate sample size to ensure the user-specified imputation\naccuracy of the final trained model. Extensive experiments upon several real-life large-scale datasets demonstrate\nthat, SCIS can speed up the model training by 7.5x in average. Using around 9% samples, it yields competitive accu-\n-racy with the state-of-the-art GAN-based imputation methods in much shorter computation time.")
        self.Page0Contents.setStyleSheet('*{font: 20px "Microsoft YaHei UI";margin-left: 30px; margin-right: 30px;}')
        self.Page0Layout.addWidget(self.Page0Title, 0, Qt.AlignCenter)
        self.Page0Layout.addWidget(self.Page0Contents, 0, Qt.AlignCenter)
        self.Page1 = InfoCanvas1()
        self.Page2 = InfoCanvas2()
        DisplayLayout =  QHBoxLayout()
        DisplayLayout.addWidget(self.GoLeft, 0, Qt.AlignCenter)
        DisplayLayout.addLayout(self.Page0Layout, 0)
        DisplayLayout.addWidget(self.Page1, 0, Qt.AlignCenter)
        DisplayLayout.addWidget(self.Page2, 0, Qt.AlignCenter)
        DisplayLayout.addWidget(self.GoRight, 0, Qt.AlignCenter)
        self.Page1.hide()
        self.Page2.hide()
        self.PageNumber = QLabel("Page: 1/3")
        self.PageNumber.setStyleSheet('''
            font: 20px "Microsoft YaHei UI";
            margin: 20px;
        ''')
        self.Info1 = QLabel("COVID-19 trials tracker (Trail) dataset shows the clinical trail registries\non studies of COVID-19 all around the world and tracks the availability\nof the studies' results. It contains 6433 trails with 9 features, taking\nabout 9.63% missing rate. The license for Trail is ODbL.")
        self.Info2 = QLabel("Daliy weather (Weather) dataset shows the 9 daily weather attributes\nfrom the nearest station reported by National Oceanic and Atmospheric\nAdministration in specific regions. It includes 4911011 samples from\n19284 regions with 21.56% missing rate. The license for Weather is CC BY.")
        self.Info3 = QLabel("COVID-19 case surveillance public use (Surveil) dataset shows the 7\nclinical and symptom features for 22507139 cases shared by thecenters\nfor disease control and prevention, taking 47.62% missing rate. The\nlicense for Surveil is CC0.")
        self.Info4 = QLabel("Symptom search trends (Search) dataset for COVID-19 shows how Google\nsearch patterns for different symptoms change based on relative frequency\nof searches for each symptom in 2792 regions. It contains 948762 samples\nwith 424 symptoms and 81.35% missing rate. The license for Search is CC0.")
        self.Info1.setStyleSheet('*{font: 17px "Microsoft YaHei UI"; margin-left: 10px; margin-right: 10px;}')
        self.Info2.setStyleSheet('*{font: 17px "Microsoft YaHei UI"; margin-left: 10px; margin-right: 10px;}')
        self.Info3.setStyleSheet('*{font: 17px "Microsoft YaHei UI"; margin-left: 10px; margin-right: 10px;}')
        self.Info4.setStyleSheet('*{font: 17px "Microsoft YaHei UI"; margin-left: 10px; margin-right: 10px;}')
        InformationLayout = QHBoxLayout()
        InformationLayout.addWidget(QLabel(), 1)
        InformationLayout.addWidget(self.Info1, 0)
        InformationLayout.addWidget(self.Info2, 0)
        InformationLayout.addWidget(self.Info3, 0)
        InformationLayout.addWidget(self.Info4, 0)
        InformationLayout.addWidget(QLabel(), 1)
        self.Info1.hide()
        self.Info2.hide()
        self.Info3.hide()
        self.Info4.hide()
    
        PageLayout = QHBoxLayout()
        PageLayout.addWidget(QLabel(), 1)
        PageLayout.addWidget(self.PageNumber, 0)
        OverallLayout = QVBoxLayout()
        OverallLayout.addWidget(QLabel(), 1)
        OverallLayout.addLayout(DisplayLayout, 0)
        OverallLayout.addLayout(InformationLayout, 0)
        OverallLayout.addLayout(PageLayout, 0)
        OverallLayout.addWidget(QLabel(), 1)
        self.setLayout(OverallLayout)

'''
Page4_Widget:
    Widget for the forth interface.
'''     
class Page4_Widget(QWidget):
    def __init__(self):
        super(Page4_Widget, self).__init__()
        self.initUI()
    def initUI(self): 
        placeholder = QLabel()
        placeholder.setMaximumHeight(1)
        placeholder.setMinimumHeight(1)
        # ==========================ALGORITHM UPLOAD===========================
        # (1) Row1: Title
        AlgLabel = QLabel("Upload Algorithm File")
        AlgLabel.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
                color: #666666;
                margin-top: 0px;
                margin-bottom: 0px;
                margin-left:40px;
                margin-right: 45px;
            }
        ''')
        AlgLabel.setMinimumHeight(30)
        AlgLabel.setMaximumHeight(30)
        # (2) Row2: Hint Button
        self.AlgTemplateButton = QPushButton("See Template File")
        self.AlgTemplateButton.setStyleSheet('''
            QPushButton{
                font: 18px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/hint.png);
                qproperty-iconSize: 25px 25px;
                margin: 0px;
                height: 24px;
            }
            QPushButton:hover{
                font: bold 18px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/hint.png);
                margin: 0px;
                height: 24px;
            }
        ''')
        self.AlgTemplateButton.setMaximumWidth(230)
        self.AlgTemplateButton.setMinimumWidth(230)
        # (3) Row3: SelectButton
        self.AlgUploadButton = QPushButton("Select File")
        self.AlgUploadButton.setProperty('name', 'upload_button')
        AlgSelectLayout = QVBoxLayout()
        AlgSelectLayout.addWidget(placeholder, 1)
        AlgSelectLayout.addWidget(AlgLabel, 0, Qt.AlignCenter)
        AlgSelectLayout.addWidget(self.AlgTemplateButton, 0, Qt.AlignCenter)
        AlgSelectLayout.addWidget(self.AlgUploadButton, 0, Qt.AlignCenter)
        AlgSelectLayout.addWidget(placeholder, 1)
        # (1) Row1: File Name
        FileNameLabel = QLabel("File Name")
        FileNameLabel.setStyleSheet('''
            *{
                font: 18px "Microsoft YaHei UI";
                color: #666666;
            }
        ''')
        # (2) Row2: A Horizontal Line
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setLineWidth(2)
        hline.setStyleSheet('''
            color: #666666;
        ''')
        # (3) Row3: Display and Delete Table
        self.AlgFileTable = QTableWidget()
        self.AlgFileTable.setStyleSheet('''
            border: none;
        ''')
        self.AlgFileTable.setColumnCount(2)
        self.AlgFileTable.setRowCount(0)
        self.AlgFileTable.horizontalHeader().setStretchLastSection(True)
        self.AlgFileTable.horizontalHeader().hide()
        self.AlgFileTable.verticalHeader().hide()
        self.AlgFileTable.setShowGrid(False)
        self.AlgFileTable.setStyleSheet('''
            font-size: 20px;
        ''')
        self.AlgFileTable.setSelectionMode(QAbstractItemView.NoSelection)
        self.AlgFileTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        AlgTableLayout = QVBoxLayout()
        # AlgTableLayout.addWidget(QLabel(), 1)
        AlgTableLayout.addWidget(FileNameLabel, 0)
        # AlgTableLayout.addWidget(hline, 0)
        AlgTableLayout.addWidget(self.AlgFileTable, 1)
        # AlgTableLayout.addWidget(QLabel(), 1)
        AlgLayout = QHBoxLayout()
        AlgLayout.addLayout(AlgSelectLayout, 0)
        AlgLayout.addLayout(AlgTableLayout, 1)
        # ==========================PARAMETER UPLOAD===========================
        ParSelectLayout = QVBoxLayout()
        # (1) Row1: Title
        ParLabel = QLabel("Upload Parameter File")
        ParLabel.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
                color: #666666;
                margin-top: 0px;
                margin-bottom: 0px;
                margin-left:40px;
                margin-right: 40px;
            }
        ''')
        ParLabel.setMaximumHeight(30)
        ParLabel.setMaximumHeight(30)
        # (2) Row2: Hint Button
        self.ParTemplateButton =  QPushButton("See Template File")
        self.ParTemplateButton.setStyleSheet('''
            QPushButton{
                font: 18px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/hint.png);
                qproperty-iconSize: 25px 25px;
                margin: 0px;
                height: 24px;
            }
            QPushButton:hover{
                font: bold 18px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/hint.png);
                margin: 0px;
                height: 24px;
            }
        ''')
        self.ParTemplateButton.setMaximumWidth(230)
        self.ParTemplateButton.setMinimumWidth(230)
        # (3) Row3: Select Button
        self.ParUploadButton = QPushButton("Select File")
        self.ParUploadButton.setProperty('name', 'upload_button')
        ParSelectLayout.addWidget(placeholder, 1)
        ParSelectLayout.addWidget(ParLabel, 0, Qt.AlignCenter)
        ParSelectLayout.addWidget(self.ParTemplateButton, 0, Qt.AlignCenter)
        ParSelectLayout.addWidget(self.ParUploadButton, 0, Qt.AlignCenter)
        ParSelectLayout.addWidget(placeholder, 1)
        # (1) Row1: File Name
        FileNameLabel1 = QLabel("File Name")
        FileNameLabel1.setStyleSheet('''
            *{
                font: 18px "Microsoft YaHei UI";
                color: #666666;
            }
        ''')
        # (2) Row2: A Horizontal Line
        hline1 = QFrame()
        hline1.setFrameShape(QFrame.HLine)
        hline1.setLineWidth(3)
        hline1.setStyleSheet('''
            color: #666666;
        ''')
        # (3) Row3: Display and Delete Table
        self.ParFileTable = QTableWidget()
        self.ParFileTable.setStyleSheet('''
            border: none;
        ''')
        self.ParFileTable.setColumnCount(2)
        self.ParFileTable.setRowCount(0)
        self.ParFileTable.horizontalHeader().setStretchLastSection(True)
        self.ParFileTable.horizontalHeader().hide()
        self.ParFileTable.verticalHeader().hide()
        self.ParFileTable.setShowGrid(False)
        self.ParFileTable.setStyleSheet('''
            font-size: 20px;
        ''')
        self.ParFileTable.setSelectionMode(QAbstractItemView.NoSelection)
        self.ParFileTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        ParTableLayout = QVBoxLayout()
        # ParTableLayout.addWidget(QLabel(), 1)
        ParTableLayout.addWidget(FileNameLabel1, 0)
        # ParTableLayout.addWidget(hline1, 0)
        ParTableLayout.addWidget(self.ParFileTable, 1)
        # ParTableLayout.addWidget(QLabel(), 1)   
        ParLayout = QHBoxLayout()
        ParLayout.addLayout(ParSelectLayout, 0)
        ParLayout.addLayout(ParTableLayout, 1)
        # ============================SCIS LABEL============================
        SCISLabel = QLabel("SCIS Algorithm")
        SCISLabel.setStyleSheet('''
            *{
                font: 22px "Microsoft YaHei UI";
                color: #666666;
                margin-left: 35px;
            }
        ''')
        # ============================SCIS QUERY============================
        SCISQueryLayout = QHBoxLayout()
        self.SCISInfoButton = QPushButton("Learn more about SCIS")
        self.SCISInfoButton.setStyleSheet('''
            QPushButton{
                background-color: #326AA9;
                font: 19px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 14px;
                height: 60px;
                width: 275px;
                qproperty-icon:url(fig/tips.png);
                qproperty-iconSize:35px 35px;
                text-align: center;
            }
            QPushButton:hover{
                background-color: #457FBF;
                font: bold 19px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 14px;
            }
        ''')
        SCISQueryInfo = QLabel("Apply SCIS Imputation Algorithm?")
        SCISQueryInfo.setStyleSheet('''
            *{
                font: 20px "Microsoft YaHei UI";
                margin-right: 53px;
                margin-left: 26px;
            }
        ''')
        self.SCISYesButton = QRadioButton("Yes")
        self.SCISYesButton.setStyleSheet('''
            *{
                font: 20px "Microsoft YaHei UI";
                margin-right: 20px;
            }
        ''')
        self.SCISNoButton = QRadioButton("No")
        self.SCISNoButton.setStyleSheet('''
            *{
                font: 20px "Microsoft YaHei UI";
                margin-right: 20px;
            }
        ''')
        self.SCISNoButton.setChecked(True)
        buttongroup1 = QButtonGroup()
        buttongroup1.addButton(self.SCISYesButton)
        buttongroup1.addButton(self.SCISNoButton)
        SCISQueryLayout.addWidget(SCISLabel, 0)
        SCISQueryLayout.addWidget(SCISQueryInfo, 0)
        SCISQueryLayout.addWidget(self.SCISYesButton, 0)
        SCISQueryLayout.addWidget(self.SCISNoButton, 0)
        SCISQueryLayout.addWidget(QLabel(), 1)   
        # ==========================SCIS PARAMETER==========================
        SCISParLayout = QHBoxLayout()
        self.SCISParInfo = QLabel("SCIS Parameters")
        self.SCISParInfo.setStyleSheet('''
            *{
                font: 22px "Microsoft YaHei UI";
                margin-left: 40px;
                margin-right: 63px;
                color: #666666;
            }
        ''')
        self.SCISParInfo.setToolTip("Set parameters for SCIS")
        # SCISPArInfoLayout = QVBoxLayout()
        # SCISPArInfoLayout.addWidget(SCISParInfo, 0)
        # SCISPArInfoLayout.addWidget(QLabel(), 1)

        TwoParameterLayout = QVBoxLayout()
        SampleSizeLayout = QHBoxLayout()
        self.SampleSizeLE = QLabel("Initial sample size:")
        self.SampleSizeLE.setStyleSheet('*{font: 22px "Microsoft YaHei UI";}')

        self.SampleSize = QLineEdit()
        self.SampleSize.setStyleSheet('''
            *{
                font: 21px "Microsoft YaHei UI";
                margin: 7px;
                border: 1px solid #A0A0A0;
                border-radius: 6px;
                width: 300px;
                margin-left: 100px;
            }
        ''')
        self.SampleSize.setPlaceholderText("Range: 4000 to 40000")
        intval = QIntValidator()
        self.SampleSize.setValidator(intval)
        self.SampleSize.setEnabled(False)
        SampleSizeLayout.addWidget(self.SampleSizeLE, 0)
        SampleSizeLayout.addWidget(self.SampleSize, 0)
        SampleSizeLayout.addWidget(QLabel(), 1)

        ErrorBoundLayout = QHBoxLayout()
        self.ErrorBoundLE = QLabel("User-tolerated error bound:")
        self.ErrorBoundLE.setStyleSheet('*{font: 22px "Microsoft YaHei UI";}')

        self.ErrorBound = QLineEdit()
        self.ErrorBound.setStyleSheet('''
            *{
                font: 21px "Microsoft YaHei UI";
                margin: 7px;
                border: 1px solid #A0A0A0;
                border-radius: 6px;
                width: 300px;
                margin-left: 8px;
            }
        ''')
        self.ErrorBound.setPlaceholderText("Range: 0.001 to 0.009")
        doubleval = QDoubleValidator()
        self.ErrorBound.setValidator(doubleval)
        self.ErrorBound.setEnabled(False)
        ErrorBoundLayout.addWidget(self.ErrorBoundLE, 0)
        ErrorBoundLayout.addWidget(self.ErrorBound, 0)
        ErrorBoundLayout.addWidget(QLabel(), 1)

        TwoParameterLayout.addLayout(SampleSizeLayout)
        TwoParameterLayout.addLayout(ErrorBoundLayout)
        self.SCISParFrame = QFrame()
        self.SCISParFrame.setFrameStyle(QFrame.Box)
        self.SCISParFrame.setProperty('name','outter')
        self.SCISParFrame.setLayout(TwoParameterLayout)
        self.SCISParFrame.setStyleSheet('''
            *[name='outter']{
                border: 1px solid #aaaaaa;
                border-radius: 10px
            }
        ''')
        SCISParLayout.addWidget(self.SCISParInfo, 0)
        SCISParLayout.addWidget(self.SCISParFrame, 0)
        SCISParLayout.addWidget(QLabel(), 1)

        # ==========================Alg Select=========================
        AlgSelectLabel = QLabel("Algorithm Selection")
        AlgSelectLabel.setStyleSheet('''
            *{
                font: 22px "Microsoft YaHei UI";
                margin-left: 40px;
                margin-right: 30px;
                color: #666666;
            }
        ''')
        self.AlgSelectUser = QRadioButton("Original Algorithm")
        self.AlgSelectSCIS = QRadioButton("SCIS-Original Algorithm")
        self.AlgSelectUser.setStyleSheet('''
            *{
                font: 22px "Microsoft YaHei UI";
                margin-right: 80px;
            }
        ''')
        self.AlgSelectSCIS.setStyleSheet('''
            *{
                font: 22px "Microsoft YaHei UI";
                margin-right: 61px;
            }
        ''')
        AlgCheckBoxSublayout = QHBoxLayout()
        AlgCheckBoxSublayout.addWidget(self.AlgSelectUser)
        AlgCheckBoxSublayout.addWidget(self.AlgSelectSCIS)
        AlgFrame = QFrame()
        AlgFrame.setProperty('name','outter')
        AlgFrame.setFrameStyle(QFrame.Box)
        AlgFrame.setLayout(AlgCheckBoxSublayout)
        AlgFrame.setStyleSheet('''
            *[name='outter']{
                border: 1px solid #aaaaaa;
                border-radius: 10px
            }
        ''')
        AlgCheckBoxLayout = QHBoxLayout()
        AlgCheckBoxLayout.addWidget(AlgSelectLabel, 0)
        AlgCheckBoxLayout.addWidget(AlgFrame, 0)
        AlgCheckBoxLayout.addWidget(QLabel(), 1)
        AlgCheckBoxLayout.addWidget(self.SCISInfoButton)
        # =========================NEXT-BACK BUTTON=========================
        nextbutton_layout = QHBoxLayout()
        self.NotingInfo = QLabel("Imputation model training time is:")
        self.ImputationTime = QLabel("00:00:00")
        self.NotingInfo.setStyleSheet('''*{font: 22px "Microsoft YaHei UI";}''')
        self.ImputationTime.setStyleSheet('''*{font: 22px "Microsoft YaHei UI";}''')
        self.back_button = QPushButton("Back")
        self.back_button.setProperty('name','next_back_button')       
        self.next_button = QPushButton("Run")
        self.next_button.setProperty('name','next_back_button')
        nextbutton_layout.addWidget(self.NotingInfo, 0)
        nextbutton_layout.addWidget(self.ImputationTime, 0)
        nextbutton_layout.addWidget(QLabel(), 1)
        #nextbutton_layout.addWidget(self.back_button, 0)
        nextbutton_layout.addWidget(self.next_button, 0)  
        self.NotingInfo.hide()
        self.ImputationTime.hide()
        # ============================OVERALL LAYOUT============================
        self.setStyleSheet('''
            QPushButton[name='upload_button']{
                background-color: #326AA9;
                font: 19px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 14px;
                height: 40px;
                width: 173px;
                qproperty-icon:url(fig/folder.png);
                qproperty-iconSize:32px 32px;
                text-align: center;
                margin: 0px;
            }
            QPushButton:hover[name='upload_button']{
                background-color: #457FBF;
                font: bold 19px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 14px;
                height: 40px;
                width: 173px;
            }
            QPushButton[name='next_back_button']{
                background-color: #326aa9;
                font: 20px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 40px;
                width: 100px;
            }
            QPushButton:hover[name='next_back_button']{
                background-color: #457fbf;
                font: bold 20px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 40px;
                width: 100px;
            }
        '''
        )
        placeholder_up = QLabel()
        placeholder_up.setMinimumHeight(8)
        placeholder_up.setMaximumHeight(8)
        placeholder_down = QLabel()
        placeholder_down.setMaximumHeight(5)
        placeholder_down.setMinimumHeight(5)
        self.placeholder = QLabel()
        self.placeholder.setMaximumHeight(103)
        self.placeholder.setMinimumHeight(103)
        OverallLayout = QVBoxLayout()
        OverallLayout.addLayout(AlgLayout)
        OverallLayout.addLayout(ParLayout)
        OverallLayout.addWidget(placeholder_up)
        OverallLayout.addWidget(hline)
        OverallLayout.addWidget(placeholder_down)
        OverallLayout.addLayout(AlgCheckBoxLayout)
        OverallLayout.addLayout(SCISParLayout)
        OverallLayout.addWidget(self.placeholder)
        OverallLayout.addLayout(nextbutton_layout)
        self.ParameterHide()
        self.setLayout(OverallLayout)    
        

    def ParameterHide(self):
        self.SCISParInfo.hide()
        self.SampleSizeLE.hide()
        self.SampleSize.hide()
        self.ErrorBoundLE.hide()
        self.ErrorBound.hide()
        self.SCISParFrame.hide()
        self.placeholder.show()
    
    def ParameterShow(self):
        self.placeholder.hide()
        self.SCISParInfo.show()
        self.SampleSizeLE.show()
        self.SampleSize.show()
        self.ErrorBoundLE.show()
        self.ErrorBound.show()
        self.SCISParFrame.show()

'''
Page5_Widget:
    Widget for the fifth interface.
'''     
class Page5_Widget(QWidget):
    def __init__(self, impute_df1, impute_df2, dfisnull, Thread1Time, Thread2Time, SCIS_Enable, USER_Enable):
        super(Page5_Widget, self).__init__()
        self.initUI(impute_df1, impute_df2, dfisnull, Thread1Time, Thread2Time, SCIS_Enable, USER_Enable)
    def initUI(self, impute_df1, impute_df2, dfisnull, Thread1Time, Thread2Time, SCIS_Enable, USER_Enable):
        if USER_Enable:
            num_row, num_col = impute_df1.shape
        else:
            num_row, num_col = impute_df2.shape
        # ==============================TOOL MENU============================
        Alg1Time = "User Algorithm Time: "+Thread1Time
        Alg2Time = "SCIS Algorithm Time: "+Thread2Time
        Alg1TimeLE = QLabel(Alg1Time)
        Alg2TimeLE = QLabel(Alg2Time)
        Alg1TimeLE.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
                margin-right: 40px;
            }
        ''')
        Alg2TimeLE.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
            }
        ''')
        GoToLineLE= QLabel("Go To Page")
        GoToLineLE.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
            }
        ''')
        self.InputLineNumber = QLineEdit()
        intval = QIntValidator()
        self.InputLineNumber.setValidator(intval)
        self.InputLineNumber.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
                margin: 7px;
                border: 1px solid #A0A0A0;
                border-radius: 8px;
                width: 54px;
            }
        ''')
        self.TotalNumLine = QLabel("/%i" %(int(num_row/100)+1))
        self.TotalNumLine.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
            }
        ''')
        self.RedoButton = QPushButton(" Re-impute")
        self.RedoButton.setStyleSheet('''
            QPushButton{
                font: 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #bfbfbf;
                qproperty-icon:url(fig/redo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
                width: 220px;
            }
            QPushButton:hover{
                font-size: 29px;
                font: bold;
                border: none;
                background-color: none;
                color: #bfbfbf;
                qproperty-icon:url(fig/redo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
            }
        ''')
        self.RedoButton.setEnabled(False)
        self.DownloadButton = QPushButton(" Download")
        self.DownloadButton.setStyleSheet('''
            QPushButton{
                font: 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/download.png);
                qproperty-iconSize:20px 20px;
                width: 147px;
            }
            QPushButton:hover{
                font: bold 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/download.png);
                qproperty-iconSize:20px 20px;
            }
        ''')
        ToolMenuLayout = QHBoxLayout()
        # ToolMenuLayout.addWidget(Alg1TimeLE, 0)
        # if SCIS_Enable:
        #     ToolMenuLayout.addWidget(Alg2TimeLE, 0)
        ToolMenuLayout.addWidget(QLabel(), 1)
        ToolMenuLayout.addWidget(GoToLineLE, 0)
        ToolMenuLayout.addWidget(self.InputLineNumber, 0)
        ToolMenuLayout.addWidget(self.TotalNumLine, 0)
        ToolMenuLayout.addWidget(self.DownloadButton, 0)
        # =============================MAIN WINDOW=============================
        display_num_row = min(100, num_row)
        if USER_Enable:
            self.MainWindow1 = QTableWidget()
            self.MainWindow1.setRowCount(max(30, display_num_row))
            self.MainWindow1.setColumnCount(max(3, num_col))
            self.MainWindow1.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.MainWindow1.setHorizontalHeaderLabels(list(impute_df1.columns))
        if SCIS_Enable:
            self.MainWindow2 = QTableWidget()
            self.MainWindow2.setRowCount(max(30, display_num_row))
            self.MainWindow2.setColumnCount(max(3, num_col))
            self.MainWindow2.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.MainWindow2.setHorizontalHeaderLabels(list(impute_df2.columns))
        for i in range(display_num_row):
            for j in range(num_col):
                if USER_Enable:
                    CurItem1 = QTableWidgetItem(str(impute_df1.iat[i, j]))
                    if dfisnull.iat[i, j]:
                        CurItem1.setBackground(QBrush(QColor(201,252,255))) 
                    self.MainWindow1.setItem(i, j, CurItem1)
                if SCIS_Enable:
                    CurItem2 = QTableWidgetItem(str(impute_df2.iat[i, j]))
                    if dfisnull.iat[i, j]:
                        CurItem2.setBackground(QBrush(QColor(201,252,255)))
                    self.MainWindow2.setItem(i, j, CurItem2)
        tablefont = QFont()
        tablefont.setPointSize(11)
        headerfont = QFont()
        headerfont.setPointSize(10)
        self.MainWindowTab = QTabWidget()
        excel_icon = QIcon(QPixmap('fig/excel.png'))
        tabbar_font = QFont()
        tabbar_font.setFamily("Microsoft YaHei UI")
        tabbar_font.setPointSize(12)
        self.MainWindowTab.tabBar().setFont(tabbar_font)
        self.MainWindowTab.tabBar().setIconSize(QSize(21, 21))
        if USER_Enable:
            self.MainWindow1.setFont(tablefont)
            self.MainWindow1.horizontalHeader().setFont(headerfont)
            self.MainWindow1.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.MainWindow1.setAlternatingRowColors(True)
            self.MainWindow1.setStyleSheet('''
                QTableWidget{
                    alternate-background-color:#f9f9f9;
                }
            ''')
            self.MainWindowTab.addTab(self.MainWindow1, excel_icon, "Original Algorithm Result")
        if SCIS_Enable:
            self.MainWindow2.setFont(tablefont)
            self.MainWindow2.horizontalHeader().setFont(headerfont)
            self.MainWindow2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.MainWindow2.setAlternatingRowColors(True)
            self.MainWindow2.setStyleSheet('''
                QTableWidget{
                    alternate-background-color:#f9f9f9;
                }
            ''')
            self.MainWindowTab.addTab(self.MainWindow2, excel_icon, "SCIS-Original Algorithm Result")
        # ===============================BACK BUTTON============================
        backbutton_layout = QHBoxLayout()
        self.NotingInfo = QLabel("Start downloading the imputed file")
        self.NotingInfo.setStyleSheet('''*{font: 20px "Microsoft YaHei UI";}''')
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 20px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 20px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
        ''')  
        self.next_button = QPushButton("Start to Predict")
        self.next_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 20px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 40px;
                width: 200px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 20px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 40px;
                width: 200px;
            }
        ''')  
        backbutton_layout.addWidget(self.NotingInfo)
        backbutton_layout.addWidget(QLabel(), 1)
        backbutton_layout.addWidget(self.back_button)
        backbutton_layout.addWidget(self.next_button)
        self.NotingInfo.hide()
        # ============================OVERALL LAYOUT===========================
        OverallLayout = QVBoxLayout()
        OverallLayout.addLayout(ToolMenuLayout)
        OverallLayout.addWidget(self.MainWindowTab)
        OverallLayout.addLayout(backbutton_layout)
        self.setLayout(OverallLayout)

class Page6Canvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = fig.add_subplot(121)
        self.axes2 = fig.add_subplot(122)
        super(Page6Canvas, self).__init__(fig)

    def ChangeAxes(self, task='Regression'):
        x = np.arange(3)
        if task == 'Regression':
            self.axes1.cla()
            self.axes2.cla()
            labels1 = ['Avg MAE', 'MAX MAE', 'MIN MAE']
            self.axes1.set_ylabel('MAE')
            self.axes1.set_title('Mean Absolute Error')
            self.axes1.set_xticks(x)
            self.axes1.set_xticklabels(labels1)
            self.axes1.set_ylim(0, 1)
            labels2 = ['Avg RMSE', 'MAX RMSE', 'MIN RMSE']
            self.axes2.set_ylabel('RMSE')
            self.axes2.set_title('Root Mean Squared Error')
            self.axes2.set_xticks(x)
            self.axes2.set_xticklabels(labels2)
            self.axes2.set_ylim(0, 1)       
            self.draw() 
        elif task == 'Classification':
            self.axes1.cla()
            self.axes2.cla()
            labels1 = ['Avg Accuracy', 'MAX Accuracy', 'MIN Accuracy']
            self.axes1.set_ylabel('Accuracy')
            self.axes1.set_title('Accuracy')
            self.axes1.set_xticks(x)
            self.axes1.set_xticklabels(labels1)
            self.axes1.set_ylim(0, 1)
            labels2 = ['Avg AUC', 'MAX AUC', 'MIN AUC']
            self.axes2.set_ylabel('AUC')
            self.axes2.set_title('Area Under Curve')
            self.axes2.set_xticks(x)
            self.axes2.set_xticklabels(labels2)
            self.axes2.set_ylim(0, 1)     
            self.draw()    

    def plot(self, data1, data2, task='Regression', Alg=None):
        self.axes1.cla()
        self.axes2.cla()
        if task == 'Regression':
            x  = np.arange(3)
            width = 0.35
            # ==========================First Plot==========================
            labels1 = ['Avg MAE', 'MAX MAE', 'MIN MAE']
            if len(data1) == 1:
                if Alg == 'USER':
                    rects11 = self.axes1.bar(x, data1[0], width, label='Original Algorithm')
                else:
                    rects11 = self.axes1.bar(x, data1[0], width, label='SCIS-Original Algorithm')
                for rect in rects11:
                    self.axes1.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom') 
                ylim_max = max(data1[0]) * 1.55
            elif len(data1) == 2:
                rects11 = self.axes1.bar(x - width/2, data1[0], width, label='Original Algorithm')
                rects12 = self.axes1.bar(x + width/2, data1[1], width, label='SCIS-Original Algorithm')
                for rect in rects11:
                    self.axes1.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom')
                for rect in rects12:
                    self.axes1.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom')
                ylim_max = max(max(data1[0]), max(data1[1])) * 1.55
            self.axes1.set_ylabel('MAE')
            self.axes1.set_title('Mean Absolute Error')
            self.axes1.set_xticks(x)
            self.axes1.set_xticklabels(labels1)
            self.axes1.set_ylim(0, ylim_max)
            self.axes1.legend()
            # ==========================Second Plot==========================
            labels2 = ['Avg RMSE', 'MAX RMSE', 'MIN RMSE']
            if len(data2) == 1:
                if Alg == 'USER':
                    rects21 = self.axes2.bar(x, data2[0], width, label='Original Algorithm')
                else:
                    rects21 = self.axes2.bar(x, data2[0], width, label='SCIS-Original Algorithm')
                for rect in rects21:
                    self.axes2.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom')
                ylim_max = max(data2[0]) * 1.55
            elif len(data2) == 2:
                rects21 = self.axes2.bar(x - width/2, data2[0], width, label='Original Algorithm')
                rects22 = self.axes2.bar(x + width/2, data2[1], width, label='SCIS-Original Algorithm')
                for rect in rects21:
                    self.axes2.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom')
                for rect in rects22:
                    self.axes2.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom')
                ylim_max = max(max(data2[0]), max(data2[1])) * 1.55
            self.axes2.set_ylabel('RMSE')
            self.axes2.set_title('Root Mean Squared Error')
            self.axes2.set_xticks(x)
            self.axes2.set_xticklabels(labels2)
            self.axes2.set_ylim(0, ylim_max)
            self.axes2.legend()
        elif task == 'Classification':
            x  = np.arange(3)
            width = 0.35
            # ==========================First Plot==========================
            labels1 = ['Avg Accuracy', 'MAX Accuracy', 'MIN Accuracy']
            if len(data1) == 1:
                if Alg == "USER":
                    rects11 = self.axes1.bar(x, data1[0], width, label='Original Algorithm')
                else:
                    rects11 = self.axes1.bar(x, data1[0], width, label='SCIS-Original Algorithm')
                for rect in rects11:
                    self.axes1.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom') 
                ylim_max = max(data1[0]) * 1.55
            elif len(data1) == 2:
                rects11 = self.axes1.bar(x - width/2, data1[0], width, label='Original Algorithm')
                rects12 = self.axes1.bar(x + width/2, data1[1], width, label='SCIS-Original Algorithm')
                for rect in rects11:
                    self.axes1.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom')
                for rect in rects12:
                    self.axes1.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom')
                ylim_max = max(max(data1[0]), max(data1[1])) * 1.55
            self.axes1.set_ylabel('Accuracy')
            self.axes1.set_title('Accuracy')
            self.axes1.set_xticks(x)
            self.axes1.set_xticklabels(labels1)
            self.axes1.set_ylim(0, ylim_max)
            self.axes1.legend()
            # ==========================Second Plot==========================
            labels2 = ['Avg ACU', 'MAX ACU', 'MIN ACU']
            if len(data2) == 1:
                if Alg == 'USER':
                    rects21 = self.axes2.bar(x, data2[0], width, label='Original Algorithm')
                else:
                    rects21 = self.axes2.bar(x, data2[0], width, label='SCIS-Original Algorithm')
                for rect in rects21:
                    self.axes2.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom')
                ylim_max = max(data2[0]) * 1.55
            elif len(data2) == 2:
                rects21 = self.axes2.bar(x - width/2, data2[0], width, label='Original Algorithm')
                rects22 = self.axes2.bar(x + width/2, data2[1], width, label='SCIS-Original Algorithm')
                for rect in rects21:
                    self.axes2.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom')
                for rect in rects22:
                    self.axes2.text(rect.get_x(), rect.get_height(), rect.get_height(), ma='center', va='bottom')
                ylim_max = max(max(data2[0]), max(data2[1])) * 1.55
            self.axes2.set_ylabel('ACU')
            self.axes2.set_title('Area Under Curve')
            self.axes2.set_xticks(x)
            self.axes2.set_xticklabels(labels2)
            self.axes2.set_ylim(0, ylim_max)
            self.axes2.legend()
        self.draw()

class Page6_Widget(QWidget):
    def __init__(self):
        super(Page6_Widget, self).__init__()
        self.initUI()
    def initUI(self):
        # ==================TASK SELECTION & ALG SELECTION=====================
        PredictionLE = QLabel("Prediction Task: ")
        PredictionLE.setStyleSheet('''
            *{
                font: 20px "Microsoft YaHei UI";
                margin-right: 15px;
            }
        ''')
        self.PredictionReg = QRadioButton("Regression")
        self.PredictionCla = QRadioButton("Classification")
        self.PredictionReg.setStyleSheet('''
            *{
                font: 20px "Microsoft YaHei UI";
                margin-right: 7px;
            }
        ''')
        self.PredictionCla.setStyleSheet('''
            *{
                font: 20px "Microsoft YaHei UI";
                margin-right: 7px;
            }
        ''')
        self.PredictionReg.setChecked(True)
        # self.PredictionTask = QComboBox()
        # self.PredictionTask.addItems(['Regression', 'Classification'])
        # self.PredictionTask.setStyleSheet('''
        #     QComboBox{
        #         font: 18px "Microsoft YaHei UI";
        #         border: 1px solid #A0A0A0;
        #         border-radius: 6px;
        #         height: 33px;
        #         width: 147px;
        #     }
        # ''')
        PredictionLayout = QHBoxLayout()
        PredictionLayout.addWidget(PredictionLE)
        PredictionLayout.addWidget(self.PredictionReg)
        PredictionLayout.addWidget(self.PredictionCla)
        PredictionFrame = QFrame()
        PredictionFrame.setProperty('name','outter')
        PredictionFrame.setFrameStyle(QFrame.Box)
        PredictionFrame.setLayout(PredictionLayout)
        PredictionFrame.setStyleSheet('''
            *[name='outter']{
                border: 1px solid #aaaaaa;
                border-radius: 7px;
            }
        ''')
        AlgorithmLE = QLabel("Prediction Model: ")
        AlgorithmLE.setStyleSheet('''
            *{
                font: 20px "Microsoft YaHei UI";
                margin-right: 15px;
            }
        ''')
        self.DefaultAlgorithm = QRadioButton("Default Model")
        self.CustomizedAlgorithm = QRadioButton("Customized Model")
        self.DefaultAlgorithm.setStyleSheet('''
            *{
                font: 20px "Microsoft YaHei UI";
                margin-right: 15px;
            }
        ''')
        self.CustomizedAlgorithm.setStyleSheet('''
            *{
                font: 20px "Microsoft YaHei UI";
                margin-right: 15px;
            }
        ''')
        self.DefaultAlgorithm.setChecked(True)
        AlgorithmLayout = QHBoxLayout()
        AlgorithmLayout.addWidget(AlgorithmLE)
        AlgorithmLayout.addWidget(self.DefaultAlgorithm)
        AlgorithmLayout.addWidget(self.CustomizedAlgorithm)
        AlgorithmFrame = QFrame()
        AlgorithmFrame.setFrameStyle(QFrame.Box)
        AlgorithmFrame.setProperty('name', 'outter')
        AlgorithmFrame.setLayout(AlgorithmLayout)
        AlgorithmFrame.setStyleSheet('''
            *[name='outter']{
                border: 1px solid #aaaaaa;
                border-radius: 7px;
                margin-left: 175x;
                margin-right: 0px;
            }
        ''')
        Predict_Alg_Layout = QHBoxLayout()
        Predict_Alg_Layout.addWidget(PredictionFrame, 0)
        Predict_Alg_Layout.addWidget(AlgorithmFrame, 0)
        # ===========================LABEL UPLOADER=========================
        # (1) Row1: Title
        TitleLabel = QLabel("Upload Files")
        TitleLabel.setStyleSheet('''
            *{
                font: 18px "Microsoft YaHei UI";
                color: #666666;
            }
        ''')
        # (2) Row2: SelectButton
        self.LabelUploadButton = QPushButton("Select Label File")
        self.LabelUploadButton.setStyleSheet('''
            QPushButton{
                background-color: #326AA9;
                font: 18px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 15px;
                height: 35px;
                width: 226px;
                qproperty-icon:url(fig/folder.png);
                qproperty-iconSize:30px 30px;
                text-align: center;
                margin-bottom: 2px;
                margin-top: 0px;
                margin-left: 7px;
                margin-right: 7px;
            }
            QPushButton:hover{
                background-color: #457FBF;
                font: bold 18px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 10px;
            }
        ''')
        self.AlgUploadButton = QPushButton("Select Model File")
        self.AlgUploadButton.setStyleSheet('''
            QPushButton{
                background-color: #326AA9;
                font: 18px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 15px;
                height: 35px;
                width: 226px;
                qproperty-icon:url(fig/folder.png);
                qproperty-iconSize:30px 30px;
                text-align: center;
                margin-top: 2px;
                margin-bottom: 2px;
                margin-left: 7px;
                margin-right: 7px;
            }
            QPushButton:hover{
                background-color: #457FBF;
                font: bold 18px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 10px;
            }
        ''')
        self.AlgUploadButton.setEnabled(False)
        self.HinitButton = QPushButton("See Template File")
        self.HinitButton.setStyleSheet('''
            QPushButton{
                font: 17px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/hint.png);
                qproperty-iconSize: 25px 25px;
                margin-right: 0px;
                height: 25px;
                width: 200px;
            }
            QPushButton:hover{
                font: bold 17px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/hint.png);
                margin-right: 0px;
                height: 25px;
                width: 200px;
            }
        ''')
        self.HinitButton.setEnabled(False)
        self.placeholder = QLabel()
        self.placeholder.setMinimumHeight(27)
        self.placeholder.setMaximumHeight(27)
        LabelSelectLayout = QVBoxLayout()
        LabelSelectLayout.addWidget(self.placeholder, 0, Qt.AlignCenter)
        LabelSelectLayout.addWidget(self.LabelUploadButton, 0, Qt.AlignCenter)
        LabelSelectLayout.addWidget(self.AlgUploadButton, 0, Qt.AlignCenter)
        LabelSelectLayout.addWidget(self.HinitButton, 0, Qt.AlignCenter)
        self.placeholder.hide()
        self.AlgUploadButton.hide()
        self.HinitButton.hide()
        # (1) Row1: File Name
        FileNameLabel = QLabel(" File Type                     File Name")
        FileNameLabel.setStyleSheet('''
            *{
                font: 19px "Microsoft YaHei UI";
                color: #666666;
            }
        ''')
        # (2) Row2: Display and Delete Table
        self.LabelFileTable = QTableWidget()
        self.LabelFileTable.setStyleSheet('''
            border: none;
        ''')
        self.LabelFileTable.setColumnCount(3)
        self.LabelFileTable.setRowCount(0)
        self.LabelFileTable.horizontalHeader().setStretchLastSection(True)
        self.LabelFileTable.horizontalHeader().hide()
        self.LabelFileTable.verticalHeader().hide()
        self.LabelFileTable.setShowGrid(False)
        self.LabelFileTable.setStyleSheet('''
            font-size: 20px;
            height: 40px;
        ''')
        self.LabelFileTable.setSelectionMode(QAbstractItemView.NoSelection)
        self.LabelFileTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # placeholder1 = QLabel()
        # placeholder1.setMinimumHeight(29)
        # placeholder1.setMaximumHeight(29)
        TableLayout= QVBoxLayout()
        TableLayout.addWidget(FileNameLabel, 0)
        TableLayout.addWidget(self.LabelFileTable, 1)
        # TableLayout.addWidget(placeholder1, 0)
        UploaderLayout = QHBoxLayout()
        UploaderLayout.addLayout(LabelSelectLayout, 0)
        UploaderLayout.addLayout(TableLayout, 1)
        UploaderFrame = QFrame()
        UploaderFrame.setProperty('name', 'outter')
        UploaderFrame.setFrameStyle(QFrame.Box)
        UploaderFrame.setLayout(UploaderLayout)
        UploaderFrame.setStyleSheet('''
            *[name='outter']{
                border: 1px solid #aaaaaa;
                border-radius: 7px;
            }
        ''')

        self.RunButton = QPushButton("Run")
        self.RunButton.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 18px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 40px;
                width: 83px;
                margin: 0px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 18px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
        ''') 
        RunButtonLayout = QHBoxLayout()
        RunButtonLayout.addWidget(QLabel(), 1)
        RunButtonLayout.addWidget(self.RunButton, 0)


        # ================================PLOT==============================
        self.Page6Plot = Page6Canvas()
        self.Page6Plot.ChangeAxes(task='Regression')
        # self.Page6Plot.axes1.plot([0,1,2,3,4], [10,1,20,3,40])
        # self.Page6Plot.axes2.plot([0,1,2,3,4], [10,1,20,3,40])
        # ==============================BACK BUTTON=========================
        backbutton_layout = QHBoxLayout()
        self.NotingInfo = QLabel("Prediction task has started, current time is:")
        self.PredictionTime = QLabel("00:00:00")
        self.NotingInfo.setStyleSheet('''*{font: 23px "Microsoft YaHei UI";}''')
        self.PredictionTime.setStyleSheet('''*{font: 23px "Microsoft YaHei UI";}''')
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 20px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 20px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 7px;
                height: 40px;
                width: 83px;
            }
        ''') 
        backbutton_layout.addWidget(self.NotingInfo, 0)
        backbutton_layout.addWidget(self.PredictionTime, 0)
        backbutton_layout.addWidget(QLabel(), 1)
        backbutton_layout.addWidget(self.back_button, 0)
        self.NotingInfo.hide()
        self.PredictionTime.hide()

        OverallLayout = QVBoxLayout()
        OverallLayout.addLayout(Predict_Alg_Layout, 0)
        OverallLayout.addWidget(UploaderFrame, 0)
        OverallLayout.addLayout(RunButtonLayout, 0)
        OverallLayout.addWidget(self.Page6Plot, 0)
        OverallLayout.addLayout(backbutton_layout, 0)
        self.setLayout(OverallLayout)

    def ChangeAxes(self, task='Regression'):
        self.Page6Plot.ChangeAxes(task=task)
        self.Page6Plot.show()

class Imputation_System(QWidget):
    def __init__(self, DEMO_MODE):
        super(Imputation_System, self).__init__()
        self.DEMO_MODE = DEMO_MODE
        self.initUI()

    def initUI(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle("SEMI: A Scalable and Extendible Generative Adversarial Imputatioin Toolbox")
        self.setStyleSheet('''
            #MainWindow{
                background-color: #f3f2f1;
            }
        ''')

        '''
        Step_Widget - Global Step Widget for all 5 pages: It is used to display the previous 
        and current steps. User can also uses it to change to previous step. 
        '''
        self.Step_Widget = StepWidget()

        # self.Step_Widget.step_layout_button3.hide()
        self.Step_Widget.step_layout_button4.hide()
        # self.Step_Widget.step_layout_button5.hide()
        self.Step_Widget.step_layout_button6.hide()
        self.Step_Widget.step_layout_button1.clicked.connect(self.GOTO_Page1)
        # self.Step_Widget.step_layout_button2.clicked.connect(self.GOTO_Page2)
        # self.Step_Widget.step_layout_button3.clicked.connect(self.GOTO_Page3)
        self.Step_Widget.step_layout_button4.clicked.connect(self.GOTO_Page4)
        # self.Step_Widget.step_layout_button5.clicked.connect(self.GOTO_Page5)
        self.Step_Widget.step_layout_button6.clicked.connect(self.GOTO_Page6)
        # style for selected button of the step widget
        self.button_select_style = '''
            QPushButton{
                border: none;
                height: 55px;
                font-size:21px;
                font: large "Microsoft YaHei UI";
                qproperty-icon:url(fig/go_right.png);
                qproperty-iconSize:18px 18px;
                text-align: left;
                background-color: #0e83c2;
            }
            QPushButton:hover{
                height: 55px;
                font-size:21px;
                font: bold large "Microsoft YaHei UI";
                background-color: #0e83c2;
            }
        '''
        # style for the released button of the step widget
        self.button_release_style = '''
            QPushButton{
                border: none;
                height: 55px;
                font-size:21px;
                font: large "Microsoft YaHei UI";
                qproperty-icon:url(fig/go_right.png);
                qproperty-iconSize:18px 18px;
                text-align: left;
                background-color: #0c6ca1;
            }
            QPushButton:hover{
                height: 55px;
                font-size:21px;
                font: bold large "Microsoft YaHei UI";
                background-color: #0e83c2;
            }
        '''
        self.Step_Widget.step_layout_button1.setStyleSheet(self.button_select_style)

        '''
        Page1_Widget - Local Widget for first page: It is used to upload files and delete them if needed
        '''
        self.pagenum = 1                        # current displayed page number
        self.FirstFileIndex = 0
        self.page2_combobox_init = False
        self.page2_init = True                  # whether page2 is initialized
        self.page3_init = False                 # whether page3 is initialized
        self.page4_init = False                 # whether page4 is initialized
        self.page5_init = False                 # whether page5 is initialized
        self.page6_init = False                 # whether page6 is initialized
        self.filenames = []                     # path of all uploaded files
        self.file_validflag = []                # indicate whether the file is still there or deleted
        self.num_allfiles = 0                   # number of all ever-uploaded files
        self.Page1_Widget = Page1_Widget()
        self.Page2_Widget = Page2_Widget(self.filenames, self.file_validflag, self.DEMO_MODE)
        self.AlgorithmFile = None               # path of algorithm file
        self.ParameterFile = None               # path of parameter file
        self.Page1_Widget.select_button.clicked.connect(self.upload_files)
        self.Page2_Widget.next_button.clicked.connect(self.GOTO_Page3)
        # self.Page1_Widget.next_button.clicked.connect(self.GOTO_Page2)
        '''
        Overall Display
        '''
        self.Overall_Layout = QVBoxLayout()
        # placeholder = QLabel()
        # placeholder.setMinimumHeight(5)
        # placeholder.setMaximumHeight(5)
        # self.Overall_Layout.addWidget(placeholder, 0)
        self.Overall_Layout.addWidget(self.Step_Widget, 0)
        self.Overall_Layout.addWidget(self.Page1_Widget, 0)
        self.Overall_Layout.addWidget(self.Page2_Widget, 1)
        self.setLayout(self.Overall_Layout)

    def upload_files(self):
        '''
        upload_files:
            This function is invoked when the "Select File(s)" button of Page 1 is clicked. It is used to 
            record the uploaded files and display the uploaded files on the screen.
        '''
        newfiles, _ = QFileDialog.getOpenFileNames(self, 'Open Files', '.', '(*.csv)')
        self.filenames.extend(newfiles)
        self.file_validflag += [True]*(len(newfiles))
        # Display the uploaded files
        current_num_row = self.Page1_Widget.file_table.rowCount()
        self.Page1_Widget.file_table.setRowCount(current_num_row+len(newfiles))
        for i in range(len(newfiles)):
            self.Page1_Widget.file_table.setItem(current_num_row+i, 0, QTableWidgetItem(newfiles[i]))
            self.Page1_Widget.file_table.setCellWidget(current_num_row+i, 1, FileDeleteButton(self.num_allfiles+i))
            self.Page1_Widget.file_table.cellWidget(current_num_row+i, 1).delete_button.clicked.connect(self.delete_file)
        self.Page1_Widget.file_table.resizeColumnsToContents()
        self.Page1_Widget.file_table.horizontalHeader().setStretchLastSection(True)
        # Display the first one hundred lines in the page2 widget
        if self.page2_init:
            for i in range(len(newfiles)):
                newfile_name_idx = max(newfiles[i].rfind('/'), newfiles[i].rfind('\\'))
                newfilename_abbr = newfiles[i][newfile_name_idx+1:]
                self.Page2_Widget.alltabwidgets.append(DataPreviewWidget(newfiles[i], self.num_allfiles+i, self.DEMO_MODE))
                excel_icon = QIcon(QPixmap('fig/excel.png'))
                self.Page2_Widget.datadisplay_tab.addTab(self.Page2_Widget.alltabwidgets[-1], excel_icon, newfilename_abbr)
                tabbar_font = QFont()
                tabbar_font.setFamily("Microsoft YaHei UI")
                tabbar_font.setPointSize(10)
                self.Page2_Widget.datadisplay_tab.tabBar().setFont(tabbar_font)   
                self.Page2_Widget.datadisplay_tab.tabBar().setIconSize(QSize(21,21))
                self.Page2_Widget.alltabwidgets[-1].filedelete_button.delete_button.clicked.connect(self.delete_file_page2)
            # if self.page2_init:
            #     self.Step_Widget.step_layout_button3.hide()
            if self.page3_init:
                self.page3_init = False
                self.Step_Widget.step_layout_button4.hide()
            if self.page4_init:
                self.page4_init = False
                # self.Step_Widget.step_layout_button5.hide()
            if self.page5_init:
                self.page5_init = False         
                self.Step_Widget.step_layout_button6.hide()   
            if self.page6_init:
                self.page6_init = False
        self.num_allfiles += len(newfiles)

    def delete_file(self):
        '''
        delete_file:
            This function is invoked when the Delete button in Page1 is clicked. It will delete the 
            corresponding row in the display table. If Page2 is already initialized, it will also
            delete the preview table in the second page.
        '''
        insert_idx = self.sender().property('idx')
        row_idx = sum(self.file_validflag[:insert_idx])
        self.file_validflag[insert_idx] = False
        self.Page1_Widget.file_table.removeRow(row_idx)
        if self.page2_init:
            del self.Page2_Widget.alltabwidgets[row_idx]
            self.Page2_Widget.datadisplay_tab.removeTab(row_idx) 
            # if self.page2_init:
            #     self.Step_Widget.step_layout_button3.hide()
            if self.page3_init:
                self.page3_init = False
                self.Step_Widget.step_layout_button4.hide()
            if self.page4_init:
                self.page4_init = False
                # self.Step_Widget.step_layout_button5.hide()
            if self.page5_init:
                self.page5_init = False       
                self.Step_Widget.step_layout_button6.hide()
            if self.page6_init:
                self.page6_init = False 

    def delete_file_page2(self):
        '''
        delete_file_page2:
            This function is invoked when the Delete button in Page2 is clicked. It will delete 
            the preview table in Page2 and the corresponding file-display line in Page1.
        '''
        insert_idx = self.sender().property('idx')
        row_idx = sum(self.file_validflag[:insert_idx])
        self.file_validflag[insert_idx] = False
        self.Page1_Widget.file_table.removeRow(row_idx)
        del self.Page2_Widget.alltabwidgets[row_idx]
        self.Page2_Widget.datadisplay_tab.removeTab(row_idx)  
        if self.page3_init:
            self.page3_init = False
            self.Step_Widget.step_layout_button4.hide()
        if self.page4_init:
            self.page4_init = False
            # self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init = False 
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def GOTO_Page1(self):
        '''
        GOTO_Page1:
            It will be invoked when user click "File Upload" button on the top, or click "Back" 
            button in the second page.
        '''
        if self.pagenum == 1:
            return
        else:
            previous_pagenum = self.pagenum
            self.pagenum = 1
        # if previous_pagenum == 2:
        #     self.Page2_Widget.hide()
        #     self.Step_Widget.step_layout_button2.setStyleSheet(self.button_release_style)
        #     if self.page3_init == False:
        #         self.Step_Widget.step_layout_button3.hide()
        if previous_pagenum == 3:
            self.Page3_Widget.hide()
            # self.Step_Widget.step_layout_button3.setStyleSheet(self.button_release_style)
            if self.page4_init == False:
                self.Step_Widget.step_layout_button4.hide()
        elif previous_pagenum == 4:
            self.Page4_Widget.hide()
            self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
            # if self.page5_init == False:
            #     self.Step_Widget.step_layout_button5.hide()
        elif previous_pagenum == 5:
            self.Page5_Widget.hide()
            self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
        elif previous_pagenum == 6:
            self.Page6_Widget.hide()
            self.Step_Widget.step_layout_button6.setStyleSheet(self.button_release_style)
        self.Step_Widget.step_layout_button1.setStyleSheet(self.button_select_style)
        self.Page1_Widget.show()
        self.Page2_Widget.show()

    def GOTO_Page2(self):
        '''
        GOTO_Page2:
            It will be invoked when user clicks "File Preview" button on the top, or clicks "Back" 
            button in the third page, or clicks "Next" button in the first page.
        '''
        # First check whether jump from elsewhere
        if self.pagenum == 2:
            return
        else:
            previous_pagenum = self.pagenum
            self.pagenum = 2
        # This part only applies when jump from the first page: if there is no file uploaded
        num_upload_file = sum(self.file_validflag)
        if num_upload_file == 0:
            QMessageBox.information(self, "No File Uploaded", "There are no files uploaded. Please upload at least one file.", QMessageBox.Yes, QMessageBox.Yes)
            self.pagenum = 1
            return
        # Check whether page2 has already been initalized
        if self.page2_init == False:
            self.page2_init = True
            self.Page2_Widget = Page2_Widget(self.filenames, self.file_validflag, self.DEMO_MODE)
            for i in range(len(self.Page2_Widget.alltabwidgets)):
                self.Page2_Widget.alltabwidgets[i].filedelete_button.delete_button.clicked.connect(self.delete_file_page2)
            self.Page2_Widget.back_button.clicked.connect(self.GOTO_Page1)
            self.Page2_Widget.next_button.clicked.connect(self.GOTO_Page3)
            self.Overall_Layout.addWidget(self.Page2_Widget)
        if previous_pagenum == 1:
            self.Page1_Widget.hide()
            self.Step_Widget.step_layout_button1.setStyleSheet(self.button_select_style)
            # self.Step_Widget.step_layout_button3.show()
        elif previous_pagenum == 3:
            self.Page3_Widget.hide()
            # self.Step_Widget.step_layout_button3.setStyleSheet(self.button_release_style)
            if self.page4_init == False:
                self.Step_Widget.step_layout_button4.hide()
        elif previous_pagenum == 4:
            self.Page4_Widget.hide()
            self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
            # if self.page5_init == False:
            #     self.Step_Widget.step_layout_button5.hide()
        elif previous_pagenum == 5:
            self.Page5_Widget.hide()
            self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
        elif previous_pagenum == 6:
            self.Page6_Widget.hide()
            self.Step_Widget.step_layout_button6.setStyleSheet(self.button_release_style)
        # self.Step_Widget.step_layout_button2.setStyleSheet(self.button_select_style)
        self.Page2_Widget.show()

    def GOTO_Page3(self):
        '''
        GOTO_Page3:
            It will be invoked when user clicks "Display and Preprocess" button on the top, or clicks "Back" 
            button in the forth page, or clicks "Next" button in the second page.
        '''
        # This part only applies when jumping from the first and second part: when no files are 
        # uploaded or all of them are deleted
        num_upload_file = sum(self.file_validflag)
        if num_upload_file == 0:
            QMessageBox.information(self, "No File Uploaded", "There are no files uploaded. Please upload at least one file.", QMessageBox.Yes, QMessageBox.Yes)
            return        
        # Check whether jump from elsewhere
        if self.pagenum == 3:
            return
        else:
            previous_pagenum = self.pagenum
            self.pagenum = 3
        # Check whether page3 has already be initialized
        if self.page3_init == False:
            self.page3_init = True
            self.page3_MergeFileThread = MergeFileThread(self.filenames, self.file_validflag, previous_pagenum, DEMO_MODE=self.DEMO_MODE)
            self.page3_MergeFileThread.MergeFileFinish.connect(self.Page3_DisplayTable)
            self.DisablePage2Buttons()
            self.Page2_Timer = QTimer()
            self.Page2_Timer.timeout.connect(self.Page2TimerDisplay)
            self.Page2TimeCounter = 0
            self.page3_MergeFileThread.start()
            self.Page2_Timer.start(500)
        else: 
            if previous_pagenum == 1:
                self.Page1_Widget.hide()
                # self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
                self.Page2_Widget.hide()
            elif previous_pagenum == 4:
                self.Page4_Widget.hide()
                self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
                # if self.page5_init == False:
                #     self.Step_Widget.step_layout_button5.hide()
            elif previous_pagenum == 5:
                self.Page5_Widget.hide()
                self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
            elif previous_pagenum == 6:
                self.Page6_Widget.hide()
                self.Step_Widget.step_layout_button6.setStyleSheet(self.button_release_style)
            self.Step_Widget.step_layout_button4.show()
            self.Step_Widget.step_layout_button1.setStyleSheet(self.button_select_style)
            self.Page3_Widget.show()

    def GOTO_Page4(self):
        '''
        GOTO_Page4:
            It will be invoked when user clicks "Algorithm Upload" button on the top, or clicks "Back" 
            button in the forth page, or clicks "Next" button in the second page.
        '''
        # First check whether jump from elsewhere
        if self.pagenum == 4:
            return
        else:
            previous_pagenum = self.pagenum
            self.pagenum = 4
        # Check whether page4 has already been initialized
        if self.page4_init == False:
            self.page4_init = True
            self.AlgorithmFile = None
            self.ParameterFile = None
            self.SampleSize = None
            self.ErrorBound = None
            self.SCIS_Enabled = False
            self.USER_Enabled = False
            self.Page4_Widget = Page4_Widget()
            self.Overall_Layout.addWidget(self.Page4_Widget)
            self.Page4_Widget.AlgUploadButton.clicked.connect(self.Page4UploadAlg)
            self.Page4_Widget.ParUploadButton.clicked.connect(self.Page4UploadPar)
            self.Page4_Widget.AlgTemplateButton.clicked.connect(self.OpenAlgTemp)
            self.Page4_Widget.ParTemplateButton.clicked.connect(self.OpenParTemp)
            self.Page4_Widget.back_button.clicked.connect(self.GOTO_Page3)
            self.Page4_Widget.next_button.clicked.connect(self.GOTO_Page5)
            self.Page4_Widget.SampleSize.editingFinished.connect(self.setSampleSize)
            self.Page4_Widget.ErrorBound.editingFinished.connect(self.setErrorBound)
            # self.Page4_Widget.SCISYesButton.clicked.connect(self.EnableSCIS)
            # self.Page4_Widget.SCISNoButton.clicked.connect(self.DisableSCIS)
            self.Page4_Widget.SCISInfoButton.clicked.connect(self.GenerateSCISInfo)
            self.Page4_Widget.AlgSelectUser.clicked.connect(self.AssertEnableUser)
            self.Page4_Widget.AlgSelectSCIS.clicked.connect(self.AssertEnableSCIS)
        if previous_pagenum == 1:
            self.Page1_Widget.hide()
            self.Page2_Widget.hide()
            self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
        # elif previous_pagenum == 2:
        #     self.Page2_Widget.hide()
        #     self.Step_Widget.step_layout_button2.setStyleSheet(self.button_release_style)
        #     self.Step_Widget.step_layout_button4.show()
        elif previous_pagenum == 3:
            self.Page3_Widget.hide()
            self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
        elif previous_pagenum == 5:
            self.Page5_Widget.hide()
            # self.Step_Widget.step_layout_button5.setStyleSheet(self.button_release_style)
        elif previous_pagenum == 6:
            self.Page6_Widget.hide()
            self.Step_Widget.step_layout_button6.setStyleSheet(self.button_release_style)
        self.Step_Widget.step_layout_button4.setStyleSheet(self.button_select_style)
        self.Page4_Widget.show()
        # self.Step_Widget.step_layout_button5.show()

    def GOTO_Page5(self):
        '''
        GOTO_Page5:
            It will be invoked when user clicks "Imputation Result" button on the top, or clicks 
            "Next" button in the forth page.
        '''
        # ====================================VALIDITY CHECK======================================
        # (1) First Check: Whether user selects at least one algorithm
        if self.USER_Enabled == False and self.SCIS_Enabled == False:
            QMessageBox.information(self, "No Algorithm Selected", "There are no algorithm selected. Please select one.", QMessageBox.Yes, QMessageBox.Yes)
            return 

        # (2) Second Check: If USER_Enabled - Whether user uploads files
        if self.USER_Enabled:
            if self.AlgorithmFile == None:
                QMessageBox.information(self, "No File Uploaded", "There are no algorithm file uploaded. Please upload one.", QMessageBox.Yes, QMessageBox.Yes)
                return 
            if self.ParameterFile == None:
                QMessageBox.information(self, "No File Uploaded", "There are no parameter file uploaded. Please upload one.", QMessageBox.Yes, QMessageBox.Yes)
                return 

        # (3) Third Check: If SCIS_Enabled - Whether user enters parameters
        if self.SCIS_Enabled:
            if self.SampleSize == None:
                QMessageBox.information(self, "No Sample Size", "There are no sample size entered.", QMessageBox.Yes, QMessageBox.Yes)
                return 
            if self.ErrorBound == None:
                QMessageBox.information(self, "No Error Bound", "There are no error bound entered.", QMessageBox.Yes, QMessageBox.Yes)
                return 

        # ====================================START MAIN LOGIC===================================
        # Check whether jump from elsewhere
        if self.pagenum == 5:
            return
        else:
            previous_pagenum = self.pagenum
        # Check whether page5 has already been initialized
        if self.page5_init == False:
            self.impute_df1 = pd.DataFrame()
            self.impute_df2 = pd.DataFrame()
            self.Thread1Time = '0'
            self.Thread2Time = '0'
            # Run User's Algorithm
            if self.USER_Enabled:
                self.page5_ImputationThread1 = ImputationThread(self.df, self.FeatureDeleteFlag, self.CategoricalFlag, self.CategoricalTransformDict, self.SelectionIndex, self.AlgorithmFile, self.ParameterFile, 1, False, self.SampleSize, self.ErrorBound)
                self.page5_ImputationThread1.ImputationFinish.connect(self.Page5_DisplayTable)
                self.Thread1Finished = False
            if self.SCIS_Enabled:
                self.page5_ImputationThread2 = ImputationThread(self.df, self.FeatureDeleteFlag, self.CategoricalFlag, self.CategoricalTransformDict, self.SelectionIndex, self.AlgorithmFile, self.ParameterFile, 2, True, self.SampleSize, self.ErrorBound)
                self.page5_ImputationThread2.ImputationFinish.connect(self.Page5_DisplayTable)
                self.Thread2Finished = False
            self.DisablePage4Buttons()
            self.Page4_Timer = QTimer()
            self.Page4Counter = 0
            self.ImputationTime = "00:00:00"
            self.Page4_Timer.timeout.connect(self.Page4TimerDisplay)
            if self.USER_Enabled:
                self.page5_ImputationThread1.start()
            if self.SCIS_Enabled:
                self.page5_ImputationThread2.start()
            self.Page4_Timer.start(1000)
        else:
            self.pagenum = 5
            if previous_pagenum == 1:
                self.Page1_Widget.hide()
                self.Page2_Widget.hide()
                self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
            # elif previous_pagenum == 2:
            #     self.Page2_Widget.hide()
            #     self.Step_Widget.step_layout_button2.setStyleSheet(self.button_release_style)
            elif previous_pagenum == 3:
                self.Page3_Widget.hide()
                self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
            elif previous_pagenum == 4:
                self.Page4_Widget.hide()
                self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
            elif previous_pagenum == 6:
                self.Page6_Widget.hide()
                self.Step_Widget.step_layout_button6.setStyleSheet(self.button_release_style)
            self.Step_Widget.step_layout_button4.setStyleSheet(self.button_select_style)
            self.Page5_Widget.show()
            self.Step_Widget.step_layout_button6.show()

    def GOTO_Page6(self):
        # First check whether jump from elsewhere
        if self.pagenum == 6:
            return
        else:
            previous_pagenum = self.pagenum
            self.pagenum = 6
        # Check whether page6 has already been initialized 
        if self.page6_init == False:
            self.page6_init = True
            self.Page6_Widget = Page6_Widget()
            self.Overall_Layout.addWidget(self.Page6_Widget)
            self.LabelFile = None
            self.PredictAlgFile = None
            self.Page6FileIdx = -1
            self.Page6LabelFileIdx = float('inf')
            self.Page6AlgFileIdx = float('inf')
            self.PredictionTask = 'Regression'
            self.PredictionAlgorithm = 'Default'
            # connect signals (TO BE FILLED)
            self.Page6_Widget.back_button.clicked.connect(self.GOTO_Page5)
            self.Page6_Widget.LabelUploadButton.clicked.connect(self.Page6UploadLabelFile)
            self.Page6_Widget.AlgUploadButton.clicked.connect(self.Page6UploadAlgorithmFile)
            self.Page6_Widget.PredictionReg.clicked.connect(self.ChangePredictionTaskToReg)
            self.Page6_Widget.PredictionCla.clicked.connect(self.ChangePredictionTaskToCla)
            self.Page6_Widget.DefaultAlgorithm.clicked.connect(self.SetDefaultPredictionAlg)
            self.Page6_Widget.CustomizedAlgorithm.clicked.connect(self.SetCustomizedPredictAlg)
            self.Page6_Widget.RunButton.clicked.connect(self.RunPredictionTask)
            self.Page6_Widget.HinitButton.clicked.connect(self.OpenPredictionAlgTemp)

        if previous_pagenum == 1:
            self.Page1_Widget.hide()
            self.Page2_Widget.hide()
            self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
        elif previous_pagenum == 3:
            self.Page3_Widget.hide()
            self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
        elif previous_pagenum == 4:
            self.Page4_Widget.hide()
            self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
        elif previous_pagenum == 5:
            self.Page5_Widget.hide()
            self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
        self.Page6_Widget.show()
        self.Step_Widget.step_layout_button6.setStyleSheet(self.button_select_style)


    def Page3_DisplayTable(self, df, previous_pagenum):
        '''
        Page3_DisplayTable:
        It only will be invoked when the merging process finishes. This function displays 
        the third interface.
        '''
        self.EnablePage2Buttons()
        self.Page2_Timer.stop()
        self.Page2TimeCounter = 0
        self.Page2_Widget.NotingInfo.hide()
        self.df = df                                            # merged dataframe
        num_sample, num_feature = df.shape
        self.FeatureDeleteFlag = [False] * num_feature          # list of indicators: whether certain features are deleted
        self.CategoricalFlag = [False] * num_feature            # list of indicators: whether certain features are transformed
        self.CategoricalTransformDict = [None] * num_feature    # list of dictionaries for transformation
        self.SelectionFlag = [False] * num_feature              # list of indicators: whether certain features are selected
        self.SelectionMethod = None                             # selection method: Range or Condition
        self.SelectionRangeUp = None                            # upper bound of the range
        self.SelectionRangeDown = None                          # lower bound of the range
        self.SelectionCondition = 0                             # indicates which conditions is chosen
        self.SelectionValue = None                              # value for the condition
        self.SelectionIndicator = [None] * num_feature          
        self.SelectionIndex = np.array([i for i in range(num_sample)])
        self.ActionStack = []                                   # list of performed operations (in order)
        self.Page3_Widget = Page3_Widget(self.df)
        self.Page3_Widget.back_button.clicked.connect(self.GOTO_Page1)
        self.Page3_Widget.next_button.clicked.connect(self.GOTO_Page4)
        self.Page3_Widget.InputLineNumber.editingFinished.connect(lambda: self.ChangePageDisplay())
        self.Page3_Widget.MainWindow.customContextMenuRequested.connect(self.generateMenu)
        self.Page3_Widget.DownloadButton.clicked.connect(self.DownloadPreprocessFile)
        self.Page3_Widget.UndoButton.clicked.connect(self.UndoPreprocess)
        self.Overall_Layout.addWidget(self.Page3_Widget)
        if previous_pagenum == 1:
            self.Page1_Widget.hide()
            self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
            self.Page2_Widget.hide()
        self.Step_Widget.step_layout_button4.show()
        self.Step_Widget.step_layout_button1.setStyleSheet(self.button_select_style)
        self.Page3_Widget.show()

    def Page2TimerDisplay(self):
        '''
        Page2TimerDisplay:
        It is used to display the "Waiting message" at the bottom of Page2.
        '''
        self.Page2TimeCounter += 1
        dots = '.' * (self.Page2TimeCounter%7)
        if self.Page2TimeCounter == 1:
            self.Page2_Widget.NotingInfo.show()
        self.Page2_Widget.NotingInfo.setText('Start loading the uploaded files'+dots)

    def ChangePageDisplay(self):
        '''
        ChangePageDisplay:
        It is invoked whenever there are some changes in the third interface.
        It is used to change the display of the third interface.
        '''
        # =======================Dealing with page-jumping======================
        num_row, num_col = self.df.shape
        display_num_row = len(self.SelectionIndex)
        try:
            PageNumber = int(self.Page3_Widget.InputLineNumber.text())
        except:
            PageNumber = 1
        TotalPage = int(display_num_row/100) + 1
        self.Page3_Widget.TotalNumLine.setText("/"+str(TotalPage))
        if PageNumber > TotalPage:
            PageNumber = TotalPage
            self.Page3_Widget.InputLineNumber.setText(str(TotalPage))
            QMessageBox.warning(self, "Warning", "Page number out of range, jump to the last page!", QMessageBox.Yes, QMessageBox.Yes)
        elif PageNumber <= 0:
            PageNumber = 1
            self.Page3_Widget.InputLineNumber.setText(str(1))
            QMessageBox.warning(self, "Warning", "Page number out of range, jump to the first page!", QMessageBox.Yes, QMessageBox.Yes)
        display_num_col = num_col - sum(self.FeatureDeleteFlag)
        cols = np.argwhere(np.array(self.FeatureDeleteFlag)-1).reshape(-1,)
        missingnum = self.df.iloc[self.SelectionIndex, cols].isnull().sum().sum()
        NewMissingRate = missingnum * 100 / (display_num_col*display_num_row)
        self.Page3_Widget.MissingRateLE.setText("Missing Rate: %.2f%%" %NewMissingRate)
        self.Page3_Widget.SampleNumLE.setText("Sample Number: %i" %display_num_row)
        self.Page3_Widget.FeatureNumLE.setText("Feature Number: %i" %display_num_col)
        self.Page3_Widget.MainWindow.clear()
        self.Page3_Widget.MainWindow.setColumnCount(max(3, display_num_col))
        HorizontalHeader = [' '] * max(3, display_num_col)
        i_idx = 0
        for i in range(len(self.df.columns)):
            if self.FeatureDeleteFlag[i] == False:
                HorizontalHeader[i_idx] = self.df.columns[i]
                i_idx += 1
        self.Page3_Widget.MainWindow.setHorizontalHeaderLabels(HorizontalHeader)
        start_row = 100 * (PageNumber - 1)
        end_row = min(display_num_row, PageNumber * 100)
        for i in range(start_row, end_row):
            i_df = self.SelectionIndex[i]
            j_idx = 0
            for j in range(num_col):
                if self.FeatureDeleteFlag[j]:
                    continue
                if self.CategoricalFlag[j]:
                    if str(self.df.iat[i_df, j]) == 'nan':
                        CurItem = 'nan'
                    else:
                        try: 
                            CurItem = str(self.CategoricalTransformDict[j][self.df.iat[i_df, j]])
                        except:
                            print(j)
                            print(len(self.CategoricalTransformDict))
                            print(self.df.iat[i_df,j])
                            print(self.df.shape)
                            str(self.CategoricalTransformDict[j][self.df.iat[i_df, j]])
                else:
                    CurItem = str(self.df.iat[i_df, j])
                if CurItem == 'nan':
                    EmptyTableItem = QTableWidgetItem()
                    EmptyTableItem.setBackground(QBrush(QColor(201,252,255)))
                    self.Page3_Widget.MainWindow.setItem(i-start_row, j_idx, EmptyTableItem)
                    j_idx += 1  
                    continue
                self.Page3_Widget.MainWindow.setItem(i-start_row, j_idx, QTableWidgetItem(CurItem))         
                j_idx += 1   
        vertical_header = [str(i+1) for i in range(start_row, start_row+self.Page3_Widget.MainWindow.rowCount())]
        self.Page3_Widget.MainWindow.setVerticalHeaderLabels(vertical_header)
        self.Page3_Widget.MainWindow.verticalHeader().show()
        self.Page3_Widget.MainWindow.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def DisablePage2Buttons(self):
        '''
        DisablePage2Buttons:
        Disable all the buttons in Page2, it is invoked when the merge thread starts.
        '''
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(False)
        # self.Step_Widget.step_layout_button2.setEnabled(False)
        # self.Step_Widget.step_layout_button3.setEnabled(False)
        self.Step_Widget.step_layout_button4.setEnabled(False)
        # self.Step_Widget.step_layout_button5.setEnabled(False)
        self.Step_Widget.step_layout_button6.setEnabled(False)
        # ========================DELETION BUTTONS=============================
        for i in range(len(self.Page2_Widget.alltabwidgets)):
            self.Page2_Widget.alltabwidgets[i].filedelete_button.delete_button.setEnabled(False)
            self.Page1_Widget.file_table.cellWidget(i, 1).delete_button.setEnabled(False)
        # ========================BACK-NEXT BUTTON=============================
        self.Page2_Widget.back_button.setEnabled(False)
        self.Page2_Widget.next_button.setEnabled(False)

        self.Page1_Widget.select_button.setEnabled(False)

    def EnablePage2Buttons(self):
        '''
        EnablePage2Buttons:
        Enable all the buttons in Page2, it is invoked when the merge thread finishes.
        '''
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(True)
        # self.Step_Widget.step_layout_button2.setEnabled(True)
        # self.Step_Widget.step_layout_button3.setEnabled(True)
        self.Step_Widget.step_layout_button4.setEnabled(True)
        # self.Step_Widget.step_layout_button5.setEnabled(True)
        self.Step_Widget.step_layout_button6.setEnabled(True)
        # ========================DELETION BUTTONS=============================
        for i in range(len(self.Page2_Widget.alltabwidgets)):
            self.Page2_Widget.alltabwidgets[i].filedelete_button.delete_button.setEnabled(True)
            self.Page1_Widget.file_table.cellWidget(i, 1).delete_button.setEnabled(True)
        # ========================BACK-NEXT BUTTON=============================
        self.Page2_Widget.back_button.setEnabled(True)
        self.Page2_Widget.next_button.setEnabled(True)    

        self.Page1_Widget.select_button.setEnabled(True)

    def generateMenu(self, pos):
        '''
        generateMenu:
        It is invoked when the right mouse is clicked to generate the context menu
        '''
        colNum = float('inf')
        for i in self.Page3_Widget.MainWindow.selectionModel().selection().indexes():
            colNum = i.column()
        _, num_col = self.df.shape
        num_col = num_col - sum(self.FeatureDeleteFlag)
        if colNum < num_col:
            menu = QMenu()
            self.InfoAction = menu.addAction("Feature Operations")
            self.DeleteAction = menu.addAction("Delete")
            self.TransformAction = menu.addAction("Transform")
            self.SelectAction = menu.addAction("Select")
            self.InfoAction.setEnabled(False)
            self.InfoAction.setCheckable(False)
            self.DeleteAction.setIcon(QIcon(QPixmap('fig/delete.png')))
            self.TransformAction.setIcon(QIcon(QPixmap('fig/transform.png')))
            self.SelectAction.setIcon(QIcon(QPixmap('fig/selection.png')))
            self.DeleteAction.triggered.connect(lambda: self.DeletePreprocessQuery(colNum))
            self.TransformAction.triggered.connect(lambda: self.TransformPreprocessQuery(colNum))
            self.SelectAction.triggered.connect(lambda: self.SelectPreprocess(colNum))
            menu.setStyleSheet('*{font-size: 17px;}')
            screenpos = self.Page3_Widget.MainWindow.mapToGlobal(pos)
            _ = menu.exec_(screenpos)

    def DeletePreprocessQuery(self, colNum):
        '''
        DeletePreprocessQuery:
        It is invoked when user clicks "Delete" button. It queries user whether to delete the column.
        '''
        DeleteQueryButtonYes = QMessageBox.Yes
        DeleteQueryButtonNo = QMessageBox.No
        result = QMessageBox.question(self, "Comfirmation", "Delete this column?", DeleteQueryButtonYes | DeleteQueryButtonNo, DeleteQueryButtonYes)
        if result == QMessageBox.Yes:
            self.DeletePreprocess(colNum)

    def DeletePreprocess(self, colNum):
        '''
        DeletePreprocess:
        Delete certain column/feature.
        '''
        num_remain = 0
        for i in range(len(self.FeatureDeleteFlag)):
            if self.FeatureDeleteFlag[i] == False:
                num_remain += 1
            if num_remain == colNum + 1:
                break
        self.FeatureDeleteFlag[i] = True
        self.ActionStack.append(('Delete', i)) # delete col-i
        if len(self.ActionStack) == 1:
            self.Page3_Widget.UndoButton.setStyleSheet('''
            QPushButton{
                font: 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:20px 20px;
                margin-left: 24px;
                width: 80px;
            }
            QPushButton:hover{
                font: bold 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:20px 20px;
                margin-left: 24px;
            }
        ''')
            self.Page3_Widget.UndoButton.setEnabled(True)
        self.ChangePageDisplay()
        # Whenever there are operations in page3 and page4/5 already initialized, we have to re-initializa them
        if self.page4_init:
            self.page4_init = False
            # self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init =False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def TransformPreprocessQuery(self, colNum):
        '''
        TransformPreprocessQuery:
        It is invoked when user clicks "Transform" button. It queries user whether to transform the column.
        '''
        result = QMessageBox.question(self, "Comfirmation", "Transform this column to categorical data?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            self.TransformPreprocess(colNum)        

    def TransformPreprocess(self, colNum):
        '''
        TransformPreprocess:
        Transform the certain column to categorical data.
        '''
        num_remain = 0
        for i in range(len(self.FeatureDeleteFlag)):
            if self.FeatureDeleteFlag[i] == False:
                num_remain += 1
            if num_remain == colNum + 1:
                break        
        if self.CategoricalFlag[i]:
            return
        self.CategoricalFlag[i] = True
        self.ActionStack.append(('Transform', i))
        if len(self.ActionStack) == 1:
            self.Page3_Widget.UndoButton.setStyleSheet('''
            QPushButton{
                font: 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:20px 20px;
                margin-left: 24px;
                width: 80px;
            }
            QPushButton:hover{
                font: bold 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:20px 20px;
                margin-left: 24px;
            }
        ''')
            self.Page3_Widget.UndoButton.setEnabled(True)
        if self.CategoricalTransformDict[i] == None:
            self.TransformPreprocessThread = TransformThread(self.df.iloc[:,i], i)
            self.TransformPreprocessThread.TransformFinish.connect(self.TransformDisplay)
            self.TransformPreprocessThread.TenPercent.connect(self.TransformProgressDisplay)
            self.DisablePage3Buttons()
            self.TransformPreprocessThread.start()
        else:
            self.ChangePageDisplay()
            # Whenever there are operations in page3 and page4/5 already initialized, we have to re-initializa them
            if self.page4_init:
                self.page4_init = False
                # self.Step_Widget.step_layout_button5.hide()
            if self.page5_init:
                self.page5_init =False
                self.Step_Widget.step_layout_button6.hide()
            if self.page6_init:
                self.page6_init = False

    def SelectPreprocess(self, colNum):
        '''
        SelectPreprocess:
        It is invoked when user clicks "Select" button. It generates the selection box.
        '''
        self.selectionbox = SelectionDialog()
        self.selectionbox.SelectButton1.clicked.connect(self.AssertRangeSelection)
        self.selectionbox.SelectButton2.clicked.connect(self.AssertConditionSelection)
        self.selectionbox.RangeLE1.editingFinished.connect(self.ChangeRangeDown)
        self.selectionbox.RangeLE2.editingFinished.connect(self.ChangeRangeUp)
        self.selectionbox.SelectionMethodComboBox.currentIndexChanged.connect(self.ChangeConditionMethod)
        self.selectionbox.ValueLE.editingFinished.connect(self.ChangeValueLE)
        self.selectionbox.CancelButton.clicked.connect(self.SelectionCancel)
        self.selectionbox.OKButton.clicked.connect(lambda: self.AssertSelectionPreprocess(colNum))
        self.selectionbox.show()
        self.selectionbox.exec_()

    def TransformDisplay(self, TransformDict, col_idx):
        '''
        TransformDisplay:
        Display the result of transformation.
        '''
        self.EnablePage3Buttons()
        self.CategoricalTransformDict[col_idx] = TransformDict
        self.ChangePageDisplay()
        # Whenever there are operations in page3 and page4/5 already initialized, we have to re-initializa them
        if self.page4_init:
            self.page4_init = False
            # self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init =False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def TransformProgressDisplay(self, counter):
        '''
        TransformProgreeDisplay:
        Display the progress bar for transformation operation.
        '''
        if counter == 0:
            self.Page3_Widget.PreprocessProgressLE.setText("Transform Process: ")
            self.Page3_Widget.PreprocessProgressLE.show()
            self.Page3_Widget.progressbar.setMaximum(10)
            self.Page3_Widget.progressbar.show()
            self.Page3_Widget.progressbar.setValue(counter)
        else:
            self.Page3_Widget.progressbar.setValue(counter)
            if counter == 10:
                self.Page3_Widget.progressbar.hide()
                self.Page3_Widget.PreprocessProgressLE.hide()

    def DownloadPreprocessFile(self):
        '''
        DownloadPreprocessFile:
        It is invoked when the "Download" button in Page3 is clicked.
        It is used to download the preprocessed file.
        '''
        savefile, _ = QFileDialog.getSaveFileName(self, 'Save File', '.', '(*.csv)')
        if savefile == '':
            return
        self.DownloadThread = DownloadThread(self.df, self.FeatureDeleteFlag, self.CategoricalFlag, self.CategoricalTransformDict, self.SelectionIndex, savefile)
        self.DownloadThread.Progress.connect(self.DownloadProgressDisplay)
        self.DownloadThread.DownloadFinish.connect(self.DownloadFinish)
        self.DisablePage3Buttons()
        self.DownloadThread.start()

    def AssertRangeSelection(self):
        '''
        AssertRangeSelection:
        It is invoked when the "Range Selection" method is chosen.
        '''
        # Enable the range-selection LEs
        self.selectionbox.RangeLE1.setEnabled(True)
        self.selectionbox.RangeLE2.setEnabled(True)
        # Disable the condition-selection LEs
        self.selectionbox.SelectionMethodComboBox.setEnabled(False)
        self.selectionbox.ValueLE.setEnabled(False)
        # Change the value of some variables
        self.SelectionMethod = "Range"
    
    def AssertConditionSelection(self):
        '''
        AssertConditionSelection:
        It is invoked when the "Condition Selection" method is chosen.
        '''
        # Enable the condition-selection LEs
        self.selectionbox.SelectionMethodComboBox.setEnabled(True)
        self.selectionbox.ValueLE.setEnabled(True)
        # Disable the range-selection LEs
        self.selectionbox.RangeLE1.setEnabled(False)
        self.selectionbox.RangeLE2.setEnabled(False)
        # Change the value of some variable
        self.SelectionMethod = "Condition"

    def ChangeRangeDown(self):
        try:
            self.SelectionRangeDown = float(self.selectionbox.RangeLE1.text())
        except:
            pass
    
    def ChangeRangeUp(self):
        try:
            self.SelectionRangeUp = float(self.selectionbox.RangeLE2.text())
        except:
            pass

    def ChangeConditionMethod(self, i):
        self.SelectionCondition = i

    def ChangeValueLE(self):
        self.SelectionValue = self.selectionbox.ValueLE.text()
        if self.SelectionCondition in [2,3,4,5]:
            try:
                self.SelectionValue = float(self.SelectionValue)
            except:
                self.selectionbox.ValueLE.setText('0')
                self.SelectionValue = 0
                QMessageBox.critical(self.selectionbox, "Error", "Please enter a numerical value!", QMessageBox.Yes, QMessageBox.Yes)
                
    def SelectionCancel(self):
        '''
        SelectionCancel:
        It is invoked when the "Cancel" button in the selection box is clicked.
        '''
        # None-lize all the variables
        self.SelectionMethod = None
        self.SelectionRangeUp = None
        self.SelectionRangeDown = None
        self.SelectionCondition = 0
        self.SelectionValue = None
        # Close the QDialog
        self.selectionbox.close()

    def AssertSelectionPreprocess(self, colNum):
        '''
        AssertSelectionPreprocess:
        It is invoked when the "OK" button in the selectin box is clicked.
        '''
        # Check for "Empty" conditions
        if self.SelectionMethod == None:
            QMessageBox.critical(self.selectionbox, "Error", "Please choose one selection method!", QMessageBox.Yes, QMessageBox.Yes)
            return
        elif self.SelectionMethod == 'Range' and self.SelectionRangeDown == None and self.SelectionRangeUp == None:
            QMessageBox.critical(self.selectionbox, "Error", "Please specify at least one bound for the range!", QMessageBox.Yes, QMessageBox.Yes)
            return
        elif self.SelectionMethod == 'Condition' and self.SelectionValue == None:
            QMessageBox.critical(self.selectionbox, "Error", "Please specify the value for the chosen condition!", QMessageBox.Yes, QMessageBox.Yes)
            return
        # Start the Selection
        self.selectionbox.close()
        num_remain = 0
        for i in range(len(self.FeatureDeleteFlag)):
            if self.FeatureDeleteFlag[i] == False:
                num_remain += 1
            if num_remain == colNum + 1:
                break  
        if self.CategoricalFlag[i] == False:
            feature = self.df.iloc[:,i]
        else:
            feature = self.df.iloc[:,i].map(lambda x:self.TransformToCategorical(x, i))
        self.SelectionPreprocessThread = SelectionThread(feature, self.SelectionMethod, self.SelectionRangeDown, self.SelectionRangeUp, self.SelectionCondition, self.SelectionValue, self.SelectionIndex, i)
        self.SelectionPreprocessThread.SelectionFinish.connect(self.SelectionDisplay)
        self.SelectionPreprocessThread.TenPercent.connect(self.SelectionProgressDisplay)
        self.DisablePage3Buttons()
        self.SelectionPreprocessThread.start()
        self.SelectionPreprocessThread.exec_()

    def TransformToCategorical(self, x, i):
        '''
        TransformToCategorical:
        It is used to transform feature to categorical data.
        '''
        if str(x) == 'nan':
            return np.nan
        else:
            return self.CategoricalTransformDict[i][x]

    def SelectionDisplay(self, SelectedIndicator, SelectionIndex, col_idx):
        '''
        SelectionDisplay:
        Display the data after selection.
        '''
        self.EnablePage3Buttons()
        self.SelectionIndex = SelectionIndex
        if self.SelectionFlag[col_idx] == False:
            self.SelectionFlag[col_idx] = True
            self.SelectionIndicator[col_idx] = [SelectedIndicator]
            self.ActionStack.append(('Select', col_idx))
            if len(self.ActionStack) == 1:
                self.Page3_Widget.UndoButton.setStyleSheet('''
                QPushButton{
                    font: 19px "Microsoft YaHei UI";
                    border: none;
                    background-color: none;
                    color: #3B8BB9;
                    qproperty-icon:url(fig/undo.png);
                    qproperty-iconSize:20px 20px;
                    margin-left: 24px;
                    width: 80px;
                }
                QPushButton:hover{
                    font: bold 19px "Microsoft YaHei UI";
                    border: none;
                    background-color: none;
                    color: #3B8BB9;
                    qproperty-icon:url(fig/undo.png);
                    qproperty-iconSize:20px 20px;
                    margin-left: 24px;
                }
            ''')
                self.Page3_Widget.UndoButton.setEnabled(True)
        else:
            self.SelectionIndicator[col_idx].append(SelectedIndicator)
            self.ActionStack.append(('Select', col_idx))
            if len(self.ActionStack) == 1:
                self.Page3_Widget.UndoButton.setStyleSheet('''
                QPushButton{
                    font: 19px "Microsoft YaHei UI";
                    border: none;
                    background-color: none;
                    color: #3B8BB9;
                    qproperty-icon:url(fig/undo.png);
                    qproperty-iconSize:20px 20px;
                    margin-left: 24px;
                    width: 80px;
                }
                QPushButton:hover{
                    font: bold 19px "Microsoft YaHei UI";
                    border: none;
                    background-color: none;
                    color: #3B8BB9;
                    qproperty-icon:url(fig/undo.png);
                    qproperty-iconSize:20px 20px;
                    margin-left: 24px;
                }
            ''')
                self.Page3_Widget.UndoButton.setEnabled(True)
        self.ChangePageDisplay()
        # self.selectionbox.close()
        self.SelectionMethod = None
        self.SelectionRangeUp = None
        self.SelectionRangeDown = None
        self.SelectionCondition = 0
        self.SelectionValue = None
        # Whenever there are operations in page3 and page4/5 already initialized, we have to re-initializa them
        if self.page4_init:
            self.page4_init = False
            # self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init = False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def SelectionProgressDisplay(self, counter):
        '''
        SelectionProgressDisplay:
        Display the progress bar for selection operation.
        '''
        if counter == 0:
            self.Page3_Widget.PreprocessProgressLE.setText("Selection Process: ")
            self.Page3_Widget.PreprocessProgressLE.show()
            self.Page3_Widget.progressbar.setMaximum(10)
            self.Page3_Widget.progressbar.show()
            self.Page3_Widget.progressbar.setValue(counter)
        else:
            self.Page3_Widget.progressbar.setValue(counter)
            if counter == 10:
                self.Page3_Widget.progressbar.hide()
                self.Page3_Widget.PreprocessProgressLE.hide()        

    def DownloadProgressDisplay(self, counter):
        '''
        DownloadProgressDisplay:
        Display the progress bar of download process.
        '''
        if counter == 0:
            self.Page3_Widget.PreprocessProgressLE.setText("Download Process: ")
            self.Page3_Widget.PreprocessProgressLE.show()
            self.Page3_Widget.progressbar.setMaximum(6)
            self.Page3_Widget.progressbar.show()
            self.Page3_Widget.progressbar.setValue(counter)
        else:
            self.Page3_Widget.progressbar.setValue(counter)
            if counter == 6:
                self.Page3_Widget.progressbar.hide()
                self.Page3_Widget.PreprocessProgressLE.hide()

    def DownloadFinish(self, OK):
        self.EnablePage3Buttons()
        if OK:
            QMessageBox.information(self, "Download Successfully", "The file has been downloaded successfully!", QMessageBox.Yes, QMessageBox.Yes)
        else:
            QMessageBox.critical(self, "Permission Error", "Permission Error: The saved file is currently opened!", QMessageBox.Yes, QMessageBox.Yes)

    def UndoPreprocess(self):
        '''
        UndoPreprocess:
        It is invoked when the "Undo" button in page3 is clicked.
        It is used to undo the previous operation.
        '''
        LastAction, col_idx = self.ActionStack[-1]
        self.ActionStack.pop(-1)
        if len(self.ActionStack) == 0:
            self.Page3_Widget.UndoButton.setStyleSheet('''
            QPushButton{
                font: 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:20px 20px;
                margin-left: 24px;
                width: 80px;
            }
            QPushButton:hover{
                font: bold 19px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:20px 20px;
                margin-left: 24px;
            }
        ''')
            self.Page3_Widget.UndoButton.setEnabled(False)
        if LastAction == 'Delete':
            self.FeatureDeleteFlag[col_idx] = False
            AddMissingNum = self.df.iloc[:,col_idx].isnull().sum()
        elif LastAction == 'Transform':
            self.CategoricalFlag[col_idx] = False
        elif LastAction ==  'Select':
            self.SelectionIndicator[col_idx].pop(-1)
            if len(self.SelectionIndicator[col_idx]) == 0:
                self.SelectionFlag[col_idx] = False
                self.SelectionIndicator[col_idx] = None
            num_sample, _ = self.df.shape
            OverallSelectionIndicator = np.array([True]*num_sample)
            for i in range(len(self.SelectionIndicator)):
                if self.SelectionIndicator[i] == None:
                    continue
                else:
                    for indicator in self.SelectionIndicator[i]:
                        try:
                            OverallSelectionIndicator *= indicator
                        except:
                            print(OverallSelectionIndicator[0:20])
                            print(indicator[0:20])
                            OverallSelectionIndicator *= indicator
            self.SelectionIndex = np.argwhere(OverallSelectionIndicator).reshape(-1,)
        self.ChangePageDisplay()
        # Whenever there are operations in page3 and page4/5 already initialized, we have to re-initializa them
        if self.page4_init:
            self.page4_init = False
            # self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init = False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False
        
    def DisablePage3Buttons(self):
        '''
        DisablePage3Buttons:
        Disable all the buttons in Page3, it is invoked when any thread starts.
        '''
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(False)
        # self.Step_Widget.step_layout_button3.setEnabled(False)
        self.Step_Widget.step_layout_button4.setEnabled(False)
        # self.Step_Widget.step_layout_button5.setEnabled(False)        
        self.Step_Widget.step_layout_button6.setEnabled(False)
        # ======================BUTTONS IN THE TOOLBAR=========================
        self.Page3_Widget.InputLineNumber.setEnabled(False)
        self.Page3_Widget.UndoButton.setEnabled(False)
        self.Page3_Widget.DownloadButton.setEnabled(False)
        # ========================BACK-NEXT BUTTONS============================
        self.Page3_Widget.back_button.setEnabled(False)
        self.Page3_Widget.next_button.setEnabled(False)

    def EnablePage3Buttons(self):
        '''
        EnablePage3Buttons:
        Enable all the buttons in Page3, it is invoked when any thread finishes.
        '''
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(True)
        # self.Step_Widget.step_layout_button3.setEnabled(True)
        self.Step_Widget.step_layout_button4.setEnabled(True)
        # self.Step_Widget.step_layout_button5.setEnabled(True)  
        self.Step_Widget.step_layout_button6.setEnabled(True)        
        # ======================BUTTONS IN THE TOOLBAR=========================
        self.Page3_Widget.InputLineNumber.setEnabled(True)
        if len(self.ActionStack) == 0:
            self.Page3_Widget.UndoButton.setEnabled(False)
        else:
            self.Page3_Widget.UndoButton.setEnabled(True)
        self.Page3_Widget.DownloadButton.setEnabled(True)
        # ========================BACK-NEXT BUTTONS============================
        self.Page3_Widget.back_button.setEnabled(True)
        self.Page3_Widget.next_button.setEnabled(True)        

    def Page4UploadAlg(self):
        '''
        Page4UploadAlg:
        It is used to upload the algorithm file.
        '''
        if self.AlgorithmFile != None:
            QMessageBox.critical(self, "Multiple File Error", "You can only upload one file", QMessageBox.Yes, QMessageBox.Yes)
            return
        newfile, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', '(*.py)')
        if newfile == '':
            return
        self.AlgorithmFile = newfile
        current_num_row = self.Page4_Widget.AlgFileTable.rowCount()
        self.Page4_Widget.AlgFileTable.setRowCount(current_num_row+1)
        self.Page4_Widget.AlgFileTable.setItem(current_num_row, 0, QTableWidgetItem(newfile))
        self.Page4_Widget.AlgFileTable.setCellWidget(current_num_row, 1, FileDeleteButton(0))
        self.Page4_Widget.AlgFileTable.resizeColumnsToContents()
        self.Page4_Widget.AlgFileTable.horizontalHeader().setStretchLastSection(True)
        self.Page4_Widget.AlgFileTable.cellWidget(current_num_row, 1).delete_button.clicked.connect(self.Page4DeleteAlgFile)

    def Page4UploadPar(self):
        '''
        Page4UploadPar:
        It is used to upload parameter file.
        '''
        if self.ParameterFile != None:
            QMessageBox.critical(self, "Multiple File Error", "You can only upload one file", QMessageBox.Yes, QMessageBox.Yes)
            return
        newfile, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', '(*.py)')
        if newfile == '':
            return
        self.ParameterFile = newfile
        current_num_row = self.Page4_Widget.ParFileTable.rowCount()
        self.Page4_Widget.ParFileTable.setRowCount(current_num_row+1)
        self.Page4_Widget.ParFileTable.setItem(current_num_row, 0, QTableWidgetItem(newfile))
        self.Page4_Widget.ParFileTable.setCellWidget(current_num_row, 1, FileDeleteButton(0))
        self.Page4_Widget.ParFileTable.resizeColumnsToContents()
        self.Page4_Widget.ParFileTable.horizontalHeader().setStretchLastSection(True)    
        self.Page4_Widget.ParFileTable.cellWidget(current_num_row, 1).delete_button.clicked.connect(self.Page4DeleteParFile)

    def OpenAlgTemp(self):
        subprocess.call(["atom", 'template/algorithm_template.py'])

    def OpenParTemp(self):
        subprocess.call(["atom", 'template/parameter_template.py'])

    def setSampleSize(self):
        self.SampleSize = float(self.Page4_Widget.SampleSize.text())
        if self.SampleSize < 4000:
            self.SampleSize = 4000
            self.Page4_Widget.SampleSize.setText('4000')
            QMessageBox.warning(self, "Warning", "Sample size out of range, set it to the 4000!", QMessageBox.Yes, QMessageBox.Yes)
        if self.SampleSize > 40000:
            self.SampleSize = 40000
            self.Page4_Widget.SampleSize.setText('40000')
            QMessageBox.warning(self, "Warning", "Sample size out of range, set it to the 40000!", QMessageBox.Yes, QMessageBox.Yes)
        if self.page5_init:
            self.page5_init = False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def setErrorBound(self):
        self.ErrorBound = float(self.Page4_Widget.ErrorBound.text())
        if self.ErrorBound < 0.001:
            self.ErrorBound = 0.001
            self.Page4_Widget.ErrorBound.setText('0.001')
            QMessageBox.warning(self, "Warning", "Error bound out of range, set it to the 0.001!", QMessageBox.Yes, QMessageBox.Yes)
        if self.ErrorBound > 0.009:
            self.ErrorBound = 0.009
            self.Page4_Widget.ErrorBound.setText('0.009')
            QMessageBox.warning(self, "Warning", "Error bound out of range, set it to the 0.009!", QMessageBox.Yes, QMessageBox.Yes)
        if self.page5_init:
            self.page5_init = False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def EnableSCIS(self):
        self.SCIS_Enabled = True
        self.Page4_Widget.ErrorBound.setEnabled(True)
        self.Page4_Widget.SampleSize.setEnabled(True)
        if self.page5_init:
            self.page5_init = False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def AssertEnableUser(self):
        # if self.Page4_Widget.AlgSelectUser.isChecked():
        #     self.USER_Enabled = True
        # else:
        #     self.USER_Enabled = False
        self.Page4_Widget.ParameterHide()
        self.USER_Enabled = True
        self.SCIS_Enabled = False
        self.Page4_Widget.ErrorBound.clear()
        self.Page4_Widget.ErrorBound.setEnabled(False)
        self.ErrorBound = None
        self.Page4_Widget.SampleSize.clear()
        self.Page4_Widget.SampleSize.setEnabled(False)
        self.SampleSize = None
        if self.page5_init:
            self.page5_init = False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def AssertEnableSCIS(self):
        # if self.Page4_Widget.AlgSelectSCIS.isChecked():
        #     self.SCIS_Enabled = True
        #     self.Page4_Widget.ErrorBound.setEnabled(True)
        #     self.Page4_Widget.SampleSize.setEnabled(True)
        # else:
        #     self.SCIS_Enabled= False
        #     self.Page4_Widget.ErrorBound.clear()
        #     self.Page4_Widget.ErrorBound.setEnabled(False)
        #     self.ErrorBound = None
        #     self.Page4_Widget.SampleSize.clear()
        #     self.Page4_Widget.SampleSize.setEnabled(False)
        #     self.SampleSize = None
        self.Page4_Widget.ParameterShow()
        self.SCIS_Enabled = True
        self.USER_Enabled = False
        self.Page4_Widget.ErrorBound.setEnabled(True)
        self.Page4_Widget.SampleSize.setEnabled(True)
        if self.page5_init:
            self.page5_init = False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def DisableSCIS(self):
        self.SCIS_Enabled = False
        self.ErrorBound = None
        self.Page4_Widget.ErrorBound.clear()
        self.Page4_Widget.ErrorBound.setEnabled(False)
        self.SampleSize = None
        self.Page4_Widget.SampleSize.clear()
        self.Page4_Widget.SampleSize.setEnabled(False)
        if self.page5_init:
            self.page5_init = False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def GenerateSCISInfo(self):
        self.SCISInfoBox = SCISInfoDialog()
        self.SCISInfoBoxPageNum = 1
        self.SCISInfoBox.GoLeft.clicked.connect(self.SCISInfoBoxGoLeft)
        self.SCISInfoBox.GoRight.clicked.connect(self.SCISInfoBoxGoRight)
        self.SCISInfoBox.showMaximized()
        self.SCISInfoBox.exec_()

    def SCISInfoBoxGoLeft(self):
        if self.SCISInfoBoxPageNum == 1:
            QMessageBox.critical(self.SCISInfoBox, "Page Number Error", "It is already the first page!", QMessageBox.Yes, QMessageBox.Yes)
            return
        elif self.SCISInfoBoxPageNum == 2:
            self.SCISInfoBoxPageNum = 1
            self.SCISInfoBox.Page1.hide()
            self.SCISInfoBox.Info1.hide()
            self.SCISInfoBox.Info2.hide()
            self.SCISInfoBox.Page0Title.show()
            self.SCISInfoBox.Page0Contents.show()
            self.SCISInfoBox.PageNumber.setText("Page: 1/3")
        elif self.SCISInfoBoxPageNum == 3:
            self.SCISInfoBoxPageNum = 2
            self.SCISInfoBox.Page2.hide()
            self.SCISInfoBox.Page1.show()
            self.SCISInfoBox.Info3.hide()
            self.SCISInfoBox.Info4.hide()
            self.SCISInfoBox.Info1.show()
            self.SCISInfoBox.Info2.show()
            self.SCISInfoBox.PageNumber.setText("Page: 2/3")
            
    def SCISInfoBoxGoRight(self):
        if self.SCISInfoBoxPageNum == 1:
            self.SCISInfoBoxPageNum = 2
            self.SCISInfoBox.Page0Contents.hide()
            self.SCISInfoBox.Page0Title.hide()
            self.SCISInfoBox.Page1.show()
            self.SCISInfoBox.Info1.show()
            self.SCISInfoBox.Info2.show()
            self.SCISInfoBox.PageNumber.setText("Page: 2/3")
        elif self.SCISInfoBoxPageNum == 2:
            self.SCISInfoBoxPageNum = 3
            self.SCISInfoBox.Page1.hide()
            self.SCISInfoBox.Page2.show()
            self.SCISInfoBox.Info1.hide()
            self.SCISInfoBox.Info2.hide()
            self.SCISInfoBox.Info3.show()
            self.SCISInfoBox.Info4.show()
            self.SCISInfoBox.PageNumber.setText("Page: 3/3")
        elif self.SCISInfoBoxPageNum == 3:
            QMessageBox.critical(self.SCISInfoBox, "Page Number Error", "It is already the last page!", QMessageBox.Yes, QMessageBox.Yes)
            return

    def Page4DeleteAlgFile(self):
        '''
        Page4DeleteAlgFile:
        It is used to delete the uploaded algorithm file.
        '''
        self.Page4_Widget.AlgFileTable.removeRow(0)
        self.AlgorithmFile = None
        if self.page5_init:
            self.page5_init = False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def Page4DeleteParFile(self):
        '''
        Page4DeleteParFile:
        It is used to delete the uploaded parameter file.
        '''
        self.Page4_Widget.ParFileTable.removeRow(0)
        self.ParameterFile = None
        if self.page5_init:
            self.page5_init = False
            self.Step_Widget.step_layout_button6.hide()
        if self.page6_init:
            self.page6_init = False

    def DisablePage4Buttons(self):
        '''
        DisablePage4Buttons:
        Disable all the buttons in Page4, it is invoked when imputation thread starts.
        '''
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(False)
        # self.Step_Widget.step_layout_button3.setEnabled(False)
        self.Step_Widget.step_layout_button4.setEnabled(False)
        # self.Step_Widget.step_layout_button5.setEnabled(False)
        self.Step_Widget.step_layout_button6.setEnabled(False)
        # ==========================SELECT BUTTONS=============================
        self.Page4_Widget.AlgUploadButton.setEnabled(False)
        self.Page4_Widget.ParUploadButton.setEnabled(False)
        # ==========================DELETE BUTTONS=============================
        try:
            self.Page4_Widget.AlgFileTable.cellWidget(0, 1).delete_button.setEnabled(False)
            self.Page4_Widget.ParFileTable.cellWidget(0, 1).delete_button.setEnabled(False)
        except:
            pass
        # ========================BACK-NEXT BUTTONS============================
        self.Page4_Widget.back_button.setEnabled(False)
        self.Page4_Widget.next_button.setEnabled(False)
        # ==============================SCIS QUERY=============================
        self.Page4_Widget.SCISYesButton.setEnabled(False)
        self.Page4_Widget.SCISNoButton.setEnabled(False)
        self.Page4_Widget.SampleSize.setEnabled(False)
        self.Page4_Widget.ErrorBound.setEnabled(False)

    def EnablePage4Buttons(self):
        '''
        EnablePage4Buttons:
        Enable all the buttons in Page4, it is invoked when imputation thread finishes.
        '''
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(True)
        # self.Step_Widget.step_layout_button3.setEnabled(True)
        self.Step_Widget.step_layout_button4.setEnabled(True)
        # self.Step_Widget.step_layout_button5.setEnabled(True)
        self.Step_Widget.step_layout_button6.setEnabled(True)
        # ==========================SELECT BUTTONS=============================
        self.Page4_Widget.AlgUploadButton.setEnabled(True)
        self.Page4_Widget.ParUploadButton.setEnabled(True)
        # ==========================DELETE BUTTONS=============================
        try:
            self.Page4_Widget.AlgFileTable.cellWidget(0, 1).delete_button.setEnabled(True)
            self.Page4_Widget.ParFileTable.cellWidget(0, 1).delete_button.setEnabled(True)
        except:
            pass
        # ========================BACK-NEXT BUTTONS============================
        self.Page4_Widget.back_button.setEnabled(True)
        self.Page4_Widget.next_button.setEnabled(True)   
        # ==============================SCIS QUERY=============================
        self.Page4_Widget.SCISYesButton.setEnabled(True)
        self.Page4_Widget.SCISNoButton.setEnabled(True)
        self.Page4_Widget.SampleSize.setEnabled(True)
        self.Page4_Widget.ErrorBound.setEnabled(True)

    def Page4TimerDisplay(self):
        '''
        Page4TimerDisplay:
        It is used to display the total time spent for imputation.
        '''
        self.Page4Counter += 1
        second = self.Page4Counter % 60
        minute = int(self.Page4Counter/60) % 60
        hour = int(int(self.Page4Counter/60) / 60)
        if second < 10:
            second = '0' + str(second)
        else:
            second = str(second)
        if minute < 10:
            minute = '0' + str(minute)
        else:
            minute = str(minute)
        if hour < 10:
            hour = '0' + str(hour)
        else:
            hour = str(hour)
        if self.Page4Counter == 1:
            self.Page4_Widget.NotingInfo.show()
            self.Page4_Widget.ImputationTime.show()
        self.ImputationTime = hour+":"+minute+":"+second
        self.Page4_Widget.ImputationTime.setText(self.ImputationTime)

    def Page5_DisplayTable(self, df, dfisnull, OK, error_msg, ThreadIndex):
        '''
        Page5_DisplayTable:
            Displays the fifth interface
        Args:
            - df: Imputed dataframe
            - dfisnull: dataframe which indicates whether each position is originally missing
            - OK: indicates whether the imputation is successful or not
            - error_msg: used to indicate type of error, valid only if OK == False
        '''
        # When error occurs
        if OK == False:
            if ThreadIndex == 1:
                if self.SCIS_Enabled:
                    self.page5_ImputationThread2.quit()
                    self.Thread2Finished == False
            elif ThreadIndex == 2:
                if self.USER_Enabled:
                    self.page5_ImputationThread1.quit()
                    self.Thread1Finished == False
            self.EnablePage4Buttons()
            self.Page4_Timer.stop()
            self.Page4Counter = 0
            self.Page4_Widget.ImputationTime.hide()
            self.Page4_Widget.NotingInfo.hide()            
            if error_msg == 0:
                QMessageBox.critical(self, "Import Error", "The algorithm file is imported unsuccessfully!", QMessageBox.Yes, QMessageBox.Yes)
                return
            elif error_msg == 1:
                QMessageBox.critical(self, "Import Error", "The paramter file is imported unsuccessfully!", QMessageBox.Yes, QMessageBox.Yes)
                return
            elif error_msg == 2:
                QMessageBox.critical(self, "Attribute Error", "The paramter file does not contain parameter() class!", QMessageBox.Yes, QMessageBox.Yes)
                return
            elif error_msg == 3:
                QMessageBox.critical(self, "Attribute Error", "The algorithm file does not contain main() function!", QMessageBox.Yes, QMessageBox.Yes)
                return
            elif error_msg == 4:
                QMessageBox.critical(self, "Error", "Error occurs in the imputation process!", QMessageBox.Yes, QMessageBox.Yes)
                return
        # When the imputation is finished successfully
        if ThreadIndex == 1:
            self.Thread1Finished = True
            self.Thread1Time = self.ImputationTime
            self.impute_df1 = np.round(df, 6)
        elif ThreadIndex == 2:
            self.Thread2Finished = True
            self.Thread2Time = self.ImputationTime
            self.impute_df2 = np.round(df, 6)
        # Check whether both threads are finished
        if self.SCIS_Enabled and self.USER_Enabled:
            if (self.Thread1Finished == False or self.Thread2Finished == False):
                return
        self.Thread1Finished = False
        self.Thread2Finished = False
        self.EnablePage4Buttons()
        self.Page4_Timer.stop()
        self.Page4Counter = 0
        self.Page4_Widget.ImputationTime.hide()
        self.Page4_Widget.NotingInfo.hide()
        self.page5_init = True
        self.pagenum = 5
        self.dfisnull = dfisnull
        self.RewriteNewValue = None
        all_name = list(self.df.columns)
        feature_name = []
        for i in range(len(self.FeatureDeleteFlag)):
            if self.FeatureDeleteFlag[i] == False:
                feature_name.append(all_name[i])
        if self.USER_Enabled:
            self.impute_df1.columns = feature_name
        if self.SCIS_Enabled:
            self.impute_df2.columns = feature_name
        self.Page5_Widget = Page5_Widget(self.impute_df1, self.impute_df2, self.dfisnull, self.Thread1Time, self.Thread2Time, self.SCIS_Enabled, self.USER_Enabled)
        self.Overall_Layout.addWidget(self.Page5_Widget)
        self.Page5_Widget.back_button.clicked.connect(self.GOTO_Page4)
        self.Page5_Widget.next_button.clicked.connect(self.GOTO_Page6)
        self.Page5_Widget.DownloadButton.clicked.connect(self.Page5DownloadFile)
        self.Page5_Widget.MainWindowTab.tabBarClicked.connect(self.ChangeTabIndex)
        self.Page5TabIndex = 0
        self.Page5_Widget.InputLineNumber.editingFinished.connect(self.Page5ChangeDisplay)
        # It seems that people can only jump to page5 from page4
        self.Page4_Widget.hide()
        self.Step_Widget.step_layout_button4.setStyleSheet(self.button_select_style)
        # self.Step_Widget.step_layout_button5.setStyleSheet(self.button_select_style)
        self.Page5_Widget.show()
        self.Step_Widget.step_layout_button6.show()
        
    def ChangeTabIndex(self, index):
        self.Page5TabIndex = index

    def Page5DownloadFile(self):
        '''
        Page5DownloadFile:
        It is used to download the imputed file.
        '''
        savefile, _ = QFileDialog.getSaveFileName(self, 'Save File', '.', '(*.csv)')
        if savefile == '':
            return
        if self.Page5TabIndex == 0 and self.USER_Enabled:
            page5download = Page5DownloadThread(self.impute_df1, savefile)
        else:
            page5download = Page5DownloadThread(self.impute_df2, savefile)
        page5download.DownloadFinish.connect(self.Page5DownloadFinish)
        self.Page5_Timer = QTimer()
        self.Page5_Timer.timeout.connect(self.Page5TimerDisplay)
        self.Page5Counter = 0
        self.Page5_Timer.start(500)
        self.DisablePage5Buttons()
        page5download.start()
        page5download.exec_()
        
    def Page5TimerDisplay(self):
        '''
        Page5TimerDisplay:
        It is used to display time spent on imputation task.
        '''
        self.Page5Counter += 1
        dots = '.' * (self.Page5Counter%7)
        if self.Page5Counter == 1:
            self.Page5_Widget.NotingInfo.show()
        self.Page5_Widget.NotingInfo.setText("Start downloading the imputed file"+dots)

    def Page5DownloadFinish(self,OK):
        '''
        Page5DownloadFinish:
        It is invoked when the download process in page5 finishes.
        '''
        self.EnablePage5Buttons()
        self.Page5_Timer.stop()
        self.Page5Counter = 0
        self.Page5_Widget.NotingInfo.hide()
        if OK:
            QMessageBox.information(self, "Download Successfully", "The file has been downloaded successfully!", QMessageBox.Yes, QMessageBox.Yes) 
        else:
            QMessageBox.critical(self, "Permission Error", "Permission Error: The saved file is currently opened!", QMessageBox.Yes, QMessageBox.Yes)

    def DisablePage5Buttons(self):
        '''
        DisablePage5Buttons:
        Disable all the buttons in Page5, it is invoked when download thread starts.
        '''
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(False)
        # self.Step_Widget.step_layout_button3.setEnabled(False)
        self.Step_Widget.step_layout_button4.setEnabled(False)
        # self.Step_Widget.step_layout_button5.setEnabled(False)
        self.Step_Widget.step_layout_button6.setEnabled(False)
        # =============================OTHERS==================================
        self.Page5_Widget.DownloadButton.setEnabled(False)
        self.Page5_Widget.back_button.setEnabled(False)        

    def EnablePage5Buttons(self):
        '''
        EnablePage5Buttons:
        Enable all the buttons in Page5, it is invoked when download thread finishes.
        '''
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(True)
        # self.Step_Widget.step_layout_button3.setEnabled(True)
        self.Step_Widget.step_layout_button4.setEnabled(True)
        # self.Step_Widget.step_layout_button5.setEnabled(True)
        self.Step_Widget.step_layout_button6.setEnabled(True)
        # ==============================OTHERS================================
        self.Page5_Widget.DownloadButton.setEnabled(True)
        self.Page5_Widget.back_button.setEnabled(True)      

    def Page5ChangeDisplay(self):
        '''
        Page5ChangeDisplay:
        It is used change the display when the page number changes.
        '''
        if self.USER_Enabled:
            num_row, num_col = self.impute_df1.shape
        else:
            num_row, num_col = self.impute_df2.shape
        try:
            PageNumber = int(self.Page5_Widget.InputLineNumber.text())
        except:
            PageNumber = 1
        TotalPage = int(num_row/100)+1
        if PageNumber > TotalPage:
            PageNumber = TotalPage
            self.Page5_Widget.InputLineNumber.setText(str(TotalPage))
            QMessageBox.warning(self, "Warning", "Page number out of, jump to the last page!", QMessageBox.Yes, QMessageBox.Yes)
        elif PageNumber <= 0:
            PageNumber = 1
            self.Page5_Widget.InputLineNumber.setText(str(1))
            QMessageBox.warning(self, "Warning", "Page number out of, jump to the first page!", QMessageBox.Yes, QMessageBox.Yes)
        if self.Page5TabIndex == 0 and self.USER_Enabled:
            self.Page5_Widget.MainWindow1.clear()
        else:
            self.Page5_Widget.MainWindow2.clear()
        start_row = 100 * (PageNumber - 1)
        end_row = min(num_row, 100 * PageNumber)
        for i in range(start_row, end_row):
            for j in range(num_col):
                if self.Page5TabIndex == 0 and self.USER_Enabled:
                    CurItem = QTableWidgetItem(str(self.impute_df1.iat[i, j]))
                else:
                    CurItem = QTableWidgetItem(str(self.impute_df2.iat[i, j]))
                if self.dfisnull.iat[i, j]:
                    if self.Page5TabIndex == 0 and self.USER_Enabled:
                        if str(self.impute_df1.iat[i, j]) == 'nan':
                            CurItem = QTableWidgetItem()
                    else:
                        if str(self.impute_df2.iat[i, j]) == 'nan':
                            CurItem = QTableWidgetItem()
                    CurItem.setBackground(QBrush(QColor(201,252,255)))
                if self.Page5TabIndex == 0 and self.USER_Enabled:
                    self.Page5_Widget.MainWindow1.setItem(i-start_row, j, CurItem)
                else:
                    self.Page5_Widget.MainWindow2.setItem(i-start_row, j, CurItem)
        if self.Page5TabIndex == 0 and self.USER_Enabled:
            vertical_header = [str(i+1) for i in range(start_row, start_row+self.Page5_Widget.MainWindow1.rowCount())]
            self.Page5_Widget.MainWindow1.setVerticalHeaderLabels(vertical_header)
            self.Page5_Widget.MainWindow1.verticalHeader().show()
            self.Page5_Widget.MainWindow1.setHorizontalHeaderLabels(list(self.impute_df1.columns))
            self.Page5_Widget.MainWindow1.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        else:
            vertical_header = [str(i+1) for i in range(start_row, start_row+self.Page5_Widget.MainWindow2.rowCount())]
            self.Page5_Widget.MainWindow2.setVerticalHeaderLabels(vertical_header)
            self.Page5_Widget.MainWindow2.verticalHeader().show()
            self.Page5_Widget.MainWindow2.setHorizontalHeaderLabels(list(self.impute_df2.columns))
            self.Page5_Widget.MainWindow2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def Page6UploadLabelFile(self):
        if self.LabelFile != None:
            QMessageBox.critical(self, "Multiple File Error", "You can only upload one label file", QMessageBox.Yes, QMessageBox.Yes)
            return
        newfile, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', '(*.csv)')
        if newfile == '':
            return
        self.LabelFile = newfile
        self.Page6FileIdx += 1
        self.Page6LabelFileIdx = self.Page6FileIdx
        current_num_row = self.Page6_Widget.LabelFileTable.rowCount()
        self.Page6_Widget.LabelFileTable.setRowCount(current_num_row + 1)
        self.Page6_Widget.LabelFileTable.setItem(current_num_row, 0, QTableWidgetItem("Label File               "))
        self.Page6_Widget.LabelFileTable.setItem(current_num_row, 1, QTableWidgetItem(newfile))
        self.Page6_Widget.LabelFileTable.setCellWidget(current_num_row, 2, FileDeleteButton(0))
        self.Page6_Widget.LabelFileTable.resizeColumnsToContents()
        self.Page6_Widget.LabelFileTable.horizontalHeader().setStretchLastSection(True)
        self.Page6_Widget.LabelFileTable.cellWidget(current_num_row, 2).delete_button.clicked.connect(self.Page6DeleteLabelFile)

    def Page6DeleteLabelFile(self):
        if self.Page6LabelFileIdx < self.Page6AlgFileIdx:
            self.Page6_Widget.LabelFileTable.removeRow(0)
        else:
            self.Page6_Widget.LabelFileTable.removeRow(1)
        self.LabelFile = None
        self.Page6LabelFileIdx = float('inf')
        
    def Page6UploadAlgorithmFile(self):
        if self.PredictAlgFile != None:
            QMessageBox.critical(self, "Multiple File Error", "You can only upload one algorithm file", QMessageBox.Yes, QMessageBox.Yes)
            return
        newfile, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', '(*.py)')
        if newfile == '':
            return
        self.PredictAlgFile = newfile
        self.Page6FileIdx += 1
        self.Page6AlgFileIdx = self.Page6FileIdx
        current_num_row = self.Page6_Widget.LabelFileTable.rowCount()
        self.Page6_Widget.LabelFileTable.setRowCount(current_num_row + 1)
        self.Page6_Widget.LabelFileTable.setItem(current_num_row, 0, QTableWidgetItem("Algorithm File      "))
        self.Page6_Widget.LabelFileTable.setItem(current_num_row, 1, QTableWidgetItem(newfile))
        self.Page6_Widget.LabelFileTable.setCellWidget(current_num_row, 2, FileDeleteButton(0))
        self.Page6_Widget.LabelFileTable.resizeColumnsToContents()
        self.Page6_Widget.LabelFileTable.horizontalHeader().setStretchLastSection(True)
        self.Page6_Widget.LabelFileTable.cellWidget(current_num_row, 2).delete_button.clicked.connect(self.Page6DeleteAlgFile)

    def Page6DeleteAlgFile(self):
        if self.Page6AlgFileIdx < self.Page6LabelFileIdx:
            self.Page6_Widget.LabelFileTable.removeRow(0)
        else:
            self.Page6_Widget.LabelFileTable.removeRow(1)
        self.PredictAlgFile = None
        self.Page6AlgFileIdx = float('inf')

    def ChangePredictionTask(self, i):
        if i == 0:
            self.PredictionTask = 'Regression'
            self.Page6_Widget.ChangeAxes(task="Regression")
            self.Page6_Widget.DefaultAlgorithm.setEnabled(True)
        elif i == 1:
            self.PredictionTask = 'Classification'
            self.Page6_Widget.ChangeAxes(task="Classification")
            if self.PredictionAlgorithm == 'Default':
                QMessageBox.information(self, "No Default Algorithm", "There is no default classification algorithm provided. Please upload one.", QMessageBox.Yes, QMessageBox.Yes)
            self.Page6_Widget.DefaultAlgorithm.setChecked(False)
            self.Page6_Widget.CustomizedAlgorithm.setChecked(True)
            self.Page6_Widget.DefaultAlgorithm.setEnabled(False)
            self.SetCustomizedPredictAlg()

    # ==================================================
    def ChangePredictionTaskToReg(self):
        self.PredictionTask = 'Regression'
        self.Page6_Widget.ChangeAxes(task='Regression')
        self.Page6_Widget.DefaultAlgorithm.setEnabled(True)

    def ChangePredictionTaskToCla(self):
        self.PredictionTask = 'Classification'
        self.Page6_Widget.ChangeAxes(task='Classification')
        if self.PredictionAlgorithm == 'Default':
            QMessageBox.information(self, "No Default Algorithm", "There is no default classification algorithm provided. Please upload one.", QMessageBox.Yes, QMessageBox.Yes)
        self.Page6_Widget.DefaultAlgorithm.setChecked(False)
        self.Page6_Widget.CustomizedAlgorithm.setChecked(True)
        self.Page6_Widget.DefaultAlgorithm.setEnabled(False)
        self.SetCustomizedPredictAlg()
    # ==================================================

    def SetDefaultPredictionAlg(self):
        self.PredictionAlgorithm = 'Default'
        self.Page6_Widget.AlgUploadButton.setEnabled(False)
        self.Page6_Widget.HinitButton.setEnabled(False)
        self.Page6_Widget.AlgUploadButton.hide()
        self.Page6_Widget.HinitButton.hide()
        self.Page6_Widget.placeholder.hide()
        self.Page6DeleteAlgFile()

    def SetCustomizedPredictAlg(self):
        self.PredictionAlgorithm = 'Custom'
        self.Page6_Widget.AlgUploadButton.setEnabled(True)
        self.Page6_Widget.HinitButton.setEnabled(True)
        self.Page6_Widget.AlgUploadButton.show()
        self.Page6_Widget.HinitButton.show()
        self.Page6_Widget.placeholder.show()

    def DisablePage6Buttons(self):
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(False)
        # self.Step_Widget.step_layout_button3.setEnabled(False)
        self.Step_Widget.step_layout_button4.setEnabled(False)
        # self.Step_Widget.step_layout_button5.setEnabled(False)
        self.Step_Widget.step_layout_button6.setEnabled(False)
        # ============================OTHER BUTTONS============================
        self.Page6_Widget.LabelUploadButton.setEnabled(False)
        self.Page6_Widget.back_button.setEnabled(False)
        self.Page6_Widget.RunButton.setEnabled(False)      
        self.Page6_Widget.AlgUploadButton.setEnabled(False)
        self.Page6_Widget.HinitButton.setEnabled(False)
        self.Page6_Widget.DefaultAlgorithm.setEnabled(False)
        self.Page6_Widget.CustomizedAlgorithm.setEnabled(False)
        self.Page6_Widget.PredictionReg.setEnabled(False)
        self.Page6_Widget.PredictionCla.setEnabled(False)
        # ============================DELETE BUTTONS===========================
        num_file = self.Page6_Widget.LabelFileTable.rowCount()
        for i in range(num_file):
            self.Page6_Widget.LabelFileTable.cellWidget(i, 2).delete_button.setEnabled(False)


    def EnablePage6Buttons(self):
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(True)
        # self.Step_Widget.step_layout_button3.setEnabled(True)
        self.Step_Widget.step_layout_button4.setEnabled(True)
        # self.Step_Widget.step_layout_button5.setEnabled(True)
        self.Step_Widget.step_layout_button6.setEnabled(True)
        # ============================OTHER BUTTONS============================
        self.Page6_Widget.LabelUploadButton.setEnabled(True)
        self.Page6_Widget.back_button.setEnabled(True)
        self.Page6_Widget.RunButton.setEnabled(True) 
        if self.PredictionTask == 'Regression':
            self.Page6_Widget.DefaultAlgorithm.setEnabled(True)
        self.Page6_Widget.CustomizedAlgorithm.setEnabled(True)
        if self.PredictionAlgorithm == 'Custom':
            self.Page6_Widget.HinitButton.setEnabled(True)
            self.Page6_Widget.AlgUploadButton.setEnabled(True)
        self.Page6_Widget.PredictionReg.setEnabled(True)
        self.Page6_Widget.PredictionCla.setEnabled(True)
        # ============================DELETE BUTTONS===========================
        num_file = self.Page6_Widget.LabelFileTable.rowCount()
        for i in range(num_file):
            self.Page6_Widget.LabelFileTable.cellWidget(i, 2).delete_button.setEnabled(True)

    def RunPredictionTask(self):
        # Check whether there is label-file uploaded
        if self.LabelFile == None:
            QMessageBox.information(self, "No File Uploaded", "There are no label file uploaded. Please upload one.", QMessageBox.Yes, QMessageBox.Yes)
            return 
        # When prediction task is "Regression"
        if self.PredictionTask == "Regression":
            if self.PredictionAlgorithm == 'Default':
                self.DisablePage6Buttons()
                # =====Thread Start======
                if self.USER_Enabled:
                    num_sample, _ = self.impute_df1.shape
                else:
                    num_sample, _ = self.impute_df2.shape
                num_sample = min(num_sample, 400)
                if self.USER_Enabled and not self.SCIS_Enabled:
                    self.page6_PredictionThread = PredictionThread(self.impute_df1.iloc[:num_sample,:], None, self.LabelFile, False, True)
                elif not self.USER_Enabled and self.SCIS_Enabled:
                    self.page6_PredictionThread = PredictionThread(None, self.impute_df2.iloc[:num_sample,:], self.LabelFile, True, False)
                elif self.USER_Enabled and self.SCIS_Enabled:
                    self.page6_PredictionThread = PredictionThread(self.impute_df1.iloc[:num_sample,:], self.impute_df2.iloc[:num_sample,:], self.LabelFile, True, True)
                self.page6_PredictionThread.PredictionFinished.connect(self.Page6_DisplayStat)
                # =======================
                self.Page6_Timer = QTimer()
                self.Page6Counter = 0
                self.Page6_Timer.timeout.connect(self.Page6TimerDisplay)
                self.Page6_Timer.start(1000)
                self.page6_PredictionThread.start()
            elif self.PredictionAlgorithm == 'Custom':
                self.DisablePage6Buttons()
                if self.PredictAlgFile == None:
                    QMessageBox.information(self, "No File Uploaded", "There are no prediction algorithm file uploaded. Please upload one.", QMessageBox.Yes, QMessageBox.Yes)
                    self.EnablePage6Buttons()
                    return 
                if self.USER_Enabled:
                    num_sample, _ = self.impute_df1.shape
                else:
                    num_sample, _ = self.impute_df2.shape
                num_sample = min(num_sample, 400)
                if self.USER_Enabled and not self.SCIS_Enabled:
                    self.page6_PredictionThread_user = PredictionThreadUser(self.impute_df1.iloc[:num_sample,:], None, self.LabelFile, self.PredictAlgFile, False, True)
                elif not self.USER_Enabled and self.SCIS_Enabled:
                    self.page6_PredictionThread_user = PredictionThreadUser(None, self.impute_df2.iloc[:num_sample,:], self.LabelFile, self.PredictAlgFile, True, False)
                elif self.SCIS_Enabled and self.USER_Enabled:
                    self.page6_PredictionThread_user = PredictionThreadUser(self.impute_df1.iloc[:num_sample,:], self.impute_df2.iloc[:num_sample,:], self.LabelFile, self.PredictAlgFile, True, True)
                self.page6_PredictionThread_user.PredictionFinished.connect(self.Page6_DisplayStat)
                self.Page6_Timer = QTimer()
                self.Page6Counter = 0
                self.Page6_Timer.timeout.connect(self.Page6TimerDisplay)
                self.Page6_Timer.start(1000)
                self.page6_PredictionThread_user.start()
        elif self.PredictionTask == 'Classification':
            self.DisablePage6Buttons()
            if self.PredictAlgFile == None:
                QMessageBox.information(self, "No File Uploaded", "There are no prediction algorithm file uploaded. Please upload one.", QMessageBox.Yes, QMessageBox.Yes)
                self.EnablePage6Buttons()
                return 
            if self.USER_Enabled:
                num_sample, _ = self.impute_df1.shape
            else:
                num_sample, _ = self.impute_df2.shape
            num_sample = min(num_sample, 400)
            if self.USER_Enabled and not self.SCIS_Enabled:
                self.page6_PredictionThread_user = PredictionThreadUser(self.impute_df1.iloc[:num_sample,:], None, self.LabelFile, self.PredictAlgFile, False, True)
            elif not self.USER_Enabled and self.SCIS_Enabled:
                self.page6_PredictionThread_user = PredictionThreadUser(None, self.impute_df2.iloc[:num_sample,:], self.LabelFile, self.PredictAlgFile, True, False)
            elif self.SCIS_Enabled and self.USER_Enabled:
                self.page6_PredictionThread_user = PredictionThreadUser(self.impute_df1.iloc[:num_sample,:], self.impute_df2.iloc[:num_sample,:], self.LabelFile, self.PredictAlgFile, True, True)
            self.page6_PredictionThread_user.PredictionFinished.connect(self.Page6_DisplayStat)
            self.Page6_Timer = QTimer()
            self.Page6Counter = 0
            self.Page6_Timer.timeout.connect(self.Page6TimerDisplay)
            self.Page6_Timer.start(1000)
            self.page6_PredictionThread_user.start()


    def OpenPredictionAlgTemp(self):
        subprocess.call(["atom", 'template/prediction_algorithm_template.py'])

    def Page6TimerDisplay(self):
        self.Page6Counter += 1
        second = self.Page6Counter % 60
        minute = int(self.Page6Counter/60) % 60
        hour = int(int(self.Page6Counter/60) / 60)
        if second < 10:
            second = '0' + str(second)
        else:
            second = str(second)
        if minute < 10:
            minute = '0' + str(minute)
        else:
            minute = str(minute)
        if hour < 10:
            hour = '0' + str(hour)
        else:
            hour = str(hour)
        if self.Page6Counter == 1:
            self.Page6_Widget.NotingInfo.show()
            self.Page6_Widget.PredictionTime.show()
        self.PredictionTime = hour+":"+minute+":"+second
        self.Page6_Widget.PredictionTime.setText(self.PredictionTime)

    def Page6_DisplayStat(self, data1, data2, SuccessFlag, ErrorNumber):
        if SuccessFlag == False:
            self.EnablePage6Buttons()
            self.Page6_Timer.stop()
            self.Page6Counter = 0
            self.Page6_Widget.NotingInfo.hide()
            self.Page6_Widget.PredictionTime.hide()
            if ErrorNumber == 0:
                QMessageBox.critical(self, "Label File Error", "The label file does not contain exactly one label!", QMessageBox.Yes, QMessageBox.Yes)
                return
            elif ErrorNumber == 1:
                QMessageBox.critical(self, "Prediction Error", "Error occurs in the regression process!", QMessageBox.Yes, QMessageBox.Yes)
                return
            elif ErrorNumber == 2:
                QMessageBox.critical(self, "Import Error", "The uploaded algorithm file cannot be imported!", QMessageBox.Yes, QMessageBox.Yes)
                return
            elif ErrorNumber == 3:
                QMessageBox.critical(self, "Prediction Error", "Error occurs in the prediction process!", QMessageBox.Yes, QMessageBox.Yes)
                return
        data1 = np.round(data1, 3)
        data2 = np.round(data2, 3)
        self.EnablePage6Buttons()
        self.Page6_Timer.stop()
        self.Page6Counter = 0
        self.Page6_Widget.NotingInfo.hide()
        self.Page6_Widget.PredictionTime.hide()
        if self.SCIS_Enabled and self.USER_Enabled:
            self.Page6_Widget.Page6Plot.plot(data1, data2, self.PredictionTask)
        elif self.USER_Enabled and not self.SCIS_Enabled:
            self.Page6_Widget.Page6Plot.plot([data1[0]], [data2[0]], self.PredictionTask, 'USER')
        elif not self.USER_Enabled and self.SCIS_Enabled:
            self.Page6_Widget.Page6Plot.plot([data1[1]], [data2[1]], self.PredictionTask, 'SCIS')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('fig/zju.png'))
    DEMO_MODE = False
    main = Imputation_System(DEMO_MODE)
    main.showMaximized()
    sys.exit(app.exec_())
    