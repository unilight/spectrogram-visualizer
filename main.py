import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtWidgets,QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget, QGraphicsView, QGraphicsScene

from PyQt5.QtWidgets import QTextEdit, QAction, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QGridLayout, QPushButton, QListWidget, QLineEdit

import sys

import librosa
import numpy as np

FFT_SIZE = 2048
SHIFT_MS = 5
LENGTH_MS = 25

def get_spectrograms(sound_file, fs=22050, fft_size=FFT_SIZE):
    # Loading sound file
    y, _ = librosa.load(sound_file) # or set sr to hp.sr.

    # stft. D: (1+n_fft//2, T)
    linear = librosa.stft(y=y,
                     n_fft=FFT_SIZE,
                     hop_length=int(fs*0.001*SHIFT_MS),
                     win_length=int(fs*0.001*LENGTH_MS),
                     )

    # magnitude spectrogram
    mag = np.abs(linear) #(1+n_fft/2, T)

    # to decibel
    mag = np.log10(np.maximum(1e-5, mag))

    return np.transpose(mag.astype(np.float32))

# 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplotlib的关键
class Figure_Canvas(FigureCanvas):
    def __init__(self, sgram, parent=None, width=6.4, height=4.8, dpi=100):

        # 创建一个Figure，注意：该Figure为matplotlib下的figure，不是matplotlib.pyplot下面的figure
        #fig = Figure(figsize=(self.w//100, self.h//100), dpi=100)
        fig = Figure(figsize=(width, height), dpi=dpi)

        # initialize parent class
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        # call the `add_subplot` method of `figure`, similar to `subplot` of `matplotlib.pyplot`
        self.axes = fig.add_subplot(111)
        self.axes.imshow(np.flipud(sgram.T), interpolation='none')
        self.axes.tick_params(labelsize=4)
        #self.axes.axis('off')
        self.axes.get_yaxis().set_visible(False)
        fig.subplots_adjust(left=0.89, right=0.9, top=0.9, bottom=0.89)
        fig.tight_layout()

class Visualizer(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 设置窗口标题
        self.setWindowTitle('Spectrogram Visualizer')
        self.setFixedSize(800, 600)

        # Set menus
        openFile = QAction(QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        self.layout_widget = LayoutWidget(self)
        self.setCentralWidget(self.layout_widget)

        # set open button action
        self.layout_widget.open_btn.clicked.connect(self.showDialog)

        # set filelist addAction
        self.layout_widget.filelist.currentItemChanged.connect(self.on_filelist_change)

        #
        self.w = self.layout_widget.graphicview.frameGeometry().width() / 100 - 0.2
        self.h = self.layout_widget.graphicview.frameGeometry().height() / 100 - 0.2

        self.graphicscene = QGraphicsScene()
        self.layout_widget.graphicview.setScene(self.graphicscene)
        self.layout_widget.graphicview.show()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '.')
        print('Reading: ', fname[0])

        if fname[0]:
            # print on the filepath_edit
            self.layout_widget.filepath_edit.setText(fname[0])

            # add to the list
            self.layout_widget.filelist.addItem(fname[0])

            # draw
            self.draw_sgram(fname[0])

    def on_filelist_change(self, curr, prev):
        self.draw_sgram(curr.text())

    def draw_sgram(self, fname):
        # Perform stft
        lin_sgram = get_spectrograms(fname)

        #实例化一个FigureCanvas
        dr = Figure_Canvas(lin_sgram, width=self.w, height=self.h)
        self.graphicscene.addWidget(dr)


class LayoutWidget(QWidget):

     def __init__(self, parent):
        super(LayoutWidget, self).__init__(parent)

        # Main Layout
        grid = QGridLayout()
        grid.setSpacing(10)

        # Open button
        open_btn = QPushButton('Open')
        grid.addWidget(open_btn, 0, 0)

        # File Path
        filepath_edit = QLineEdit()
        filepath_edit.setDisabled(True)
        grid.addWidget(filepath_edit, 0, 1,)

        # List
        filelist = QListWidget()
        grid.addWidget(filelist, 1, 0)

        # spectrogram
        graphicview = QGraphicsView()
        graphicview.setObjectName("graphicview")
        grid.addWidget(graphicview, 1, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 6)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 5)

        self.setLayout(grid)

        self.grid = grid
        self.open_btn = open_btn
        self.filepath_edit = filepath_edit
        self.filelist = filelist
        self.graphicview = graphicview


if __name__ == '__main__':

    app = QApplication(sys.argv)

    visualizer = Visualizer()
    visualizer.show()
    app.exec_()
