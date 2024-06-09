import sys
import pcl
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer


class PointCloudViewer(QMainWindow):
    def __init__(self, filename):
        super().__init__()
        self.point_cloud = pcl.load(filename)
        self.points = np.asarray(self.point_cloud)
        self.current_index = 0
        
        self.initUI()
        
    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Point Cloud Viewer')
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.label = PointCloudLabel()
        self.layout.addWidget(self.label)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showNextPoint)
        self.timer.start(100)  # adjust speed here
        
    def showNextPoint(self):
        if self.current_index < len(self.points):
            point = self.points[self.current_index]
            self.label.addPoint(point)
            self.current_index += 1
        else:
            self.timer.stop()
        
class PointCloudLabel(QWidget):
    def __init__(self):
        super().__init__()
        self.image = QImage(800, 600, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        
    def addPoint(self, point):
        painter = QPainter(self.image)
        painter.setPen(QPen(Qt.black, 1))
        x, y = int(point[0]), int(point[1])
        painter.drawPoint(x, y)
        painter.end()
        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = PointCloudViewer("KITTI_sequence_2.ply")
    viewer.show()
    sys.exit(app.exec_())

