from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

__author__ = "Atinderpal Singh"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "atinderpalap@gmail.com"


class ImageViewer:
    ''' Basic image viewer class to show an image with zoom and pan functionaities.
        Requirement: Qt's Qlabel widget name where the image will be drawn/displayed.
    '''

    def __init__(self, qlabel):
        self.qlabel_image = qlabel  # widget/window name where image is displayed (I'm usiing qlabel)
        self.qimage_scaled = QImage()  # scaled image to fit to the size of qlabel_image
        self.qpixmap = QPixmap()  # qpixmap to fill the qlabel_image
        self.qimage=None
        self.zoomX = 1  # zoom factor w.r.t size of qlabel_image
        self.position = [0, 0]  # position of top left corner of qimage_label w.r.t. qimage_scaled
        self.leftButton = None
        self.startpoint = None

        self.qlabel_image.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.__connectEvents()

    def __connectEvents(self):
        # Mouse events
        self.qlabel_image.mousePressEvent = self.mousePressAction
        self.qlabel_image.mouseMoveEvent = self.mouseMoveAction
        self.qlabel_image.mouseReleaseEvent = self.mouseReleaseAction
        self.qlabel_image.wheelEvent = self.wheelEvent

    def onResize(self):
        ''' things to do when qlabel_image is resized '''
        self.qpixmap = QPixmap(self.qlabel_image.size())
        self.qpixmap.fill(QtCore.Qt.gray)
        self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX,
                                                self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()

    def loadImage(self, imagePath):
        ''' To load and display new image.'''
        self.qimage = QImage(imagePath)
        # print(self.qimage.width(), self.qimage.height())
        self.qpixmap = QPixmap(self.qlabel_image.size())
        if not self.qimage.isNull():
            # reset Zoom factor and Pan position
            self.zoomX = 1
            self.position = [0, 0]
            self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width(), self.qlabel_image.height(),
                                                    QtCore.Qt.KeepAspectRatio)
            self.update()
        else:
            self.statusbar.showMessage('Cannot open this image! Try another one.', 5000)

    def update(self):
        ''' This function actually draws the scaled image to the qlabel_image.
            It will be repeatedly called when zooming or panning.
            So, I tried to include only the necessary operations required just for these tasks.
        '''
        if not self.qimage_scaled.isNull():
            # check if position is within limits to prevent unbounded panning.
            px, py = self.position
            px = px if (px <= self.qimage_scaled.width() - self.qlabel_image.width()) else (
                    self.qimage_scaled.width() - self.qlabel_image.width())
            py = py if (py <= self.qimage_scaled.height() - self.qlabel_image.height()) else (
                    self.qimage_scaled.height() - self.qlabel_image.height())
            px = px if (px >= 0) else 0
            py = py if (py >= 0) else 0
            self.position = (px, py)

            if self.zoomX == 1:
                self.qpixmap.fill(QtCore.Qt.white)

            # the act of painting the qpixamp
            painter = QPainter()
            painter.begin(self.qpixmap)
            painter.drawImage(QtCore.QPoint(0, 0), self.qimage_scaled,
                              QtCore.QRect(self.position[0], self.position[1], self.qlabel_image.width(),
                                           self.qlabel_image.height()))
            if self.leftButton:
                painter.drawLine(self.leftButton, self.pressed)
            painter.end()

            self.qlabel_image.setPixmap(self.qpixmap)
        else:
            pass

    def mousePressAction(self, QMouseEvent):
        if self.qimage:
            x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
            self.pressed = QMouseEvent.pos()  # starting point of drag vector
            if QMouseEvent.buttons() & Qt.RightButton:
                self.anchor = self.position  # save the pan position when panning starts
            elif QMouseEvent.buttons() & Qt.LeftButton:
                self.leftButton = QMouseEvent.pos()

    def mouseMoveAction(self, QMouseEvent):
        if self.qimage:
            x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
            if QMouseEvent.buttons() & Qt.RightButton and self.pressed:
                dx, dy = x - self.pressed.x(), y - self.pressed.y()  # calculate the drag vector
                self.position = self.anchor[0] - dx, self.anchor[1] - dy  # update pan position using drag vector
                self.update()  # show the image with udated pan position
            elif QMouseEvent.buttons() & Qt.LeftButton and self.leftButton:
                self.leftButton = QMouseEvent.pos()
                self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX,
                                                        self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
                self.update()

    def mouseReleaseAction(self, QMouseEvent):
        if self.qimage:
            self.startpoint = self.pressed
            self.pressed = None  # clear the starting point of drag vector
            self.leftButton = None
            x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
            x = x if (x <= self.qlabel_image.width()) else (self.qlabel_image.width())
            y = y if (y <= self.qlabel_image.height()) else (self.qlabel_image.height())
            x = x if (x >= 0) else 0
            y = y if (y >= 0) else 0
            self.released = (x, y)

    def calculatePix(self):
        scale = min(self.qlabel_image.width() * self.zoomX / self.qimage.width(),
                    self.qlabel_image.height() * self.zoomX / self.qimage.height())
        x1, y1 = (self.startpoint.x() + self.position[0]) / scale, (self.startpoint.y() + self.position[1]) / scale

        dx, dy = (self.released[0] - self.startpoint.x()) / scale, (self.released[1] - self.startpoint.y()) / scale
        x2, y2 =x1+dx, y1+dy
        return round(x1), round(y1), round(x2), round(y2)

    def wheelEvent(self, event):
        if self.qimage:
            if event.angleDelta().y() > 0:
                self.zoomPlus()
            elif event.angleDelta().y() < 0:
                self.zoomMinus()

    def zoomPlus(self):
        if self.zoomX < 10:
            self.zoomX += 1
            px, py = self.position
            px += self.qlabel_image.width() / 2
            py += self.qlabel_image.height() / 2
            self.position = (px, py)
            self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX,
                                                    self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
            self.update()

    def zoomMinus(self):
        if self.zoomX > 1:
            self.zoomX -= 1
            px, py = self.position
            px -= self.qlabel_image.width() / 2
            py -= self.qlabel_image.height() / 2
            self.position = (px, py)
            self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX,
                                                    self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
            self.update()

    def resetZoom(self):
        self.zoomX = 1
        self.position = [0, 0]
        self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX,
                                                self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()
