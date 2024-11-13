import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QGridLayout
from urllib.request import urlopen
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

from PyQt5.QtCore import Qt
import random
import cv2
import numpy as np


# PyQt5 화면 띄우기 (영상과 버튼 만드는 과정)
class App(QMainWindow):
    
    ip = '192.168.137.153'

    def __init__(self):
        super().__init__()
        self.stream = urlopen('http://' + App.ip +':81/stream')
        self.buffer = b""
        urlopen('http://' + App.ip + "/action?go=speed80")
        self.initUI()
        
        '''
        # OpenCV 얼굴 인식 모델 로딩
        '''
        self.face = cv2.CascadeClassifier("models\haarcascade_frontalface_default.xml")
        self.face_active = False  # 얼굴 인식 활성화 상태 변수 True 면 작동중 False 면 비작동중 기본상태를 비작동중으로 설정하였다.
        
        '''
        QLabel: 실시간 카메라 영상을 표시하는 부분입니다. 비디오 스트림을 받아서 QLabel에 그려줍니다.
        QPushButton: 다양한 제어 버튼들(예: "Speed 40", "Forward", "Backward")을 만들어서 사용자가 버튼을 클릭하면 차량이 동작하게 만듭니다.
        QVBoxLayout 및 QHBoxLayout: 버튼들을 세로, 가로로 배치하는 데 사용됩니다.
        '''
    def initUI(self):
        
        widget = QWidget() 
        #QLabel로 비디오 스트리밍 출력
        self.video_label = QLabel(self)
        self.video_label.setGeometry(0, 0, 800, 600)

        # 타이머 설정하여 일정 간격으로 프레임 업데이트
        # update_frame 함수는 비디오 스트리밍을 처리하는 함수입니다. 이 함수가 매번 호출될 때마다 새로운 카메라 프레임을 받아오고, 그 프레임에서 얼굴을 인식할 수 있도록 해줍니다.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)  
        self.timer.start(5) 
        
        # 버튼생성할거야!
        btn_speed40 = QPushButton('Speed 40', self)
        btn_speed40.resize(100, 50)

        btn_speed50 = QPushButton('Speed 50', self)
        btn_speed50.resize(100, 50) 

        btn_speed60 = QPushButton('Speed 60', self)
        btn_speed60.resize(100, 50) 

        btn_speed80 = QPushButton('Speed 80', self)
        btn_speed80.resize(100, 50) 

        btn_speed100 = QPushButton('Speed 100', self)
        btn_speed100.resize(100, 50) 

        
        ## 전진. 왼쪽돌기, 왼쪽, 멈춤, 오른쪽 , 오른쪽 돌기, 후진
        btn_forward = QPushButton('forward', self)     # 전진
        btn_forward.resize(100, 50) 
        
        btn_turn_left = QPushButton("turn_left", self)   # 왼쪽돌기
        btn_turn_left.resize(100, 50)
        
        btn_left = QPushButton("left", self)           # 왼쪽
        btn_left.resize(100, 50)

        btn_stop = QPushButton("stop", self)           # 멈춤
        btn_stop.resize(100, 50)

        btn_right = QPushButton("right", self)         # 오른쪽
        btn_right.resize(100, 50)

        btn_turn_right = QPushButton("trun_right", self) #오른쪽돌기 
        btn_turn_right.resize(100, 50)

        btn_backward = QPushButton('BACKWARD', self)   # 후진
        btn_backward.resize(100, 50) 

        # Haar 얼굴 검출기능을 추가하자
        btn_harr = QPushButton('harr', self)   
        btn_harr .resize(100, 50) 
        

        ## 버튼을 누름(작동 후) 때면 동시에 멈춤
        btn_speed40.pressed.connect(self.speed40)
        btn_speed40.released.connect(self.stop)

        btn_speed50.pressed.connect(self.speed50)
        btn_speed50.released.connect(self.stop)

        btn_speed60.pressed.connect(self.speed60)
        btn_speed60.released.connect(self.stop)

        btn_speed80.pressed.connect(self.speed80)
        btn_speed80.released.connect(self.stop)

        btn_speed100.pressed.connect(self.speed100)
        btn_speed100.released.connect(self.stop)

        btn_forward.pressed.connect(self.forward)
        btn_forward.released.connect(self.stop)

        btn_turn_left.pressed.connect(self.turn_left)
        btn_turn_left.released.connect(self.stop)

        btn_left.pressed.connect(self.left)
        btn_left.released.connect(self.stop)

        btn_stop.pressed.connect(self.stop)
        btn_stop.released.connect(self.stop)

        btn_right.pressed.connect(self.right)
        btn_right.released.connect(self.stop)

        btn_turn_right.pressed.connect(self.turn_right)
        btn_turn_right.released.connect(self.stop)

        btn_backward.pressed.connect(self.backward)
        btn_backward.released.connect(self.stop)

        btn_harr.clicked.connect(self.haar)   # 얼굴 인식을 시작
       

        # 레이아웃 설정
        hbox1 = QHBoxLayout()  # 가로 방향 레이아웃
        hbox1.addStretch(2)  # 왼쪽 여백
        for button in [btn_speed40, btn_speed50, btn_speed60, btn_speed80, btn_speed100]:
            hbox1.addSpacing(15)
            hbox1.addWidget(button)  # 버튼 추가
            hbox1.addSpacing(15)
        hbox1.addStretch(2)  # 오른쪽 여백
        
        hbox2 = QHBoxLayout()  # 가로 방향 레이아웃
        hbox2.addStretch(1) # addStretch : 여백 추가 
        hbox2.addWidget(btn_forward) # 요소 추가 
        hbox2.addStretch(1) # addStretch : 여백 추가 


        hbox3 = QHBoxLayout()  # 가로 방향 레이아웃
        hbox3.addStretch(2) # addStretch : 여백 추가                 
        for button in [btn_turn_left, btn_left, btn_stop, btn_right, btn_turn_right]:
            hbox3.addSpacing(15)
            hbox3.addWidget(button)  # 버튼 추가
            hbox3.addSpacing(15)
        hbox3.addStretch(2) 

        hbox4 = QHBoxLayout()  # 가로 방향 레이아웃
        hbox4.addStretch(1)
        hbox4.addWidget(btn_backward)
        hbox4.addStretch(1)
        
        hbox5 = QHBoxLayout()  # 가로 방향 레이아웃
        hbox5.addStretch(1)
        hbox5.addWidget(btn_harr)
        hbox5.addStretch(1)
        


        vbox = QVBoxLayout(widget) # 세로 방향 레이아웃 
        vbox.addWidget(self.video_label) 
        vbox.addLayout(hbox1) 
        vbox.addLayout(hbox2) 
        vbox.addLayout(hbox3) 
        vbox.addLayout(hbox4) 
        vbox.addLayout(hbox5) 
        vbox.addStretch(1)

        # self 는 현재 MainWindow 
        # 아래 코드는 MainWindow 안에다 Widget 추가하기 코드! 
        self.setCentralWidget(widget)

        self.setWindowTitle('AI CAR CONTROL WINDOW')
        self.move(600, 400)   # 윈도우 위치
        self.resize(200, 200) # 윈도우 크기 조정
        self.show()           # 화면에 표시
    
    def update_frame(self):
        self.buffer += self.stream.read(4096)
        head = self.buffer.find(b'\xff\xd8')
        end = self.buffer.find(b'\xff\xd9')

        try: 
            if head > -1 and end > -1:
                jpg = self.buffer[head:end+2]
                self.buffer = self.buffer[end+2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                img = cv2.flip(img, -1)  # 수평 및 수직 반전 ##
                
                if self.face_active:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face.detectMultiScale(gray, 1.3, 5)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 얼굴 영역에 사각형 그리기
                        # 얼굴 영역 위에 "Face" 텍스트 추가
                        cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)



                # OpenCV의 BGR 이미지를 RGB로 변환
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # OpenCV의 이미지를 QImage로 변환
                height, width, channels = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

                # QPixmap을 QLabel에 표시
                pixmap = QPixmap.fromImage(q_image)
                self.video_label.setPixmap(pixmap)
        except Exception as e :
            print(e)
    
    def toggle_face_detection(self):
        """얼굴 인식 기능을 켜거나 끄는 함수"""
        self.face_detection_active = not self.face_detection_active
        status = "enabled" if self.face_detection_active else "disabled"
        print(f"Face detection {status}")


    def closeEvent(self, event):
        event.accept()
    
    def keyPressEvent(self, event):
        # print("pressed!", ord("q"))
        # print(event.key())
        # Qt.Key_3
        """키가 눌렸을 때 동작"""
        
        

        if event.key() == 16777216:  # ESC 는 27 = close
            self.close()
        
        elif event.key() == 87:  # W 키는 전진
            self.forward()

        elif event.key() == 65:  # A 키는 왼쪽
            self.left() 

        elif event.key() == 83:  # S키는 후진
            self.backward()

        elif event.key() == 68:  # D 키는 오른쪽
            self.right()

        elif event.key() == 81:  # Q 는 왼쪽돌기
            self.turn_left()

        elif event.key() == 69:  # E 는 오른쪽돌기
            self.turn_right()

        elif event.key() == Qt.Key_1:  # speed40
            self.speed40()
        
        elif event.key() == Qt.Key_2:  # speed50
            self.speed50()
        
        elif event.key() == Qt.Key_3:  # speed60
            self.speed60()

        elif event.key() == Qt.Key_4:  # speed80
            self.speed80()

        elif event.key() == Qt.Key_5:  # speed100
            self.speed100()
        
        elif event.key() == Qt.Key_6:  # harr
            self.haar()  # 얼굴 인식 활성화/비활성화
        

            '''
            위에 동작이 키입력을 통하여 움직이는거 까지 확인을햿어

                    if event.key() == Qt.Key_W:  # q 키는 끄기
                        self.forward()

                    elif event.key() == 68:  # D 키는 오른쪽
                        self.right()
            '''

    # 속도
    def speed40(self) :
        urlopen('http://' + App.ip + "/action?go=speed40")

    def speed50(self) :
        urlopen('http://' + App.ip + "/action?go=speed50")    

    def speed60(self) :
        urlopen('http://' + App.ip + "/action?go=speed60")

    def speed80(self) :
        urlopen('http://' + App.ip + "/action?go=speed80")
    
    def speed100(self) :
        urlopen('http://' + App.ip + "/action?go=speed100")
    
    # 방향
    def forward(self) :
        urlopen('http://' + App.ip + "/action?go=forward")

    def turn_left(self) :
        urlopen('http://' + App.ip + "/action?go=turn_left")
    
    def left(self) :
        urlopen('http://' + App.ip + "/action?go=left")

    # 멈춤
    def stop(self) :
        urlopen('http://' + App.ip + "/action?go=stop")

    def right(self) :
        urlopen('http://' + App.ip + "/action?go=right")

    def turn_right(self) :
        urlopen('http://' + App.ip + "/action?go=turn_right")

    def backward(self) :
        urlopen('http://' + App.ip + "/action?go=backward")
    
    def haar(self):
        """얼굴 인식 기능을 켜거나 끄는 함수"""
        self.face_active = not self.face_active  # 얼굴 인식 상태 토글
        status = "enabled" if self.face_active else "disabled"   # 활성화 또는 비활성화 상태를 텍스트로 표시
        print(f"Face{status}")  # 상태를 콘솔에 출력

if __name__ == '__main__':
    print(sys.argv)
    app = QApplication(sys.argv)
    view = App()
    sys.exit(app.exec_())
