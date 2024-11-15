import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QGridLayout, QSplitter
from urllib.request import urlopen
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

import time
from PyQt5.QtCore import Qt
import random
import cv2
import numpy as np



# PyQt5 화면 띄우기 (영상과 버튼 만드는 과정)
class App(QMainWindow):
    ip = '192.168.137.150'    
    def __init__(self):
        super().__init__()
        self.stream = urlopen('http://' + App.ip +':81/stream')
        self.buffer = b""
        urlopen('http://' + App.ip + "/action?go=speed80")
        self.initUI()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)  
        self.timer.start(1) 


        # OpenCV 얼굴 인식 모델 로딩
        self.face = cv2.CascadeClassifier("models\haarcascade_frontalface_default.xml")
        self.face_active = False  # 얼굴 인식 활성화 상태 변수 True 면 작동중 False 면 비작동중 기본상태를 비작동중으로 설정하였다.
        
        # 라인트레이싱 기능 기본값 비활성화
        self.line_drive_active = False
        self.initUI()
        '''
        QLabel: 실시간 카메라 영상을 표시하는 부분입니다. 비디오 스트림을 받아서 QLabel에 그려줍니다.
        QPushButton: 다양한 제어 버튼들(예: "Speed 40", "Forward", "Backward")을 만들어서 사용자가 버튼을 클릭하면 차량이 동작하게 만듭니다.
        QVBoxLayout 및 QHBoxLayout: 버튼들을 세로, 가로로 배치하는 데 사용됩니다.
        '''


    def initUI(self):

        widget = QWidget() 
        # 비디오 스트리밍과 라인 추적을 표시할 QLabel 생성
        self.video_label = QLabel(self)
        self.video_label.setGeometry(0, 0, 800, 600)
        
        # QLabel로 제목 표시
        self.title = QLabel("강영수의 AI CAR CONTROL", self)
        self.title.setAlignment(Qt.AlignCenter)
        # hbox 레이아웃을 사용을 안할경우 self.title.setGeometry(x좌표, y좌표, 가로길이, 세로길이)
        font = self.title.font()
        font.setPointSize(35)
        font.setBold(True)
        self.title.setFont(font)

        # 라인 추적 결과를 표시할 QLabel 생성
        self.line_label = QLabel(self)
        self.line_label.setGeometry(0, 0, 800, 600)

        # 타이머 설정하여 일정 간격으로 프레임 업데이트
        # update_frame 함수는 비디오 스트리밍을 처리하는 함수입니다. 이 함수가 매번 호출될 때마다 새로운 카메라 프레임을 받아오고, 그 프레임에서 얼굴을 인식할 수 있도록 해줍니다.


        '''
        버튼 생성
        '''
        # 자동차가 움직이는 속도를 조절하는 버튼을 만들거야!
        # 속도 40
        btn_speed40 = QPushButton('Speed 40', self)   
        btn_speed40.resize(100, 50)
        ## 버튼을 누름(작동 후) 때면 동시에 멈춤
        btn_speed40.pressed.connect(self.speed40)
        btn_speed40.released.connect(self.stop)

        # 속도 50
        btn_speed50 = QPushButton('Speed 50', self)    
        btn_speed50.resize(100, 50) 
        btn_speed50.pressed.connect(self.speed50)
        btn_speed50.released.connect(self.stop)

        # 속도 60
        btn_speed60 = QPushButton('Speed 60', self)
        btn_speed60.resize(100, 50) 
        btn_speed60.pressed.connect(self.speed60)
        btn_speed60.released.connect(self.stop)

        # 속도 80
        btn_speed80 = QPushButton('Speed 80', self)
        btn_speed80.resize(100, 50) 
        btn_speed80.pressed.connect(self.speed80)
        btn_speed80.released.connect(self.stop)

        # 속도 100
        btn_speed100 = QPushButton('Speed 100', self)
        btn_speed100.resize(100, 50) 
        btn_speed100.pressed.connect(self.speed100)
        btn_speed100.released.connect(self.stop)        
        

        # 자동차가 움직이는 방향전환 키를 추가할거야~!        
        # 전진
        btn_forward = QPushButton('Forward = W', self)
        btn_forward.resize(100, 50) 
        btn_forward.pressed.connect(self.forward)
        btn_forward.released.connect(self.stop)
        


        # 왼쪽돌기
        btn_turn_left = QPushButton("Turn_left", self) 
        btn_turn_left.resize(100, 50)
        btn_turn_left.pressed.connect(self.turn_left)
        btn_turn_left.released.connect(self.stop) 

        # 왼쪽
        btn_left = QPushButton("Left = A", self)           
        btn_left.resize(100, 50)
        btn_left.pressed.connect(self.left)
        btn_left.released.connect(self.stop)

        # 멈춤
        btn_stop = QPushButton("Stop", self)           
        btn_stop.resize(100, 50)
        btn_stop.pressed.connect(self.stop)
        btn_stop.released.connect(self.stop)


        # 오른쪽
        btn_right = QPushButton("Right = D", self)         
        btn_right.resize(100, 50)
        btn_right.pressed.connect(self.right)
        btn_right.released.connect(self.stop)

        
        # 오른쪽 돌기
        btn_turn_right = QPushButton("Trun_right", self) 
        btn_turn_right.resize(100, 50)
        btn_turn_right.pressed.connect(self.turn_right)
        btn_turn_right.released.connect(self.stop)
        
        # 후진
        btn_backward = QPushButton('Backwdrd = S', self)   
        btn_backward.resize(100, 50) 
        btn_backward.pressed.connect(self.backward)
        btn_backward.released.connect(self.stop)

        # Haar 얼굴 검출기능을 추가하자
        btn_harr = QPushButton('Harr', self)   
        btn_harr.resize(100, 50) 
        btn_harr.clicked.connect(self.haar)   # 얼굴 인식을 시작        
        
        # 라인 트레이싱 자율주행 기능을 추가하자~!
        btn_line_drive = QPushButton("Line", self)
        btn_line_drive.resize(100,50)
        btn_line_drive.clicked.connect(self.line_drive) # 라인 트레이싱 자율주행 시작
        

        # 레이아웃 설정
        hbox1 = QHBoxLayout()  # 가로 방향 레이아웃
        hbox1.addStretch(2)  # 왼쪽 여백
        for button in [btn_speed40, btn_speed50, btn_speed60, btn_speed80, btn_speed100]:
            hbox1.addSpacing(15)
            hbox1.addWidget(button)  # 버튼 추가
            hbox1.addSpacing(15)
        hbox1.addStretch(2)  # 오른쪽 여백
        
        hbox2 = QHBoxLayout()  # 가로 방향 레이아웃
        hbox2.addStretch(2) # addStretch : 여백 추가 
        hbox2.addWidget(btn_forward) # 요소 추가 
        hbox2.addStretch(2) # addStretch : 여백 추가 


        hbox3 = QHBoxLayout()  # 가로 방향 레이아웃
        hbox3.addStretch(2) # addStretch : 여백 추가                 
        for button in [btn_turn_left, btn_left, btn_stop, btn_right, btn_turn_right]:
            hbox3.addSpacing(15)
            hbox3.addWidget(button)  # 버튼 추가
            hbox3.addSpacing(15)
        hbox3.addStretch(2) 

        hbox4 = QHBoxLayout()  # 가로 방향 레이아웃
        hbox4.addStretch(2)
        hbox4.addWidget(btn_backward)
        hbox4.addStretch(2)
        
        hbox5 = QHBoxLayout()  # 가로 방향 레이아웃
        hbox5.addStretch(1)
        hbox5.addWidget(btn_harr)
        hbox5.addStretch(6)
        hbox5.addWidget(btn_line_drive)
        hbox5.addStretch(1)
        
        vbox = QVBoxLayout(widget) # 세로 방향 레이아웃 
        vbox.addWidget(self.title) 
        vbox.addWidget(self.video_label)   # 실시간 화면 송출
        vbox.addWidget(self.line_label)    # 라인 추적 화면 송출
        vbox.addStretch(1)
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
                
                # Haar 얼굴 검출 인식기능
                if self.face_active :
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.face.detectMultiScale(gray, 1.3, 5)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 얼굴 영역에 사각형 그리기
                        # 얼굴 영역 위에 "Face" 텍스트 추가
                        cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                

                # 라인 추적 (검정색 선 추출)
                if self.line_drive_active :
                    # 아랫부분의 반만 자르기
                    height, width, _ = img.shape
                    img = img[height //2:, :]

                    # 색상 필터링으로 검정색 선 추출
                    lower_bound = np.array([0, 0, 0])       # 검정색 하한선
                    upper_bound = np.array([255, 255, 80])  # 검정색 상한선으로 좁혀짐
                    mask = cv2.inRange(img, lower_bound, upper_bound)

                    # 무게중심 계산
                    M = cv2.moments(mask)  # 모멘트 계산
                    if M["m00"] > 1000:  # 선을 찾았을 때 (임계값을 두어 작은 선은 무시)
                        cX = int(M["m10"] / M["m00"])  # x 좌표 (중심)
                        cY = int(M["m01"] / M["m00"])  # y 좌표 (중심)
                    else:
                        cX, cY = 0, 0  # 선이 없을 때
                    center_offset = img.shape[1] // 2 - cX  # 이미지 중앙에서 선 중심까지의 오차 계산

                    # 녹색점 화면의 중앙을 가리킴
                    cv2.circle(img, (cX, cY),10, (0,255,0),-1)

                    # 라인 추적 로직 (중앙에서 얼마나 벗어났는지에 따라 로봇을 제어)
                    if center_offset > 10:  # 오른쪽으로 많이 벗어나면 오른쪽으로
                        print("오른쪽으로 이동")
                        urlopen("http://" + App.ip + "/action?go=right")
                    elif center_offset < -10:  # 왼쪽으로 많이 벗어나면 왼쪽으로
                        print("왼쪽으로 이동")
                        urlopen("http://" + App.ip + "/action?go=left")
                    else:  # 중앙에 가까우면 직진
                        print("직진")
                        urlopen("http://" + App.ip + "/action?go=forward")
                    
                    cv2.imshow("mask", mask)  # 처리된 mask를 시각화

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
    
    def closeEvent(self, event):
        event.accept()
    
    """키가 눌렸을 때 동작"""
    def keyPressEvent(self, event):
        # print("pressed!", ord("q"))
        # print(event.key())
        # Qt.Key_3

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
        
        elif event.key() == Qt.Key_M:  # 자유주행은  M 으로 지정하겠다
            self.line_drive()  # 자유주행기능 활성화/비활성화
        


            '''
            위에 동작이 키입력을 통하여 움직이는거 까지 확인을햿어

                    if event.key() == Qt.Key_W:  # W 키는 앞으로
                        self.forward()

                    elif event.key() == 68:  # D 키는 오른쪽
                        self.right()
            '''

    # 동작 기능별 함수를 정의하였다.
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
        
    def right(self) :
        urlopen('http://' + App.ip + "/action?go=right")

    def turn_right(self) :
        urlopen('http://' + App.ip + "/action?go=turn_right")

    def backward(self) :
        urlopen('http://' + App.ip + "/action?go=backward")
    
    # 멈춤
    def stop(self) :
        urlopen('http://' + App.ip + "/action?go=stop")


    def haar(self):
        """얼굴 인식 기능을 켜거나 끄는 함수"""
        self.face_active = not self.face_active  # 얼굴 인식 상태를 토글
        status = "enabled" if self.face_active else "disabled"  # 활성화/비활성화 상태
        print(f"Face detection {status}")  # 상태를 콘솔에 출력

        # 얼굴 인식 상태에 따라 로봇에 명령을 보낼 수 있습니다.
        if self.face_active:
        # 얼굴 인식 활성화에 관련된 코드 추가 (예: 카메라를 활성화하거나 얼굴 추적 시작)
            pass
        else:
        # 얼굴 인식 비활성화에 관련된 코드 추가 (예: 얼굴 추적 중지)
            pass

    def line_drive(self):
        urlopen('http://' + App.ip + "/action?go=stop")
        """라인 트레이싱 자율주행 기능을 켜거나 끄는 함수"""
        self.line_drive_active = not self.line_drive_active  # 자율주행 상태를 토글
        status = "자율주행 활성화중" if self.line_drive_active else "자율주행 비활성화중"
        print(f"Line drive {status}")  # 상태 출력

if __name__ == '__main__':
    print(sys.argv)
    app = QApplication(sys.argv)
    view = App()
    sys.exit(app.exec_())
