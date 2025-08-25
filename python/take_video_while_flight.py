from djitellopy import Tello
import cv2
import time

# Tello 객체 생성 및 연결
tello = Tello()
tello.connect()

# 배터리 잔량 확인 (안전 비행을 위해 확인하는 것이 좋습니다)
print(f"배터리 잔량: {tello.get_battery()}%")

# 스트리밍 시작
tello.streamon()
frame_read = tello.get_frame_read()

# 영상 저장을 위한 설정
# 코덱을 'mp4v'로 설정하여 .mp4 형식으로 저장합니다.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 저장될 파일명, 코덱, 초당 프레임 수(FPS), 해상도를 설정합니다.
out = cv2.VideoWriter('result/tello_flight_recording.mp4', fourcc, 30.0, (960, 720))

# --- 비행 및 촬영 시작 ---

try:
    # 1. 이륙
    print("이륙합니다...")
    tello.takeoff()
    # 이륙 후 안정화를 위해 잠시 대기합니다.
    time.sleep(2)

    # 2. 지정된 높이(2.5m)로 이동
    print("2.5미터 상공으로 이동합니다...")
    # move_up 명령어의 단위는 cm이므로 250으로 설정합니다.
    tello.move_up(250)
    # 이동 후 안정화를 위해 잠시 대기합니다.
    time.sleep(2)

    # 3. 영상 녹화
    record_duration = 10  # 녹화 시간 (초)
    start_time = time.time()
    print(f"{record_duration}초 동안 녹화를 시작합니다...")

    while (time.time() - start_time) < record_duration:
        # Tello로부터 현재 프레임(영상 조각)을 받아옵니다.
        frame = frame_read.frame

        if frame is not None:
            # 영상 프레임을 파일에 씁니다.
            out.write(frame)
            
            # 실시간으로 드론이 보는 화면을 컴퓨터에 표시합니다.
            cv2.imshow("Tello Camera", frame)
        
        # 'q' 키를 누르면 녹화를 중단하고 비상 착륙 절차를 시작합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("사용자에 의해 녹화가 중단되었습니다.")
            break
            
    print("녹화가 완료되었습니다.")

finally:
    # --- 정리 및 착륙 ---
    # 4. 착륙
    print("착륙합니다...")
    tello.land()

    # 5. 사용한 자원 해제
    # 비디오 파일 저장을 완료합니다.
    out.release()
    # 열려있는 모든 OpenCV 창을 닫습니다.
    cv2.destroyAllWindows()
    # 비디오 스트리밍을 끕니다.
    tello.streamoff()
    # Tello와의 연결을 종료합니다.
    tello.end()
    print("모든 과정이 완료되었습니다.")

