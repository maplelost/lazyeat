import os
import sys
import threading

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from MyDetector import MyDetector, \
    show_toast  # https://github.com/maplelost/lazyeat/issues/15 线程中 import mediapipe as mp 出错

if hasattr(sys, 'frozen'):
    # pyinstaller打包成exe时，sys.argv[0]的值是exe的路径
    # os.path.dirname(sys.argv[0])可以获取exe的所在目录
    # os.chdir()可以将工作目录更改为exe的所在目录
    os.chdir(os.path.dirname(sys.argv[0]))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# config
class Config:
    show_detect_window: bool = False  # 显示检测窗口
    camera_index: int = 0  # 当前摄像头索引


CONFIG = Config()
my_detector: MyDetector = None

work_thread_lock = threading.Lock()
work_thread: threading.Thread = None
flag_work = False


@app.get("/")
def read_root():
    return "ready"


@app.get("/get_all_cameras")
def get_all_cameras() -> dict:
    # {0: '组合摄像头', 1: 'Xiaomi 12S（前置）', 2: 'Xiaomi 12S（后置）', 3: 'USB webcam'}
    from pygrabber.dshow_graph import FilterGraph

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras


def thread_init():
    global my_detector
    my_detector = MyDetector(maxHands=2)


def thread_detect():
    thread_cam = cv2.VideoCapture(CONFIG.camera_index, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 优化帧率
    thread_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 设置编码格式

    while True:
        success, img = thread_cam.read()

        if my_detector is None:
            show_toast(
                title='初始化中',
                msg='初始化中',
                duration=1
            )
            continue

        if not flag_work:
            break

        if not success:
            continue

        if CONFIG.show_detect_window:
            all_hands, img = my_detector.findHands(img, draw=True)
            if all_hands:
                my_detector.process(all_hands)
                img = my_detector.draw_mouse_move_box(img)
                cv2.imshow("Lazyeat Detect Window", img)
                cv2.waitKey(1)
        else:
            all_hands = my_detector.findHands(img, draw=False)
            if all_hands:
                my_detector.process(all_hands)
                # not CONFIG.show_detect_window 改变需要关闭窗口
                try:
                    cv2.destroyAllWindows()
                except:
                    pass

    # 结束取图，释放资源
    try:
        thread_cam.release()
        cv2.destroyAllWindows()
    except:
        pass


@app.get("/toggle_work")
def toggle_work():
    global flag_work, work_thread
    with work_thread_lock:
        if work_thread is None or not work_thread.is_alive():
            flag_work = True

            work_thread = threading.Thread(target=thread_detect)
            work_thread.daemon = True
            work_thread.start()
            return "started"
        else:
            flag_work = False
            return "stopped"


@app.post("/update_config")
def update_config(data: dict):
    from pinia_store import PINIA_STORE

    CONFIG.show_detect_window = data.get("show_window", False)

    camera_index = int(data.get("camera_index", 0))
    if camera_index != CONFIG.camera_index:
        CONFIG.camera_index = camera_index

    # 更新四个手指同时竖起发送的按键
    new_key = data.get("four_fingers_up_send")
    if new_key:
        gesture_sender = PINIA_STORE.gesture_sender
        gesture_sender.set_gesture_send(gesture_sender.four_fingers_up, new_key)


@app.get("/shutdown")
def shutdown():
    try:
        cv2.destroyAllWindows()
    except:
        pass

    import signal
    import os
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == '__main__':
    print("Initializing...")
    t_init = threading.Thread(target=thread_init, daemon=True)
    t_init.start()

    port = 62334

    print(f"Starting server at http://localhost:{port}/docs")
    uvicorn.run(app, host="127.0.0.1", port=port)
