import cv2
import time
from loguru import logger

cam = cv2.VideoCapture(3, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cam.set(cv2.CAP_PROP_FPS, 30)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
# cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# 输出相机信息
print("Camera info:")
print('Width:', cam.get(cv2.CAP_PROP_FRAME_WIDTH))
print('Height:', cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('FPS:', cam.get(cv2.CAP_PROP_FPS))

while True:

    start = time.time()
    ret, frame = cam.read()
    end = time.time()

    cost_time = (end - start) * 1000
    if cost_time > 30:
        logger.error(f"时间大于30ms: {cost_time:.2f} ms")
    else:
        print(f"时间小于30ms: {cost_time:.2f} ms")

    cv2.imshow("Frame", frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
