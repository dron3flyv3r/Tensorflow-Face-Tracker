import os
import time
import uuid
import cv2

img_path = os.path.join("data", "images")
num_img = 30

cam = cv2.VideoCapture(0)
for imgnum in range(num_img):
    print(f"Capturing image {imgnum}")
    ret, frame = cam.read()
    cv2.imwrite(os.path.join(img_path, "img" + str(uuid.uuid1()) + ".jpg"), frame)
    cv2.imshow("frame", frame)
    time.sleep(.5)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()