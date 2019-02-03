"""
Used to prepare custom dataset
"""
import numpy as np
import cv2
import os

#makes directories for storing digits
def make_dirs():
    if not os.path.exists("./Dataset"):
        os.makedirs("Dataset")
    for i in range(10):
        if not os.path.exists(f"./Dataset/{i}"):
            os.makedirs("./Dataset/{i}")
    return True


if __name__ == "__main__":
    make_dirs()
    cap = cv2.VideoCapture(0)
    digits_captured = [0]*10

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (100,100), (300,300), (20,34,255),2)
        img = frame[100:300,100:300]
        img = cv2.resize(img, (100,100))
        cv2.imshow("image", frame)    
        key = cv2.waitKey(1)
        if key>=48 and key<58:
            digit = key-48
            digits_captured[digit]+=1
            print(digits_captured)
            cv2.imwrite(f"./Dataset/{digit}/{digits_captured[digit]}.jpg", img)
        elif key==27:
            break

    cap.release()
    cv2.destroyAllWindows()
