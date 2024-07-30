import numpy as np
from PIL import ImageGrab
import cv2

vertices = np.array([[120, 540], [150, 350], [12, 324], [360, 216], [600, 216], [960, 324], [800, 400], [800, 450], [960, 540]])

image = []


def roi(img):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def processimage(image):
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    processed_image = roi(processed_image)
    return processed_image



while (True):
    printscreen_pil = np.array(ImageGrab.grab(bbox=(0, 0, 1920, 1080)))
    # printscreen_pil = processimage(printscreen_pil)
    cv2.imshow('window', printscreen_pil)
    image.append(printscreen_pil)
    # cv2.imshow('edgewindow', new_edge_screen)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destryoyAllWindows()
        break
