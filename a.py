import numpy as np
from PIL import ImageGrab
import cv2
import sys
import vgamepad as vg

gamepad = vg.VX360Gamepad()


vertices = np.array(
    [[120, 540], [150, 350], [12, 324], [360, 216], [600, 216], [960, 324], [800, 400], [800, 450], [960, 540]])


def roi(img):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


height, width = 540, 960


def extract_line_points(processed_image):
    points = []
    for y in range(height - 1, 215, -1):
        e = np.average(np.where(processed_image[y, :] == 255))
        if not np.isnan(e):
            points.append(e)
    return points


def find_direction(x_coords):
    x_coords = np.array(x_coords)
    x_coords -= 480
    x = np.average(x_coords)

    if x > 0:
        direction = 0
        acc_howmuch = (0.5 - (x + 480)*0.2 / 960)
    else:
        direction = 1
        acc_howmuch = (0.5 - (480-x)*0.2 / 960)
    if np.isnan(acc_howmuch):
        acc_howmuch = 0.2
    magnitude = x
    gmpaddir = ((magnitude + 480) * 65536 / 960) - 32768
    print(acc_howmuch)
    gamepad.right_trigger_float(value_float=acc_howmuch)
    # if np.isnan(gmpaddir):
        # gamepad.left_joystick(x_value=0, y_value=0)
    # else:
    if not np.isnan(gmpaddir):
        gamepad.left_joystick(x_value=int(gmpaddir), y_value=0)
    gamepad.update()
    return direction, magnitude


def processimage(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 50, 160])
    upper_blue = np.array([50, 255, 255])

    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(image, image, mask=mask_cleaned)

    processed_image = roi(result)
    processed_image = cv2.Canny(processed_image, 100, 200, apertureSize=3)

    return processed_image


if __name__ == "__main__":
    while (True):
        printscreen_pil = np.array(ImageGrab.grab(bbox=(0, 40, 960, 580)))
        new_edge_screen = processimage(printscreen_pil)
        lines = extract_line_points(new_edge_screen)
        with open("output.txt", "a") as txt_file:
            for line in lines:
                txt_file.write(" ".join(str(line)))
            txt_file.write("\n")
        dir, mag = find_direction(lines)
        print(dir, mag)
        #cv2.imshow('window', printscreen_pil)
        cv2.imshow('edgewindow', new_edge_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destryoyAllWindows()
            break
