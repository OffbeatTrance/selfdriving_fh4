import numpy as np
from PIL import ImageGrab
import cv2
import sys
import vgamepad as vg
import math

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


def translate_log(value):
    if value or not np.isnan(value):
        multiplier = 1
        if (value < 0):
            multiplier = -1
            value = -value
        x = (np.arctan(value/120))*1000*multiplier
        normalized_value = ((x + 1000) * 65536)/2000 - 32768
        print(value, normalized_value)
        return int(normalized_value)
    else:
        return 0


def find_direction(x_coords):
    x_coords = np.array(x_coords)
    x_coords -= 480
    magnitude = np.average(x_coords)

    if magnitude > 0:
        direction = 0
    else:
        direction = 1
    gmpaddir = translate_log(magnitude)
    gamepad.right_trigger_float(value_float=0.5)
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
        # print(dir, mag)
        cv2.imshow('edgewindow', new_edge_screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destryoyAllWindows()
            break
