import pyautogui as pag
import time
import os
import mss
import mss.tools
import cv2
from PIL import Image


class Think_or_swim:
    def __init__(self):
        self.loop_num = 1
        self.png_size = (224, 224)
        self.test_dir = 'predict'
        self.ori_dir = 'screenshot_ori'
        self.class_name = 'ball_'

    def select_next_stock(self):
        pag.click(160, 800)
        time.sleep(0.5)
        pag.press('down')
        time.sleep(4)



    def rename_file(self, class_name):
        # help to start wheere you end
        cwd = os.getcwd()
        sc_path = os.path.join(cwd, 'scrshots')
        fname = class_name + str(self.loop_num) + '.jpg'
        output = os.path.join(sc_path, fname)
        while os.path.exists(output):
            self.loop_num += 1
            fname = class_name + str(self.loop_num) + '.jpg'
            output = os.path.join(sc_path, fname)
        # output = os.path.join('screenShot', fname)
        return output

    def take_screenshot_original(self):
        with mss.mss() as sct:
            monitor_number = 3
            mon = sct.monitors[monitor_number]
            monitor = {
                "top": mon["top"] + 0,  # 100px from the top
                "left": mon["left"] + 0,  # 100px from the left
                "width": 350,
                "height": 520,
                "mon": monitor_number,
            }
            output = self.rename_file(self.class_name)
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img.save(output, 'jpeg')
            # mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
            time.sleep(5)
            print("screen captured, file name: " + output)
            # self.select_next_stock()
            # self.loop_num+=1

    def take_screenshot_learning(self):
        with mss.mss() as sct:
            monitor_number = 1
            mon = sct.monitors[monitor_number]
            monitor = {
                "top": mon["top"] + 150,  # 100px from the top
                "left": mon["left"] + 320,  # 100px from the left
                "width": 700,
                "height": 700,
                "mon": monitor_number,
            }
            output = self.rename_file(self.learn_dir)
            sct_img = sct.grab(monitor)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
            img = cv2.imread(output)
            res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(output, res)
            print("screen captured, file name: " + output)
            self.select_next_stock()
            self.loop_num += 1

    def take_one_screenshot(self):
        with mss.mss() as sct:
            monitor_number = 1
            mon = sct.monitors[monitor_number]
            monitor = {
                "top": mon["top"] + 150,  # 100px from the top
                "left": mon["left"] + 645,  # 100px from the left
                "width": 700,
                "height": 700,
                "mon": monitor_number,
            }
            output = 'predict/test.png'
            sct_img = sct.grab(monitor)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
            img = cv2.imread(output)
            res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(output, res)
            print("screen captured, file name: " + output)


if __name__ == '__main__':
    run = Think_or_swim()
    selection = input("enter 1 to analyze current chart, 2 for screenshot: ")
    print(type(selection))
    while selection != '1' and selection != '2':
        selection = input("enter 1 to analyze current chart, 2 for screenshot: ")

    if selection == "1":
        run.take_one_screenshot()
    elif selection == "2":
        input("press enter to continue")
        iter = 0
        while iter < 500:
            run.take_screenshot_original()
            # run.take_screenshot_learning()
            iter += 1
