#  import ml algo
# take screenshot
# send to ml algo
# get the coords

import pyautogui as pag
import time, sys, os
import numpy as np
import threading, queue
import mss
import mss.tools
from ball_detection import BallDetection
from PIL import Image


class AutoPlayAncientBricks:
    def __init__(self):
        self.cwd = os.getcwd()
        self.kill_num = 0
        self.now_ball = 0
        self.prev_ball = 0
        self.now_ball_x = 0
        self.prev_ball_x = 0
        self.now_ball_y = 0
        self.prev_ball_y = 0
        self.bat_pos = None
        self.first_run = True
        self.start_thread = True
        self.new_batx = None
        self.df_row_index = 0
        self.run_deep_learn = BallDetection()
        self.q1 = queue.Queue()
        self.q2 = queue.Queue()
        self.q3 = queue.Queue()

    def test_bat_position(self):

        began = time.time()
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
            output = "sc.png"
            sct_img = sct.grab(monitor)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        bat = pag.locate('bat.png', "sc.png", confidence=0.7)  # here is the error
        print(pag.center(bat))
        print(time.time() - began)

    def get_bat_screenshot(self):
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
            output = "sc_bat.png"
            sct_img = sct.grab(monitor)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

    def take_screenshot(self, q1):

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
            output = self.cwd + '/content/datalab/test_image/image1.png'
            while True:
                # print("taking screenshot")
                sct_img = sct.grab(monitor)
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
                # img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                # img.save(output, 'jpeg')
                pos = self.run_deep_learn.set_up_object_detection_api()
                if not q1.empty():
                    q1.get()
                q1.put(pos)



    def get_bat_pos(self):
        self.get_bat_screenshot()
        try:
            bat = pag.locate('bat.png', "sc_bat.png", confidence=0.8)
        except:
            return
        self.bat_pos = pag.center(bat)

    # problem: move the bat contiguously with

    def move_bat(self, bat_pos, new_batx, q2):
        # print("moving bat")
        if 63 <= new_batx <= 299:
            q2.put(True)
            pag.dragTo(new_batx, bat_pos.y, 0.101, button="left")

        elif 13 <= new_batx <= 63:
            q2.put(True)
            pag.dragTo(63, bat_pos.y, 0.101, button="left")
            # if the new pos is outside the area move bat the edge where
            # the ball is moving
        elif 349 >= new_batx >= 299:
            q2.put(True)
            pag.dragTo(299, bat_pos.y, 0.101, button="left")
        if not q2.empty():
            q2.get()
        q2.put(False)




    def compute_pos_and_move_bat(self):
        # print("computing pos")
        # x and y are intentionally reversed to match  y = mx+b
        if 340 < self.now_ball_y < 470:
            x = [self.prev_ball_y, self.now_ball_y]
            y = [self.prev_ball_x, self.now_ball_x]

            z = np.polyfit(x, y, 1)
            m, b = z
            self.new_batx = int(self.bat_pos.y * m + b)

            # print(self.bat_pos)
            wait = self.q2.get()
            if not wait:
                thread2 = threading.Thread(target=self.move_bat, \
                                           args=(self.bat_pos, self.new_batx, self.q2))
                thread2.daemon = True  # Daemonize thread
                thread2.start()

    def ball_direction(self):
        if self.start_thread:
            self.start_thread = False
            self.q2.put(False)
            print("thread began")
            thread1 = threading.Thread(target=self.take_screenshot, args=(self.q1,))
            thread1.daemon = True  # Daemonize thread
            thread1.start()

        if self.first_run:
            self.get_bat_pos()
        # # offset 20px is intentional - identifying only once side of the bat
            pag.click(self.bat_pos.x + 20, self.bat_pos.y)
        if not self.first_run:
            self.prev_ball = self.now_ball
        # ball pos is updated here
        self.now_ball = self.q1.get()
        if self.first_run:
            self.prev_ball = self.now_ball
            self.first_run = False



        # print("printing self nowball")
        # print(self.now_ball)
        # print(self.prev_ball)

        self.now_ball_x, self.now_ball_y = self.now_ball
        self.prev_ball_x, self.prev_ball_y = self.prev_ball
        # if the ball is going right and down
        if self.now_ball_x <= self.prev_ball_x and self.now_ball_y > self.prev_ball_y:
            # print("goin down right")
            self.compute_pos_and_move_bat()  # threading didnt work . mose movement shaky

        # if the ball is going left and down
        elif self.now_ball_x >= self.prev_ball_x and self.now_ball_y > self.prev_ball_y:
            # print("going down left")
            self.compute_pos_and_move_bat()


if __name__ == "__main__":
    # series_shot()
    run = AutoPlayAncientBricks()
    # run.test_bat_position()

    while True:
        run.ball_direction()
