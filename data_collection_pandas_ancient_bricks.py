import pyautogui as pag
import time, sys, os
import numpy as np
import threading, queue
import mss
import mss.tools
import pandas as pd
import tensorflow as tf


class Auto_Play_Ancient_bricks:
    def __init__(self):
        self.kill_num = 0
        self.now_ball = 0
        self.prev_ball = 0
        self.prev_2_ball = 0
        self.prev_3_ball = 0
        self.bat_pos = None
        self.first_run = True
        self.start_thread = True
        self.new_batx = None
        self.df_row_index = 0
        self.df_col_index = 0
        self.q1 = queue.Queue()
        self.q2 = queue.Queue()
        self.q3 = queue.Queue()
        self.df1 = pd.DataFrame(columns=['x0', 'y0', 'x1', 'y1','hit_x', 'hit_y'])

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
        bat = pag.locate('bat.png',"sc.png", confidence=0.7) # here is the error
        print(pag.center(bat))
        print(time.time()-began)


    def take_screenshot(self, scName):
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
            output = scName
            sct_img = sct.grab(monitor)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)

    def get_ball_pos(self, q1, q3, scName):
        while True:
            self.take_screenshot(scName)
            try:
                ball = pag.locate('ball.png',scName, confidence=0.6)
            except:
                try:
                    ball = pag.locate('ball_ice.png',scName, confidence=0.61)
                except Exception as f:
                    try:
                        ball = pag.locate('close.png',scName, confidence=0.81)
                        q3.put("kill")
                        sys.exit()
                        return
                    except:
                        try:
                            ball = pag.locate('next.png',scName, confidence=0.81)
                            q3.put("kill")
                            sys.exit()
                            return
                        except:
                            continue

            ball_pos = pag.center(ball)
            if not q1.empty():
                q1.get()
            q1.put(ball_pos)



##problem: move the bat contiguoulsy with

    def move_bat(self, bat_pos, new_batx, now_ball, prev_ball, q2):
        if 63 <=new_batx <=299:
            pag.dragTo(new_batx, bat_pos.y,0.11, button="left")

        elif 13 <= new_batx <= 63:
            pag.dragTo(63, bat_pos.y,0.11, button="left")
            #if the new pos is outside the area move bat the edge where
            #the ball is moving
        elif 349>= new_batx >= 299:
            pag.dragTo(299, bat_pos.y,0.11, button="left")
        if not q2.empty():
            q2.get()
        q2.put(False)

    def append_to_df(self):
        #collect coords of the ball keep on adding to the csv with pandas
        # record and go to the next line when ball comes to tbe bottom
        #x and y are intentionally reversed to match  y = mx+b
        if 370 < self.now_ball.y.item() < 410:
            self.df1.at[self.df_row_index, 'x0'] = self.prev_ball.x.item()
            self.df1.at[self.df_row_index, 'y0'] = self.prev_ball.y.item()
            self.df1.at[self.df_row_index, 'x1'] = self.now_ball.x.item()
            self.df1.at[self.df_row_index, 'y1'] = self.now_ball.y.item()
        elif 478 < self.now_ball.y.item() < 482:
            self.df1.at[self.df_row_index, 'hit_x'] = self.now_ball.x.item()
            self.df1.at[self.df_row_index, 'hit_y'] = self.now_ball.y.item()
            self.df_row_index+=1
            if self.df_row_index%2 == 0:
                self.df1.to_csv('ml_data.csv', index=False)
                print('df saved at row:' +str(self.df_row_index))



    def ball_direction(self):

        if self.start_thread:
            if os.path.exists('ml_data.csv'):
                self.df1 = pd.read_csv('ml_data.csv')
                self.df_row_index = len(self.df1.index)
            self.start_thread=False
            self.q2.put(False)
            print("thread began")
            thread1 = threading.Thread(target=self.get_ball_pos,\
                            args=(self.q1, self.q3, "sc_ball_1.png"))
            thread1.daemon = True                            # Daemonize thread
            thread1.start()

        if not self.first_run:
            self.prev_ball = self.now_ball

        #ball pos is updated here
        self.now_ball = self.q1.get()
        if self.first_run:
            self.prev_ball = self.now_ball
            self.first_run = False
        #if the ball is going right and down
        if self.now_ball.x <= self.prev_ball.x and self.now_ball.y > self.prev_ball.y:
            self.append_to_df() # threading didnt work . mose movement shaky

        #if the ball is going left and down
        elif self.now_ball.x >= self.prev_ball.x and self.now_ball.y > self.prev_ball.y:
            self.append_to_df()
    #ball pos is updated here
        # print(self.now_ball)

        if not self.q3.empty():
            sys.exit()




if __name__ == "__main__":
    # series_shot()

    run = Auto_Play_Ancient_bricks()
    # run.test_bat_position()

    while True:
        run.ball_direction()
