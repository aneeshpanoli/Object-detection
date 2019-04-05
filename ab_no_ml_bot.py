import pyautogui as pag
import time, sys, os
import numpy as np
import threading, queue
import mss
import mss.tools
import tensorflow as tf
import pandas as pd


class AutoPlayAncientBricks:
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
        self.q1 = queue.Queue()
        self.q2 = queue.Queue()
        self.q3 = queue.Queue()
        self.model = tf.keras.models.load_model('ancient_bricks.model')
        self.tf_graph = tf.get_default_graph()
        self.df1 = pd.DataFrame(columns=['x0', 'y0', 'x1', 'y1', 'hit_x'])



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
                ball = pag.locate('ball.png', scName, confidence=0.615)
            except:
                try:
                    ball = pag.locate('ball_ice.png', scName, confidence=0.61)
                except:
                    try:
                        ball = pag.locate('ball_red.png', scName, confidence=0.61)
                    except:
                        try:
                            ball = pag.locate('ball_fire.png', scName, confidence=0.55)
                        except:
                            try:
                                ball = pag.locate('close.png', scName, confidence=0.81)
                                q3.put("kill")
                                sys.exit()
                                return
                            except:
                                try:
                                    ball = pag.locate('next.png', scName, confidence=0.81)
                                    q3.put("kill")
                                    sys.exit()
                                    return
                                except:
                                    continue

            ball_pos = pag.center(ball)
            if not q1.empty():
                q1.get()
            q1.put(ball_pos)

    def get_single_batShot(self):
        self.take_screenshot("sc_bat.png")
        try:
            bat = pag.locate('bat.png', "sc_bat.png", confidence=0.8)
        except:
            return
        self.bat_pos = pag.center(bat)

    # problem: move the bat contiguously with

    def move_bat(self, bat_pos, new_batx, now_ball, prev_ball, q2):
        if 63 <= new_batx <= 299:
            pag.dragTo(new_batx, bat_pos.y, 0.101, button="left")

        elif 13 <= new_batx <= 63:
            pag.dragTo(63, bat_pos.y, 0.101, button="left")
            # if the new pos is outside the area move bat the edge where
            # the ball is moving
        elif 349 >= new_batx >= 299:
            pag.dragTo(299, bat_pos.y, 0.101, button="left")
        if not q2.empty():
            q2.get()
        q2.put(False)

    def compute_pos_and_move_bat(self):
        # x and y are intentionally reversed to match  y = mx+b
        if 370 < self.now_ball.y.item() < 470:
            x = [self.prev_ball.y.item(), self.now_ball.y.item()]
            y = [self.prev_ball.x.item(), self.now_ball.x.item()]
            self.df1.at[self.df_row_index, 'x0'], self.df1.at[self.df_row_index, 'x1'] = x
            self.df1.at[self.df_row_index, 'y0'], self.df1.at[self.df_row_index, 'y1'] = y

            z = np.polyfit(x, y, 1)
            m, b = z
            self.new_batx = int(self.bat_pos.y * m + b)
            self.df1.at[self.df_row_index, 'hit_x'] = self.new_batx

            # print(self.bat_pos)
            wait = self.q2.get()
            if not wait:
                self.q2.put(True)
                thread2 = threading.Thread(target=self.move_bat, \
                                           args=(self.bat_pos, self.new_batx, self.now_ball, self.prev_ball, self.q2))
                thread2.daemon = True  # Daemonize thread
                thread2.start()
                self.df_row_index += 1
                if self.df_row_index % 5 == 0:
                    self.df1 = self.df1.dropna()
                    self.df1.to_csv('ml_data.csv', index=False)
                    print('df saved at row:' + str(self.df_row_index))

    def predict_pos_and_move_bat(self):
        # x and y are intentionally reversed to match  y = mx+b
        if 370 < self.now_ball.y.item() < 470:
            column_names = ['x0', 'y0', 'x1', 'y1']
            df_pred = pd.DataFrame(columns=column_names)
            df_pred.at[0, 'y0'] = self.prev_ball.x.item()
            df_pred.at[0, 'x0'] = self.prev_ball.y.item()
            df_pred.at[0, 'y1'] = self.now_ball.x.item()
            df_pred.at[0, 'x1'] = self.now_ball.y.item()
            df_pred = df_pred.apply(pd.to_numeric)
            pred_data_norm = tf.keras.utils.normalize(df_pred, axis=1)
            self.new_batx = self.predict_pos(pred_data_norm)
            wait = self.q2.get()
            if not wait:
                self.q2.put(True)
                thread2 = threading.Thread(target=self.move_bat, \
                                           args=(self.bat_pos, self.new_batx, self.now_ball, self.prev_ball, self.q2))
                thread2.daemon = True  # Daemonize thread
                thread2.start()

    def append_to_df(self):
        # collect coords of the ball keep on adding to the csv with pandas
        # record and go to the next line when ball comes to tbe bottom
        # x and y are intentionally reversed to match  y = mx+b
        if 370 < self.now_ball.y.item() < 470:
            self.df1.at[self.df_row_index, 'x0'] = self.prev_ball.x.item()
            self.df1.at[self.df_row_index, 'y0'] = self.prev_ball.y.item()
            self.df1.at[self.df_row_index, 'x1'] = self.now_ball.x.item()
            self.df1.at[self.df_row_index, 'y1'] = self.now_ball.y.item()
        elif 475 < self.now_ball.y.item() < 490:
            self.df1.at[self.df_row_index, 'hit_x'] = self.now_ball.x.item()
            self.df1.at[self.df_row_index, 'hit_y'] = self.now_ball.y.item()
            self.df_row_index += 1
            if self.df_row_index % 2 == 0:
                self.df1.to_csv('ml_data.csv', index=False)
                print('df saved at row:' + str(self.df_row_index))

    def ball_direction(self):
        if self.start_thread:
            if os.path.exists('ml_data.csv'):
                self.df1 = pd.read_csv('ml_data.csv')
                self.df_row_index = len(self.df1.index)
            self.start_thread = False
            self.q2.put(False)
            print("thread began")
            thread1 = threading.Thread(target=self.get_ball_pos, \
                                       args=(self.q1, self.q3, "sc_ball_1.png"))
            thread1.daemon = True  # Daemonize thread
            thread1.start()
        if self.first_run:
            self.get_single_batShot()
            # offset 20px is intentional - identifying only once side of the bat
            pag.click(self.bat_pos.x + 20, self.bat_pos.y)
        if not self.first_run:
            self.prev_ball = self.now_ball

        # ball pos is updated here
        self.now_ball = self.q1.get()
        if self.first_run:
            self.prev_ball = self.now_ball
            self.first_run = False
        # if the ball is going right and down
        if self.now_ball.x <= self.prev_ball.x and self.now_ball.y > self.prev_ball.y:
            self.compute_pos_and_move_bat()  # threading didnt work . mose movement shaky
            # self.predict_pos_and_move_bat()
            # self.append_to_df()

        # if the ball is going left and down
        elif self.now_ball.x >= self.prev_ball.x and self.now_ball.y > self.prev_ball.y:
            self.compute_pos_and_move_bat()
            # self.predict_pos_and_move_bat()
            # self.append_to_df()

        if not self.q3.empty():
            sys.exit()

    def predict_pos(self, predict_data):
        # prepare prediction data as def fname(arg)
        # load model

        with self.tf_graph.as_default():
            predictions = self.model.predict(predict_data)
        pred = predictions[0][0]

        return pred


if __name__ == "__main__":
    # series_shot()
    print(os.getcwd())
    run = AutoPlayAncientBricks()
    # run.test_bat_position()

    while True:
        run.ball_direction()
