
import pyautogui as pag
from Xlib import display
import cv2
import time, sys
import multiprocessing as mp
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import threading, queue
import mss
import mss.tools
# pag.FAILSAFE = True

class Auto_Play_Ancient_bricks_old:
    def __init__(self):
        self.exit_timer = 0
        self.ball_first_pos = 0
        self.ball_second_pos = 0
        self.ball = None
        self.bat = None
        self.bat_pos = None
        self.ballCenter = None
        self.batCenter =None
        self.first_run = True
        self.start_thread = True
        self.new_batx = None
        self.q1 = queue.Queue()
        self.q2 = queue.Queue()
        self.q3 = queue.Queue()
        self.q4 = queue.Queue()
    def draw_line(self):
        lines = [
        (50, 50, 40, 35),
        (52, 52, 42, 37),
        (38, 30, 25, 15),
        ]
        x = []
        y = []
        for x1, y1, x2, y2 in lines:
            x += [x1, x2]
            y += [y1, y2]
        z = np.polyfit(x, y, 1)
        print(z)
        # for x1, y1, x2, y2 in lines:
        #     plt.plot((x1, x2), (y1, y2), 'g')
        #
        # plt.axis([0, 60, 0, 60])
        # plt.show()


    def get_screenshot(self):

        try:
            self.ball = pag.locateOnScreen('ball.png',region=(0,0, 350,520), confidence=0.7)
            self.bat =  pag.locateOnScreen('bat.png',region=(0,0, 350,520), confidence=0.7)
            pag.click(pag.center(bat))
            pag.dragTo(pag.center(ball).x, None, 0.2, button="left")
            self.batCenter = pag.center(bat)
            self.ballCenter = pag.center(ball)
            print(self.batCenter)
            print(self.ballCenter)
            self.exit_timer =0
        except:
            self.exit_timer+=1
            if self.exit_timer > 10:
                sys.exit()
    def get_scrot(self):
        began = time.time()
        im = pag.screenshot(region=(0,0, 350,520))
        im.save('sc.png')
        print(time.time()-began)

    def bat_position(self):
        began = time.time()
        print(pag.locateCenterOnScreen('bat.png',region=(0,0, 350,520), confidence=0.7))
        print(time.time()-began)

    def mss_scrot(self):
        began = time.time()
        with mss.mss() as sct:
    # Get information of monitor 2
            monitor_number = 3
            mon = sct.monitors[monitor_number]

            # The screen part to capture
            monitor = {
                "top": mon["top"] + 0,  # 100px from the top
                "left": mon["left"] + 0,  # 100px from the left
                "width": 350,
                "height": 520,
                "mon": monitor_number,
            }
            output = "sc.png"

            # Grab the data
            sct_img = sct.grab(monitor)

            # Save to the picture file
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
            ball_pos = pag.locate('ball.png',"sc.png", confidence=0.7)
        print(time.time()-began)

    def add_scond_ball_pos(self, pos, q3):
        q3.put(pos)
        print("from second ball pos")

    def get_ball_pos(self, q1, q3):
        timer_kil = 0
        while True:
            # began = time.time()
            try:
                with mss.mss() as sct:
            # Get information of monitor 3
                    monitor_number = 3
                    mon = sct.monitors[monitor_number]

                    # The screen part to capture
                    monitor = {
                        "top": mon["top"] + 0,  # 100px from the top
                        "left": mon["left"] + 0,  # 100px from the left
                        "width": 350,
                        "height": 520,
                        "mon": monitor_number,
                    }
                    output = "sc.png"

                    # Grab the data
                    sct_img = sct.grab(monitor)

                    # Save to the picture file
                    mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
                ball = pag.locate('ball.png',"sc.png", confidence=0.7)
                # self.ball = pag.locate(name,'sc.png', confidence=0.7)
                ball_pos = pag.center(ball)
                # return_dict['first_ball_pos'] = self.ball
                # q3.put(q1.get())
                if q1.empty():
                    # fn_to_call(self.ball, q3)
                    q3.put(ball_pos)
                    print("q empty")
                else:
                    # fn_to_call(q1.get(), q3)
                    q3.put(q1.get())
                    print("q not empty")
                q1.put(ball_pos)
                timer_kil =0
            except:
                timer_kil+=1
                if timer_kil > 15:
                    sys.exit()
            # print(began-time.time())
    def get_bat_pos(self, q2):
        timer_kil = 0
        while True:
            try:
                with mss.mss() as sct:
            # Get information of monitor 3
                    monitor_number = 3
                    mon = sct.monitors[monitor_number]

                    # The screen part to capture
                    monitor = {
                        "top": mon["top"] + 0,  # 100px from the top
                        "left": mon["left"] + 0,  # 100px from the left
                        "width": 350,
                        "height": 520,
                        "mon": monitor_number,
                    }
                    output = "sc1.png"

                    # Grab the data
                    sct_img = sct.grab(monitor)

                    # Save to the picture file
                    mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
                bat = pag.locate('bat.png',"sc1.png", confidence=0.7)
                # self.ball = pag.locate(name,'sc.png', confidence=0.7)
                bat_pos = pag.center(bat)
                # bat_pos = pag.locateCenterOnScreen('bat.png',region=(0,0, 350,520), confidence=0.6)
                # self.bat = pag.locate(name,'sc.png', confidence=0.7)
                # self.bat_pos = pag.center(self.bat)
                # return_dict['bat_pos'] = self.bat
                q2.put(bat_pos)
                timer_kil = 0
            except:
                timer_kil+=1
                if timer_kil > 15:
                    sys.exit()

    async def async_task(self, return_dict):
        jobs =[]
        # p4=mp.Process(target=self.get_scrot)
        # jobs.append(p4)
        # p4.start()
        p1=mp.Process(target=self.get_ball_pos, args=(return_dict,))
        jobs.append(p1)
        p1.start()
        p3=mp.Process(target=self.get_bat_pos, args=(return_dict,))
        jobs.append(p3)
        p3.start()


        for job in jobs:
            job.join()

    def keyboard_interrupt(self, q4):
        while True:
            if keyboard.is_pressed('esc'):
                q4.put("kill")

    async def ball_direction(self):
        try:
            # manager = mp.Manager()
            # return_dict = manager.dict()
            # await asyncio.gather(self.async_task(return_dict))
            if self.start_thread:
                # self.q1.put(0)
                # await asyncio.gather(self.async_task(return_dict))
                print("thread began")
                thread1 = threading.Thread(target=self.get_ball_pos, args=(self.q1, self.q3))
                thread1.daemon = True                            # Daemonize thread
                thread1.start()
                thread2 = threading.Thread(target=self.get_bat_pos, args=(self.q2,))
                thread2.daemon = True                            # Daemonize thread
                thread2.start()
                # thread4 = threading.Thread(target=self.keyboard_interrupt, args=(self.q4,))
                # thread4.daemon = True                            # Daemonize thread
                # thread4.start()
                self.start_thread=False

            # await asyncio.gather(self.get_ball_pos('ball.png',return_dict),
            # self.get_bat_pos('bat.png',return_dict)
            # )
            # self.async_task(return_dict)
            # self.ball_first_pos = return_dict['first_ball_pos']
            # self.bat_pos = return_dict['bat_pos']


            self.ball_first_pos = self.q1.get()
            self.q1.put(self.ball_first_pos)
            self.ball_second_pos = self.q3.get()

            print(self.ball_first_pos)
            print(self.ball_second_pos)
            y_minus_y = self.ball_first_pos.y - self.ball_second_pos.y
            print(y_minus_y)
            # if y_minus_y >0:
                # print(type(self.ball_first_pos.x.item()))
                #reveresed x y becuase what we want is x
                # from the coefficients calculate ball x from bat y (506)
                #y= mx+b
            x=[]
            y=[]
            x+=[self.ball_first_pos.y.item(), self.ball_second_pos.y.item()]
            y+=[self.ball_first_pos.x.item(), self.ball_second_pos.x.item()]
            z = np.polyfit(x, y, 1)
            print(z)
            m, b = z
            self.new_batx = int(510*m+b)
            print(self.new_batx)

            # pag.moveTo(self.bat_pos.x, self.bat_pos.y, 0.5)
            if self.first_run:
                pag.click(self.bat_pos)
                time.sleep(0.1)
            if 72< self.new_batx <287:
                pag.click(self.bat_pos)
                # time.sleep(0.1)
                pag.dragTo(self.new_batx, 510,0.3, button="left")
                self.exit_timer =0
            else:
                self.exit_timer+=1

        except:
            self.exit_timer+=1
            self.first_run = True;
        if self.ball_first_pos == self.ball_second_pos:
            self.exit_timer+=1
        if self.exit_timer > 10:
            sys.exit()
            thread1.join()
            thread2.join()

if __name__ == "__main__":
    # series_shot()

    run = Auto_Play_Ancient_bricks()
    # run.bat_position()
    # run.mss_scrot()
    # run.get_scrot()
    # run.draw_line()
    while True:
        asyncio.run(run.ball_direction())
