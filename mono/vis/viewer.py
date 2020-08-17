#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

import numpy as np
import cv2

import OpenGL.GL as gl
import pangolin

from multiprocessing import Process, Queue
from .fixed_len_container import FixlengthQueueThread

def RotZ(angle):
    R = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    R[:2, :2] = np.array([[c, -s], [s, c]]).reshape((2,2))
    return R

def RotY(angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32).reshape((3, 3))
    return R

def convert_prediction_boxes(prediction):
    prediction[:, 1] -= prediction[:, -2] / 2.
    # prediction[:, 1] *= -1
    n = prediction.shape[0]
    poses = np.zeros((n, 4, 4))
    dims = np.zeros((n, 3))
    for i in range(n):
        c, s = np.cos(prediction[i, -1]), np.sin(prediction[i, -1])
        poses[i, :3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32).reshape((3, 3))
        poses[i, :3, -1] = np.array(prediction[i, :3])
        poses[i, 3, 3] = 1
        dims[i, :] = np.array(prediction[i, 3:6])
    return poses, dims


class MapViewer(object):
    def __init__(self):
        q_size = 10
        self.q_active_bg = FixlengthQueueThread(max_len=q_size)
        self.q_active_box = FixlengthQueueThread(max_len=q_size)
        self.q_active_img = FixlengthQueueThread(max_len=q_size)
        self.q_active_velo = FixlengthQueueThread(max_len=q_size)

        self.view_thread = Process(target=self.view)
        self.view_thread.start()

    def update(self, tmp_pc=None, img=None, boxes=None, velo=None, refresh=False):
        if tmp_pc is not None:
            self.q_active_bg.put(tmp_pc)

        if img is not None:
            self.q_active_img.put(img)

        if boxes is not None:
            self.q_active_box.put(boxes)

        if velo is not None:
            self.q_active_velo.put(velo)

    # def stop(self):
    #     self.update(refresh=True)
    #     self.view_thread.join()
    #
    #     qtype = type(Queue())
    #     for x in self.__dict__.values():
    #         if isinstance(x, qtype):
    #             while not x.empty():
    #                 _ = x.get()
    #     print('viewer stopped')

    def view(self):
        pangolin.CreateWindowAndBind('Main', 1280, 960)
        gl.glEnable(gl.GL_DEPTH_TEST)
        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(1080, 720, 1250, 1250, 614, 350, 0.2, 100),
            pangolin.ModelViewLookAt(1, 1, -10, 1, 1, 0, pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(scam)

        # Create Interactive View in window
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
        dcam.SetHandler(handler)

        # width = 890
        # height = 500
        # dimg_monitor = pangolin.Display('monitor')
        # dimg_monitor.SetBounds(0.8, 0.99, 0.8, 0.99, width / height)
        #
        # dimg_monitor.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)
        # texture_moniotr = pangolin.GlTexture(width, height, gl.GL_RGB, False,
        #                                      0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        last_car_points = None
        last_bg = None
        last_monitor = None
        last_box = None
        last_velo = None

        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)

            gl.glPointSize(3)
            gl.glColor3f(1.0, 0.0, 0.0)

            gl.glLineWidth(5)
            gl.glColor(1.0, 0.0, 0.0)
            pangolin.DrawLine(np.array([[0, 0, 0], [10, 0, 0]]))
            gl.glColor(0.0, 1.0, 0.0)
            pangolin.DrawLine(np.array([[0, 0, 0], [0, 10, 0]]))
            gl.glColor(0.0, 0.0, 1.0)
            pangolin.DrawLine(np.array([[0, 0, 0], [0, 0, 10]]))

            gl.glLineWidth(2)
            for i in range(10, 80, 5):
                pangolin.DrawLine(np.array([[-10, 0, i], [10, 0, i]]))
            for i in range(-10, 10, 5):
                pangolin.DrawLine(np.array([[i, 0, 80], [i, 0, 0]]))

            if not self.q_active_bg.empty():
                gl.glPointSize(5)
                bg = self.q_active_bg.get()

                pangolin.DrawPoints(bg[:, :3], bg[:, 3:])
                last_bg = bg.copy()
            else:
                if last_bg is not None:
                    pangolin.DrawPoints(last_bg[:, :3], last_bg[:, 3:])

            if not self.q_active_velo.empty():
                gl.glPointSize(2)
                velo = self.q_active_velo.get()

                pangolin.DrawPoints(velo[:, :3], velo[:, 3:])
                last_velo = velo.copy()

            else:
                if last_velo is not None:
                    gl.glPointSize(2)
                    pangolin.DrawPoints(last_velo[:, :3], last_velo[:, 3:])

            if not self.q_active_box.empty():
                box = self.q_active_box.get()
                last_box = box_view = box
            else:
                box_view = last_box

            if box_view is not None:
                box_view_ = box_view.copy()
                gl.glColor3f(1.0, 0.0, 1.0)
                gl.glLineWidth(3)

                poses, dims = convert_prediction_boxes(box_view_)

                pangolin.DrawBoxes(poses, dims)

                # if not self.q_active_img.empty():
                #     monitor_img = self.q_active_img.get()
                #     monitor_img = monitor_img[::-1, :, ::-1]
                # texture_moniotr.Upload(monitor_img, gl.GL_RGB,
                #                        gl.GL_UNSIGNED_BYTE)
                # dimg_monitor.Activate()
                # gl.glColor3f(1.0, 1.0, 1.0)
                # texture_moniotr.RenderToViewport()
                # last_monitor = monitor_img
            # else:
            #     if last_monitor is not None:
            #         texture_moniotr.Upload(last_monitor, gl.GL_RGB,
            #                                gl.GL_UNSIGNED_BYTE)
            #         dimg_monitor.Activate()
            #         gl.glColor3f(1.0, 1.0, 1.0)
            #         texture_moniotr.RenderToViewport()

            pangolin.FinishFrame()
