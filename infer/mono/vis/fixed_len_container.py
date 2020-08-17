#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

import numpy as np
from collections import deque
from multiprocessing import Process, Queue, Lock


class FixLengthDeque(object):
    def __init__(self, max_len):
        self.queue = deque()
        self.max_len = max_len

    def append(self, x):
        if self.queue.__len__() >= self.max_len:
            self.queue.popleft()

        self.queue.append(x)

    def __getitem__(self, index):
        assert index < self.max_len
        tmp_list = list(self.queue)
        return tmp_list[index]

    def __len__(self):
        return self.queue.__len__()

    def mean(self):
        return np.mean(np.array(list(self.queue)), axis=0)

    def is_full(self):
        return len(self) >= self.max_len


class FixlengthQueueThread(object):
    def __init__(self, max_len):
        self.queue = Queue()
        self.max_len = max_len
        self.last_item = None
        self.lock = Lock()

    def put(self, x):
        with self.lock:  # make sure only one put can do clean job
            if len(self) > self.max_len:
                while len(self) > 0:
                    try:
                        # clean queue
                        # if other process just got the very last item between our >0 check and here
                        # will not hang by nowait
                        self.queue.get_nowait()
                    except:
                        pass

        self.queue.put(x)

    def get(self, resuse_last=False):
        if not resuse_last:
            return self.queue.get()

        # if not self.queue.empty():  # FIXME, empty is not the same as qsize?
        if len(self) > 0 or self.last_item is None:
            self.last_item = self.queue.get()
        # else -> now empty, and last_item is already assigned value, return it is okay

        return self.last_item

    def __len__(self):
        return self.queue.qsize()

    def empty(self):
        return len(self) == 0
        # return self.queue.empty()


if __name__ == '__main__':
    d = FixLengthDeque(3)
    d.append(1)
    d.append(2)
    d.append(3)
    print(len(d))
    print(d.__len__())
    print(d.is_full())
    print(d[-1])

    d.append([1, 2])
    d.append([3, 4])
    d.append([5, 6])
    print()
    print(len(d))
    print(d[0])
    print(d[1])
    print(d[2])
    print(d.mean())

    q = FixlengthQueueThread(3)
    q.put(1)
    q.put(2)
    q.put(3)
    q.put(4)
    q.put(5)

    for i in range(100):
        print(i, q.get(resuse_last=True))
        # print(i, q.get(resuse_last=False))
