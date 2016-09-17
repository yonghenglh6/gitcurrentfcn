import threading
import random
import time
import Queue


class KCFTracker(threading.Thread):
    def __init__(self, thread_name,callback):
        self.semaphore = threading.Semaphore(0)
        threading.Thread.__init__(self, name=thread_name)
        self.data = Queue.Queue()
        self.running = False
        self.callback=callback;

    def run(self):
        self.running = True
        while self.running:
            frame = self.data.get(block=True)
            if not self.running:
                break
            print('dealing Frame:' + str(frame))
            self.callback(frame);

    def put_frame(self, frame):
        self.data.put(frame)

    def stop(self):
        self.running = False
        self.data.put(None)

    def init_param(self):
        pass


def show_frame(frame):
    print('return:'+str(frame));


if __name__ == '__main__':
    tracker1 = KCFTracker('tracker1',show_frame)
    tracker2 = KCFTracker('tracker2',show_frame)
    tracker1.start();
    tracker2.start();
    tracker1.put_frame(13)
    tracker2.put_frame(17)
    tracker1.stop()
    tracker2.stop()
