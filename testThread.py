import threading
import random
import time

class MyThread(threading.Thread):
    availableTables=['A','B','C','D','E']

    def __init__(self,threadName,semaphore):
        self.interval =random.randrange(1,6)
        self.semaphore =semaphore
        threading.Thread.__init__(self,name=threadName)

    def run(self):
        self.semaphore.acquire()
        #acquire a semaphore
        table = MyThread.availableTables.pop()
        print "%s entered;seated at table %s." %(self.getName(),table)
        time.sleep(self.interval)

        #free a table
        print "%s exiting,freeing table %s." %(self.getName(),table)
        MyThread.availableTables.append(table)

        self.semaphore.release()

mySemaphore = threading.Semaphore(len(MyThread.availableTables))

def Test():
    threads=[]

    for i in range(1,10):
        threads.append(MyThread("thread"+str(i),mySemaphore))

    for i in range(len(threads)):
        threads[i].start()

if __name__ == '__main__':
    Test()