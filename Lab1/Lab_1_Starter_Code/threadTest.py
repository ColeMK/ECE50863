import threading
import time

def task1(testDict):
    while True:
        time.sleep(.3)
        print(f"beginning of task 1: {testDict}")
        testDict["Value"] = "live"
        print(f"TASK1 Value: {testDict}")
def task2(testDict):
    while True:
        time.sleep(2)
        print(f"beginning of task 2: {testDict}")
        testDict["Value"] = "dead"
        print(f"TASK2 Value: {testDict}")

testDict = {"Value": "dead"}

thread1 = threading.Thread(target=task1, args=(testDict,))
thread2 = threading.Thread(target=task2, args=(testDict,))

thread1.start()
thread2.start()
