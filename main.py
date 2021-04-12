import numpy as np
import scipy
import requests
import hmi
from queue import PriorityQueue
import multiprocessing
from multiprocessing import Queue
from time import sleep
import binascii
# CHANNEL L
#           X 50 Y 16 W 414 H 98
#         R
#           X 50 Y 120 W 414 H 98
# LEVEL L
#         X 50 Y 234
#       R
#         X 50 Y 272
#       Pixel Width 8px per level [0, 49]
# PEAK 
#      L X 450 Y 234
#      R X 450 Y 272

# ACCENT RGB(67, 217, 150)
# PEAK RGB(255, 251, 1)
# PEAK_WARNING RGB(245, 70, 70)

SERIAL = 'COM3'

WINDOW_SIZE = 128

PEAK_POSITION = np.linspace(50, 442, 50).astype('int')
LEVEL_WIDTH = (np.arange(0, 400, 8) - 350).astype('int')

PEAK_WARN_L = 'vis pkl,1'
PEAK_WARN_R = 'vis pkr,1'
PEAK_DWARN_L = 'vis pkl,0'
PEAK_DWARN_R = 'vis pkr,0'

window = [
    [],
    []
]

def calculateRMS(window):
    # got list ranged in [0, 1]
    return np.sqrt(np.mean(window ** 2))

def calculatePEAK(window):
    # got list ranged in [0, 1]
    return np.max(np.abs(window))

def generateWaveSerialCommand(data):
    # got data array [L, R] ranged in [0, 1]
    height = [98 * data[0], 98 * data[1]]
    # print(height)
    cmds = []
    cmds.append('add 1,0,%i' % int(height[0]))
    cmds.append('add 2,0,%i' % int(height[1]))
    return cmds

def generateWaveLevelCommand(data):
    cmds = []

    windowLinearRMSL = calculateRMS(np.array(data[0]))
    windowLinearRMSR = calculateRMS(np.array(data[1]))
    windowPeakL = calculatePEAK(np.array(data[0]))
    windowPeakR = calculatePEAK(np.array(data[1]))
    
    rmsL = int(50 * windowLinearRMSL)
    rmsR = int(50 * windowLinearRMSR)

    peakL = 49 if int(50 * windowPeakL) == 50 else int(50 * windowPeakL)
    peakR = 49 if int(50 * windowPeakR) == 50 else int(50 * windowPeakR)

    if rmsL == 50:
        cmds.append(PEAK_WARN_L)
        rmsL = 49
    else:
        cmds.append(PEAK_DWARN_L)

    if rmsR == 50:
        cmds.append(PEAK_WARN_R)
        rmsR = 49
    else:
        cmds.append(PEAK_DWARN_R)

    cmds.append('ll.x=%i' % LEVEL_WIDTH[rmsL])
    cmds.append('lr.x=%i' % LEVEL_WIDTH[rmsR])
    # cmds.append('pl.x=%i' % PEAK_POSITION[peakL])
    # cmds.append('pr.x=%i' % PEAK_POSITION[peakR])
    return cmds

def normal(data):
    return np.linalg.norm(data)

def queueHandler(queue, device):
    device = hmi.openport(SERIAL)
    def send(port, content):
        cmd = binascii.hexlify(content.encode('utf-8')).decode('utf-8')
        cmd = bytes.fromhex(cmd+'ff ff ff')
        port.write(cmd)

    send(device, 'page 3')

    while True:
        if not queue.empty():
            try:
                msg = queue.get()
                send(device, msg)
                print(msg)
            except Exception as exc:
                print(exc)
                # break

def sendSerialCommandList(commands):
    for item in commands:
        q.put(item)
    return len(commands)

# def sendSerialCommand(priority, command):
#     q.put(priority, command)
def waveHandler(audio):
    global window
    window[0].append(float(audio[0]))
    window[1].append(float(audio[1]))
    levelWave = generateWaveSerialCommand(audio)
    # print(window)
    sendSerialCommandList(levelWave)
    # print(len(window[0]))
    if len(window[0]) == WINDOW_SIZE:
        levelCmds = generateWaveLevelCommand(window)
        print(levelCmds)
        result = sendSerialCommandList(levelCmds)
        window = [[],[]]


if __name__ == '__main__':
    device = 'COM3'
    q = Queue(maxsize=0)

    # simulate sinwave
    t = np.linspace(0, 2, 1000)
    x = 2 * np.pi * t

    # signal = [
    #     normal(np.sin(x) * np.sin(3 * x)),
    #     normal(np.cos(x) * np.cos(3 * x))
    # ]

    # print(signal)

    handler = multiprocessing.Process(target=queueHandler, args=(q, device))
    handler.start()

    # hmi.send(device, 'page 3')

    while True:
        for i in x:
            sample = [0.5 * (np.sin(i) * np.sin(3 * i)) + 0.5, 0.5 * (np.cos(i) * np.cos(3 * i)) + 0.5]
            waveHandler(sample)
            # print(sample)
            sleep(0.01)