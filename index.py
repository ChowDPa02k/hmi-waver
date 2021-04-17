import numpy as np
# import scipy
import serial
# import time, datetime
# import binascii
import multiprocessing
import pyaudio
import sys
from tabulate import tabulate
# from matplotlib import pyplot as plt
# from apscheduler.schedulers import background
from multiprocessing import Process, Queue, Pool, Manager
import pyfftw

window = [
    [],
    []
]
peak_record = [0, 0]
peak_position = np.linspace(72, 742, 68).astype('int')
peak_position = np.append(peak_position, 742)
freq_seperate = [1,  3,  7, 14, 29,  58, 117, 233, 466, 931, 1857]
amp_scale = 100 / (np.log10(1000) - np.log10(0.025))
np.set_printoptions(linewidth=400)
# commands = []

def arrayRMS(window):
    # got list ranged in [-1, 1]
    return np.sqrt(np.mean(window ** 2))

def arrayPEAK(window):
    # got list ranged in [-1, 1]
    return np.max(np.abs(window))

def pcm2float(byte, dtype='float32'):
    """
    Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    byte : bytes from stream.read()
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.frombuffer(byte, dtype='<i2',).reshape(-1, 2)
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

def sampler(in_data, volume, queue):
    global pool, window
    wave = pcm2float(in_data).reshape(2,-1)
    window[0].append(wave[0]) 
    window[1].append(wave[1])
    # cmds = command_generator_level(volume, window)
    if (len(window[0]) % 2) == 0:
        pool.apply_async(command_generator_level, args=(volume, [np.hstack(window[0][-2:]), np.hstack(window[1][-2:])], queue,))

    if len(window[0]) >= 4:
        pool.apply_async(command_generator_spectrum, args=(window, queue,))
        window = [[],[]]

def command_generator_level(volume, window, queue):
    global peak_record
    commands = []
    vol_scale = 100. / float(volume)
    # print(window)
    buffer_l = window[0]
    buffer_r = window[1]

    buffer_l = np.array(buffer_l)
    buffer_r = np.array(buffer_r)
    # calculate RMS
    rms_l = arrayRMS(buffer_l)
    rms_r = arrayRMS(buffer_r)
    rms_l_pct = int(rms_l * vol_scale * 100)
    rms_r_pct = int(rms_r * vol_scale * 100)
    # RMS to level
    commands.append('b[1].val=%i' % (rms_l_pct if rms_l_pct < 100 else 100))
    commands.append('b[2].val=%i' % (rms_r_pct if rms_r_pct < 100 else 100))
    # peak point
    peak_l = 68 if (int(arrayPEAK(buffer_l) * 68 * vol_scale) >= 68) else int(arrayPEAK(buffer_l) * 68 * vol_scale)
    peak_r = 68 if (int(arrayPEAK(buffer_r) * 68 * vol_scale) >= 68) else int(arrayPEAK(buffer_r) * 68 * vol_scale)
    commands.append('b[3].x=%i' % peak_position[peak_l])
    commands.append('b[4].x=%i' % peak_position[peak_r])
    # peak sign
    if rms_l_pct >= 99:
        if peak_record[0] == 0:
            commands.append('vis 5,1')
        peak_record[0] = 1
    else:
        if peak_record[0] == 1:
            commands.append('vis 5,0')
        peak_record[0] = 0
    if rms_r_pct >= 99:
        if peak_record[1] == 0:
            commands.append('vis 6,1')
        peak_record[1] = 1
    else:
        if peak_record[1] == 1:
            commands.append('vis 6,0')
        peak_record[1] = 0

    print('\rVOL\t', '|' * int(np.mean([rms_l_pct, rms_r_pct]) * 0.4), ' ' * int(40 - 0.4 * np.mean([rms_l_pct, rms_r_pct])), end='')
    queue.put(commands)
    return commands

def command_generator_spectrum(window, queue):
    global freq_seperate, amp_scale

    commands = ['ref stop']
    left = np.hstack(window[0])
    right = np.hstack(window[1])
    left_spectrum = np.abs(pyfftw.interfaces.numpy_fft.fft(left))
    right_spectrum = np.abs(pyfftw.interfaces.numpy_fft.fft(right))
    # print(np.max(left_spectrum), np.max(right_spectrum))
    left_spectrum_seperated = []
    right_spectrum_seperated = []

    for i in range(10):
        rms_fl = np.log10(arrayRMS(left_spectrum[freq_seperate[i] : freq_seperate[i+1]]))
        rms_fr = np.log10(arrayRMS(right_spectrum[freq_seperate[i] : freq_seperate[i+1]]))
        left_spectrum_seperated.append(rms_fl)
        right_spectrum_seperated.append(rms_fr)
    left_spectrum_seperated = np.array(left_spectrum_seperated)
    right_spectrum_seperated = np.array(right_spectrum_seperated)

    # print(np.min(left_spectrum_seperated), np.max(left_spectrum_seperated), '\n', np.min(right_spectrum_seperated), np.max(right_spectrum_seperated), '\n')
    # print(np.max(left_spectrum_seperated), np.max(right_spectrum_seperated))

    left_val = (left_spectrum_seperated + 1) * amp_scale
    right_val = (right_spectrum_seperated  + 1) * amp_scale
    left_val[left_val>100] = 100
    right_val[right_val>100] = 100
    left_val[left_val<0] = 0
    right_val[right_val<0] = 0
    # print(left_val, '\n', right_val)

    for i in range(10):
        commands.append('b[%i].val=%i' % (16-i, left_val[i]))
        commands.append('b[%i].val=%i' % (i+17, right_val[i]))
    commands.append('ref star')
    queue.put(commands)
    # debug
    # print(commands)
    return 0

# def send(port, content):
#     cmd = binascii.hexlify(content.encode('utf-8')).decode('utf-8')
#     cmd = bytes.fromhex(cmd+'ff ff ff')
#     port.write(cmd)

def serial_sender(port, queue):
    device = serial.Serial(port, 115200, timeout=1)
    print('Serial', device.name, 'opened\n')

    # def send(device, content):
    #     cmd = binascii.hexlify(content.encode('utf-8')).decode('utf-8')
    #     cmd = bytes.fromhex(cmd+'ff ff ff')
    #     device.write(bytes(content, encoding='utf-8') + b'\xff\xff\xff')
    
    device.write(bytes('baud=512000', encoding='utf-8') + b'\xff\xff\xff')
    device.close()
    device = serial.Serial(port, 512000, timeout=1)  
    device.write(bytes('sendxy=0', encoding='utf-8') + b'\xff\xff\xff')
    device.write(bytes('bkcmd=0', encoding='utf-8') + b'\xff\xff\xff')
    
    while True:
        # if not queue.empty():
        commands = queue.get()
        # print(commands)
        for item in commands:
            try:
                device.write(bytes(item, encoding='utf-8') + b'\xff\xff\xff')
                # print(item, end=' ')
            except Exception as exc:
                sys.stderr.write(exc)
            # print(device.read())

if __name__ == '__main__':

    SYSVOL = 15
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    BUFFER = 1024
    DEBUG = 0
    SERIAL = 'COM3'

    p = pyaudio.PyAudio()

    # list audio devices
    audio_devices = []
    for i in range(p.get_device_count()):
        device = p.get_device_info_by_index(i)
        audio_devices.append([device['index'], device['name'], device['defaultSampleRate'], device['maxInputChannels'], device['maxOutputChannels']])
    print()
    sys.stdout.write(tabulate(audio_devices, headers=['Index', 'Device Name', 'Sample Rate', 'Input Channels', 'Output Channels']))
    print('\n')
    if DEBUG:
        exit()
    
    # open queue
    manager = Manager()
    q = manager.Queue(maxsize=0)
    pool = Pool(5)
    # open stream
    stream = p.open(rate=RATE, channels=CHANNELS, format=FORMAT, frames_per_buffer=BUFFER,
                    input=True, input_device_index=1)
    # init sender
    pool.apply_async(serial_sender, args=(SERIAL, q,))

    try:
        while True:
            sampler(stream.read(BUFFER), SYSVOL, q)
    except KeyboardInterrupt:
        stream.close()
        pool.close()

