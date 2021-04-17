import numpy as np
import serial
from index import send
from time import sleep

spectrum_level = 10
sample_rate = 44100
buffer_size = 4096

scale = sample_rate / buffer_size
a = np.geomspace(20, 20000, num=spectrum_level+1)
a_scale = a / scale
print(a_scale.astype('int'))

b = [np.array([1,2,2]), np.array([2,5,6])]
print(np.hstack(b))

# device = serial.Serial('COM3', 115200)
# send(device,'page 3')
# while True:
#     for x in range(20):
#         # send(device, 'ref stop')
#         for n in range(20):
#             cmd = 'j%i.val=%i' % (n, int(np.random.random()*100))
#             send(device,cmd)
        # send(device, 'ref star')
    # sleep(0.1)