import serial
import binascii

def openport(device):
    port = serial.Serial(device, 115200, timeout=1)
    return port

def send(port, content):
    cmd = binascii.hexlify(content.encode('utf-8')).decode('utf-8')
    cmd = bytes.fromhex(cmd+'ff ff ff')
    port.write(cmd)