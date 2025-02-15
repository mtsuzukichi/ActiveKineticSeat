
import serial

ser = serial.Serial("COM18",921600)

ser.write(bytearray([ord('p'), 0,0]))
