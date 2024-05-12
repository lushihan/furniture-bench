import serial
import threading
import time
import numpy as np

class FsrReader:
    def __init__(self, port, num_channels, baud_rate=230400, buffer_size=100):
        self.ser = serial.Serial(port, baud_rate, timeout=1)
        self.ser.flushInput()
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.fsr_values = np.zeros(4, dtype=int)
        # self.fsr_values_buffer = [deque(maxlen=buffer_size) for _ in range(num_channels)]
        self.fsr_values_buffer = np.zeros((4, buffer_size), dtype=int)
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.read_from_port)
        self.thread.daemon = True  # Ensure thread exits when main program does
        self.thread.start()

    def read_from_port(self):
        while True:
            try:
                fsr_values_str = self.ser.readline().decode('utf-8').strip()
                if fsr_values_str:
                    fsr_values = [int(value) for value in fsr_values_str.split(',')]
                    if len(fsr_values) == self.num_channels:
                        with self.lock:
                            self.fsr_values = fsr_values
                            self.fsr_values_buffer = np.roll(self.fsr_values_buffer, -1, axis=1)
                            self.fsr_values_buffer[:, -1] = fsr_values
                            # for i, value in enumerate(fsr_values):
                                # self.fsr_values_buffer[i].append(value)
            except (UnicodeDecodeError, ValueError):
                pass

    def get_latest_value(self):
        with self.lock:
            return self.fsr_values

    def get_all_values(self):
        with self.lock:
            return self.fsr_values_buffer
            # return np.array(list(self.fsr_values_buffer)) # Return a list of all values in the buffer

    def close(self):
        self.ser.close()  # Close the serial port

def read_detect_force_array(forceArray: FsrReader): # output size=(4, 100, 1)
    return np.expand_dims(forceArray.get_all_values(), axis=2)

if __name__ == '__main__':
    FSRreader = FsrReader('/dev/ttyACM0', 4)  # Adjust the COM port as needed
    time.sleep(3)
    # print(read_detect_force_array(FSRreader))
    # time.sleep(0.1)
    # print(read_detect_force_array(FSRreader))

    while True:
    #     latest_value = reader.get_latest_value()
    #     # if latest_value is not None:
    #     print("Latest Sensor Value:", latest_value)
    #     # Optionally, print all values in the buffer
        # all_values = FSRreader.get_all_values()
        # print("All Sensor Values:", all_values[3])
        print(read_detect_force_array(FSRreader)[3])
        time.sleep(0.1)
