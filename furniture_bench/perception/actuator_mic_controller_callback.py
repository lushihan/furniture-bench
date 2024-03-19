# this script generate the emit signals to the speaker 
# and record the received signals by the mic in callback mode
# it used PyAudio, a python wrapper for Portaudio library

# import sys
import queue
import numpy
import pyaudio
import wave
import time

import librosa
import librosa.core
import librosa.display

# import alsaaudio # for audio jack volume control thru Linux alsa sound driver 
                 # (only work for Linux. Install: )

from matplotlib import pyplot as plt
from matplotlib.mlab import window_hanning,specgram
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

from scipy.signal import chirp, sweep_poly
from scipy.io.wavfile import write, read

from scipy.fft import rfft, rfftfreq

import pdb

# numpy.set_printoptions(threshold=sys.maxsize)


class ActiveAcousticSensor(object):
    def __init__(
        self,
        sample_rate, # P.get_device_info_by_index(0)['defaultSampleRate']
        frame_rate,  # 10/30 Hz for observation space?
        # frames_per_buffer,
        excitation_mode # 'impulse', 'linear', 'exponential'
        ):
        self.channels = 1
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.frames_per_buffer = int(self.sample_rate / self.frame_rate)
        self.streamOpen = False
        self.excitation_mode = excitation_mode
        self.p = pyaudio.PyAudio()
        self._stream = None

        # Temporary variables
        self.current_frame = numpy.zeros(self.frames_per_buffer)
        self.input_buffer = []
        self.q = queue.Queue()
        self.plotdata = None # raw mic data visualization
        self.lines = None # raw mic data visualization

        self.SAMPLES_PER_FRAME = 4
        self.specdata = None
        self.im = None

        # for debugging
        # print(self.p.get_device_info_by_index(0)['defaultSampleRate']) # important, otherwise may crash
        # info = self.p.get_host_api_info_by_index(0)
        # numdevices = info.get('deviceCount')

        # for i in range(0, numdevices):
        #     if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
        #         print("Output Device id ", i, " - ", self.p.get_device_info_by_host_api_device_index(0, i).get('name'))

    def streaming(self):
        self._stream = self.p.open(rate=self.sample_rate, 
                                format=pyaudio.paFloat32, 
                                channels=self.channels, 
                                output=True, 
                                input=True, 
                                output_device_index=2,
                                input_device_index=2, # depends on the system
                                frames_per_buffer=self.frames_per_buffer,
                                stream_callback=self.get_callback())
    
    def is_streaming(self):
        if self._stream.is_active():
            return True
        else:
            return False

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            numpydata = numpy.frombuffer(in_data, dtype=numpy.float32) # input to mic
            
            self.current_frame = numpydata.copy()
            # self.input_buffer.append(numpydata)
            # print(numpydata)
            self.q.put(numpydata) # for visualization

            out_data = numpy.float32(numpy.vstack(self.excitation_signal()).ravel()) # output to actuator
            return (out_data, pyaudio.paContinue)
        return callback

    def excitation_signal(self):
        if self.excitation_mode == 'impulse':
            return numpy.vstack(self.channels * [numpy.float64(librosa.core.clicks(times=numpy.array([0.0]), sr=self.sample_rate, click_duration=0.01, length=int(1.0 / self.frame_rate * self.sample_rate)))]).T
        if self.excitation_mode == 'linear':
            return numpy.vstack(self.channels * [numpy.float64(librosa.core.chirp(20, 10000, self.sample_rate, duration=float(1 / self.frame_rate), linear=True))]).T
        if self.excitation_mode == 'exponential':
            return numpy.vstack(self.channels * [numpy.float64(librosa.core.chirp(20, 10000, self.sample_rate, duration=float(1 / self.frame_rate)))]).T      

    def get_window(self):
        return self.current_frame

    def visualize_input(self):
        fig, ax = plt.subplots(figsize=(8,4))

        # lets set the title
        ax.set_title("Mic Input")

        # Make a matplotlib.lines.Line2D plot item of color green
        self.plotdata =  numpy.zeros((44100, 1))
        self.lines = ax.plot(self.plotdata, color = (0,1,0.29))
        ax.set_xlim([0,44100])
        ax.set_ylim([-0.1, 0.1])

        ani = FuncAnimation(fig, self.update_plot, interval=100, blit=True)
        plt.show()

    def update_plot(self, frame):
        while True:
            try:
                data = self.q.get_nowait()
                # print(data)
            except queue.Empty:
                break
            shift = len(data)
            self.plotdata = numpy.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = [[i] for i in data]
        for column, line in enumerate(self.lines):
            line.set_ydata(self.plotdata[:, column])
        return self.lines

    # def update_fig_spectrogram(self, n):
    #     while True:
    #         try:
    #             data = self.q.get_nowait()
    #         except queue.Empty:
    #             break
    #         arr2D, freqs, bins = self.get_specgram(data, self.sample_rate)
    #         # print(arr2D)
    #         im_data = self.im.get_array()
    #         if n < self.SAMPLES_PER_FRAME:
    #             im_data = numpy.hstack((im_data, arr2D))
    #             self.im.set_array(im_data)
    #         else:
    #             keep_block = arr2D.shape[1] * (self.SAMPLES_PER_FRAME - 1)
    #             im_data = numpy.delete(im_data, numpy.s_[:-keep_block], 1)
    #             im_data = numpy.hstack((im_data, arr2D))
    #             self.im.set_array(im_data)
    #     return self.im,

    def update_fig_spectrogram(self, n):
        while True:
            try:
                data = self.q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self.specdata = numpy.roll(self.specdata, -shift)
            self.specdata[-shift:] = data
            arr2D, freqs, bins = self.get_specgram(self.specdata, self.sample_rate)
            self.im.set_array(arr2D)
        return self.im,

    def visualize_spectrogram(self):
        fig = plt.figure(figsize=(5,2))
        # specdata = self.get_window()
        self.specdata = numpy.zeros(44100)
        arr2D_init, freqs, bins = self.get_specgram(self.specdata, self.sample_rate)
        # librosa.feature.melspectrogram(y=all_rec_np, sr=SR_rec)
        # print(arr2D_init)
        extent = (bins[0],bins[-1]*self.SAMPLES_PER_FRAME,freqs[-1],freqs[0])
        self.im = plt.imshow(arr2D_init, aspect='auto', extent=extent, 
                            interpolation="none", cmap = 'summer',
                            norm=LogNorm(vmin=10e-13, vmax=10e-8))
                            # norm=LogNorm(vmin=.000000000001, vmax=0.000000001))

        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Real Time Spectogram')
        plt.gca().invert_yaxis()

        ani = FuncAnimation(fig, self.update_fig_spectrogram, 
                            interval=10, blit=True)
        plt.show()

    def get_specgram(self, signal, rate):
        arr2D, freqs, bins = specgram(signal, window=window_hanning,
                                Fs=rate, NFFT=256,
                                noverlap=192)
        return arr2D,freqs,bins

    def get_fft(self):
        signal = numpy.squeeze(self.get_window(), axis=0)
        yf = rfft(signal)
        # xf = rfftfreq(len(signal), 1 / self.sample_rate)
        return numpy.abs(yf) #, print(xf)

    def get_output(self):
        """
        raw_signal: (1, length of sequence) 
        signal_fft: (1, length of sequence / 2 + 1)
        arr2D: (num of freq bins, time stamps, 1)
        """
        raw_signal = self.get_window()
        # _signal = numpy.squeeze(raw_signal, axis=0)
        signal_fft = numpy.abs(rfft(raw_signal))
        arr2D, _, _ = specgram(raw_signal, window=window_hanning,
                        Fs=self.sample_rate, NFFT=256,
                        noverlap=192)
        return (
            numpy.expand_dims(raw_signal, axis=0), 
            numpy.expand_dims(signal_fft, axis=0), 
            numpy.expand_dims(arr2D, axis=2)
            )
        # return print(numpy.shape(numpy.expand_dims(raw_signal, axis=0))), print(numpy.shape(numpy.expand_dims(signal_fft, axis=0))), print(numpy.shape(numpy.expand_dims(arr2D, axis=2))) 
        # return print(numpy.expand_dims(raw_signal, axis=0)), print(numpy.expand_dims(signal_fft, axis=0)), print(numpy.expand_dims(arr2D, axis=2))     

    def save_buffer(self):
        pass

    def close(self):
        self._stream.close()
        self.p.terminate()
        self.save_buffer()

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

def read_detect_active_acous(activeAcous: ActiveAcousticSensor):
    return activeAcous.get_output()  # (raw signal, fft, spectrogram)

def play_sound():  # add parameters

    def callback(in_data, frame_count, time_info, status):

        numpydata = numpy.frombuffer(in_data, dtype=numpy.float32)
        all_rec.append(numpydata)
        out_data = w
        # If len(data) is less than requested frame_count, PyAudio automatically
        # assumes the stream is finished, and the stream stops.
        return (out_data, pyaudio.paContinue)

    SR = 44100
    CHANNELS = 1 # 1
    DURATION = 0.1  # (sec)
    PICKLE_PROTOCOL = 2
    HIGH_FREQ = 10000 # 500
    LOW_FREQ = 20
    buffer_size = int(SR * DURATION)

    all_rec = []

    silence = numpy.zeros((int(SR), CHANNELS))
    # step = numpy.vstack(CHANNELS * [librosa.core.clicks(times=numpy.array([0.0]), sr=SR, click_freq=1, click_duration=0.5)]).T  # click_freq=2
    step = numpy.vstack(CHANNELS * [numpy.float64(librosa.core.clicks(times=numpy.array([0.0]), sr=SR, click_duration=0.01, length=int(DURATION * SR)))]).T
    linear = numpy.vstack(CHANNELS * [numpy.float64(librosa.core.chirp(LOW_FREQ, HIGH_FREQ, SR, duration=DURATION, linear=True))]).T  # linear signal
    sweep = numpy.vstack(CHANNELS * [numpy.float64(librosa.core.chirp(LOW_FREQ, HIGH_FREQ, SR, duration=DURATION))]).T  # sweeping signal

    w = numpy.vstack(step).ravel()
    # w = numpy.vstack(linear).ravel()
    # w = numpy.vstack(sweep).ravel()

    w = numpy.float32(w) # double check

    # print("step: ", type(librosa.core.clicks(times=numpy.array([0.0]), sr=SR, click_duration=0.01, length=22050)[0]))
    # print("linear: ", len(w))

    # Instantiate PyAudio
    P = pyaudio.PyAudio()

    # for debugging
    # print(P.get_device_info_by_index(0)['defaultSampleRate']) # important, otherwise may crash
    # info = P.get_host_api_info_by_index(0)
    # numdevices = info.get('deviceCount')

    # for i in range(0, numdevices):
    #     if (P.get_device_info_by_host_api_device_index(0, i).get('maxOutputChannels')) > 0:
    #         print("Output Device id ", i, " - ", P.get_device_info_by_host_api_device_index(0, i).get('name'))

    # callback mode
    stream = P.open(rate=SR, 
                    format=pyaudio.paFloat32, 
                    channels=CHANNELS, 
                    output=True, 
                    input=True, 
                    output_device_index=None,
                    input_device_index=12,
                    frames_per_buffer = buffer_size,
                    stream_callback=callback)


    # plot data
    # plt.plot(w)
    # plt.xlim(0, 44100)
    # # librosa.display.waveshow(w, sr=SR, label='Step signal')
    # plt.show()

    # plt.figure()
    # emit_signal = numpy.tile(w, 10)
    # S = librosa.feature.melspectrogram(y=emit_signal, sr=SR)
    # # print("len of S:", len(S[0]))
    # ax = plt.subplot(2,1,2)
    # librosa.display.specshow(librosa.power_to_db(S, ref=numpy.max), sr=SR,
    #                          x_axis='time', y_axis='mel')
    # plt.subplot(2,1,1, sharex=ax)
    # librosa.display.waveshow(emit_signal, sr=SR)
    # # plt.plot(all_rec_np)
    # plt.legend()
    # plt.xlim(0, 0.5)
    # plt.ylim(-1.0, 1.0)
    # plt.tight_layout()
    # plt.show()

    # write to plain txt file
    # numpy.savetxt('emit_15_step.txt', w, fmt='%8.8f')  # 2->4->5

    # write to .wav file
    # write('emit_15_step.wav', SR, w.astype(numpy.float32))

    # callback mode
    time_start = time.time()
    while stream.is_active() and time.time() - time_start < 5.0:
        # get_window = stream.read(4410)
        time.sleep(0.01)

    # write to .wav file through opening a wave class (wav looks not right)
    wf = wave.open("test_callback_recording.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(P.get_sample_size(pyaudio.paFloat32))
    wf.setframerate(SR)
    wf.writeframes(b''.join(all_rec)) # write all recording to file at once

    stream.close()
    P.terminate()
    wf.close()


    all_rec_np = numpy.hstack(all_rec)

    ## plot data    
    plt.figure()
    S = librosa.feature.melspectrogram(y=all_rec_np, sr=SR)
    # print("len of S:", len(S[0]))
    ax = plt.subplot(2,1,2)
    librosa.display.specshow(librosa.power_to_db(S, ref=numpy.max), sr=SR,
                             x_axis='time', y_axis='mel')
    plt.subplot(2,1,1, sharex=ax)
    librosa.display.waveshow(all_rec_np, sr=SR)
    plt.legend()
    plt.xlim(0, 5)
    plt.ylim(-0.1, 0.1)
    plt.tight_layout()
    plt.show()
    # plt.plot(all_rec_np)
    # plt.subplot(3,1,3) # fft plot
    
    # scipy fft
    record_sec = 5
    plt.figure()
    n_fft = 256
    hop_length = 128
    win_length = 256
    # aud_ft= numpy.abs(librosa.stft(all_rec_np, n_fft = n_fft, hop_length = hop_length,win_length=win_length)) 
    yf = rfft(all_rec_np)
    xf = rfftfreq(len(all_rec_np), 1 / SR)
    plt.plot(xf, numpy.abs(yf))
    plt.show()

if __name__=="__main__":

    # volume control for 3.5mm audio jack (actuator and mic) in system level
    # for i in alsaaudio.mixers():
    #     print('**', i)

    # actuator = alsaaudio.Mixer('Master')
    # actuator.setvolume(60)

    # # set the gain for the microphone
    # mic = alsaaudio.Mixer('Capture')
    # # print('Current mic level: ', int(mic.getvolume()[0]))
    # mic.setvolume(100)

    # play_sound()
    activeAcoustic = ActiveAcousticSensor(sample_rate=44100, frame_rate=10, excitation_mode='exponential')
    activeAcoustic.streaming()
    # activeAcoustic.visualize_input()
    activeAcoustic.visualize_spectrogram()
    # activeAcoustic.get_window()
    read_detect_active_acous(activeAcoustic)

    # while activeAcoustic.is_streaming():
        # print(activeAcoustic.get_window())
        
        # time.sleep(0.001)