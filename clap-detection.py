import pyaudio
import struct
import math
import time
import numpy as np

INITIAL_TAP_THRESHOLD = 0.005
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = (1.0/32768.0)
CHANNELS = 2
RATE = 44100
INPUT_BLOCK_TIME = 0.1
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)
# if we get this many noisy blocks in a row, increase the threshold
OVERSENSITIVE = 15.0/INPUT_BLOCK_TIME
# if we get this many quiet blocks in a row, decrease the threshold
UNDERSENSITIVE = 120.0/INPUT_BLOCK_TIME
# if the noise was longer than this many blocks, it's not a 'tap'
MAX_TAP_BLOCKS = 0.4/INPUT_BLOCK_TIME

def get_rms( block ):
    # RMS amplitude is defined as the square root of the
    # mean over time of the square of the amplitude.
    # so we need to convert this string of bytes into
    # a string of 16-bit samples...

    # we will get one short out for each
    # two chars in the string.
    count = len(block)/2
    format = "%dh"%(count)
    shorts = struct.unpack( format, block )

    # iterate over the block.
    sum_squares = 0.0
    for sample in shorts:
        # sample is a signed short in +/- 32768.
        # normalize it to 1.0
        n = sample * SHORT_NORMALIZE
        sum_squares += n*n

    return math.sqrt( sum_squares / count )

def is_clap(block):
    count = len(block) // 2
    format = "%dh" % (count)
    shorts = struct.unpack(format, block)

    # Calculate FFT and obtain the magnitude of frequency components
    fft_data = np.fft.rfft(np.array(shorts))
    magnitude = np.abs(fft_data)

    # Set a frequency threshold to differentiate claps from other sounds
    low_freq_threshold = 1750
    high_freq_threshold = 2350
    freq_resolution = RATE / len(magnitude)
    low_freq_index = int(low_freq_threshold // freq_resolution)
    high_freq_index = int(high_freq_threshold // freq_resolution)

    magnitude_range = magnitude[low_freq_index:high_freq_index]

    # Adjust this threshold to fine-tune the sensitivity to clap sounds
    clap_magnitude_threshold = 60000000

    magnitude_sum = np.sum(magnitude_range)
    clap_detected = magnitude_sum > clap_magnitude_threshold
    peak_frequency_index = np.argmax(magnitude_range) + low_freq_index
    detected_frequency = peak_frequency_index * freq_resolution

    return clap_detected, detected_frequency




class TapTester(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.tap_threshold = INITIAL_TAP_THRESHOLD
        self.noisycount = MAX_TAP_BLOCKS+1
        self.quietcount = 0
        self.errorcount = 0
        self.last_clap_time = None
        self.clap_count = 0
        self.clap_count_goal = 4
        self.rapid_clap_interval = 3  # time threshold for rapid claps in seconds


    def stop(self):
        self.stream.close()

    def find_input_device(self):
        device_index = None
        for i in range( self.pa.get_device_count() ):
            devinfo = self.pa.get_device_info_by_index(i)
            print( "Device %d: %s"%(i,devinfo["name"]) )

            for keyword in ["mic","input"]:
                if keyword in devinfo["name"].lower():
                    print( "Found an input: device %d - %s"%(i,devinfo["name"]) )
                    device_index = i
                    return device_index

        if device_index == None:
            print( "No preferred input found; using default input device." )

        return device_index

    def open_mic_stream( self ):
        device_index = self.find_input_device()

        stream = self.pa.open(   format = FORMAT,
                                 channels = CHANNELS,
                                 rate = RATE,
                                 input = True,
                                 input_device_index = device_index,
                                 frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

        return stream

    def tapDetected(self, detected_frequency):
        current_time = time.time()
        if self.last_clap_time is None or current_time - self.last_clap_time >= self.rapid_clap_interval:
            self.clap_count = 1  # reset clap count if it's the first clap or if the interval is too long
            print(str(self.clap_count) + " Detected / Reset Count. Detected Frequency: {:.2f} Hz".format(detected_frequency))
        else:
            self.clap_count += 1  # increment clap count if it's a rapid clap
            print(str(self.clap_count) + " Clap Detected. Detected Frequency: {:.2f} Hz".format(detected_frequency))

        if self.clap_count == self.clap_count_goal:
            print(f"{self.clap_count} rapid claps detected")
            self.clap_count = 0  # reset clap count after two rapid claps detected
        self.last_clap_time = current_time  # update the last clap time


    def listen(self):
        try:
            block = self.stream.read(INPUT_FRAMES_PER_BLOCK)
        except e:
            # dammit.
            self.errorcount += 1
            print( "(%d) Error recording: %s"%(self.errorcount,e) )
            self.noisycount = 1
            return

        amplitude = get_rms(block)
        is_clap_sound, detected_frequency = is_clap(block)

        if amplitude > self.tap_threshold and is_clap_sound:
            # noisy block
            self.quietcount = 0
            self.noisycount += 1
        else:
            # quiet block.

            if 1 <= self.noisycount <= MAX_TAP_BLOCKS:
                self.tapDetected(detected_frequency)
            self.noisycount = 0
            self.quietcount += 1

if __name__ == "__main__":
    tt = TapTester()

    while True:
        tt.listen()
