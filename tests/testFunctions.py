

import numpy as np
import pyedflib
import os.path
import matplotlib.pyplot as plt
import audioop 
import scipy.signal as signal

#Load: we would like to load EDF files and print out the first data structure 
#Note: code was used in Gustavo Chavez laptop thus directory should be changed as needed.



#This script is intended to test signal, butter, EDF reader to use for feature extraction.
def loadSingleEDF():
	print("We are loading data found in chb01_01\n\n")
	directory = "/Users/gustavochavez/Documents/GitHub/CS221Project/tests/testData/chb01_01.edf"
	f = pyedflib.EdfReader(directory)
	n = f.signals_in_file
	print("The file has "+ str(n) +" signals in file")
	signal_labels = f.getSignalLabels()
	print("Signal labels are "+str(signal_labels)+ "which correspond to channels of EEG")
	print("The 5th signal is " + signal_labels[4])
	print("which should return same " + str(signal_labels.index(signal_labels[4])))
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
		sigbufs[i, :] = f.readSignal(i)
	print("signal_labels has entry type " + str(type(signal_labels[0])))
	print("the number of signals in each channel is "+ str(f.getNSamples()[0]))
	print("sigbufs is " + str(sigbufs) +" and has a length of "+ str(len(sigbufs)))
	listI = [1,2,3]
	subseg = [sigbufs[i] for i in listI]
	print("shape of subseg is " + str(len(subseg)))
	return sigbufs

#Takes in data created from loadSingleEDF file and makes a pyplot of first 5 seconds
def makePlotChannel1FiveSeconds(data):
	channel_1 = data[0,0:1280]
	#Sampled at 256 hz so interval is at .00390625 = 1/256

	t = np.arange(0.0, 5.0, .00390625)
	s = channel_1
	plt.plot(t, s)

	plt.xlabel('time (s)')
	#I think that voltage is not the correct y variable but whatever
	plt.ylabel('voltage ')
	plt.title('First 5 seconds of EEG channel 1 of Pt1')
	plt.grid(True)

	#plt.show()
	return channel_1

def calculateRMS(signal):
	timeLength = 2 # We are calating the energy of signal per second, over a 2 second period
	return np.sqrt(np.dot(signal,signal) / timeLength)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y
def makePlotWithFiltered(unfiltered):
	#Make butterworth filters one for 1-12 hz 
	fs = 256.0
	low = 1
	med= 12
	high = 24
	t = np.arange(0.0,5.0,.00390625)
	plt.plot(t,unfiltered, label="Unfiltered")
	filtered = butter_bandpass_filter(channel_1,low,med,fs)
	plt.plot(t,filtered,label="low band")
	filtered2 = butter_bandpass_filter(channel_1,med,high,fs)
	plt.plot(t,filtered2,label="high band")
	plt.show()


data = loadSingleEDF()
channel_1 = makePlotChannel1FiveSeconds(data)

width = len(channel_1)
#makePlotWithFiltered(channel_1)




















