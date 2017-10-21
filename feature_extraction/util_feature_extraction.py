
import numpy as np
import pyedflib
import scipy.signal as signal
import os

#Funciton to load 1 file of EEG data 
def loadSingleEDF(file_dir):
	#name should be relative directory from where feature_extraction.py is located
	
	f = pyedflib.EdfReader(file_dir)
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
		sigbufs[i, :] = f.readSignal(i)
	print(sigbufs.shape)
	return sigbufs,signal_labels

#Takes in data created from loadSingleEDF file and makes a pyplot of first 5 seconds

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

def extract_features_2(raw_signal,signal_labels):
	#This function filters raw_sigals using two butterworthfilters of ranges .5 to 12
	#hz and 12 hz to 24 hz following Shoeb et. al paper 
	#returns a matrix with features with row a 2 second segment 
	fs = 256.0
	low = .5
	med= 12
	high = 24
	signals = ["FP1-F7","F7-T7"]

	indexes = []

	for signal in signals:
		indexes.append(signal_labels.index(signal))
	channels = [raw_signal[i,:] for i in indexes]

	num_2_sec_segments = raw_signal.getNSamples()[0] / 512
	num_features_per_segment = len(channel) * len(signal_labels)
	feature_matrix = np.zeros(num_2_sec_segments,num_features_per_segment)

	filtered = []
	for channel in channels:
		#figure how how to make list of functions that does filtering 
		filtered.extend([butter_bandpass_filter(channel,low,med,fs),
		 butter_bandpass_filter(channel_1,med,high,fs)])

	for seg in xrange(num_2_sec_segments):
		for i,filtered_signal in enumerate(filtered):
			feature_matrix[seg,i] = calculateRMS(filtered_signal[seg*512,seg*512 + 512])
	
	return feature_matrix


