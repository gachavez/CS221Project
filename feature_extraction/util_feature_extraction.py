import numpy as np
import pyedflib
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

#Funciton to load 1 file of EEG data
def loadSingleEDF():
	#name should be relative directory from where feature_extraction.py is located
	file_dir = "/Users/gustavochavez/Documents/GitHub/CS221Project/test_data"
	f = pyedflib.EdfReader(file_dir)
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
		sigbufs[i, :] = f.readSignal(i)
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

def makePlotWithFiltered():
	#Make butterworth filters one for 1-12 hz
	file_dir = "/Users/gustavochavez/documents/Data/CS221_project_data/physionet.org/pn6/temp/chb01/chb01_03.edf"
	f = pyedflib.EdfReader(file_dir)
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
		sigbufs[i, :] = f.readSignal(i)
	fs = 256.0
	low = 1
	med= 12
	high = 24
	unfiltered = sigbufs[2,:]

	t = np.arange(0.0,5.0,.00390625)
	l = len(t)

	#plt.plot(t,unfiltered[:l], label="Unfiltered")
	#plt.show()
	#filtered = butter_bandpass_filter(unfiltered[:l],low,med,fs)
	#plt.plot(t,filtered,label="low band")
	#plt.show()
	filtered2 = butter_bandpass_filter(unfiltered[:l],med,high,fs)
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
	num_filters = 2
	#note that gustavo deleted , "F8-T8" which was the 17th common channel in all data files because 16 was planned for
	signals = ["F4-C4", "C3-P3", "P7-O1", "P8-O2", "F7-T7", "FZ-CZ", "F3-C3", "FP2-F8", "FP2-F4", "P3-O1", "T7-P7", "CZ-PZ", "C4-P4", "P4-O2", "FP1-F7", "FP1-F3"]

	indexes = []

	for signal in signals:
		indexes.append(signal_labels.index(signal))
	channels = [raw_signal[i,:] for i in indexes]

	num_2_sec_segments = len(raw_signal[0]) / 512
	num_features_per_segment = len(channels) * num_filters
	feature_matrix = np.zeros((num_2_sec_segments,num_features_per_segment))

	filtered = []
	for channel in channels:
		#figure how how to make list of functions that does filtering
		filtered.extend([butter_bandpass_filter(channel,low,med,fs)])
		filtered.extend([butter_bandpass_filter(channel,med,high,fs)])

	for seg in xrange(0,num_2_sec_segments):
		for i,filtered_signal in enumerate(filtered):
			feature_matrix[seg,i] = calculateRMS(filtered_signal[seg*512:seg*512 + 512])

	return feature_matrix


def extract_features_8(raw_signal,signal_labels):
	#This function filters raw_sigals using two butterworthfilters of ranges .5 to 12
	#hz and 12 hz to 24 hz following Shoeb et. al paper
	#returns a matrix with features with row a 2 second segment
	fs = 256.0
	samples_per_seg = 512
	low = 1
	high = 25
	num_filters = 8
	filter_band_width = float((high - low)) / float(num_filters) # 24/8 = 3.125
	signals = ["FP1-F7","F7-T7"]

	indexes = []

	for signal in signals:
		indexes.append(signal_labels.index(signal))
	channels = [raw_signal[i,:] for i in indexes]

	num_2_sec_segments = len(raw_signal[0]) / samples_per_seg
	num_features_per_segment = len(channels) * num_filters
	feature_matrix = np.zeros((num_2_sec_segments,num_features_per_segment))

	filtered = []
	for channel in channels:
		#figure how how to make list of functions that does filtering
		for i in xrange(0,8):
			filtered.extend([butter_bandpass_filter(channel,i*filter_band_width + low,(i + 1) * filter_band_width + low,fs)])

	for seg in xrange(0,num_2_sec_segments):
		for i,filtered_signal in enumerate(filtered):
			feature_matrix[seg,i] = calculateRMS(filtered_signal[seg*samples_per_seg:seg*samples_per_seg + samples_per_seg])

	return feature_matrix
def representsInt(s):
	try:
		int(s)
		return True
	except ValueError:
		return False
def loadLabels():
	file_object  = open("/Users/gustavochavez/Documents/GitHub/CS221Project/feature_extraction/labels.txt", 'r')
	text = file_object.readlines()
	end = len(text)
	current_line = 0
	labels = {}
	while(current_line < end):
		key = text[current_line]
		#remove \n symbol
		key = key.replace("\n", "")
		key = key.split("/")[1]
		value = []
		current_line = current_line + 1
		while(text[current_line][0] != "c"):
			index = 3
			if (representsInt(text[current_line].split(" ")[3])):
				index = 3
			else:
				index = 4
			start_time = text[current_line].split(" ")[index]
			end_time   = text[current_line + 1].split(" ")[index]
			value.append((int(start_time),int(end_time)))
			current_line = current_line + 2
			if current_line >= end:
				break

		labels[key] = value
	return labels
def label(directory, labels):

	data = np.loadtxt(open(directory, "rb"), delimiter=",")
	H,W = data.shape
	label_col = np.zeros((H,1))
	label_col_oracle =  np.zeros((H,1))
	for label in labels:
		start_row = label[0] // 2
		end_row   = label[1] // 2
		label_col[start_row:end_row + 1] = 1
		oracle_start = start_row - 30
		oracle_end = end_row + 30
		if oracle_start < 0:
			oracle_start = 0
		if oracle_start >= H - 1:
			oracle_end =  H - 1
		label_col_oracle[oracle_start:oracle_end + 1] = 1

	data = np.hstack((data,label_col))
	data = np.hstack((data,label_col_oracle))
	np.savetxt(directory,data,delimiter =",")


def countChannels(directory, channels): 
	f = pyedflib.EdfReader(directory)
	n = f.signals_in_file
	signal_labels = f.getSignalLabels()
	for channel in signal_labels:
		if channel in channels.keys():
			channels[channel] = channels[channel] + 1
		else:
			channels[channel] = 0
