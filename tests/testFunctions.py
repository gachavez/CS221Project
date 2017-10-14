

import numpy as np
import pyedflib
import os.path
import matplotlib.pyplot as plt

#Load: we would like to load EDF files and print out the first data structure 
#Note: code was used in Gustavo Chavez laptop thus directory should be changed as needed.

def loadSingleEDF():
	print("We are loading data found in chb01_01\n\n")
	directory = "/Users/gustavochavez/Documents/GitHub/CS221Project/tests/testData/chb01_01.edf"
	f = pyedflib.EdfReader(directory)
	n = f.signals_in_file
	print("The file has "+ str(n) +" signals in file")
	signal_labels = f.getSignalLabels()
	print("Signal labels are "+str(signal_labels)+ "which correspond to channels of EEG")
	sigbufs = np.zeros((n, f.getNSamples()[0]))
	for i in np.arange(n):
		sigbufs[i, :] = f.readSignal(i)

	print("the number of signals in each channel is "+ str(f.getNSamples()[0]))
	print("sigbufs is " + str(sigbufs) +" and has a length of "+ str(len(sigbufs)))
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

	plt.show()

data = loadSingleEDF()
makePlotChannel1FiveSeconds(data)