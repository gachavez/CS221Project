#imports
import os
import util_feature_extraction as util
import numpy as np

def main():

	#Script was run on Gustavo Chavez's laptop and directory should change to where data is stored locally.
	parent_dir = "/Users/gustavochavez/Documents/Data/CS221_project_data/physionet.org/pn6/chbmit"
	patients = [s for s in os.listdir(parent_dir) if "chb" in s]

	for patient in patients:
		patient_dir = parent_dir +"/"+  patient
		#print(patient_dir)
		files = [s for s in os.listdir(patient_dir) if "chb" in s and ".txt" not in s and ".seizures" not in s]
		for file in files:
			print("Processing file:" + file)
			file_dir = patient_dir +"/"+  file
			#print("\t" + file_dir)
			raw_signal, signal_labels = util.loadSingleEDF(file_dir)
			feature_matrix = util.extract_features_2(raw_signal,signal_labels)
			np.savetxt("/Users/gustavochavez/Documents/GitHub/CS221Project/feature_extraction_output/" + file + ".csv",feature_matrix,delimiter =",")

def data_labeling():
	#get labels from txt files 
	#find feature vectors from feature_extraction_output folder
		#for each feature vector = find intervals where seizure occured
			#label as 1 
		#else 
			#label as 0 
		#resave csv file 
	#map that maps files to list of tuples which tuples is start,stop 

	seizure_labels_map = util.loadLabels()
	directory = "/Users/gustavochavez/Documents/GitHub/CS221Project/feature_extraction_output/"
	files = [s for s in os.listdir(directory) if "chb" in s]
	for file in files:
		if file[:-4] in seizure_labels_map.keys():
			util.label(directory+file,seizure_labels_map[file[:-4]])
		else: util.label(directory + file, [])
			
def findCommonChannels():
	parent_dir = "/Users/gustavochavez/Documents/Data/CS221_project_data/physionet.org/pn6/chbmit"
	patients = [s for s in os.listdir(parent_dir) if "chb" in s]
	channels = {}
	for patient in patients:
		patient_dir = parent_dir +"/"+  patient
		#print(patient_dir)
		files = [s for s in os.listdir(patient_dir) if "chb" in s and ".txt" not in s and ".seizures" not in s]
		
		for file in files:
			print("Processing file:" + file)
			file_dir = patient_dir +"/"+  file
			#print("\t" + file_dir)
			util.countChannels(file_dir,channels)
	result = [] 
	print(str(channels) + str(max(channels, key=channels.get)))
	for v,k in channels.items():
		if k >= 682:
			result.append(v)
	print("the result is " + str(result) + " with a length of " + str(len(result)))
#main()

main()

