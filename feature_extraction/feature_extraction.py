#imports
import os
import util_feature_extraction as util

#main 
	
	#open each patient directory 
		# for each data file ending in edf
			#create csv file with name pt_#_file_#.csv = extract_features(file)
			#---> csv each row is feature vector of length 32 = 16 
			#channels * 2 filters/

def main():

	#Script was run on Gustavo Chavez's laptop and directory should change to where data is stored locally.
	parent_dir = "/Users/gustavochavez/Documents/Data/CS221_project_data/physionet.org/pn6/chbmit"
	patients = [s for s in os.listdir(parent_dir) if "chb" in s]

	for patient in patients:
		patient_dir = parent_dir +"/"+  patient
		#print(patient_dir)
		files = [s for s in os.listdir(patient_dir) if "chb" in s and ".txt" not in s]
		for file in files:
			file_dir = patient_dir +"/"+  file
			#print("\t" + file_dir)
			raw_signal, signal_labels = util.loadSingleEDF(file_dir,signal_labels)
			feature_matrix = util.extract_features_2(raw_signal,signal_labels)
			np.savetxt(file + ".csv",feature_matrix,delimiter =",")
main()