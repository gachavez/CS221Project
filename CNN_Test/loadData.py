import torch
import numpy as np
#import cPickle

class EpilepsyTrain():

	def __init__(self):
		#x = cPickle.load(open("CNN_data/xtrain3d.pkl", "rb"))
		x = np.loadtxt(open("/Users/gustavochavez/Desktop/Data/chb01_03.edf.csv", "rb"), delimiter=",")
		#y = cPickle.load(open("CNN_data/ytrain.pkl", "rb"))
		y = np.loadtxt(open("/Users/gustavochavez/Documents/GitHub/CS221Project/feature_extraction_output/chb01_03.edf.csv", "rb"), delimiter=",")
		y = y[:,32]
		self.len = x.shape[0]
		self.x_data = torch.DoubleTensor(torch.from_numpy(x))
		self.y_data = torch.DoubleTensor(torch.from_numpy(y))
		

	def __getitem__(self, index):
		return self.x_data[index,0:256].resize_(1,16,16), self.y_data[index]

	def __len__(self):
		return self.len

class EpilepsyTest():

	def __init__(self):
		#x = cPickle.load(open("CNN_data/xtest3d.pkl", "rb"))
		x = np.loadtxt(open("/Users/gustavochavez/Desktop/Data/chb01_04.edf.csv", "rb"), delimiter=",")
		#y = cPickle.load(open("CNN_data/ytest.pkl", "rb"))
		y = np.loadtxt(open("/Users/gustavochavez/Documents/GitHub/CS221Project/feature_extraction_output/chb01_04.edf.csv", "rb"), delimiter=",")
		y = y[:,32]
		self.len = x.shape[0]
		self.x_data = torch.DoubleTensor(torch.from_numpy(x))
		self.y_data = torch.DoubleTensor(torch.from_numpy(y))

	def __getitem__(self, index):
		return self.x_data[index].resize_(1,16,16), self.y_data[index]

	def __len__(self):
		return self.len

# class EpilepsyVal():

# 	def __init__(self):
# 		xy = np.loadtxt('separated/epilepsy_val.csv', delimiter=',', dtype = np.float32)
# 		self.len = xy.shape[0]
# 		self.x_data = torch.from_numpy(xy[:,0:-2])
# 		self.y_data = torch.from_numpy(xy[:,[-1]])

# 	def __getitem__(self, index):
# 		return self.x_data[index], self.y_data[index]

# 	def __len__(self):
# 		return self.len

