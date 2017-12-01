import torch
import numpy as np
import cPickle

class EpilepsyTrain():

	def __init__(self):
		x = cPickle.load(open("CNN_data/xtrain3d.pkl", "rb"))
		y = cPickle.load(open("CNN_data/ytrain.pkl", "rb"))
		self.len = x.shape[0]
		self.x_data = torch.from_numpy(x) 
		self.y_data = torch.from_numpy(y) 
		

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len

class EpilepsyTest():

	def __init__(self):
		x = cPickle.load(open("CNN_data/xtest3d.pkl", "rb"))
		y = cPickle.load(open("CNN_data/ytest.pkl", "rb"))
		self.len = x.shape[0]
		self.x_data = torch.from_numpy(x)
		self.y_data = torch.from_numpy(y)

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

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

