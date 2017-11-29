import torch
import numpy as np
import os
import torch.nn
from torch.autograd import Variable
from torch.utils.data import Dataset


class model(torch.nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.l1 = torch.nn.Linear(32,32)
        self.l2 = torch.nn.Linear(32,16)
        self.l3 = torch.nn.Linear(16,1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))

        return y_pred

def loadTrainingData():
    #code works in Gustavo's laptop and will require change
    #directory = "/Users/gustavochavez/Documents/GitHub/CS221Project/feature_extraction_output/"
    #directory = "/Users/gustavochavez/Documents/GitHub/CS221Project/test_data/"

    directory = "/data"
    files = [s for s in os.listdir(directory) if "epilepsy_train" in s]
    df = np.zeros([1,34])
    for file in files:
        print("the file is "+ str(file))
        curr_df = np.loadtxt(directory +"/" +  file, delimiter = ",",dtype= np.float32)
        curr_df = curr_df[:,:34]
        df = np.vstack((df,curr_df))
    df = np.delete(df, (0), axis=0)
    print(df.shape)
    return df


def loadTestData():
    directory = "/data"
    files = [s for s in os.listdir(directory) if "epilepsy_test" in s]
    df = np.zeros([1, 34])
    for file in files:
        print("loading" + str(file))
        curr_df = np.loadtxt(directory + "/" + file, delimiter=",", dtype=np.float32)
        curr_df = curr_df[:, :34]
        df = np.vstack((df, curr_df))
    df = np.delete(df, (0), axis=0)
    print(np.sum(df[:, -2]))
    print(df.shape)
    return df


class EEG_Dataset(Dataset):
    def __init__(self):
        xy = loadTrainingData()
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-2]).type(torch.FloatTensor)
        self.y_data = torch.from_numpy(xy[:,-2].reshape((xy.shape[0],1))).type(torch.FloatTensor)

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
