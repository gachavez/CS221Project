import torch
import util_simple_neural as util
from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import train_test_split

from torch.autograd import Variable

dtype = torch.FloatTensor
X_test = util.loadTestData()
X_train = util.loadTrainingData()

y_train = X_train[:, -2]
X_train = X_train[:, :-2]

y_test = X_test[:,-2]
X_test = X_test[:,:-2]

X_train = Variable(torch.from_numpy(X_train).type(dtype))
y_train = Variable(torch.from_numpy(y_train).type(dtype))

X_test = Variable(torch.from_numpy(X_test).type(dtype))
y_test = Variable(torch.from_numpy(y_test).type(dtype))

#split training / validation / test sets (60-20-20 split between train, validation, test)

print(X_train.data.shape)
print(X_test.data.shape)

model = util.model()

criterion = torch.nn.BCELoss(size_average = True)
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

#start to learn

for epoch in range(2000):
	y_pred = model(X_train)
	loss = criterion(y_pred,y_train)
	print(epoch,loss.data[0])
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


y_pred = model(X_test)
loss = criterion(y_pred,y_test)
print("Testing accuracy was " + str(loss.data[0])) #0.02064555510878563 for iter of 1000\
#recall score
