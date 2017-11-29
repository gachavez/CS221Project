import torch
import util_simple_neural as util
from torch.utils.data import DataLoader
from torch.autograd import Variable

model = util.model()

criterion = torch.nn.BCELoss(size_average = True)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)




dataset = util.EEG_Dataset()

train_loader = DataLoader(dataset=dataset,
                                           batch_size=100000,
                                           shuffle=True,
                                           num_workers=2)
dtype = torch.FloatTensor

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels.type(dtype))
        y_pred = model(inputs)
        y_pred = y_pred.type(dtype)
        loss = criterion(y_pred,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch, loss.data[0])

X_test = util.loadTestData()
y_test = X_test[:, -2]
X_test = X_test[:, :-2]

X_test = Variable(torch.from_numpy(X_test).type(dtype))
y_test = Variable(torch.from_numpy(y_test).type(dtype))

y_pred = model(X_test)
loss = criterion(y_pred,y_test)
print("Testing accuracy was " + str(loss.data[0])) #0.02064555510878563 for iter of 1000\
#0.46427321434020996

torch.save(model.state_dict(), 'model')