import loadData
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable

# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

print 'loading training set...'
train_dataset = loadData.EpilepsyTrain()
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)
print 'training set loaded'

print 'loading test set...'
test_dataset = loadData.EpilepsyTest()
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle=False)
print 'test set loaded'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(4, 4, kernel_size=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(1))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(1))
        self.fc = nn.Linear(7*7*32, 2)
        
    def forward(self, x):
        out = self.layer1(x)
        print 'layer1'
        out = self.layer2(out)
        print 'layer2'
        out = out.view(out.size(0), -1)
        print 'layer3'
        out = self.fc(out)
        return out

cnn = CNN()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
#torch.stack([torch.atan(a), torch.exp(a), a], dim=1)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        print 'here1'
        outputs = cnn(images.float())
        print 'here2'
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

















