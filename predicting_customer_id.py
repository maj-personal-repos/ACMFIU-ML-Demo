import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from datasets import CustomerData
from models import IdNet
from torch import FloatTensor, LongTensor, max, device, cuda
import time



# Let's load the data and take a look
train_df = pd.read_pickle("data/train_data.pickle")
cmap = cm.get_cmap('Spectral')
train_df.sample(frac=0.05).plot.scatter(x='st_weekday',
                                        y='st_seconds',
                                        c='customerId',
                                        cmap=cmap,
                                        edgecolor=None,
                                        alpha=0.5,
                                        # s=100,
                                        marker="_")
plt.show()

# Setup the data for training the model
train_dataset = CustomerData('data/train_data.pickle')
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Setup the data for testing the model
test_dataset = CustomerData('data/test_data.pickle')
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# hyperparameters
num_epochs = 10
num_labels = 4
num_features = 2
log_interval = 100
n_hidden = 128
ngpu = 1

device = device("cuda:0" if (cuda.is_available() and ngpu > 0) else "cpu")
classifier = IdNet(num_features, n_hidden, num_labels)
classifier.to(device)
optimizer = SGD(classifier.parameters(), lr=0.01, momentum=0.9)
criterion = CrossEntropyLoss()

for epoch in range(num_epochs):
    losses = []
    count = 0
    correct_cnt = 0
    total_cnt = 0
    for batch_id, (data, labels) in enumerate(train_dataloader):
        n_batch = len(labels)
        optimizer.zero_grad()
        data, labels = data.type(FloatTensor).to(device), labels.type(LongTensor).to(device)
        out = classifier(data)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        _, pred_label = max(out.data, 1)
        total_cnt += data.data.size()[0]
        correct_cnt += (pred_label == labels.data).sum()
        count += n_batch
        # losses.append(loss.data.mean())
        if (batch_id + 1) % log_interval == 0:
            accuracy = correct_cnt.data.cpu().numpy() * 1.0 / total_cnt
            # losses_mean = np.mean(losses)
            message = "{}\t Epoch {}:\t[{}/{}]\tloss: {:.6f}, acc: {:.3f}".format(time.ctime(), epoch + 1, count,
                                                                                  len(train_dataset), loss.item(),
                                                                                  accuracy)
            print(message)
