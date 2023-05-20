from model import *
from traineval import *
from dataset import *
import os

from torch.utils.data import DataLoader

train_path = '/Users/h1de0us/uni/coursework/project/EMOPIA_1.0/midis/Q1/'
test_path = '/Users/h1de0us/uni/coursework/project/EMOPIA_1.0/midis/Q1/'
train_files = [train_path + file for file in os.listdir(train_path) if file.endswith('.mid')]
test_files = [test_path + file for file in os.listdir(test_path) if file.endswith('.mid')]


input_dim = 256
hidden_dim = 512
latent_dim = 32
batch_size = 64
num_epochs = 20

train_dataset = MidiDataset(train_files, input_dim)
test_dataset = MidiDataset(test_files, input_dim)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, num_epochs+1):
    train_loss = train(model, optimizer, train_loader, device)
    test_loss = test(model, test_loader, device)
    print('Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch, train_loss, test_loss))
