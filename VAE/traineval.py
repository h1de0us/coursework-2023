# from model import *
from tqdm import tqdm
import torch


def train(model, optimizer, scheduler, train_loader, val_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        # print('epoch:', epoch)
        model.train()
        total_loss = 0
        for i, x in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mean, logvar = model.forward(x)
            loss = model.loss_function(x, x_hat, mean, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader.dataset)
        print("Train epoch [%d/%d], Loss: %.4f" % (
                epoch + 1, num_epochs, total_loss))
        train_losses.append(total_loss)


        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, x in enumerate(val_loader):
                x = x.to(device)
                x_hat, mean, logvar = model.forward(x)
                test_loss += model.loss_function(x, x_hat, mean, logvar).item()
        test_loss /= len(val_loader.dataset)
        print("Test epoch [%d/%d], Loss: %.4f" % (
            epoch + 1, num_epochs, test_loss))
        test_losses.append(test_loss)
        scheduler.step()

    return train_losses, test_losses


def inference(model, num_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    z = torch.randn(num_samples, model.latent_dim).to(device)
    with torch.no_grad():
        generated_samples = model.decode(z)
    return generated_samples


def get_midi(dataset, samples):
    midi = []
    for sample in samples:
        pass

