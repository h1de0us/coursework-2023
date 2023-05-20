import torch
import torch.nn.functional as F

def loss_function(output, target, mean, logvar):
    BCE = F.binary_cross_entropy_with_logits(output, target, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, mean, logvar = model(data)
        loss = loss_function(output, data, mean, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output, mean, logvar = model(data)
            loss = loss_function(output, data, mean, logvar)
            test_loss += loss.item()
    return test_loss / len(test_loader.dataset)

