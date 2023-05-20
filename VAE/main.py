import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from dataset import *
from model import *
from traineval import *


'''
TODO: для DataLoader нужно задать collate_fn
TODO: прочекать Pytorch-VAE? мне кажется там может быть что-то полезное
TODO: разобраться с VQ-VAE и куда-то это записать
'''

def collate_fn(batch):
    X = [item.detach() for item in batch]
    return torch.stack(X, 0)

# def collate_fn(batch):
#     max_len = max(len(seq) for seq in batch)
#     padded_seqs = []
#     for seq in batch:
#         padded_seq = np.zeros((max_len, ), dtype=np.int32)
#         padded_seq[:len(seq)] = seq
#         padded_seqs.append(padded_seq)
#
#     return np.array(padded_seqs)


if __name__ == '__main__':
    DEBUG_PATH = '/Users/h1de0us/uni/coursework/project/EMOPIA_1.0/midis/test'

    PATH_Q1 = '/Users/h1de0us/uni/coursework/project/EMOPIA_1.0/midis/Q1'
    PATH_Q2 = '/Users/h1de0us/uni/coursework/project/EMOPIA_1.0/midis/Q2'
    PATH_Q3 = '/Users/h1de0us/uni/coursework/project/EMOPIA_1.0/midis/Q3'
    PATH_Q4 = '/Users/h1de0us/uni/coursework/project/EMOPIA_1.0/midis/Q4'
    # TODO: + как-то разбить на train и test
    VAL_PATH_Q1 = PATH_Q1
    VAL_PATH_Q2 = ''
    VAL_PATH_Q3 = ''
    VAL_PATH_Q4 = ''

    SEQ_LEN = 250  # подбиралось опытным путём

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 64

    NUM_EPOCHS = 20

    NUM_WORKERS = 1

    train_dataset = MidiDataset(data_path=PATH_Q1,
                                seq_len=SEQ_LEN,
                                device=DEVICE,
                                )
    val_dataset = MidiDataset(data_path=VAL_PATH_Q1,
                              seq_len=SEQ_LEN,
                              device=DEVICE)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              collate_fn=collate_fn,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            collate_fn=collate_fn,
                            num_workers=NUM_WORKERS)

    model = VAE(
        input_dim=128,
        latent_dim=32,  # советуют от 10 до 100?
        hidden_dims=512,  # было 512
        batch_size=BATCH_SIZE
    )


    # TODO: перебрать гиперпараметры для оптимайзера, виды оптимайзеров и шедулер
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_losses, val_losses = train(model, optimizer, scheduler, train_loader, val_loader, num_epochs=NUM_EPOCHS)

    generated_samples = inference(model, 10)
    torch.save({
                'epoch': 20,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
                'samples': generated_samples,
                }, 'base')
