from cards_class import CardsDataset
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from model import SSD300, MultiBoxLoss
from utils import unpackBoundigBox, showBatch, collate_fn, train, save_checkpoint, adjust_learning_rate
import sys
import os

def main():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    if '/home/alonso/' in file_dir:
        images_path = '/home/alonso/Developer/datasets/playing_cards/images'
        anotations_path = '/home/alonso/Developer/datasets/playing_cards/anotations'
        labels_path = '/home/alonso/Developer/datasets/playing_cards/classes.txt'
        model_path = '/home/alonso/Developer/datasets/playing_cards/models/cards_2.pth.tar'
    elif '/Users/acano/' in file_dir:
        images_path = '/Volumes/Cano/datasets/playing_cards/images'
        anotations_path = '/Volumes/Cano/datasets/playing_cards/anotations'
        labels_path = '/Volumes/Cano/datasets/playing_cards/classes.txt'
        model_path = '/Volumes/Cano/datasets/playing_cards/models/cards_2.pth.tar'
    elif '/home/acano/' in file_dir:
        images_path = '/home/acano/Developer/datasets/playing_cards/images'
        anotations_path = '/home/acano/Developer/datasets/playing_cards/anotations'
        labels_path = '/home/acano/Developer/playing_cards/classes.txt'
        model_path = '/home/acano/Developer/datasets/playing_cards/models/cards_2.pth.tar'
    elif '/floyd/home' in file_dir:
        images_path = '/floyd/input/cards/images'
        anotations_path = '/floyd/input/cards/anotations'
        labels_path = '/floyd/input/cards/general_labels/classes.txt'
        model_path = 'cards_2.pth.tar'
    else:
        images_path = 'dataset/playing_cards/images'
        anotations_path = 'dataset/playing_cards/anotations'
        labels_path = 'dataset/playing_cards/classes.txt'
        model_path = 'datasets/playing_cards/models/cards_2.pth.tar'




    train_dataset = CardsDataset(images_path, anotations_path, labels_path)
    print('Dataset lenght:\n', len(train_dataset))
    num_classes = len(list(train_dataset.labels.keys()))
    print('Classes:\n', train_dataset.labels)
    print('Number of classes:\n', num_classes)
    # label_map = dataset.invLabels

    # showBatch(train_dataset[500], 1)
    # sys.exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Availabele GPU:\n', device)
    ''' ----------------Learning parameters--------------'''
    # path to checkpoint if there is one, set None if there isnt
    checkpoint = None # model_path
    # batch size
    batch_size = 4
    # number of epochs to train
    epochs = 40
    # number of workers for loading data in the DataLoader
    workers = 4
    # print training status every __ batches
    print_freq = 50
    # learning rate
    lr = 1e-4
    # decay learning rate after these many iterations
    decay_lr_at = [int(epochs * .7), int(epochs * .9)]
    # decay learning rate to this fraction of the existing learning rate
    decay_lr_to = 0.1
    # momentum
    momentum = 0.9
    # weight decay
    weight_decay = 5e-4
    # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
    grad_clip = None

    cudnn.benchmark = True

    
    print('Number of epochs:\n', epochs)
    print('Batch size:\n', batch_size)
    print('Learning Rate:\n', lr)
    print('Decaying learning rate at epochs:\n', decay_lr_at)

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=num_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('Loaded checkpoint from epoch %d.' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=workers,  collate_fn=collate_fn, drop_last=True)


     # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader, model=model,criterion=criterion,
              optimizer=optimizer,epoch=epoch, print_freq=print_freq, grad_clip=grad_clip)

        # Save checkpoint
        if epoch % 2 == 0 and model_path is not None:
            save_checkpoint(model_path,epoch, model, optimizer)


if __name__ == '__main__':
    main()






