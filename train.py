from cards_class import CardsDataset
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from model import SSD300, MultiBoxLoss
from utils import unpackBoundigBox, showBatch, collate_fn, train, save_checkpoint, adjust_learning_rate
import sys

def main():

    train_dataset = CardsDataset('data/images', 'data/txt_cards', 'data/general_labels/classes.txt')
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
    # path to model, set None if dont want to save
    model_path = None
    # path to checkpoint if there is one, set None if there isnt
    checkpoint = None 
    # batch size
    batch_size = 2
    # number of iterations to train
    iterations = 12000
    # number of workers for loading data in the DataLoader
    workers = 4
    # print training status every __ batches
    print_freq = 50
    lr = 1e-4  # learning rate
    decay_lr_at = [8000, 10000]  # decay learning rate after these many iterations
    decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
    momentum = 0.9  # momentum
    weight_decay = 5e-4  # weight decay
    grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

    cudnn.benchmark = True

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


    epochs = iterations // (len(train_dataset) // batch_size)
    epochs = 1
    print('Number of epochs:\n', epochs)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]
    print('Decaying learning rate at epochs:\n', decay_lr_at)
    

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






