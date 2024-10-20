
# import library
import argparse
import os
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from torch import nn
import torch.optim as optim
from collections import OrderedDict
from time import time
import json

# 1. Load and process the image dataset
# 1.1. parse argument
def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path for data directory')
    parser.add_argument('--arch', type=str, help='Training Model', default='vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate, default 0.001')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units, default 512')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', default='gpu', help='use GPU or CPU for training')

    return parser.parse_args()

# 1.2. Loading data
def loading_data(path='flowers'):
    data_path = path
    train_path = path + '/train'
    valid_path = path + '/valid'
    test_path = path + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    transforms_train = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])
    print("====> Transform train done")

    transform_test = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    print("====> Transform test done")
    
    transform_valid = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])
    print("====> Transform valid done")

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_path, transform=transforms_train)
    print("====> Train datasets done")
    test_datasets = datasets.ImageFolder(test_path, transform=transform_test)
    print("====> Test datasets done")
    valid_datasets = datasets.ImageFolder(valid_path, transform=transform_valid)
    print("====> Valid datasets done")

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    print("====> Train loader done")
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    print("====> Valid loader done")
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    print("====> Test loader done")

    return train_datasets, train_loader, valid_loader, test_loader

# check path exist 
def path_is_exist(path):
    exist = False
    if path == '':
        return False
    if not os.path.exists(path):
        print(f"Path {path} does not exist!")
        return False
    return True

# check computer gpu available or not
def cpu_or_gpu(gpu):
    if gpu and torch.cuda.is_available():
        print(" ======> GPU CUDA activated")
        device = torch.device("cuda")
    else:
        print("====> GPU not found !!!")
        device = torch.device("cpu")
    return device

# load model for pre-train
def model_pre_train(arch, hidden_units):
    if arch == 'vgg16':
        print("Model VGG16")
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        print("Model densenet121")
        model = models.densenet121(pretrained=True)
    else:
        print("Unsupport model")
        return
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model.classifier[0].in_features, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 256)),
        ('output', nn.LogSoftmax(dim=1))    
    ]))
    print("=====> Model pretrain ")
    print(model)
    print("===========================")
    return model

# 2. Train the image classifier on your dataset
def main():
    args = parser_arguments()
    if not path_is_exist(args.data_dir):
        return
    print(f"Path {args.data_dir} found!")
    device = cpu_or_gpu(args.gpu)
    epochs = args.epochs
    hidden_units = args.hidden_units
    arch = args.arch
    learning_rate = args.learning_rate
    steps = 0
    running_loss = 0
    print_every = 10
    criterion = nn.NLLLoss()
    model = model_pre_train(arch=arch, hidden_units=hidden_units)
    model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    train_data, train_loader, valid_loader, test_loader = loading_data(args.data_dir)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # 3. Use the trained classifier to predict image content
    for epoch in range(epochs):
        t = time()
        for inputs, labels in train_loader:
            steps+= 1
            # Move input & label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            # clear gradient
            optimizer.zero_grad()
            print(f" Done clear gradients step: {steps} ")

            # forward pass
            logps = model.forward(inputs)
            print(" Done Move forward")
            loss = criterion(logps, labels)
            # backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                loss_val = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        loss_batch = criterion(logps, labels)
                        loss_val += loss_batch.item()

                        # calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(
                    f"======> Epoch {epoch + 1}/{epoch}.. "
                    f"======> Train loss: {running_loss/print_every:.3f}.."
                    f"======> Validation loss: {loss_val/len(valid_loader):.3f}.."
                    f"======> Validation accuracy: {accuracy/len(valid_loader):.3f}"
                    )
                running_loss = 0
                model.train()
        print(f" ========> Elapse time for epoch: { epoch+1, time() - t}")

    print("=============== Finish training ==================================")

    # TODO: Save the checkpoint
    save_checkpoint_path = 'checkpoint_path.pth'
    model.class_to_idx = train_data.class_to_idx
    torch.save(
        {
            'class_to_idx': model.class_to_idx,
            'model_state_dict': model.state_dict(),
            'arch': arch
        }, 
        save_checkpoint_path
    )



if __name__ == '__main__':
    main()
    
# Command
# python train.py flowers --gpu --epochs 10
# python train.py flowers --learning_rate 0.001 --hidden_units 512 --epochs 3 --arch vgg16 --gpu