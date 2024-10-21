# import library
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import os
from PIL import Image
import json
import torch.nn as nn
from collections import OrderedDict

# 1. Load and process the image dataset
# 1.1. parse argument
def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2img', type=str, help='Path to image')
    parser.add_argument('--chkpoint', type=str, help='path to checkpoint', default='checkpoint_path.pth')
    parser.add_argument('--cat2json', type=str, help='path to category to json file')
    parser.add_argument('--topk', type=int, default=5, help='Number of top k class, default 5')
    parser.add_argument('--gpu', action='store_true', default='gpu', help='use GPU or CPU for training')
    
    return parser.parse_args()

# check computer gpu available or not
def cpu_or_gpu(gpu):
    if gpu and torch.cuda.is_available():
        print(" ======> GPU CUDA activated")
        device = torch.device("cuda")
    else:
        print("====> GPU not found !!!")
        device = torch.device("cpu")
    return device

# check path exist 
def path_is_exist(path):
    exist = False
    if path == '':
        return False
    if not os.path.exists(path):
        print(f"Path {path} does not exist!")
        return False
    return True

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # TODO: Process a PIL image for use in a PyTorch model
    picture = Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    picture = transform(picture)
    return picture

def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    img = process_image(image_path)
    device = cpu_or_gpu(gpu)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pre_probs = model(img)
    output_probs = torch.exp(pre_probs)
    probs, indeces = output_probs.topk(topk)

    return probs, indeces

def load_checkpoint_data(path):
    # check path to image is exist
    if not path_is_exist(path):
        return
    checkpoint = torch.load(path)
    if checkpoint['structure'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['structure'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    else:
        raise ValueError('Model load arch error.')
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(model.classifier[0].in_features, 512)),
                          ('relu', nn.ReLU()),
                          ('d_out1', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(512, 256)),
                          ('d_out2', nn.Dropout(p=0.3)),
                          ('relu', nn.ReLU()),
                          ('output', nn.LogSoftmax(dim=1))
                        ]))
    
    print(checkpoint.keys())

    #model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])    
        
    return model

def img_prediction_show(cat2name, probs, indeces, topk = 5):    
    probs = probs[0].cpu().numpy()
    indeces = indeces[0].cpu().numpy()
    class_names = [cat2name[str(idx + 1)] for idx in indeces]
    print("Class names: ")
    print(class_names)
    #print("Probs: " + probs)
    for i in range(topk):
        print(f"{class_names[i]}: {probs[i] * 100:.2f}%")

# 2. Train the image classifier on your dataset
def main():
    args = parser_arguments()
    # check path to image is exist
    if not path_is_exist(args.path2img):
        return
    # check check point exist
    if not path_is_exist(args.chkpoint):
        return
    
    device = cpu_or_gpu(args.gpu)
    chkpoint = args.chkpoint
    img_path = args.path2img
    topk = args.topk
    cat2name = args.cat2json
    model = load_checkpoint_data(chkpoint)
    model.to(device)

    probs, indeces = predict(img_path, model, args.gpu, topk=topk)
    if cat2name:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
    img_prediction_show(cat_to_name, probs, indeces)

    
if __name__ == '__main__':
    main()
    
# Command
# python predict.py --path2img flowers/test/1/image_06743.jpg --chkpoint checkpoint_path.pth --gpu
# python predict.py --path2img flowers/test/1/image_06743.jpg --chkpoint checkpoint_path.pth --cat2json cat_to_name.json --topk 5
# python predict.py --path2img flowers/test/1/image_06743.jpg --chkpoint checkpoint_path.pth --cat2json cat_to_name.json --topk 5 --gpu

