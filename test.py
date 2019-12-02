import pandas as pd
from torch.utils.data import DataLoader
import torch
import numpy as np
import torchvision.transforms as tt
from PIL import Image

from vgg16 import VGG16
from datasets import InferenceDataset, GDataset
import sys, time
from ransac_matching import *

def test_hard_thresholding(index, dir_, model_file):
    start = time.time()

    df = pd.read_csv(index)
    dataset = InferenceDataset(df, dir_)

    train_df = pd.read_csv("train_try.csv")

    model = VGG16()
    model.load_state_dict(torch.load(model_file))
    model.cuda()
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    transform = tt.Compose([
            tt.Resize((128, 128)),
            tt.ToTensor(),
            tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    accuracy = 0
    for x, t_cpu in dataset:

        x_cpu = np.array(x)
        x, t = transform(x).unsqueeze(0).cuda(), t_cpu.unsqueeze(0).cuda()
        z = model(x)
        y = softmax(z)
        top_p, top_class = y.topk(1, dim=1)
        if top_p[:, 0].item() < 0.45:
            accuracy += t_cpu.item() == 1000
        else:
            accuracy += (top_class[:, 0] == t).sum().item()

    print("Top 1 Accuracy:", accuracy / len(dataset))
    end = time.time()
    print("Time:", end - start)
    
    
    
def test_ransac_surf(index, dir_, model_file):
    start = time.time()
    
    df = pd.read_csv(index)
    
    model = VGG16()
    model.load_state_dict(torch.load(model_file))
    model.cuda()
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    transform = tt.Compose([
            tt.Resize((128, 128)),
            tt.ToTensor(),
            tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    counter = 0
    test_index = pd.read_csv("train_try.csv")
    # dict_kps, dict_des = prepare_features(test_index)
    for row in df.iterrows():
        print(counter, end="\r")
        id_ = row[1]['id']
        img_name = "/".join([dir_, id_+".jpg"])
        image = Image.open(img_name).convert('RGB')
        x = transform(image).unsqueeze(0).cuda()
        t = torch.tensor(row[1]['class_id']).long().unsqueeze(0).cuda()
        z = model(x)
        y = softmax(z)
        top_p, top_class = y.topk(5, dim=1)
        if top_p[0, 0].item() > 0.75:
            counter += top_class[0, 0].item() == row[1]['class_id']
        else:
            top_class = randsac_matching_metrics(top_class[0].tolist(), 
                                                           img_name, 
                                                           test_index, 
                                                           "index")
#             top_class = randsac_matching_metrics_thresholding(top_class[0].tolist(), 
#                                                               img_name, 
#                                                               test_index, 
#                                                               "index", 
#                                                               dict_kp,
#                                                               dict_des)
            counter += top_class == row[1]['class_id']
    print("Accuracy:", counter / len(df))
    end = time.time()
    print("Time:", end - start)
    
    
def query_image(img_name, dir_, model_file):
    start = time.time()
    
    model = VGG16()
    model.load_state_dict(torch.load(model_file))
    model.cuda()
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    transform = tt.Compose([
            tt.Resize((128, 128)),
            tt.ToTensor(),
            tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    test_index = pd.read_csv("test_index.csv")
    image = Image.open(img_name).convert('RGB')
    x = transform(image).unsqueeze(0).cuda()
    z = model(x)
    y = softmax(z)
    top_p, top_class = y.topk(5, dim=1)
    if top_p[0, 0].item() > 0.75:
        top_class = top_class[0, 0].item()
    else:
        top_class = randsac_matching_metrics(top_class[0].tolist(), 
                                             img_name, 
                                             test_index, 
                                             "index")
    print("Prediction", top_class)
    end = time.time()
    print("Time:", end - start)
    
    
if __name__ == "__main__":
    print("Loading dataset:", sys.argv[1])
    print("Dataset folder:", sys.argv[3])
    print("Model:", sys.argv[2])

    index = sys.argv[1]
    dir_ = sys.argv[2]
    model_file = sys.argv[3] if sys.argv[3] else ""
    test_ransac_surf(sys.argv[1], sys.argv[2], sys.argv[3])
    #query_image(sys.argv[1], sys.argv[2], sys.argv[3])
