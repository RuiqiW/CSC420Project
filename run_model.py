import pandas as pd
from torch.utils.data import DataLoader
import torch

from vgg16 import VGG16
from datasets import InferenceDataset, GDataset
import sys


def train_model(index, dir_, model_file="", 
                epochs=32, 
                model_save=False, model_save_name="vgg16_iter"):
    """ Training the model
        @param index: index file of the train data
        @param dir_: directory name that stores the properly formatted data
        @param model_file: model state file for continuous training
        @param epochs: number of iterations to run
        @param model_save: whether to save as a model file
        @param model_save_name: name of the saving model file
    """
    
    df = pd.read_csv(index).sample(frac=0.5)
    dataset = GDataset(df, dir_)
    
    batch_size = 108
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = VGG16()
    
    if model_file != "":
        model.load_state_dict(torch.load(model_file))
        for param in model.features[:34].parameters():
            param.requires_grad = False
    
    model.cuda()
    
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    optimizer = torch.optim.Adam(model.parameters())

        # train the model
    for e in range(epochs):

        train_loss = 0
        accuracy = 0
        counter = 0
        for x, t in dataloader:
            counter += 1
            print(round(counter / len(dataloader) * 100, 2), "%  ", end="\r")
            x, t = x.cuda(), t.cuda()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, t)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            y = softmax(z)
            top_p, top_class = y.topk(1, dim=1)
            accuracy += (top_class[:, 0] == t).sum().item()

        print(e, train_loss / len(dataloader), accuracy / len(dataset))
        if model_save:
            torch.save(model.state_dict(), model_save_name + str(e) +".pth")
    

    
    
def inference(index, dir_, model_file):
    """ Inferencing the model
        @param index: index file of the train data
        @param dir_: directory name that stores the properly formatted data
        @param model_file: model state file for continuous training
    """
    df = pd.read_csv(index).sample(frac=0.5)
    dataset = InferenceDataset(df, dir_)

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = VGG16()
    model.load_state_dict(torch.load(model_file))
    model.cuda()
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    loss_sum = 0
    accuracy = 0
    top_5_accuracy = 0
    counter = 0
    for x, t in dataloader:
        counter += 1
        print(round(counter / len(valid_loader) * 100, 2), "%  ", end="\r")
        x, t = x.cuda(), t.cuda()
        z = model(x)
        loss = criterion(z, t)
        loss_sum += loss.item()

        y = softmax(z)
        top_p, top_class = y.topk(5, dim=1)
        accuracy += (top_class[:, 0] == t).sum().item()
        top_5_accuracy += (top_class == t.unsqueeze(1).repeat(1, 5)).max(axis=1).values.sum().item()

    print("Loss:", loss_sum / len(dataloader))
    print("Top 1 Accuracy:", accuracy / len(dataset))
    print("Top 5 accuracy:", top_5_accuracy / len(dataset))
    
if __name__ == "__main__":
    print("Loading dataset:", sys.argv[1])
    print("Dataset folder:", sys.argv[3])
    print("Model:", sys.argv[2])
    
    index = sys.argv[1]
    dir_ = sys.argv[2]
    model_file = sys.argv[3] if sys.argv[3] else ""
    # inference(sys.argv[1], sys.argv[2], sys.argv[3])
    train_model(index, dir_, model_file=model_file, epochs=32, model_save=False)
