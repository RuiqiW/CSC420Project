import torchvision.transforms as tt
from torch.utils.data import Dataset
import torch
from PIL import Image

class GDataset(Dataset):
    def __init__(self, df, directory):

        self.df = df
        self.dir = directory
        self.transform = tt.Compose([
            tt.Resize((160, 160)),
            tt.RandomCrop((96, 96)),
            tt.ToTensor(),
            tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index):
        
        id_ = self.df.iloc[index]['id']
        img_name = "/".join([self.dir, id_[:3], id_+".jpg"])
        image = Image.open(img_name)
        image = self.transform(image)
        return image, torch.tensor(int(self.df.iloc[index]['class_id'])).long()
        
