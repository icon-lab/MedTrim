from torch.utils.data import Dataset
from PIL import Image

class TripletDataset(Dataset):
    def __init__(self, data, img_transform=None):
        self.data = data
        self.img_transform = img_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load and optionally transform images
        img_a = Image.open(item['img_a']).convert('RGB')
        img_p = Image.open(item['img_p']).convert('RGB')
        img_n = Image.open(item['img_n']).convert('RGB')

        if self.img_transform:
            img_a = self.img_transform(img_a)
            img_p = self.img_transform(img_p)
            img_n = self.img_transform(img_n)

        return {
            'img_a': img_a,
            'img_p': img_p,
            'img_n': img_n,
            'text_a': item['text_a'],
            'text_p': item['text_p'],
            'text_n': item['text_n']
        }
