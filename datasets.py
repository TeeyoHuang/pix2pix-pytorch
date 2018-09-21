import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, args, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.args = args
        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        #print(self.files)
        #input()

    def __getitem__(self, index):

        img = Image.open(self.files[index])
        w, h = img.size

        if self.args.which_direction == 'AtoB':
            img_A = img.crop((0, 0, w/2, h))
            img_B = img.crop((w/2, 0, w, h))
        else:
            img_B = img.crop((0, 0, w/2, h))
            img_A = img.crop((w/2, 0, w, h))


        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)


# Configure dataloaders
def Get_dataloader(args):
    transforms_ = [ transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    train_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name), transforms_=transforms_,mode='train'),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)

    test_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name), transforms_=transforms_, mode='test'),
                            batch_size=10, shuffle=True, num_workers=1, drop_last=True)

    val_dataloader = DataLoader(ImageDataset(args, "%s/%s" % (args.data_root,args.dataset_name), transforms_=transforms_, mode='val'),
                            batch_size=10, shuffle=True, num_workers=1, drop_last=True)

    return train_dataloader, test_dataloader, val_dataloader
