
import glob # File operations
import json
from PIL import Image # Pillow
import numpy as np
import torch
import tensorboard
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


#Dataset对数据集进行封装，提供索引的方式对数据样本进行读取

class DataSet(Dataset):
    def __init__(self,img_path,img_label,transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    # 通过给定索引值获取数据和标签
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 定长字符识别策略，使用10来进行填充;size == 6
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (6 - len(lbl)) * [10]

        return img, torch.from_numpy(np.array(lbl)) #numpy2tensor

    #获取数据集大小
    def __len__(self):
        return len(self.img_path)


# Train Data
train_path = glob.glob('./Data/mchar_train/*.png')
train_path.sort()
train_json = json.load(open('./Data/mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json]


# DataLoder对Dataset进行封装，提供批量读取的迭代读取
# 加入DataLoader之后，数据按照批次获取，每批次调用Dataset读取单个样本进行拼接，data格式[10,3,64,128] (batchsize，chanel，height，width)
train_loader = torch.utils.data.DataLoader(
        #数据扩增
        DataSet(train_path, train_label,
                   transforms.Compose([
                       transforms.Resize((64, 128)), #图像转变
                       transforms.ColorJitter(0.3, 0.3, 0.2), #修改亮度、对比度、饱和度
                       transforms.RandomRotation(5),#随机旋转
                       transforms.ToTensor(),#转换为tensor
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #标准化
            ])),
    batch_size=64, # 每批样本个数
    shuffle=True, # 是否打乱顺序
    num_workers=2, # 读取的线程个数
)


val_path = glob.glob('./Data/mchar_val/*.png')
val_path.sort()
val_json =  json.load(open('./Data/mchar_val.json'))
val_label = [val_json[x]['label'] for x in val_json]

val_loader = torch.utils.data.DataLoader(
    DataSet(val_path,val_label,
            transforms.Compose([
                transforms.Resize((64,128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            ),
    batch_size=64,
    shuffle=False,
    num_workers=2,
)

test_path = glob.glob('./Data/mchar_test_a/*.png')
test_path.sort()
test_label = [[1]] * len(test_path)
#print(len(test_path),len(test_label))

test_loader = torch.utils.data.DataLoader(
    DataSet(test_path,test_label,
            transforms.Compose([
                transforms.Resize((64,128)),
                transforms.RandomCrop((60,120)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            ),
    batch_size=64,
    shuffle=False,
    num_workers=2,
)

