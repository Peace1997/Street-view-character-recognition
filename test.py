import json,glob,torch
import numpy as np
from PIL import Image 
from torchvision import transforms
from Model import Model_CNN
from Model import Model_Resnet
device = torch.device("cpu")
class DataSet():
    def __init__(self,data,transform=None):
        self.data =data
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    def __getitem__(self, index):
        if self.transform is not None:
            img = self.transform(self.data)
        return img

    def __len__(self):
        return 1

# test Resnet
def predict(data):
    test_loader = torch.utils.data.DataLoader(
        DataSet(data,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                ),
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )
    model = Model_Resnet().to(device)
    model.load_state_dict(torch.load('model/resnet_model.pt'))
    output = model.predict(test_loader,device)
    print(output)

image = Image.open('img/000056.png')
predict(image)

# test cnn
# import json,glob,torch
# import numpy as np
# from PIL import Image 
# from torchvision import transforms
# from Model import Model_CNN

# device = torch.device("cpu")

# class DataSet():
#     def __init__(self,img_path,img_label,transform=None):
#         self.img_path = img_path
#         self.img_label = img_label
#         if transform is not None:
#             self.transform = transform
#         else:
#             self.transform = None
#     def __getitem__(self, index):
#         img = Image.open(self.img_path[index]).convert('RGB')

#         if self.transform is not None:
#             img = self.transform(img)

#         lbl = np.array(self.img_label[index], dtype=np.int)
#         lbl = list(lbl) + (6 - len(lbl)) * [10]

#         return img, torch.from_numpy(np.array(lbl))

#     def __len__(self):
#         return len(self.img_path)



# if __name__ == '__main__':

#     img = glob.glob('000005.png')
#     predict(img)
