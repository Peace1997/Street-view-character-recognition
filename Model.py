
import torch.nn as nn
import torch
import numpy as np

import torchvision.models as models

class Model_CNN(nn.Module):
    def __init__(self):
        super(Model_CNN, self).__init__()
        # build network
        self.cnn = nn.Sequential(
            # input shape: (channel=3,height=64,width=128)
            nn.Conv2d( # convolution
                in_channels= 3,
                out_channels=16,
                kernel_size=(3,3), #filter
                stride=(1,1),
                padding = 1
            ),
            # After once convolution shape : (16,64,128)
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=2),#pooling
            # After once pooling shape：(16,32,64)

            nn.Conv2d(16,32,(3,3),(1,1),1),
            # # After twice convolution shape :（32,32,64)
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            # After twice pooling shape : (32,16,32)

            nn.Conv2d(32,64,(3,3),(1,1),1),
            # After third convolution shape:(64,16,32)
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            # After third convolution shaper: (64,8,16)

            # nn.Conv2d(64,128,(3,3),(1,1)),
            # # After fourth convolution shaper: (128,,16)
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.MaxPool2d(2),
            # # After fourth convolution shaper: (128,4,8)
        )
        print(self.cnn)
        #并联六个全连接层进行分类
        self.fc1 = nn.Linear(64*8*16,11)
        self.fc2 = nn.Linear(64*8*16,11)
        self.fc3 = nn.Linear(64*8*16,11)
        self.fc4 = nn.Linear(64*8*16,11)
        self.fc5 = nn.Linear(64*8*16,11)
        self.fc6 = nn.Linear(64*8*16,11)

    def forward(self,img):
        #print(img.shape)
        feat = self.cnn(img)

        feat = feat.view(feat.shape[0],-1) #展平 （batchsize,64*8*16）
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)

        return c1,c2,c3,c4,c5,c6

    def mytrain(self,train_loader,loss_func,optimizer,device):
        #切换模型为训练模式
        self.train()
        accuracy = []
        train_loss = []
        for i,(data,label) in enumerate(train_loader):
            c0,c1,c2,c3,c4,c5 = self(data.to(device))
            label = label.long().to(device)
            loss = loss_func(c0, label[:, 0]) + \
                   loss_func(c1, label[:, 1]) + \
                   loss_func(c2, label[:, 2]) + \
                   loss_func(c3, label[:, 3]) + \
                   loss_func(c4, label[:, 4]) + \
                   loss_func(c5, label[:, 5])

            loss /=6
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy.append((c0.argmax(1) == label[:, 0]).sum().item() * 1.0 / c0.shape[0])

        print("train_accuracy:",np.mean(accuracy))
        return np.mean(train_loss)

    def myvalidat(self,val_loader,loss_func,device):
        #切换模型为预测模式
        self.eval()

        val_loss = []
        accuracy = []
        #不记录模型梯度信息
        with torch.no_grad():
            for i,(data,label) in enumerate(val_loader):
                c0,c1,c2,c3,c4,c5 = self(data.to(device))
                label = label.long().to(device)
                loss = loss_func(c0, label[:, 0]) + \
                       loss_func(c1, label[:, 1]) + \
                       loss_func(c2, label[:, 2]) + \
                       loss_func(c3, label[:, 3]) + \
                       loss_func(c4, label[:, 4]) + \
                       loss_func(c5, label[:, 5])
                loss /= 6

                val_loss.append(loss.item())

                accuracy.append((c0.argmax(1) == label[:, 0]).sum().item() * 1.0 / c0.shape[0])

        print("val_accuracy:",np.mean(accuracy))
        return np.mean(val_loss)

    def predict(self,test_loader,device):

        self.eval()

        is_init = True

        #不记录模型梯度信息
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                c0, c1, c2, c3, c4, c5 = self(data.to(device))
                # 100 x 11->100 x 1 转换成1列 axis--第一维
                l0 = np.reshape(c0.numpy().argmax(axis=1),(-1,1))
                l1 = np.reshape(c1.numpy().argmax(axis=1),(-1,1))
                l2 = np.reshape(c2.numpy().argmax(axis=1),(-1,1))
                l3 = np.reshape(c3.numpy().argmax(axis=1),(-1,1))
                l4 = np.reshape(c4.numpy().argmax(axis=1),(-1,1))
                l5 = np.reshape(c5.numpy().argmax(axis=1),(-1,1))

                #合并 100 x 6
                tmp = np.concatenate((l0,l1,l2,l3,l4,l5),axis=1)
                if is_init:
                    pred_labels = tmp
                    is_init = False
                else:
                    pred_labels = np.concatenate((pred_labels,tmp),axis=0)
        return pred_labels

class Model_Resnet(nn.Module):
    def __init__(self):
        super(Model_Resnet, self).__init__()

        # 继承resnet18
        model_conv = models.resnet18(pretrained=True)
        # 将resnet18的最后一个池化层修改为自适应的全局平均池化层
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        # 微调，把fc层删除
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        # 自定义fc层

        self.hd_fc1 = nn.Linear(512, 128)
        self.hd_fc2 = nn.Linear(512, 128)
        self.hd_fc3 = nn.Linear(512, 128)
        self.hd_fc4 = nn.Linear(512, 128)
        self.hd_fc5 = nn.Linear(512, 128)
        self.hd_fc6 = nn.Linear(512, 128)
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.25)
        self.dropout_3 = nn.Dropout(0.25)
        self.dropout_4 = nn.Dropout(0.25)
        self.dropout_5 = nn.Dropout(0.25)
        self.dropout_6 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, 11)
        self.fc2 = nn.Linear(128, 11)
        self.fc3 = nn.Linear(128, 11)
        self.fc4 = nn.Linear(128, 11)
        self.fc5 = nn.Linear(128, 11)
        self.fc6 = nn.Linear(128, 11)

    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        # print(feat.shape)
        feat1 = self.hd_fc1(feat)
        feat2 = self.hd_fc2(feat)
        feat3 = self.hd_fc3(feat)
        feat4 = self.hd_fc4(feat)
        feat5 = self.hd_fc5(feat)
        feat6 = self.hd_fc6(feat)
        feat1 = self.dropout_1(feat1)
        feat2 = self.dropout_2(feat2)
        feat3 = self.dropout_3(feat3)
        feat4 = self.dropout_4(feat4)
        feat5 = self.dropout_5(feat5)
        feat6 = self.dropout_6(feat6)
        c1 = self.fc1(feat1)
        c2 = self.fc2(feat2)
        c3 = self.fc3(feat3)
        c4 = self.fc4(feat4)
        c5 = self.fc5(feat5)
        c6 = self.fc6(feat6)
        return c1, c2, c3, c4, c5, c6

    def mytrain(self,train_loader,loss_func,optimizer,device):
        self.train()
        train_loss = []
        accuracy = []
        for i,(data,label) in enumerate(train_loader):
            c0,c1,c2,c3,c4,c5 = self(data.to(device))
            label = label.long().to(device)
            loss = loss_func(c0, label[:, 0]) + \
                   loss_func(c1, label[:, 1]) + \
                   loss_func(c2, label[:, 2]) + \
                   loss_func(c3, label[:, 3]) + \
                   loss_func(c4, label[:, 4]) + \
                   loss_func(c5, label[:, 5])
            loss /= 6

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            accuracy.append((c0.argmax(1) == label[:, 0]).sum().item() * 1.0 / c0.shape[0])
        print("train_accuracy:",np.mean(accuracy))

        return np.mean(train_loss)

    def myvalidat(self,val_loafer,loss_func,device):
        self.eval()
        val_loss = []
        accuracy = []
        with torch.no_grad():
            for i,(data,label) in enumerate(val_loafer):
                # if use_cuda:
                #     data = data.cuda()
                #     label = label.cuda()
                c0,c1,c2,c3,c4,c5 = self(data.to(device))
                label = label.long().to(device)
                loss = loss_func(c0, label[:, 0]) + \
                       loss_func(c1, label[:, 1]) + \
                       loss_func(c2, label[:, 2]) + \
                       loss_func(c3, label[:, 3]) + \
                       loss_func(c4, label[:, 4]) + \
                       loss_func(c5, label[:, 5])
                loss /=6
                val_loss.append(loss.item())
                accuracy.append((c0.argmax(1) == label[:, 0]).sum().item() * 1.0 / c0.shape[0])

        print("val_accuracy:",np.mean(accuracy))

        return np.mean(val_loss)
    def predict(self,test_loader,device):

        # 切换模型为训练模式
        self.eval()

        is_init = True

        #不记录模型梯度信息
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                c0, c1, c2, c3, c4, c5 = self(data.to(device))
                # 100 x 11->100 x 1 转换成1列 axis--第一维
                l0 = np.reshape(c0.numpy().argmax(axis=1),(-1,1))
                l1 = np.reshape(c1.numpy().argmax(axis=1),(-1,1))
                l2 = np.reshape(c2.numpy().argmax(axis=1),(-1,1))
                l3 = np.reshape(c3.numpy().argmax(axis=1),(-1,1))
                l4 = np.reshape(c4.numpy().argmax(axis=1),(-1,1))
                l5 = np.reshape(c5.numpy().argmax(axis=1),(-1,1))

                #合并 100 x 6
                tmp = np.concatenate((l0,l1,l2,l3,l4,l5),axis=1)
                if is_init:
                    pred_labels = tmp
                    is_init = False
                else:
                    pred_labels = np.concatenate((pred_labels,tmp),axis=0)
        return pred_labels

