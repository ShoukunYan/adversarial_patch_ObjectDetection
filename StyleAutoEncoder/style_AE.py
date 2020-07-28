import torch
import torch.nn as nn
import torch.nn.functional as F

from Style_transfer import *
from torchvision import transforms, utils
from PIL import Image



import random

def conv(in_channel, out_channel, kernel_size, strides):

    padding = int(kernel_size / 2)

    return nn.Conv2d(in_channel, out_channel, kernel_size, stride=strides, padding=padding, padding_mode='reflect')

class deconv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, strides):
        super(deconv2d, self).__init__()
        self.up = nn.Upsample(scale_factor=2*strides)
        self.conv1 = conv(in_channel, out_channel, kernel_size, strides)
        self.out_channel = out_channel

    def forward(self, x):

        x = self.up(x)
        x = self.conv1(x)
        x = nn.InstanceNorm2d(self.out_channel)(x)
        x = nn.ReLU(inplace = True)(x)

        return x



class resblock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size):
        super(resblock, self).__init__()

        self.conv1 = conv(in_channel, out_channel, kernel_size, 1)
        self.conv2 = conv(in_channel, out_channel, kernel_size, 1)


    def forward(self, x):

        original = x
        x = nn.ReLU(inplace=True)(self.conv1(x))
        x = self.conv2(x)

        return x + original

class encoder(nn.Module):

    def __init__(self):
        
        super(encoder, self).__init__()
        self.image_size = 256

        self.feature = nn.Sequential(
            conv(3, 32, 9, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),

            conv(32, 64, 3, 2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            conv(64, 128, 3, 2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )

        res_list = [resblock(128,128,3) for _ in range(5)]
        self.resbackbone = nn.Sequential(*res_list)

    def forward(self, x):
        
        x = F.pad(x, (10,10,10,10), "reflect")

        x = self.feature(x)
        x = self.resbackbone(x)

        return x


class decoder(nn.Module):

    def __init__(self):
        super(decoder, self).__init__()

        self.deconv1 = deconv2d(128, 64, 3, 2)
        self.deconv2 = deconv2d(64, 32, 3, 2)
        self.final = nn.Sequential(conv(32, 3, 9, 1), nn.InstanceNorm2d(3), nn.Tanh())

    def forward(self, x):

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.final(x)

        
        img = (x[0, :, 10:266, 10:266] + 1) /2
        img = transforms.ToPILImage()(img.detach().cpu())
        img.save('sample_transfer.jpg')

        return x[:,:,10:266,10:266]


class Style_AutoEncoder(nn.Module):
    def __init__(self):
        
        super(Style_AutoEncoder, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x) 

        return x




if __name__ == "__main__":

    import sys 
    import argparse

    sys.path.append('../')
    from dataset import read_picture


    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--w_closs', type=float, default=1)
    parser.add_argument('--w_sloss', type=float, default=1)

    args = parser.parse_args()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from data_processing.DataLoader import FlatDirectoryImageDataset, \
        get_transform, get_data_loader, FoldersDistributedDataset

    data_source = FlatDirectoryImageDataset

    images_dir = "../../BMSG-GAN/sourcecode/flowers/data/jpg"
    dataset = data_source(images_dir, transform=get_transform((256,256)))
    loader = get_data_loader(dataset, 6, 4)

    ae_model = Style_AutoEncoder().to(device)

    if args.pretrain:
        print('Weights Loading....')
        ae_model.load_state_dict(torch.load('models/AutoEncoder/ae.pth'))


    optimizer = optim.Adam(ae_model.parameters(), lr=0.003, amsgrad=True)

    style_img = read_picture('../style_samples/style_pictures/sky_2.jpg', normalization=True).to(device)
    print(style_img.size())
    transformer = style_transfer(style_img).to(device)


    
    print("Success!")
    print("One epoch == > {} batch".format(len(loader)))

    
    max_loss = 999999

    epochs = 200
    for epoch in range(epochs):
        for index, img_batch in enumerate(loader):

            transformer.train()
            optimizer.zero_grad()

            img_batch = img_batch.to(device)
            out = ae_model(img_batch)

            s_loss, c_loss = transformer(img_batch, out)

            loss = args.w_sloss * s_loss + args.w_closs * c_loss

            loss.backward(retain_graph=True)

            optimizer.step()

            print('Iteration:{}, Loss:{}, S_loss:{}, C_loss:{}'.format(index+epoch*len(loader), loss, s_loss, c_loss))

            if loss < max_loss:
                max_loss = loss
                torch.save(ae_model.state_dict(), '../models/AutoEncoder/style_ae_{}_{}_.pth'.format(args.w_sloss, args.w_closs))

        torch.save(ae_model.state_dict(), '../models/AutoEncoder/style_ae_backup_{}_{}.pth'.format(args.w_sloss, args.w_closs))

        
    




