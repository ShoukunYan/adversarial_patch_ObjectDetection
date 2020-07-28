import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torchvision import transforms, utils
from PIL import Image

import os

from dataset import get_loaders, read_picture



class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


class style_transfer(nn.Module):

    def __init__(self, feature_layers, style_img, original_img):
        super(style_transfer, self).__init__()
        self.feature_extractor = models.vgg16(pretrained=True).features.cuda()
        
        self.feature_layers = feature_layers

        self.style_img = style_img
        self.style_features = self.get_features(style_img)
        self.style_gram = self.gram_matrix(self.style_features)


        self.original_features = self.get_features(original_img)

        #self.tv_loss = TotalVariation()


    def gram_matrix(self, features):

        feature_grams = []

        for feature in features:

            F = feature.view(feature.size(1), feature.size(2)*feature.size(3))

            gram = F.mm(F.transpose(0,1))
            
            feature_grams.append(gram)

        return feature_grams


    def get_features(self, img):
        
        feature_maps = []
        x = img
        for index , layer in enumerate(self.feature_extractor):
            x = layer(x)
            #print(layer)
            if index in self.feature_layers:
                #print(index)
                feature_maps.append(x)

        return feature_maps

    def forward(self, input_img):

        input_features = self.get_features(input_img)
        input_grams = self.gram_matrix(input_features)
        
        content_loss = 0
        style_loss = 0

        for index in range(len(input_features)):
            feature_size = input_features[index].size()
            gram_size = input_grams[index].size()

            if index == 2:
                content_diff = 0.5 * (input_features[index] - self.original_features[index])**2
                content_loss = content_diff.view(feature_size[0],feature_size[1],feature_size[2]*feature_size[3]).sum() / content_diff.numel()

            #if index == 4:
            #    break
            gram_diff = ((input_grams[index] - self.style_gram[index])**2) / (4 * (gram_size[0]**2) * (feature_size[2]**4))
            #gram_diff = 0.5 * ((input_grams[index] - self.style_gram[index])**2) / input_grams[index].numel()
            style_loss += gram_diff.sum()

        #tv_loss =self.tv_loss(input_img.squeeze(0))


        loss =  10 * style_loss +  1* content_loss

        return loss



if __name__ == "__main__":

    
    iteration = 900

    #dataset = 'flower'
    #data_root = os.path.join('train_data', dataset)
    #datalodar = get_loaders(data_root, batch_size=64, normalization=False, augmentation=False)
    feature_layers = [4, 9, 16, 23, 30]
    

    #input_img = read_picture('train_data/flower/image_00001.jpg')
    input_img = read_picture('sample.jpg').cuda()
    
    transfer_img = torch.full((1, 3, 300, 300), 0.5).cuda()
    #transfer_img = input_img
    transfer_img.requires_grad_(True)


    style_img = read_picture('sky_2.jpg').cuda()
    #style_img = torch.ones((1,3,300,300)) 
    #input_img = torch.zeros((1,3,300,300))
    print(style_img.size())

    transformer = style_transfer(feature_layers, style_img, input_img).cuda()


    optimizer = optim.Adam([transfer_img], lr=0.03, amsgrad=True)

    for index in range(iteration):

        optimizer.zero_grad()

        loss = transformer(transfer_img)

        loss.backward(retain_graph=True)

        optimizer.step()
       
        transfer_img.data.clamp_(0,1)

        print('Iteration:{}, Loss:{}'.format(index, loss))

        if index % 10 == 0:
            img = transforms.ToPILImage()(transfer_img.squeeze(0).detach().cpu())
            img.save('samples/sample_{}.jpg'.format(index))

    exit()
    



    for img_batch in datalodar:
        img = img_batch[1, :, :, :]
        img = transforms.ToPILImage()(img.detach().cpu())
        img.save('sample.jpg')

        print(img_batch.size())


    