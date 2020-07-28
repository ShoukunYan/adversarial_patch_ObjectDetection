"""
Training code for Adversarial patch training


"""

import PIL
from tqdm import tqdm
from argparse import ArgumentParser

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
import torch.nn as nn
from torchvision import transforms, models
#from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time

from utils_yolo import get_region_boxes
from roi import fda_loss, cos_loss, norm_loss, roi_feature
from StyleAutoEncoder.Style_transfer import style_transfer
from dataset import read_picture
from darknet import *

class faster_rcnn(nn.Module):
    def __init__(self):
        super(faster_rcnn, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().to('cuda')

    

    def faster_rcnn_loss(self, output):

        labels = output['labels']
        prediction = output['scores']

        result = prediction[(labels==1)].max()
        
        return result 

    def forward(self, p_img_batch):
        
        output = self.model(p_img_batch)

        rcnn_loss = sum(map(self.faster_rcnn_loss, output)) / len(output)

        draw_img(p_img_batch, output)

        return rcnn_loss


class yolo(nn.Module):

    def __init__(self, mode, feature_loss=None, hyper_param = [0,1]):
        super(yolo, self).__init__()
        self.config = patch_config.patch_configs[mode]()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().to(self.device)
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).to(self.device)
        
        self.ROI = roi_feature(200)

        self.feature_function = {'fda_loss':fda_loss, 'cos_loss':cos_loss, 'norm_loss':norm_loss}
        self.feature_loss = feature_loss

        self.feature_weight = hyper_param[0]
        self.det_weight = hyper_param[1]

    def forward(self, img_batch, lab_batch, p_img_batch):

        # For testing the roi extract features
        #test_a , _ = self.ROI(img_batch, lab_batch, p_img_batch)
        #img = test_a[0,:]
        #img = transforms.ToPILImage()(img.detach().cpu())
        #img.save('roi.jpg')

        with torch.no_grad():
            
            ori_outputs = self.darknet_model(img_batch)[1]
        
        attack_outputs = self.darknet_model(p_img_batch)[1]

        feature_loss = 0
        confidences = []

        for index in range(len(ori_outputs)):
            
            ori_output = ori_outputs[index]
            attack_output = attack_outputs[index]

            ori_feature, attack_feature = self.ROI(ori_output, lab_batch, attack_output)

            if self.feature_loss is not None:
                feature_diff += (self.feature_function[self.feature_loss](ori_feature, attack_feature)).sum()
            else:
                feature_diff = 0

            feature_loss += feature_diff / img_batch.size(0)

            max_prob = self.prob_extractor(attack_output, num_anchors=3).unsqueeze(0) # Max confidence in each batch
            confidences.append(max_prob)

        confidences, _ = torch.cat(confidences, dim=0).max(dim=0)
        det_loss = confidences.mean()

        feature_loss = feature_loss / len(ori_outputs)

        # why do I design this loss?
        #feature_loss_norm = norm_loss(ori_feature, attack_feature).mean() * 0.0001

        return feature_loss, det_loss

class patch_loss(nn.Module):

    def __init__(self, mode):
        super(patch_loss, self).__init__()
        self.config =  patch_config.patch_configs[mode]()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).to(self.device)
        self.total_variation = TotalVariation().to(self.device)


    def forward(self, adv_patch):
        adv_patch = adv_patch.to(self.device)
        
        nps = self.nps_calculator(adv_patch)
        tv = self.total_variation(adv_patch.squeeze(0))
        tv_loss = tv*2.5

        loss = torch.max(tv_loss, torch.tensor(0.1).to(self.device))
        return loss




class PatchTrainer(nn.Module):
    def __init__(self, mode, feature_loss='cos_loss'):
        super(PatchTrainer, self).__init__()
        self.config = patch_config.patch_configs[mode]()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.patch_applier = PatchApplier().to(self.device)
        self.patch_transformer = PatchTransformer().to(self.device)
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).to(self.device)

        self.height = 416
        self.width = 416
        #subprocess.Popen(['tensorboard', '--logdir=runs'])
        #if name is not None:
        #    time_str = time.strftime("%Y%m%d-%H%M%S")
        #    return SummaryWriter(f'runs/{time_str}_{name}')
        #else:
        #    return SummaryWriter()

    def forward(self, adv_patch_cpu, img_batch, lab_batch, single_patch=True):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        #time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        # Initial the adv-patch with gray image or random image
        #adv_patch_cpu = self.generate_patch("gray").unsqueeze(0)
        #adv_patch_cpu = torch.cat([adv_patch_cpu,adv_patch_cpu],0)

        #adv_patch_cpu_1 = torch.full((1, 3, self.config.patch_size, self.config.patch_size), 0.5)
        #adv_patch_cpu_2 = torch.full((1, 3, self.config.patch_size, self.config.patch_size), 0)
        #adv_patch_cpu = torch.cat([adv_patch_cpu_1, adv_patch_cpu_2], 0)

        '''adv_patch_cpu = G_z
        

        img = G_z[0, :, :, :]
        img = transforms.ToPILImage()(img.detach().cpu())
        img.save('patch.jpg')

        img = G_z[1, :, :, :]
        img = transforms.ToPILImage()(img.detach().cpu())
        img.save('patch_1.jpg')'''





        #for i_batch, (img_batch, lab_batch) in enumerate(train_loader):
        with autograd.detect_anomaly():
            
            img_batch = img_batch.to(self.device)
            img_size = img_batch.size(3)
            
            lab_batch = lab_batch.to(self.device)
            # lab_batch.size = (batchsize, max_lab, 5)
            loc_param = [0.2, 0, 0]

            #img_batch = img_batch.to(self.device).repeat(adv_patch_cpu.size(0),1,1,1)
            
            #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
            
            adv_patch = adv_patch_cpu.to(self.device)

            #adv_patch = torch.cat([adv_patch, adv_patch], 0)

            adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, loc_param, do_rotate=True, rand_loc=True, single_patch=single_patch)
            
            if not single_patch:
                img_batch = img_batch.expand(adv_patch.size(0), -1, -1, -1)
            
            p_img_batch = self.patch_applier(img_batch, adv_batch_t)

            # resize to the size as input of yolo_model
            p_img_batch = F.interpolate(p_img_batch, (self.height, self.width))

            #print(p_img_batch.size())
            
            img = p_img_batch[1, :, :, :]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save('sample.jpg')

            


            return p_img_batch

        #et1 = time.time()
        #ep_det_loss = ep_det_loss/len(train_loader)
        #ep_nps_loss = ep_nps_loss/len(train_loader)
        #ep_tv_loss = ep_tv_loss/len(train_loader)
        #ep_loss = ep_loss/len(train_loader)

        #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
        #plt.imshow(im)
        #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

        #scheduler.step(ep_loss)
        '''if True:
            print('  EPOCH NR: ', epoch),
            print('EPOCH LOSS: ', ep_loss)
            print('  DET LOSS: ', ep_det_loss)
            print('  NPS LOSS: ', ep_nps_loss)
            print('   TV LOSS: ', ep_tv_loss)
            print('EPOCH TIME: ', et1-et0)
            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.show()
            #im.save("saved_patches/patchnew1.jpg")
            del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
            torch.cuda.empty_cache()
        #et0 = time.time()'''

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def generate_patch(type):
    """
    Generate a random patch as a starting point for optimization.

    :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
    :return:
    """
    if type == 'gray':
        adv_patch_cpu = torch.full((1, 3, 300, 300), 0.5)
    elif type == 'random':
        adv_patch_cpu = torch.rand((1, 3, 300, 300))

    return adv_patch_cpu




def draw_img(img_batch, output):

    x_1 = int(output[0]['boxes'][0][0])
    y_1 = int(output[0]['boxes'][0][1])
    x_2 = int(output[0]['boxes'][0][2])
    y_2 = int(output[0]['boxes'][0][3])
    

    img_batch[0,:, y_1, x_1:x_2] = 0
    img_batch[0,:, y_2, x_1:x_2] = 0
    img_batch[0,:, y_1:y_2, x_1] = 0
    img_batch[0,:, y_1:y_2, x_2] = 0
    
    
    
    img = transforms.ToPILImage()(img_batch[0,...].detach().cpu())
    img.save('box.jpg')


def main(): 



    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='paper_obj')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--style', action='store_true', default=False)
    parser.add_argument('--inria', action='store_true', default=False)


    parser.add_argument('--w_feature', type=float, default=0)
    parser.add_argument('--w_det', type=float, default=1)
    parser.add_argument('--pretrain', action='store_true', default=False)

    args = parser.parse_args()
    

    inria_batch_size = args.batch_size
    max_lab = 14

    if args.inria:

        train_loader = torch.utils.data.DataLoader(InriaDataset("./inria/Train/pos/", "./inria/Train/pos/yolo-labels", max_lab, 416,
                                                                shuffle=True),
                                                   batch_size= inria_batch_size,
                                                   shuffle=True,
                                                   num_workers=10)
    else:

        train_loader = torch.utils.data.DataLoader(InriaDataset("./inria/Train/ourdataset/srcimage", "./inria/Train/ourdataset/srcimage/label", max_lab, 416,
                                                                shuffle=True),
                                                   batch_size= inria_batch_size,
                                                   shuffle=True,
                                                   num_workers=10)

    trainer = PatchTrainer(args.config)
    rcnn  = faster_rcnn()
    yolo_part = yolo(args.config)
    self_loss = patch_loss(args.config)




    input_img = read_picture('style_samples/content_pictures/sample.jpg').cuda()
    style_img = read_picture('style_samples/style_pictures/sky.jpg').cuda()
    feature_layers = [4, 9, 16, 23, 30]
    transformer = style_transfer(style_img).cuda()
    
    if args.pretrain:
        
        adv_patch_cpu = Image.open('patch/patch.jpg').convert('RGB')
        adv_patch_cpu = transforms.ToTensor()(adv_patch_cpu).unsqueeze(0).cuda()
    else:
        adv_patch_cpu = generate_patch('gray').cuda()
    
    adv_patch_cpu.requires_grad_(True)
    

    if args.optim == 'adam':
        optimizer = optim.Adam([adv_patch_cpu], lr=args.lr, amsgrad=True)
    else:
        optimizer = optim.SGD([adv_patch_cpu], lr=args.lr, momentum=0.9)

    best_attack = 1.0


    print('One epoch is {}'.format(len(train_loader)))

    num_epochs = args.epochs
    for epoch in range(num_epochs):

        acc = 0
        for index, [img_batch, lab_batch] in enumerate(train_loader):

            img_batch = img_batch.to('cuda')
            lab_batch = lab_batch.to('cuda')

            p_img_batch = trainer(adv_patch_cpu, img_batch, lab_batch)

            img = p_img_batch[0, :, :, :]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save('style_patch.jpg')
            
            feature_loss, det_loss = yolo_part(img_batch, lab_batch, p_img_batch)
            tv_loss = self_loss(adv_patch_cpu)
            
            loss = args.w_feature * feature_loss + args.w_det * det_loss + tv_loss

            #loss.backward(retain_graph=True)
            #style_loss.backward()              
            #该情况会出现需要retain_graph = True 的问题
            if args.style:
                style_loss = transformer(adv_patch_cpu)
                style_loss.backward(retain_graph=True)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
            adv_patch_cpu.data.clamp_(0,1)

            acc += det_loss

            #print("\r Epoch:{}, Batch:{}/{}, Loss:{}, Confidence:{}".format(epoch, index, len(train_loader), loss, prob), end=' ')
            print("\r Epoch:{}, Batch:{}/{}, Loss:{}, Confidence:{}".format(epoch, index, len(train_loader), loss, det_loss), end=' ')
        print('Average acc:{}'.format(acc/len(train_loader)))

        if acc/len(train_loader) < best_attack:
            best_attack = acc/len(train_loader)
            img = transforms.ToPILImage()(adv_patch_cpu[0,...].detach().cpu())
            img.save('patch/patch.jpg')



if __name__ == '__main__':
    main()


