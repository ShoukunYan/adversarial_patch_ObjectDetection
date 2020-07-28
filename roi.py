import torch 
import torch.nn as nn

from load_data import *
#from train_patch import PatchTrainer

import patch_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class roi_feature(nn.Module):
  
    def __init__(self, bin_size):
        super(roi_feature, self).__init__()
        self.bin_size = bin_size


    def forward(self, original_features, lab, attack_features):
        # Supposed the bin is odd number

        # features.size = [batch_size, channel_num, width, heights]
        # lab.size = [batch_size, max_lab, 5]

        mesh_box = torch.zeros(original_features.size(0), self.bin_size, self.bin_size, 2).to(device)

        index = (torch.arange(self.bin_size).float() - int(self.bin_size / 2)).to(device)


        for batch in range(original_features.size(0)):

            mesh_y = 2 * index * lab[batch, 0, 3] / self.bin_size + (lab[batch, 0, 1] * 2 - 1)
            mesh_x = 2 * index * lab[batch, 0, 4] / self.bin_size + (lab[batch, 0, 2] * 2 - 1)

            #mesh_x = 2 * index * lab[batch, 0, 3] / self.bin_size 
            #mesh_y = 2 * index * lab[batch, 0, 4] / self.bin_size

            grid_x, grid_y  = torch.meshgrid(mesh_x, mesh_y)
            grid_x.to(device)
            grid_y.to(device)

            mesh_box[batch, :, :, 0] = grid_y
            mesh_box[batch, :, :, 1] = grid_x


        output_1 = nn.functional.grid_sample(original_features, mesh_box)

        output_2 = nn.functional.grid_sample(attack_features, mesh_box)

        return output_1, output_2


        

def fda_loss(original_features, attack_features):

    mean = torch.cat([attack_features.mean(1).unsqueeze(1)] * attack_features.size(1) ,1)
 

    wts_good = (attack_features > mean).float()
    wts_bad  = (attack_features < mean).float()

    size = attack_features.size()

    good_features = (attack_features * wts_good).view([size[0], size[1] * size[2] * size[3]])
    bad_features  = (attack_features * wts_bad).view([size[0], size[1] * size[2] * size[3]])

    loss = torch.log(good_features.norm(2,1)) - torch.log(bad_features.norm(2,1))

    return loss

def cos_loss(original_features, attack_features):
    
    size = original_features.size()
    original_features = original_features.view(size[0], size[1]*size[2]*size[3])
    attack_features = attack_features.view(size[0], size[1]*size[2]*size[3])

    loss = torch.cosine_similarity(original_features, attack_features, dim=1)
    return loss

def norm_loss(original_features, attack_features):
    
    size = original_features.size()
    original_features = original_features.view(size[0], size[1]*size[2]*size[3])
    attack_features = attack_features.view(size[0], size[1]*size[2]*size[3])

    diff = original_features - attack_features

    loss = -torch.norm(diff, p=1, dim=1) * 0.00001

    return loss







def main():

    config = patch_config.patch_configs['paper_obj']()
    #darknet_model = Darknet(config.cfgfile)
    #darknet_model.load_weights(config.weightfile)
    #   darknet_model = darknet_model.eval().cuda()
    fda_attacker = roi_feature(255)

    inria_batch_size = 10
    max_lab = 14

    train_loader = torch.utils.data.DataLoader(InriaDataset("./inria/Train/pos", "./inria/Train/pos/yolo-labels", max_lab, 416,
                                                                shuffle=True),
                                                   batch_size= inria_batch_size,
                                                   shuffle=True,
                                                   num_workers=10)

    for batch , [img_batch, lab_batch] in enumerate(train_loader):

        img_batch = img_batch
        lab_batch = lab_batch

        ori_feature, _  = fda_attacker([img_batch], lab_batch, [img_batch])

        img = ori_feature[0,...]
        img = img = transforms.ToPILImage()(img.detach().cpu())
        img.save('sample.jpg')

        img = img_batch[0,...]
        img = img = transforms.ToPILImage()(img.detach().cpu())
        img.save('sample_1.jpg')

        print(lab_batch[0,0,...])


        print('ROI extraction completed!')

        #loss = fda_loss(ori_feature, ori_feature)
        #loss = cos_loss(ori_feature, ori_feature)
        loss = norm_loss(ori_feature, ori_feature)

        print(loss.size())

        exit()


if __name__ == "__main__":

    main()




