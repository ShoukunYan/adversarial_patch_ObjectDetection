import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import read_picture
from torchvision import transforms, utils
from PIL import Image

from load_data import *
from train_patch import PatchTrainer, yolo
import random
from StyleAutoEncoder.style_AE import *


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--det_weight', type=float, default=50)
    parser.add_argument('--style_weight', type=float, default=1)
    parser.add_argument('--content_weight', type=float, default=1)
    

    args = parser.parse_args()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from data_processing.DataLoader import FlatDirectoryImageDataset, \
        get_transform, get_data_loader, FoldersDistributedDataset

    data_source = FlatDirectoryImageDataset

    images_dir = "../BMSG-GAN/sourcecode/flowers/data/jpg"
    dataset = data_source(images_dir, transform=get_transform((256,256)))
    loader = get_data_loader(dataset, 6, 4)

    ae_model = Style_AutoEncoder().to(device)

    if args.pretrain:
        print('Weights Loading....')
        ae_model.load_state_dict(torch.load('models/AutoEncoder/ae.pth'))


    optimizer = optim.Adam(ae_model.parameters(), lr=0.003, amsgrad=True)

    style_img = read_picture('style_samples/style_pictures/sky_2.jpg', normalization=True).to(device)
    print(style_img.size())
    transformer = style_transfer(style_img).to(device)


    
    patch_attacher = PatchTrainer("paper_obj")
    yolo_part = yolo("paper_obj")

    max_lab = 14
    person_data = InriaDataset("./inria/Train/ourdataset/srcimage", "./inria/Train/ourdataset/srcimage/label", max_lab, 416,
                                                                shuffle=True)

    
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

            person_batch, lab_batch = person_data[int(random.random()*len(person_data))]
            person_batch = person_batch.unsqueeze(0).cuda()
            lab_batch = lab_batch.unsqueeze(0).cuda()
            p_img_batch = patch_attacher((out/2 + 0.5).clamp(0,1), person_batch, lab_batch, single_patch=False)

            person_batch = person_batch.expand(p_img_batch.size(0),-1,-1,-1)
            lab_batch = lab_batch.expand(p_img_batch.size(0),-1,-1)

            y_loss, det_loss = yolo_part(person_batch, lab_batch, p_img_batch)

            s_loss, c_loss = transformer(img_batch, out)

            loss = args.det_weight * y_loss + args.style_weight * s_loss + args.content_weight * c_loss

            loss.backward(retain_graph=True)

            optimizer.step()

            print('Iteration:{}, Loss:{}, S_loss:{}, C_loss:{}, Det_loss:{}'.format(index+epoch*len(loader), loss, s_loss, c_loss, det_loss))

            #input()
            if loss < max_loss:
                #max_loss = loss
                torch.save(ae_model.state_dict(), 'models/AutoEncoder/ae_1_1_{}.pth'.format(args.det_weight))

        torch.save(ae_model.state_dict(), 'models/AutoEncoder/ae_backup_1_1_{}.pth'.format(args.det_weight))

        
    




