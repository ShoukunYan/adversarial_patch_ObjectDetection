import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils_yolo import *
from darknet import *
#from load_data import PatchTransformer, PatchApplier, InriaDataset
from load_data import *
import json


def create_paths(savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(os.path.join(savedir, 'clean_images')):
        os.makedirs(os.path.join(savedir, 'clean_images'))
    if not os.path.exists(os.path.join(savedir, 'clean_labels')):
        os.makedirs(os.path.join(savedir, 'clean_labels'))


def gen_clean_padding_image(imgdir, savedir, img_size):
    print("generating clean padding images")
    file_list = os.listdir(imgdir)
    cnt = 0
    for imgfile in file_list:
        # print cnts
        cnt = cnt + 1
        print("{}/{}".format(cnt, len(file_list)), end='\r')
        # main
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]
            # padding
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')
            w, h = img.size
            if w == h:
                padded_img = img
            else:
                dim_to_pad = 1 if w < h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                    padded_img.paste(img, (int(padding), 0))
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                    padded_img.paste(img, (0, int(padding)))
            resize = transforms.Resize((img_size, img_size))
            padded_img = resize(padded_img)
            cleanname = name + ".png"
            # save file
            padded_img.save(os.path.join(savedir, 'clean_images', cleanname))
    print("done#")


def gen_clean_labels(savedir, datrknet_model):
    print("generating clean labels")
    file_list = os.listdir(os.path.join(savedir, 'clean_images'))
    cnt = 0
    for imgfile in file_list:
        # print cnts
        cnt = cnt + 1
        print("{}/{}".format(cnt, len(file_list)), end='\r')
        # open
        name = os.path.splitext(imgfile)[0]
        imgfile = os.path.abspath(os.path.join(savedir, 'clean_images', imgfile))
        img = Image.open(imgfile).convert('RGB')
        # detect
        boxes = do_detect(darknet_model, img, 0.4, 0.4, 1)
        boxes = nms(boxes, 0.4)
        txt_name = name + ".txt"
        txt_path = os.path.abspath(os.path.join(savedir, 'clean_labels', txt_name))
        textfile = open(txt_path, 'w+')
        for box in boxes:
            cls_id = box[6]
            if (cls_id == 0):  # if person
                x_center = box[0]
                y_center = box[1]
                width = box[2]
                height = box[3]
                textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
        textfile.close()
    print("done@")


def generate_patch(patch_file):
    patch_name = os.path.split(patch_file)[1].replace('.', '_')
    if not os.path.exists(os.path.join(savedir, "patch", patch_name)):
        os.makedirs(os.path.join(savedir, "patch", patch_name))
    # read patch
    patch_img = Image.open(patch_file).convert('RGB')
    #tf = transforms.Resize((patch_size, patch_img)
    #patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()
    adv_patch = adv_patch.unsqueeze(0)

    print("starting generate path for {}".format(patch_name))
    file_list = os.listdir(os.path.join(savedir, 'clean_images'))
    cnt = 0
    for imgfile in file_list:
        # print cnts
        cnt = cnt + 1
        print("{}/{}".format(cnt, len(file_list)), end='\r')

        # open image file and txt file
        name = os.path.splitext(imgfile)[0]
        imgfile = os.path.abspath(os.path.join(savedir, 'clean_images', imgfile))
        padded_img = Image.open(imgfile).convert('RGB')
        txt_name = name + ".txt"
        txt_path = os.path.abspath(os.path.join(savedir, 'clean_labels', txt_name))

        # get txt
        if os.path.getsize(txt_path):  # check to see if label file contains data.
            label = np.loadtxt(txt_path)
        else:
            label = np.ones([5])
        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        transform = transforms.ToTensor()
        padded_img = transform(padded_img).cuda()
        img_fake_batch = padded_img.unsqueeze(0)
        lab_fake_batch = label.unsqueeze(0)

        

        
        
        lab_fake_batch = lab_fake_batch.cuda()
        adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, loc_param=[0.2,0,0], do_rotate=True, rand_loc=False,single_patch=True)
        p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
        p_img = p_img_batch.squeeze(0)
        p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
        properpatchedname = name + ".png"
        p_img_pil.save(os.path.join(savedir, 'patch', patch_name, properpatchedname))
    print("done=")


def evaluate_patch(patch_file):
    patch_name = os.path.split(patch_file)[1].replace('.', '_')
    print("evaluating for {}".format(patch_name))

    eval_wid = img_size
    eval_hei = img_size

    iou_thresh = 0.5
    min_box_scale = 8. / img_size

    file_list = os.listdir(os.path.join(savedir, 'clean_images'))
    cnt = 0

    correct = 0
    total = 0
    t_iou = 0

    with torch.no_grad():
        for imgfile in file_list:
            # print cnts
            cnt = cnt + 1
            # print("{}/{}".format(cnt, len(file_list)), end='\r')
            # calculate img_path lab_path
            name = os.path.splitext(imgfile)[0]
            txt_name = name + '.txt'
            img_path = os.path.join(savedir, 'patch', patch_name, imgfile)
            lab_path = os.path.join(savedir, 'clean_labels', txt_name)
            print(lab_path)
            truths = read_truths_args(lab_path, min_box_scale)

            img = Image.open(img_path).convert('RGB')
            boxes = do_detect(darknet_model, img, 0.4, 0.4, 1)

            sub_correct = 0
            sub_t_iou = 0
            for i in range(truths.shape[0]):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0]
                best_iou = 0
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    best_iou = max(iou, best_iou)
                if best_iou > iou_thresh:
                    sub_correct = sub_correct + 1
                sub_t_iou = sub_t_iou + best_iou

            if truths.shape[0] == 0:
                avg_iou = 0
            else:
                avg_iou = sub_t_iou/truths.shape[0]
            print("{:20s} box:{:02d} correct:{:02d} iou:{:.2f}".format(name, truths.shape[0], sub_correct, avg_iou))
            total = total + truths.shape[0]
            correct = correct + sub_correct
            t_iou = t_iou + sub_t_iou

    print("=========================result=======================")
    print("box:", total)
    print("correct: {:.6f}".format(correct/total))
    print("iou: {:.5f}".format(t_iou/total))
    return (total, correct, correct/total, t_iou/total)


if __name__ == '__main__':
    print("Setting everything up")
    imgdir = "Testset"
    cfgfile = "cfg/yolov3.cfg"
    weightfile = "weights/yolov3.weights"
    pth_list = ["patch/patch.jpg",
                #"patch/patch.jpg",
                "patch/object_score.png"]
                #"patch/patch_cat.jpg",
                #"patch/patch_cat_59.jpg",
                #"patch/patch_cat_62.jpg",
                #"patch/patch_0_1.jpg",
                #"patch/patch_0_5_1.jpg",
                #"patch/patch_1_1.jpg"]
    savedir = "testing"

    darknet_model = Darknet(cfgfile)
    #darknet_model.print_network()
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    batch_size = 1
    max_lab = 14
    img_size = darknet_model.height

    patch_size = 300



    clean_results = []
    noise_results = []
    patch_results = []

    create_paths(savedir=savedir)
    #gen_clean_padding_image(imgdir, savedir, img_size)
    #gen_clean_labels(savedir, darknet_model)

    eval_d = []
    for patch in pth_list:
        generate_patch(patch)
        d = evaluate_patch(patch)
        eval_d.append({"name": patch, "d": d})

    print("************\n****************\n******************\n*************")
    for k in eval_d:
        print("{:30s}\tbox:{:03d}\tcorrect:{:03d}\tc_ratio:{:.2f}\tiou:{:.2f}".format(
            k["name"], k["d"][0], k["d"][1], k["d"][2]*100, k["d"][3]*100
        ))
