cd ..
CUDA_VISIBLE_DEVICES=3 python train_patch.py --epochs 500 --batch_size 12 --optim adam --inria --pretrain --lr 0.005
