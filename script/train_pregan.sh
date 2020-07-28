cd ..
CUDA_VISIBLE_DEVICES=2 python train_gan.py --depth=7 --latent_size=512 \
        --images_dir "../BMSG-GAN/sourcecode/flowers/data/jpg" \
        --model_dir=models/adv_20 --batch_size=8 --num_epochs=200 \
        --generator_file=models/original/GAN_GEN_0.pth \
        --generator_optim_file=models/original/GAN_GEN_OPTIM_0.pth \
        --discriminator_file=models/original/GAN_DIS_0.pth \
        --discriminator_optim_file=models/original/GAN_DIS_OPTIM_0.pth \
        --shadow_generator_file=models/original/GAN_GEN_SHADOW_0.pth \
        --sample_dir /data/ysk/gan_samples/original
