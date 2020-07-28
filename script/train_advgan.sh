cd ..
CUDA_VISIBLE_DEVICES=2 python train_gan.py --depth=7 --latent_size=512 \
        --images_dir "../BMSG-GAN/sourcecode/flowers/data/jpg" \
        --sample_dir /data/ysk/gan_samples/only_adv --model_dir=models/only_adv --batch_size=12 --num_epochs=200
        --adversarial
