cd ..
CUDA_VISIBLE_DEVICES=2 python train_gan.py --depth=7 --latent_size=512 \
        --images_dir "../BMSG-GAN/sourcecode/flowers/data/jpg" \
        --sample_dir=samples/original --model_dir=models/original --batch_size=12 --num_epochs=200
