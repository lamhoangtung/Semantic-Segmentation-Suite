python3 train.py --num_epochs 250 \
                  --epoch_start_i 0 \
                  --checkpoint_step 1 \
                  --validation_step 1 \
                  --dataset city_scape \
                  --crop_height 240 \
                  --crop_width 320 \
                  --batch_size 32 \
                  --num_val_images 32 \
                  --h_flip true \
                  --v_flip true \
                  --brightness 0.1 \
                  --rotation 5 \
                  --model MobileUNet-Skip \
                  --frontend MobileNetV2 \
                  --train_dir train_mobile-net-v2_mobile-unet-skip
