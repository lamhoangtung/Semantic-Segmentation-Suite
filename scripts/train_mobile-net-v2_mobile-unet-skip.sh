python3 train.py --num_epochs 500 \
                #   --epoch_start_i 109 \
                  --checkpoint_step 1 \
                  --validation_step 1 \
                #   --continue_training true \
                  --dataset city_scape \
                  --crop_height 224 \
                  --crop_width 320 \
                  --batch_size 16 \
                  --num_val_images 32 \
                  --h_flip true \
                  --v_flip true \
                  --brightness 0.1 \
                  --rotation 5 \
                  --model MobileUNet-Skip \
                  --frontend MobileNetV2 \
                  --train_dir train_with_background_mobile-net-v2_mobile-unet-skip
