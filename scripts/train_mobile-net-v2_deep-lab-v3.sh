python3 train.py --num_epochs 600 \
                  --checkpoint_step 1 \
                  --validation_step 1 \
                  --dataset city_scape \
                  --crop_height 224 \
                  --crop_width 320 \
                  --batch_size 64 \
                  --num_val_images 32 \
                  --h_flip true \
                  --v_flip true \
                  --brightness 0.1 \
                  --rotation 5 \
                  --model DeepLabV3 \
                  --frontend MobileNetV2 \
                  --train_dir tiny-cityscape_mobile-net-v2_deep-lab-v3
                #   --epoch_start_i 322 \
                #   --continue_training true
