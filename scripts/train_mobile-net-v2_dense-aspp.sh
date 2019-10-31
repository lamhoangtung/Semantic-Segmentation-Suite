python3 train.py --num_epochs 500 \
                  --checkpoint_step 1 \
                  --validation_step 1 \
                  --dataset city_scape \
                  --crop_height 224 \
                  --crop_width 320 \
                  --batch_size 16 \
                  --num_val_images 32 \
                  --h_flip true \
                  --v_flip true \
                  --brightness 0.1 \
                  --rotation 5 \
                  --model DenseASPP \
                  --frontend MobileNetV2 \
                  --train_dir tiny-cityscape_mobile-net-v2_dense-aspp \
                #   --epoch_start_i 96 \
                #   --continue_training true
