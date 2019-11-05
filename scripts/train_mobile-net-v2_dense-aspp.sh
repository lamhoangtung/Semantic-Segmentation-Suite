python3 train.py --num_epochs 600 \
                  --checkpoint_step 1 \
                  --validation_step 1 \
                  --dataset apolo_city_scape \
                  --crop_height 224 \
                  --crop_width 320 \
                  --batch_size 64 \
                  --num_val_images 64 \
                  --h_flip true \
                  --v_flip true \
                  --brightness 0.1 \
                  --rotation 5 \
                  --model DenseASPP \
                  --frontend MobileNetV2 \
                  --train_dir apolo-cityscape_mobile-net-v2_dense-aspp
                #   --epoch_start_i 322 \
                #   --continue_training true
