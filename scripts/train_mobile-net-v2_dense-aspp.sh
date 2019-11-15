python3 train.py --num_epochs 200 \
                  --checkpoint_step 1 \
                  --validation_step 1 \
                  --dataset apolo_city_scape_no_vehicle \
                  --crop_height 224 \
                  --crop_width 320 \
                  --batch_size 64 \
                  --num_val_images 128 \
                  --h_flip true \
                  --v_flip true \
                  --brightness 0.1 \
                  --rotation 5 \
                  --model DenseASPP \
                  --frontend MobileNetV2 \
                  --loss self_balanced_focal_loss \
                  --lr_warmup false \
                  --optimizer rmsprop \
                  --train_dir apolo-cityscape-no-vehicle_mobile-net-v2_dense-aspp_self-bl-focal_adam
                  # --learning_rate 0.001 \
                  # --lr_scheduler poly_decay \
                #   --epoch_start_i 322 \
                #   --continue_training true
