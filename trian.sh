CUDA_VISIBLE_DEVICES=0,1 python pdseg/train_wandb.py --cfg configs/ocrnet_w18_bn_remote_sensing.yaml \
                                                  --use_gpu \
                                                  --do_eval \
                                                  --use_vdl \
                                                  --vdl_log_dir train_log/more_aug