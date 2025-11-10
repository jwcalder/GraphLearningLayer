#!/usr/bin/env bash
cd ..

CP_PATH="${1:-save/PreTrain_combined/preactresnet18_bsz_512_None_ssaug_strong_gamma_0.5/pretrain_joint_ckpt_epoch_1000.pth}"
# checkpoints/fullysup_ckpt.pth
# save/_Sup_and_SS/SupCE_resnet110_bsz_512_method_SupCE_Sup_and_SS_supaug_strong_ssaug_strong/ckpt_epoch_1000.pth
# checkpoints/resnet18_SimCLR_mlp.pth 

python3 FullySup.py \
    --epoch 0 \
    --model preactresnet18 \
    --dataset cifar10 \
    --plot_freq_ss 25 \
    --cosine \
    --sup_train_type gl \
    --cp_load_path "$CP_PATH" \
    --epsilon 1 \
    --num_train None
