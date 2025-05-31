export FLUX_DEV="./pretrained_weights/models--black-forest-labs--FLUX.1-dev/flux1-dev.safetensors"
export AE="./pretrained_weights/models--black-forest-labs--FLUX.1-dev/ae.safetensors"
export T5="./pretrained_weights/models--xlabs-ai--xflux_text_encoders/"
export CLIP_VIT="./pretrained_weights/models--openai--clip-vit-large-patch14/"
accelerate launch --config_file "train_configs/deepspeed_stage2.yaml" \
    --main_process_port=$(( $RANDOM % 10000 + 10000 )) \
    train_flux_deepspeed.py --config "train_configs/train.yaml"