#!/bin/bash

# Super-resolution model argument (GFPGAN or CodeFormer)
SUPERRES=$1  # Pass the super-resolution model as the first argument (GFPGAN/CodeFormer)

# Check if the super-resolution model is specified
if [ -z "$SUPERRES" ]; then
    echo "Error: Super-resolution method (GFPGAN/CodeFormer) must be specified."
    exit 1
fi

# Ensure required paths are provided
VIDEO_PATH="assets/demo1_video.mp4"
AUDIO_PATH="assets/demo1_audio.wav"
VIDEO_OUT_PATH="video_out.mp4"

# Ensure input video and audio paths exist
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file '$VIDEO_PATH' not found."
    exit 1
fi

if [ ! -f "$AUDIO_PATH" ]; then
    echo "Error: Audio file '$AUDIO_PATH' not found."
    exit 1
fi

# Run the inference script with the selected super-resolution method
python -m scripts.inference \
    --unet_config_path "configs/unet/second_stage.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 20 \
    --guidance_scale 1.5 \
    --video_path "$VIDEO_PATH" \
    --audio_path "$AUDIO_PATH" \
    --video_out_path "$VIDEO_OUT_PATH" \
    --superres "$SUPERRES"
