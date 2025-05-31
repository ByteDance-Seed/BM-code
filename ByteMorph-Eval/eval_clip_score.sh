GEN_DATA_ROOT=./gen_images
GT_DATA_ROOT=./ByteMorph-Bench/output_bench
OUTPUT_SCORE_DIR=./benchmark_eval/clip_score
sudo chmod -R 777 $GEN_DATA_ROOT

python3 utils/copy_json.py \
    --src_json_root $GT_DATA_ROOT \
    --tgt_json_root $GEN_DATA_ROOT \

CUDA_VISIBLE_DEVICES=0 python3 clip_eval.py \
    --src_data_root $GT_DATA_ROOT  \
    --gen_data_root $GEN_DATA_ROOT \
    --output_root $OUTPUT_SCORE_DIR \