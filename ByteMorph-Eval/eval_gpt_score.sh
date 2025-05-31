GEN_DATA_ROOT=./gen_images
GT_DATA_ROOT=./ByteMorph-Bench/output_bench
OUTPUT_SCORE_DIR=./benchmark_eval/gpt_score
sudo chmod -R 777 $GEN_DATA_ROOT

python3 utils/copy_json.py \
    --src_json_root $GT_DATA_ROOT \
    --tgt_json_root $GEN_DATA_ROOT \

python3 gpt_eval.py \
    --api_key \
    --openai_api_version \
    --deployment_name \
    --base_url \
    --src_root $GEN_DATA_ROOT \
    --output_root $OUTPUT_SCORE_DIR