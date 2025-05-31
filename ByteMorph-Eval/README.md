#### 1. Save your generation results as the GT format (Left: Input; Right: Output)

1.1 Specify the generation root, gt root and metrics output directory in `eval_clip_score.sh` and `eval_gpt_score.sh`.


#### 2. Install related packages
```bash
cd ./ByteMorph-Eval
conda create -n bytemorph_eval python=3.10
conda activate bytemorph_eval
pip3 install -r requirements.txt
```

#### 3. Run the evaluation script
```bash
bash eval_clip_score.sh
```

The metric results will be save in `{OUTPUT_SCORE_DIR}/clip_score/{EDITING_TYPE}_avg.json`. 

#### 4. Additional steps for evaluation with VLM (GPT)

4.1 Configure your API key and model version in `gpt_eval.py`

4.2 After configuration, run the following script:
```bash
bash eval_gpt_score.sh
```

The final score will be save in `{OUTPUT_SCORE_DIR}/gpt_score/{EDITING_TYPE}/average_score.json`. 


