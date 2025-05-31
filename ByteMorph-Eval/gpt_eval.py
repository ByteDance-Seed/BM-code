# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from langchain_openai import AzureChatOpenAI

import re
from argparse import ArgumentParser


def encode_image(image):
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=100)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_root", required=False, type=str, default="/mnt/bn/dichang-bytenas/dichang-seed/xedit/data/evaluation_dataset/X-Edit-Bench_test/output_bench")
    parser.add_argument("--output_root", required=False, type=str, default="./")
    parser.add_argument("--api_key", required=False, type=str, default="") # your api_key here
    parser.add_argument("--openai_api_version", required=False, type=str, default="") #  openai api version 
    parser.add_argument("--deployment_name", required=False, type=str, default="") #  eval model name
    parser.add_argument("--base_url", required=False, type=str, default="") # your base_url here
    args = parser.parse_args()
    src_root = args.src_root
    output_root = args.output_root
    error_log_path = os.path.join(output_root, "error_log.txt")
    OPENAI_API_KEY = args.api_key 
    openai_api_version = args.openai_api_version 
    deployment_name = args.deployment_name 
    base_url =  args.base_url 


    EDIT_EVALUATION_PROMPT_TEMPLATE_SIMPLE_V4 = """
    You are an evaluator for image editing. You will be given a pair of images before and after editing as well as an editing instruction.
    You need to rate the editing result with a score between 0 to 100.
    A successful editing should not miss any change required by editing instruction.
    A successful editing should not have any extra changes that are not required by editing instruction. 
    The second image should have minimum change to reflect the changes made with EDIT TEXT.
    Be strict about the changes made between two images.
    Give the final response in a json format as such:
    {
        "Score": xx
    }
    Do not output anything else.
    """

    EDIT_ITEM_EXAMPLE = """EDIT TEXT: {edit_action}"""


    model = AzureChatOpenAI(
        azure_endpoint=base_url,
        openai_api_version=openai_api_version,
        model=deployment_name,
        openai_api_key=OPENAI_API_KEY,
        openai_api_type="azure",
        )

    # Traverse subfolders and process
    for subfolder in sorted(os.listdir(src_root)):
        subdir = os.path.join(src_root, subfolder)
        if not os.path.isdir(subdir):
            continue
        files = sorted(os.listdir(subdir))
        for fname in tqdm(files):
            if not fname.endswith(".png"):
                continue
            
            src_img_path = os.path.join(subdir, fname)
            json_path = src_img_path.replace(".png", ".json")
            rel_path = os.path.relpath(subdir, src_root)
            output_dir = os.path.join(output_root, rel_path)
            os.makedirs(output_dir, exist_ok=True)

            # Read prompt
            if not os.path.exists(json_path):
                error_msg = f"[Missing JSON] {fname}"
                print("⚠️", error_msg)
                with open(error_log_path, "a") as ef:
                    ef.write(error_msg + "\n")
                continue

            with open(json_path, "r") as f:
                json_data = json.load(f)
            src_prompt = json_data.get("edit", "").strip()
            if not src_prompt:
                error_msg = f"[Empty Prompt] {fname}"
                print("⚠️", error_msg)
                with open(error_log_path, "a") as ef:
                    ef.write(error_msg + "\n")
                continue
            
            image = Image.open(src_img_path)
            w,h = image.size
            image1 = image.crop((0,0,w//2,h))
            image2 = image.crop((w//2,0,w,h))
            images = [image1, image2]
            sys_message = EDIT_EVALUATION_PROMPT_TEMPLATE_SIMPLE_V4
            hum_message = EDIT_ITEM_EXAMPLE.format(edit_action=src_prompt)
            request = [{
                    "type": "text",
                    "text": hum_message
                }]
            for image in images:
                base64_image = encode_image(image)
                input_ip = {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                request.append({
                    "type": "image_url",
                    "image_url": input_ip,
                })
            
            try:
                response = model([
                        { "role": "system", "content": sys_message},
                        { "role": "user", "content": request},
                    ])
                json_string = re.findall(r'\{.*?\}', response.content, re.DOTALL)[0]
                json_data = json.loads(json_string)
                assert "Score" in json_data 
                score = json_data["Score"]
                score_path = os.path.join(output_dir, fname.replace(".png", ".score.json"))
                with open(score_path, "w") as fscore:
                    json.dump({"Score": score}, fscore, indent=2)
            except Exception as e:
                error_msg = f"[API Error] {fname} — {str(e)}"
                print("❌", error_msg)
                print("Error during API call for image: ", src_img_path)
                with open(error_log_path, "a") as ef:
                    ef.write(error_msg + "\n")
                    ef.write("Error during API call for image: " + src_img_path + "\n")
    summary = {}

    for subfolder in sorted(os.listdir(output_root)):
        subdir = os.path.join(output_root, subfolder)
        if not os.path.isdir(subdir):
            continue

        total_score = 0
        count = 0
        for fname in os.listdir(subdir):
            if fname.endswith(".score.json"):
                with open(os.path.join(subdir, fname), "r") as f:
                    try:
                        score = json.load(f)["Score"]
                        total_score += score
                        count += 1
                    except Exception as e:
                        print(f"⚠️ Failed to read score from {fname}: {e}")

        if count > 0:
            avg_score = total_score / count
            summary[subfolder] = {
                "average_score": round(avg_score, 2),
                "num_images": count
            }

    # Save summary JSONs per subfolder
    for subfolder, stats in summary.items():
        summary_path = os.path.join(output_root, f"{subfolder}_avg_score.json")
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=2)

    print("✅ Average scores saved to each subfolder.")


