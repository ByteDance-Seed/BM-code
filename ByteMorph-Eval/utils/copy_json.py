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
import shutil
from tqdm import tqdm

def copy_matching_json(src_json_root, tgt_json_root):
    for subdir, _, files in os.walk(tgt_json_root):
        for fname in tqdm(files):
            if not fname.endswith(".png"):
                continue
            rel_dir = os.path.relpath(subdir, tgt_json_root)
            json_name = fname.replace(".png", ".json")

            src_json_path = os.path.join(src_json_root, rel_dir, json_name)
            tgt_json_path = os.path.join(tgt_json_root,rel_dir)


            if os.path.exists(src_json_path):
                shutil.copy2(src_json_path, tgt_json_path)
            else:
                print(f"⚠️ Missing source JSON: {src_json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_json_root", required=True, help="Source JSON root directory")
    parser.add_argument("--tgt_json_root", required=True, help="Target PNG directory to match structure")
    args = parser.parse_args()

    copy_matching_json(args.src_json_root, args.tgt_json_root)
