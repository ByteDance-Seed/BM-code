 # Licensed under the FLUX.1 [dev] Non-Commercial License  
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at 
 #
 #  https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License. 

import cv2


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)
