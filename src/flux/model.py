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

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange

from .modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.gradient_checkpointing = True # False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor, # clip
        block_controlnet_hidden_states=None,
        guidance: Tensor | None = None,
        image_proj: Tensor | None = None, 
        ip_scale: Tensor | float = 1.0, 
        use_share_weight_referencenet=False,
        single_img_ids: Tensor | None = None,
        single_block_refnet=False,
        double_block_refnet=False,
    ) -> Tensor:
        if single_block_refnet or double_block_refnet:
            assert use_share_weight_referencenet == True
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        # print("vec shape 1:", vec.shape)
        # print("y shape 1:", y.shape)
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        # print("vec shape 1.5:", vec.shape)
        vec = vec + self.vector_in(y)
        # print("vec shape 2:", vec.shape)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        if use_share_weight_referencenet:
            # print("In img shape:", img.shape)
            img_latent_length = img.shape[1]
            single_ids = torch.cat((txt_ids, single_img_ids), dim=1)
            single_pe = self.pe_embedder(single_ids)
            if double_block_refnet and (not single_block_refnet):
                double_block_pe = pe
                double_block_img = img
                single_block_pe = single_pe
                
            elif single_block_refnet and (not double_block_refnet):
                double_block_pe = single_pe
                double_block_img = img[:, img_latent_length//2:, :]
                single_block_pe = pe
                ref_img_latent = img[:, :img_latent_length//2, :]
            else:
                print("RefNet only support either double blocks or single blocks. If you want to turn on all blocks for RefNet, please use Spatial Condition.")
                raise NotImplementedError

        if block_controlnet_hidden_states is not None:
            controlnet_depth = len(block_controlnet_hidden_states)
        for index_block, block in enumerate(self.double_blocks):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward
                if not use_share_weight_referencenet:
                    img, txt = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        img,
                        txt,
                        vec,
                        pe,
                        image_proj,
                        ip_scale,
                        use_reentrant=True,
                    )
                else:
                    double_block_img, txt = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        double_block_img,
                        txt,
                        vec,
                        double_block_pe, 
                        image_proj,
                        ip_scale,
                        use_reentrant=True,
                    )
            else:
                if not use_share_weight_referencenet:
                    img, txt = block(
                        img=img, 
                        txt=txt, 
                        vec=vec, 
                        pe=pe, 
                        image_proj=image_proj,
                        ip_scale=ip_scale, 
                    )
                else:
                    double_block_img, txt = block(
                        img=double_block_img, 
                        txt=txt, 
                        vec=vec, 
                        pe=double_block_pe, 
                        image_proj=image_proj,
                        ip_scale=ip_scale, 
                    )
            # controlnet residual
            if block_controlnet_hidden_states is not None:
                if not use_share_weight_referencenet:
                    img = img + block_controlnet_hidden_states[index_block % 2]
                else:
                    double_block_img = double_block_img + block_controlnet_hidden_states[index_block % 2]
        
        if use_share_weight_referencenet:
            mid_img = double_block_img
            # print("After double blocks img shape:",mid_img.shape)
            if double_block_refnet and (not single_block_refnet):
                single_block_img = mid_img[:, img_latent_length//2:, :]
            elif single_block_refnet and (not double_block_refnet):
                single_block_img = torch.cat([ref_img_latent, mid_img], dim=1)
            single_block_img = torch.cat((txt, single_block_img), 1)
        else:
            img = torch.cat((txt, img), 1)
        # print("single block input img shape:", single_block_img.shape)
        for block in self.single_blocks:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward
                if not use_share_weight_referencenet:
                    img = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        img,
                        vec,
                        pe,
                        use_reentrant=True,
                    )
                else:
                    single_block_img = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        single_block_img,
                        vec,
                        single_block_pe,
                        use_reentrant=True,
                    ) 
            else:
                if not use_share_weight_referencenet:
                    img = block(
                        img, 
                        vec=vec, 
                        pe=pe,
                    )
                else:
                    single_block_img = block(
                        single_block_img, 
                        vec=vec, 
                        pe=single_block_pe,
                    )
        if use_share_weight_referencenet:
            out_img = single_block_img
            if double_block_refnet and (not single_block_refnet):
                out_img = out_img[:, txt.shape[1]:, ...]
            elif single_block_refnet and (not double_block_refnet):
                out_img = out_img[:, txt.shape[1]:, ...]
                out_img = out_img[:, img_latent_length//2:, :]
            img = out_img
            # print("output img shape:", img.shape)
        else:
            img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img



# In img shape: torch.Size([1, 2048, 3072])
# After double blocks img shape: torch.Size([1, 1024, 3072])
# single block input img shape: torch.Size([1, 2560, 3072])
# output img shape: torch.Size([1, 1024, 3072])
#
# In img shape: torch.Size([1, 2048, 3072])   
# After double blocks img shape: torch.Size([1, 2048, 3072])                                                                                                                                 [78/1966]
# single block input img shape: torch.Size([1, 1536, 3072]) 
# output img shape: torch.Size([1, 1024, 3072])  