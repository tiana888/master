import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image, ImageFilter
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


from .utils import is_torch2_available, get_generator
import numpy as np


if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class IPAdapter:
    ip_stream_token_sizes = None

    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, target_blocks=["block"], instance_idx=0, prompt_tokens=[], save_path = None, ca_mask=False,save_name=''):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        if self.ip_stream_token_sizes is None:
            self.ip_stream_token_sizes = [self.num_tokens]
        self.total_ip_tokens = int(sum(self.ip_stream_token_sizes))
        self.target_blocks = target_blocks
        self.instance_idx = instance_idx
        self.prompt_tokens = prompt_tokens
        self.save_path = save_path
        self.ca_mask = ca_mask
        self.save_name = save_name

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()


        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()

    def get_layer_stream_scales(self, layer_name):
        # Default single-stream behavior (backward compatible).
        return [1.0]

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def set_ip_adapter(self, save_name=''):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            # print()
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:  # attn1.processor -> normal atten
                attn_procs[name] = AttnProcessor()
            else:  
                selected = False
                for block_name in self.target_blocks:  # ['up_blocks.0.attentions.1', 'down_blocks.2.attentions.1']
                    if block_name in name:
                        print('selected layer', name )
                        selected = True
                        break
                if selected:
                    attn_procs[name] = IPAttnProcessor( 
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.total_ip_tokens,
                        instance_idx=self.instance_idx,
                        prompt_tokens=self.prompt_tokens,
                        ca_mask = self.ca_mask,
                        save_name = f'{name}',
                        stream_token_sizes=self.ip_stream_token_sizes,
                        stream_scales=self.get_layer_stream_scales(name),
                    ).to(self.device, dtype=torch.float16)
                    attn_procs[name].save_path = self.save_path
                else:  # skip ip-adapter
                    attn_procs[name] = IPAttnProcessor(   # bug
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=self.total_ip_tokens,
                        skip=True,
                        instance_idx=self.instance_idx,
                        prompt_tokens=self.prompt_tokens,
                        stream_token_sizes=self.ip_stream_token_sizes,
                        stream_scales=self.get_layer_stream_scales(name),
            
                    ).to(self.device, dtype=torch.float16)
                    attn_procs[name].save_path = self.save_path
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.total_ip_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.total_ip_tokens))
    
    def set_ip_variables(self, dir_name, ca_mask_thres):
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        for ip_layer in ip_layers:
            ip_layer.dir_name = dir_name
            ip_layer.ca_mask_thres = ca_mask_thres


    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    @torch.inference_mode()
    def get_image_embeds(self, clr_ref_img=None, clip_image_embeds=None, content_prompt_embeds=None):
        if clr_ref_img is not None:
            if isinstance(clr_ref_img, Image.Image):
                clr_ref_img = [clr_ref_img]
            clip_image = self.clip_image_processor(images=clr_ref_img, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
        if content_prompt_embeds is not None:  
            clip_image_embeds = clip_image_embeds - content_prompt_embeds

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)   # 


        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        clr_ref_img=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        neg_content_emb=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if clr_ref_img is not None:
            num_prompts = 1 if isinstance(clr_ref_img, Image.Image) else len(clr_ref_img)
        else:
            num_prompts = clip_image_embeds.shape[0]

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            clr_ref_img=clr_ref_img, clip_image_embeds=clip_image_embeds, content_prompt_embeds=neg_content_emb
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)   # concat text and image embedding
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class IPAdapterXL(IPAdapter):
    """SDXL"""

    @torch.inference_mode()
    def get_image_embeds(self, clr_ref_img=None, clip_image_embeds=None, content_prompt_embeds=None,
                         clr_texture_ref_img=None, texture_ref_img=None, 
                         subscale=0., color_scale=1, texture_scale=0):
        
        if isinstance(clr_texture_ref_img, Image.Image):
            clr_texture_ref_img = [clr_texture_ref_img]
        if isinstance(texture_ref_img, Image.Image):
            texture_ref_img = [texture_ref_img]


        if clr_ref_img is not None:
            if isinstance(clr_ref_img, Image.Image):
                clr_ref_img = [clr_ref_img]
            clip_image = self.clip_image_processor(images=clr_ref_img, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        
        if content_prompt_embeds is not None:  
            clip_image_embeds = clip_image_embeds - content_prompt_embeds
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)   # 
            uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
             
        elif clr_texture_ref_img is not None and texture_ref_img is not None:  #  
            # ori   get texture feature
            txr_clip_image2 = self.clip_image_processor(images=texture_ref_img, return_tensors="pt").pixel_values
            txr_clip_image2 = txr_clip_image2.to(self.device, dtype=torch.float16)
            txr_clip_image_embeds2 = self.image_encoder(txr_clip_image2, output_hidden_states=True).image_embeds
            

            
            # disentangle color
            ################################# color img ###################################
            clr_clip_image = self.clip_image_processor(images=clr_ref_img, return_tensors="pt").pixel_values
            clr_clip_image = clr_clip_image.to(self.device, dtype=torch.float16)
            clr_clip_image_embeds = self.image_encoder(clr_clip_image, output_hidden_states=True).image_embeds
            
            
            ################################# color texture img ###################################
            clr_texture_ref_image = self.clip_image_processor(images=clr_texture_ref_img, return_tensors="pt").pixel_values
            clr_texture_ref_image = clr_texture_ref_image.to(self.device, dtype=torch.float16)
            clr_txr_clip_image_embeds = self.image_encoder(clr_texture_ref_image, output_hidden_states=True).image_embeds
            # remove position embed
            # clr_txr_clip_image_embeds = self.image_encoder(clr_texture_ref_image, output_hidden_states=True, is_posembed=True).hidden_states[-2]
            #########################################################################


            # disentangle color
            clr_img_embedding = color_scale*clr_clip_image_embeds - clr_txr_clip_image_embeds*subscale
            # print(clr_img_embedding.mean(2))
            # color + texture
            clip_image_embeds = clr_img_embedding + txr_clip_image_embeds2*texture_scale
            

            image_prompt_embeds = self.image_proj_model(clip_image_embeds)   



        return image_prompt_embeds, uncond_image_prompt_embeds
    

    def generate(
        self,
        clr_ref_img,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        neg_content_emb=None,
        neg_content_prompt=None,    # 
        neg_content_scale=1.0,
        instance_idx=0,

        clr_texture_ref_img = None,     # for color transfer
        substract_scale = 0.,
        texture_scale = 0.,
        color_scale = 1.,
        texture_ref_img = None,    # for texture transfer
        wct_starts_step = 0,
        wct_ends_step=0,
        csd_iter_num = 1,
        **kwargs,
    ):
        self.set_scale(scale)
        self.instance_idx = instance_idx

        
        num_prompts = 1 if isinstance(clr_ref_img, Image.Image) else len(clr_ref_img)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        if neg_content_emb is None:
            if neg_content_prompt is not None:
                with torch.inference_mode():
                    (
                        prompt_embeds_, # torch.Size([1, 77, 2048])
                        negative_prompt_embeds_,
                        pooled_prompt_embeds_, # torch.Size([1, 1280])
                        negative_pooled_prompt_embeds_,
                    ) = self.pipe.encode_prompt(
                        neg_content_prompt,
                        num_images_per_prompt=num_samples,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    pooled_prompt_embeds_ *= neg_content_scale
            else:
                pooled_prompt_embeds_ = neg_content_emb
        else:
            pooled_prompt_embeds_ = None
        # style image embedding - style neg_content_emb
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                                                                    clr_ref_img, 
                                                                    content_prompt_embeds=pooled_prompt_embeds_,
                                                                    clr_texture_ref_img=clr_texture_ref_img, 
                                                                    texture_ref_img=texture_ref_img, 
                                                                    subscale=substract_scale, 
                                                                    color_scale=color_scale, 
                                                                    texture_scale=texture_scale, 
                                                                    )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():  # text embedding
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)  # concat text and style image embedding
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)
        images = self.pipe(
            prompt_embeds=prompt_embeds,   # concat text and style image embedding  [1, 81, 2048]  [1, 77+num_img_tokens, 2048]
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,   # pooled_prompt_embeds
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class IPAdapterPlus(IPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clr_ref_img=None, clip_image_embeds=None):
        if isinstance(clr_ref_img, Image.Image):
            clr_ref_img = [clr_ref_img]
        clip_image = self.clip_image_processor(images=clr_ref_img, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class IPAdapterFull(IPAdapterPlus):
    """IP-Adapter with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


CALC_SIMILARITY = False

def punish_weight_module(wo_batch, latent_size, alpha=1., method='soft-weight'):
    if method == 'weight':
        wo_batch *= alpha
    elif method in ['alpha', 'beta', 'delete', 'soft-weight', 'highest']:
        u, s, vh = torch.linalg.svd(wo_batch)
        u = u[:,:latent_size]
        zero_idx = int(latent_size * alpha)
        # print(f'before punish:s max{s.max()} min{s.min()}')
        if method == 'alpha':
            s[:zero_idx] = 0
        elif method == 'beta':
            s[zero_idx:] = 0
        elif method == 'delete':
            s = s[zero_idx:] if zero_idx < latent_size else torch.zeros(latent_size).to(s.device)
            u = u[:, zero_idx:] if zero_idx < latent_size else u
            vh = vh[zero_idx:, :] if zero_idx < latent_size else vh
        elif method == 'soft-weight':
            if CALC_SIMILARITY:
                _s = s.clone()
                _s[zero_idx:] = 0
                _wo_batch = u @ torch.diag(_s) @ vh
                dist = cdist(wo_batch[:,0].unsqueeze(0).cpu(), _wo_batch[:,0].unsqueeze(0).cpu(), metric='cosine')
                print(f'The distance between the word embedding before and after the punishment: {dist}')
            if alpha == -.001:
                s *= (torch.exp(.001 * s) * 1.2)  # strengthen objects (our Appendix.F)
            else:
                s *= torch.exp(-alpha*s)  # suppression EOT (our main paper)
        
        # highest1_highest2_0.1
        elif method == 'highest':
            s[0:2] = s[0:2] * alpha

        
        wo_batch = u @ torch.diag(s) @ vh    # [1280,514] [514,514] [514,514]
        # print(f'after soft punish[{alpha}]:s max{s.max()} min{s.min()}')
    elif  method is None:
        return wo_batch
    else:
        raise ValueError('Unsupported method')
    return wo_batch


def get_base_layer_stream_scales(layer_name):
    if layer_name.startswith("down_blocks"):
        return (0.80, 1.35, 0.20)
    if layer_name.startswith("mid_block"):
        return (1.00, 0.85, 1.05)
    if layer_name.startswith("up_blocks"):
        return (1.15, 0.30, 1.20)
    return (1.0, 1.0, 1.0)




class IPAdapterPlusXL(IPAdapter):
    """SDXL"""
    ip_stream_token_sizes = None

    def __init__(self, *args, **kwargs):
        kwargs.pop("mode_profile", None)
        super().__init__(*args, **kwargs)
        # three independent image condition streams: color / geometry / texture
        self.ip_stream_token_sizes = [self.num_tokens, self.num_tokens, self.num_tokens]
        # rebuild processors with multi-stream token settings
        self.set_ip_adapter()
        self.load_ip_adapter()

    def get_layer_stream_scales(self, layer_name):
        equal_scale = 1.0 / 3.0
        return [equal_scale, equal_scale, equal_scale]

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    # @torch.inference_mode()
    # def get_image_embeds(self, clr_ref_img, clr_texture_ref_img=None, texture_ref_img=None,
    #                      geometry_ref_img=None,
    #                      texture_ref_img2=None, subscale=0., color_scale=1, texture_scale=0,
    #                      texture2_scale=0.0, geometry_sub_scale=1.0, geometry_scale=0.0,
    #                      geometry_color_decouple=0.35,
    #                      texture_color_decouple=0.3,
    #                      punish_weight = 0.005,    # 0.005
    #                     punish_type = 'soft-weight'):
    #     if isinstance(clr_ref_img, Image.Image):
    #         clr_ref_img = [clr_ref_img]
    #     if clr_texture_ref_img is None:
    #         # Fallback for color disentanglement branch.
    #         clr_texture_ref_img = [img.convert("L") for img in clr_ref_img]
    #     elif isinstance(clr_texture_ref_img, Image.Image):
    #         clr_texture_ref_img = [clr_texture_ref_img]
    #     if texture_ref_img is None:
    #         # Fallback neutral texture to avoid runtime crash.
    #         texture_ref_img = [Image.new("L", clr_ref_img[0].size, color=128)]
    #     elif isinstance(texture_ref_img, Image.Image):
    #         texture_ref_img = [texture_ref_img]
    #     if isinstance(geometry_ref_img, Image.Image):
    #         geometry_ref_img = [geometry_ref_img]
    #     if isinstance(texture_ref_img2, Image.Image):
    #         texture_ref_img2 = [texture_ref_img2]


    #     # disentangle color
    #     ################################# color img ###########################
    #     clr_clip_image = self.clip_image_processor(images=clr_ref_img, return_tensors="pt").pixel_values
    #     clr_clip_image = clr_clip_image.to(self.device, dtype=torch.float16)
    #     clr_clip_image_embeds = self.image_encoder(clr_clip_image, output_hidden_states=True).hidden_states[-2]


    #     ################################# color texture img #####################
    #     clr_texture_ref_image = self.clip_image_processor(images=clr_texture_ref_img, return_tensors="pt").pixel_values
    #     clr_texture_ref_image = clr_texture_ref_image.to(self.device, dtype=torch.float16)
    #     clr_txr_clip_image_embeds = self.image_encoder(clr_texture_ref_image, output_hidden_states=True).hidden_states[-2]
    #     #########################################################################


    #     ################################# texture img ###########################
    #     txr_clip_image2 = self.clip_image_processor(images=texture_ref_img, return_tensors="pt").pixel_values
    #     txr_clip_image2 = txr_clip_image2.to(self.device, dtype=torch.float16)
    #     txr_clip_image_embeds2 = self.image_encoder(txr_clip_image2, output_hidden_states=True).hidden_states[-2]

    #     # remove geometric component from texture embedding:
    #     # texture_pure = texture - proj_texture_on_geometry
    #     geo_clip_image_embeds = None
    #     if geometry_ref_img is not None:
    #         geo_clip_image = self.clip_image_processor(images=geometry_ref_img, return_tensors="pt").pixel_values
    #         geo_clip_image = geo_clip_image.to(self.device, dtype=torch.float16)
    #         geo_clip_image_embeds = self.image_encoder(geo_clip_image, output_hidden_states=True).hidden_states[-2]

    #         if geometry_sub_scale != 0:
    #             eps = 1e-6
    #             geo_norm_sq = (geo_clip_image_embeds * geo_clip_image_embeds).sum(dim=-1, keepdim=True).clamp_min(eps)
    #             proj_t_on_g = ((txr_clip_image_embeds2 * geo_clip_image_embeds).sum(dim=-1, keepdim=True) / geo_norm_sq) * geo_clip_image_embeds
    #             txr_clip_image_embeds2 = txr_clip_image_embeds2 - geometry_sub_scale * proj_t_on_g
        

    #     ###################a pure gray image is obtained by averaging the gray texture img #####################
    #     texture_img_np = np.array(texture_ref_img[0])
    #     mean_values =  texture_img_np.mean(axis=(0, 1))  
    #     texture_img_puregray = np.full_like(texture_img_np, mean_values, dtype=np.uint8)
    #     texture_img_puregray = Image.fromarray(texture_img_puregray)
    #     txr_puregray_clip_image = self.clip_image_processor(images=[texture_img_puregray], return_tensors="pt").pixel_values
    #     txr_puregray_clip_image = txr_puregray_clip_image.to(self.device, dtype=torch.float16)
    #     txr_puregray_clip_embeds = self.image_encoder(txr_puregray_clip_image, output_hidden_states=True).hidden_states[-2]


    #     is_punish_gray = punish_weight!=0 and punish_type is not None
    #     if is_punish_gray:
    #         dtype = txr_clip_image_embeds2.dtype
    #         b, t_num, ch = txr_clip_image_embeds2.shape
    #         wo_batch = torch.concat([txr_clip_image_embeds2, txr_puregray_clip_embeds], dim=1).squeeze(0)
    #         latent_size = wo_batch.shape[0]
    #         wo_batch = wo_batch.permute(1,0)   # [ch, token_num]
    #         wo_batch = punish_weight_module(wo_batch.float(), latent_size, alpha=punish_weight, method=punish_type)  # 0.005
    #         wo_batch = wo_batch.permute(1,0)   # [token_num, ch]->[ch, token_num]

    #         txr_clip_image_embeds2 = wo_batch[:t_num, :].unsqueeze(0).to(dtype)

    #     # Keep the existing optional leakage suppression on the texture branch.
    #     if texture_color_decouple != 0:
    #         eps = 1e-6
    #         c = clr_clip_image_embeds
    #         t = txr_clip_image_embeds2
    #         c_norm_sq = (c * c).sum(dim=-1, keepdim=True).clamp_min(eps)
    #         proj_t_on_c = ((t * c).sum(dim=-1, keepdim=True) / c_norm_sq) * c
    #         t_pure = t - texture_color_decouple * proj_t_on_c
    #         # preserve texture energy to prevent texture collapse
    #         t_energy = t.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    #         t_pure_energy = t_pure.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    #         txr_clip_image_embeds2 = t_pure * (t_energy / t_pure_energy)

    #     if geo_clip_image_embeds is not None and geometry_color_decouple != 0:
    #         eps = 1e-6
    #         c = clr_clip_image_embeds
    #         g = geo_clip_image_embeds
    #         c_norm_sq = (c * c).sum(dim=-1, keepdim=True).clamp_min(eps)
    #         proj_g_on_c = ((g * c).sum(dim=-1, keepdim=True) / c_norm_sq) * c
    #         g_pure = g - geometry_color_decouple * proj_g_on_c
    #         g_energy = g.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    #         g_pure_energy = g_pure.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    #         geo_clip_image_embeds = g_pure * (g_energy / g_pure_energy)

    #     # Legacy appearance formulation:
    #     # color_emb * color_scale - gray_color_emb * subscale + gray_texture_emb * texture_scale
    #     appearance_stream_embeds = (
    #         clr_clip_image_embeds * color_scale
    #         - clr_txr_clip_image_embeds * subscale
    #         + txr_clip_image_embeds2 * texture_scale
    #     )

    #     # Keep texture guidance on its own branch so mid-block attention sees
    #     # texture features instead of the fused color+texture appearance signal.
    #     texture_stream_embeds = txr_clip_image_embeds2 * texture_scale

    #     # geometry stream
    #     if geo_clip_image_embeds is not None:
    #         geometry_stream_embeds = geometry_scale * geo_clip_image_embeds
    #     else:
    #         geometry_stream_embeds = torch.zeros_like(appearance_stream_embeds)

    #     # brushstroke texture(2): use high-pass image to emphasize strokes over global tone
    #     if texture_ref_img2 is not None and texture2_scale != 0:
    #         texture2_img = texture_ref_img2[0].convert("RGB")
    #         texture2_blur = texture2_img.filter(ImageFilter.GaussianBlur(radius=2.0))
    #         texture2_np = np.array(texture2_img, dtype=np.float32)
    #         texture2_blur_np = np.array(texture2_blur, dtype=np.float32)
    #         texture2_highpass_np = np.clip(texture2_np - texture2_blur_np + 127.5, 0, 255).astype(np.uint8)
    #         texture2_highpass_img = Image.fromarray(texture2_highpass_np)

    #         txr2_clip_image = self.clip_image_processor(images=[texture2_highpass_img], return_tensors="pt").pixel_values
    #         txr2_clip_image = txr2_clip_image.to(self.device, dtype=torch.float16)
    #         txr2_clip_image_embeds = self.image_encoder(txr2_clip_image, output_hidden_states=True).hidden_states[-2]
    #         texture_stream_embeds = texture_stream_embeds + txr2_clip_image_embeds * texture2_scale
    #         appearance_stream_embeds = appearance_stream_embeds + txr2_clip_image_embeds * texture2_scale

    #     color_prompt_embeds = self.image_proj_model(appearance_stream_embeds)
    #     geometry_prompt_embeds = self.image_proj_model(geometry_stream_embeds)
    #     texture_prompt_embeds = self.image_proj_model(texture_stream_embeds)
    #     image_prompt_embeds = torch.cat([color_prompt_embeds, geometry_prompt_embeds, texture_prompt_embeds], dim=1)


    #     # unconditional embedding for each stream
    #     uncond_clip_image_embeds = self.image_encoder(
    #         torch.zeros_like(clr_clip_image), output_hidden_states=True
    #     ).hidden_states[-2]
    #     uncond_color = self.image_proj_model(uncond_clip_image_embeds)
    #     uncond_geometry = self.image_proj_model(uncond_clip_image_embeds)
    #     uncond_texture = self.image_proj_model(uncond_clip_image_embeds)
    #     uncond_image_prompt_embeds = torch.cat([uncond_color, uncond_geometry, uncond_texture], dim=1)

    #     return image_prompt_embeds, uncond_image_prompt_embeds   # [1,16,2048]

    @torch.inference_mode()
    def get_image_embeds(self, clr_ref_img=None, clr_texture_ref_img=None, texture_ref_img=None,
                         geometry_ref_img=None, texture_ref_img2=None, 
                         subscale=0., color_scale=1, texture_scale=0,
                         texture2_scale=0.0, geometry_sub_scale=0.15, texture_geometry_decouple=None, geometry_scale=0.0,
                         geometry_color_decouple=0.35, texture_color_decouple=0.08,
                         color_to_geometry_decouple=None,
                         texture_to_geometry_decouple=None, geometry_to_texture_decouple=None,
                         color_to_texture_decouple=None, texture_to_color_decouple=None,
                         punish_weight=0.005, punish_type='soft-weight'):
        if texture_geometry_decouple is None:
            texture_geometry_decouple = geometry_sub_scale
        if texture_to_geometry_decouple is None:
            texture_to_geometry_decouple = texture_geometry_decouple
        if geometry_to_texture_decouple is None:
            geometry_to_texture_decouple = texture_geometry_decouple
        if color_to_geometry_decouple is None:
            color_to_geometry_decouple = geometry_color_decouple
        if color_to_texture_decouple is None:
            color_to_texture_decouple = texture_color_decouple
        if texture_to_color_decouple is None:
            texture_to_color_decouple = texture_color_decouple
        
        # 1. 提取原始 Embedding 
        if isinstance(clr_ref_img, Image.Image):
            clr_ref_img = [clr_ref_img]
        if isinstance(clr_texture_ref_img, Image.Image):
            clr_texture_ref_img = [clr_texture_ref_img]
        if isinstance(texture_ref_img, Image.Image):
            texture_ref_img = [texture_ref_img]
        if isinstance(geometry_ref_img, Image.Image):
            geometry_ref_img = [geometry_ref_img]
        if isinstance(texture_ref_img2, Image.Image):
            texture_ref_img2 = [texture_ref_img2]

        base_ref_img = clr_ref_img or texture_ref_img or geometry_ref_img or texture_ref_img2
        if base_ref_img is None:
            raise ValueError("At least one reference image must be provided for IP-Adapter generation.")

        base_clip_image = self.clip_image_processor(images=base_ref_img, return_tensors="pt").pixel_values.to(
            self.device, dtype=torch.float16
        )
        uncond_clip = self.image_encoder(torch.zeros_like(base_clip_image), output_hidden_states=True).hidden_states[-2]
        uncond_p = self.image_proj_model(uncond_clip)

        def encode_hidden_states(ref_images):
            if ref_images is None:
                return None
            clip_image = self.clip_image_processor(images=ref_images, return_tensors="pt").pixel_values.to(
                self.device,
                dtype=torch.float16,
            )
            return self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]

        def apply_texture_punishment(texture_embeds, texture_images):
            if (
                texture_embeds is None
                or texture_images is None
                or punish_weight == 0
                or punish_type is None
            ):
                return texture_embeds

            texture_img_np = np.array(texture_images[0])
            mean_values = texture_img_np.mean(axis=(0, 1))
            texture_img_puregray = Image.fromarray(
                np.full_like(texture_img_np, mean_values, dtype=np.uint8)
            )
            txr_puregray_clip_image = self.clip_image_processor(
                images=[texture_img_puregray],
                return_tensors="pt",
            ).pixel_values.to(self.device, dtype=torch.float16)
            txr_puregray_clip_embeds = self.image_encoder(
                txr_puregray_clip_image,
                output_hidden_states=True,
            ).hidden_states[-2]

            dtype = texture_embeds.dtype
            _, token_num, _ = texture_embeds.shape
            wo_batch = torch.concat([texture_embeds, txr_puregray_clip_embeds], dim=1).squeeze(0)
            latent_size = wo_batch.shape[0]
            wo_batch = wo_batch.permute(1, 0)
            wo_batch = punish_weight_module(
                wo_batch.float(),
                latent_size,
                alpha=punish_weight,
                method=punish_type,
            )
            return wo_batch.permute(1, 0)[:token_num, :].unsqueeze(0).to(dtype)

        c_ori = None
        clr_txr_clip_embeds = None
        if clr_ref_img is not None:
            color_stream_ref_imgs = clr_ref_img
            c_ori = encode_hidden_states(color_stream_ref_imgs)
            if clr_texture_ref_img is None:
                clr_texture_ref_img = [img.convert("L") for img in color_stream_ref_imgs]
            clr_txr_clip_embeds = encode_hidden_states(clr_texture_ref_img)

        t_ori = encode_hidden_states(texture_ref_img) if texture_ref_img is not None else None

        g_ori = None
        if geometry_ref_img is not None:
            geo_clip_image = self.clip_image_processor(images=geometry_ref_img, return_tensors="pt").pixel_values.to(
                self.device, dtype=torch.float16
            )
            g_ori = self.image_encoder(geo_clip_image, output_hidden_states=True).hidden_states[-2]

        eps = 1e-6
        def proj_res(A, B, scale):
            if A is None or B is None or scale == 0: return A
            B_norm_sq = (B * B).sum(dim=-1, keepdim=True).clamp_min(eps)
            proj = ((A * B).sum(dim=-1, keepdim=True) / B_norm_sq) * B
            return A - scale * proj

        def restore(original, processed):
            if original is None or processed is None:
                return None
            return processed * (
                original.norm(p=2, dim=-1, keepdim=True)
                / processed.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
            )

        # 3. 執行解耦流程 (依序處理)
        
        # (A) 基礎預處理：Geometry 單向扣除 Color (依你建議保留)
        g_pre = proj_res(g_ori, c_ori, geometry_color_decouple) if g_ori is not None else None 
        
        # (B) 核心雙向互扣一：Texture <-> Geometry (處理構圖干擾)
        # 使用預處理後的 g_pre 進行互扣
        t_after_g = proj_res(t_ori, g_pre, texture_to_geometry_decouple)
        g_final = proj_res(g_pre, t_ori, geometry_to_texture_decouple) if g_pre is not None else None
        color_after_g = proj_res(c_ori, g_pre if g_pre is not None else g_ori, color_to_geometry_decouple)

        # (C) 核心雙向互扣二：Color <-> Texture (處理色調干擾)
        # 注意：使用剛扣完幾何的 t_after_g 來跟顏色互扣，確保層次清晰
        c_final = proj_res(color_after_g, t_after_g, color_to_texture_decouple)
        t_final = proj_res(t_after_g, c_ori, texture_to_color_decouple)

        # 4. 能量補償更新變數
        clr_clip_image_embeds = restore(c_ori, c_final)
        txr_clip_image_embeds2 = restore(t_ori, t_final)
        geo_clip_image_embeds = restore(g_ori, g_final) if g_final is not None else None
        # 5. Gray punishment and stream assembly
        is_punish_gray = (
            txr_clip_image_embeds2 is not None
            and texture_ref_img is not None
            and punish_weight != 0
            and punish_type is not None
        )
        if is_punish_gray:
            texture_img_np = np.array(texture_ref_img[0])
            mean_values = texture_img_np.mean(axis=(0, 1))
            texture_img_puregray = Image.fromarray(np.full_like(texture_img_np, mean_values, dtype=np.uint8))
            txr_puregray_clip_image = self.clip_image_processor(images=[texture_img_puregray], return_tensors="pt").pixel_values.to(self.device, dtype=torch.float16)
            txr_puregray_clip_embeds = self.image_encoder(txr_puregray_clip_image, output_hidden_states=True).hidden_states[-2]

            dtype = txr_clip_image_embeds2.dtype
            b, t_num, ch = txr_clip_image_embeds2.shape
            wo_batch = torch.concat([txr_clip_image_embeds2, txr_puregray_clip_embeds], dim=1).squeeze(0)
            latent_size = wo_batch.shape[0]
            wo_batch = wo_batch.permute(1, 0)
            wo_batch = punish_weight_module(wo_batch.float(), latent_size, alpha=punish_weight, method=punish_type)
            txr_clip_image_embeds2 = wo_batch.permute(1, 0)[:t_num, :].unsqueeze(0).to(dtype)

        color_stream_embeds = None

        if clr_clip_image_embeds is not None:
            color_stream_embeds = clr_clip_image_embeds * color_scale
            if clr_txr_clip_embeds is not None:
                color_stream_embeds = color_stream_embeds - clr_txr_clip_embeds * subscale
            # 刪除此處的 + txr_clip_image_embeds2 * texture_scale 
        texture_stream_embeds = txr_clip_image_embeds2 * texture_scale if txr_clip_image_embeds2 is not None else None
        geometry_stream_embeds = geometry_scale * geo_clip_image_embeds if geo_clip_image_embeds is not None else None

        if texture_stream_embeds is not None and texture_ref_img2 is not None and texture2_scale != 0:
            # ... (前段高通濾波處理不變) [cite: 135]
            texture2_highpass_imgs = []
            for texture2_img in texture_ref_img2:
                texture2_rgb = texture2_img.convert("RGB")
                texture2_blur = texture2_rgb.filter(ImageFilter.GaussianBlur(radius=5))
                texture2_np = np.array(texture2_rgb).astype(np.float32)
                texture2_blur_np = np.array(texture2_blur).astype(np.float32)
                texture2_highpass_np = np.clip(texture2_np - texture2_blur_np + 127.5, 0, 255).astype(np.uint8)
                texture2_highpass_imgs.append(Image.fromarray(texture2_highpass_np))

            txr2_clip_image = self.clip_image_processor(images=texture2_highpass_imgs, return_tensors="pt").pixel_values
            txr2_clip_image = txr2_clip_image.to(self.device, dtype=torch.float16)
            txr2_clip_image_embeds = self.image_encoder(txr2_clip_image, output_hidden_states=True).hidden_states[-2]
            texture_stream_embeds = texture_stream_embeds + txr2_clip_image_embeds * texture2_scale

        color_p = self.image_proj_model(color_stream_embeds) if color_stream_embeds is not None else uncond_p
        geo_p = self.image_proj_model(geometry_stream_embeds) if geometry_stream_embeds is not None else uncond_p
        tex_p = self.image_proj_model(texture_stream_embeds) if texture_stream_embeds is not None else uncond_p
        image_prompt_embeds = torch.cat([color_p, geo_p, tex_p], dim=1)

        uncond_image_prompt_embeds = torch.cat([uncond_p, uncond_p, uncond_p], dim=1)

        return image_prompt_embeds, uncond_image_prompt_embeds
    def generate(
        self,
        clr_ref_img=None,     # for color transfer
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        clr_texture_ref_img = None,     # for color transfer
        substract_scale = 0.,
        texture_scale = 0.,
        texture2_scale = 0.,
        geometry_sub_scale = 1.0,
        texture_geometry_decouple = None,
        geometry_scale = 0.0,
        geometry_color_decouple = 0.2,
        texture_color_decouple = 0.3,
        color_to_geometry_decouple = None,
        texture_to_geometry_decouple = None,
        geometry_to_texture_decouple = None,
        color_to_texture_decouple = None,
        texture_to_color_decouple = None,
        color_scale = 1.,
        texture_ref_img = None,    # for texture transfer
        geometry_ref_img = None,   # for geometry disentanglement
        texture_ref_img2 = None,   # for brushstroke transfer
        wct_starts_step = 0,
        wct_ends_step=0,
        wct_guidance=0,
        csd_iter_num = 1,
        update_latent_starts_step=0, 
        update_latent_ends_step=1,
        reg_scale=0.1,
        clr_ref_img_dir=None,
        sty_ref_img_dir=None,
        wctnoise_add_scale=0.,
        punish_weight = 0.005,    # 0.005
        punish_type = 'soft-weight',   #
        save_name = '',
        **kwargs,
    ):
        self.set_scale(scale)
        if texture_geometry_decouple is None:
            texture_geometry_decouple = geometry_sub_scale
        if texture_to_geometry_decouple is None:
            texture_to_geometry_decouple = texture_geometry_decouple
        if geometry_to_texture_decouple is None:
            geometry_to_texture_decouple = texture_geometry_decouple
        if color_to_geometry_decouple is None:
            color_to_geometry_decouple = geometry_color_decouple
        if color_to_texture_decouple is None:
            color_to_texture_decouple = texture_color_decouple
        if texture_to_color_decouple is None:
            texture_to_color_decouple = texture_color_decouple
        
        num_prompts = 1 #if isinstance(clr_ref_img, Image.Image) else len(clr_ref_img)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # cal clip img embedding      
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(clr_ref_img, 
                                                                                clr_texture_ref_img, 
                                                                                texture_ref_img, 
                                                                                geometry_ref_img,
                                                                                texture_ref_img2,
                                                                                substract_scale, 
                                                                                color_scale, 
                                                                                texture_scale,
                                                                                texture2_scale,
                                                                                geometry_sub_scale,
                                                                                texture_geometry_decouple,
                                                                                geometry_scale,
                                                                                geometry_color_decouple,
                                                                                texture_color_decouple,
                                                                                color_to_geometry_decouple,
                                                                                texture_to_geometry_decouple,
                                                                                geometry_to_texture_decouple,
                                                                                color_to_texture_decouple,
                                                                                texture_to_color_decouple,
                                                                                punish_weight = punish_weight,    # 0.005
                                                                                punish_type = punish_type)  
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        # 霈∠??捆prompt embedding
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            # 
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)  # [text emd, img emd]
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            wct_starts_step= wct_starts_step,
            wct_ends_step= wct_ends_step,
            wct_guidance=wct_guidance,
            csd_iter_num = csd_iter_num,
            clr_ref_img_dir=clr_ref_img_dir,
            sty_ref_img_dir=sty_ref_img_dir,
            wctnoise_add_scale=wctnoise_add_scale,
            save_name=save_name,
            **kwargs,
        ).images

        return images

    ## ddim Inversion
    @torch.no_grad()
    def invert(self,
        clr_ref_img,
        start_latents,
        prompt,
        negative_prompt=None,
        guidance_scale=3.5,
        num_inference_steps=80,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        seed=42,
        scale=1.0,
        imgsize=(512,512),
        cross_attention_kwargs=None,
        **kwargs,
    ):
        
        self.set_scale(scale)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        if not isinstance(prompt, List):
            prompt = [prompt]
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt]


        # Encode prompt
        # text_embeddings = self.pipe._encode_prompt(
        #     prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        # )

        # Latents are now the specified start latents
        latents = start_latents.clone()


        # 霈∠?clip img embedding
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(clr_ref_img, clr_texture_ref_img=None)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, 1, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, 1, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed, seq_len, -1)

        # 霈∠??捆 text prompt embedding
        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            # 
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)  # [text emd, img emd]
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)
         # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.pipe._get_add_time_ids(
            imgsize, (0,0), imgsize, dtype=prompt_embeds.dtype
        )
        negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device).repeat(1 * num_images_per_prompt, 1)

        # 8. Denoising loop
        # num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

    
        # We'll keep a list of the inverted latents as the process goes on
        intermediate_latents = []

        # Set num inference steps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.pipe.scheduler.timesteps)
        alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(self.device)
        for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

            # We'll skip the final iteration
            if i >= num_inference_steps - 1:
                continue

            t = timesteps[i].long()

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.pipe.unet(latent_model_input, 
                                        t, 
                                        encoder_hidden_states=prompt_embeds,
                                        cross_attention_kwargs=cross_attention_kwargs,
                                        added_cond_kwargs=added_cond_kwargs,
                                        return_dict=False)[0]

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
 
            
            
            current_t = max(0, int(t.item() - (1000 // num_inference_steps)))  # t
            next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
            alpha_t = alphas_cumprod[current_t]
            alpha_t_next = alphas_cumprod[next_t]

            # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
            latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
                1 - alpha_t_next
            ).sqrt() * noise_pred

            # Store
            intermediate_latents.append(latents)

        return torch.cat(intermediate_latents)
