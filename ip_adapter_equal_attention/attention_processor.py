# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PHASE_NAMES = ("front", "mid", "back")


# Check Flash Attention / SDPA availability

def is_sdpa_available():
    return hasattr(F, "scaled_dot_product_attention")


# def get_base_phase_mix_table(save_name):
#     if "down_blocks" in save_name:
#         return {
#             "front": (0.00, 1.00, 0.00),
#             "mid": (0.00, 1.00, 0.00),
#             "back": (0.00, 1.00, 0.00),
#         }
#     if "mid_block" in save_name:
#         return {
#             "front": (0.18, 0.64, 0.18),
#             "mid": (0.18, 0.60, 0.22),
#             "back": (0.18, 0.54, 0.28),
#         }
#     if "up_blocks.0" in save_name:
#         return {
#             "front": (0.46, 0.34, 0.20),
#             "mid": (0.34, 0.34, 0.32),
#             "back": (0.34, 0.28, 0.38),
#         }
#     if "up_blocks" in save_name:
#         return {
#             "front": (0.30, 0.46, 0.24),
#             "mid": (0.24, 0.38, 0.38),
#             "back": (0.24, 0.30, 0.46),
#         }
#     return {
#         "front": (1.0, 0.0, 0.0),
#         "mid": (1.0, 0.0, 0.0),
#         "back": (1.0, 0.0, 0.0),
#     }


# def get_phase_mix_table(save_name):
#     phase_mix = dict(get_base_phase_mix_table(save_name))
#     if "up_blocks.0.attentions.1" in save_name and save_name.endswith("attn2.processor"):
#         phase_mix["back"] = (0.40, 0.20, 0.40)
#     return phase_mix
def get_base_phase_mix_table(save_name):
    equal_mix = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return {
        "front": equal_mix,
        "mid": equal_mix,
        "back": equal_mix,
    }

def get_phase_mix_table(save_name):
    return get_base_phase_mix_table(save_name)

class AttnProcessor(nn.Module):
    def __init__(self, hidden_size=None, cross_attention_dim=None):
        super().__init__()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class IPAttnProcessor2_0(nn.Module):
    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        scale=1.0,
        num_tokens=16,
        skip=False,
        instance_idx=0,
        prompt_tokens=None,
        ca_mask=False,
        save_name="",
        stream_token_sizes=None,
        stream_scales=None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.skip = skip
        self.instance_idx = instance_idx
        self.prompt_tokens = prompt_tokens or []
        self.ca_mask = ca_mask
        self.save_name = save_name

        # Compatibility with existing caller:
        # - old caller may pass total image tokens (48)
        # - new three-stream design needs per-stream token count (16)
        if stream_token_sizes is not None:
            self.stream_token_sizes = [int(x) for x in stream_token_sizes]
            self.total_ip_tokens = int(sum(self.stream_token_sizes))
            self.per_stream_tokens = self.stream_token_sizes[0] if self.stream_token_sizes else 0
        else:
            self.total_ip_tokens = int(num_tokens) if num_tokens is not None else 16
            self.per_stream_tokens = (
                self.total_ip_tokens // 3 if self.total_ip_tokens % 3 == 0 else self.total_ip_tokens
            )
            self.total_ip_tokens = self.per_stream_tokens * 3
            self.stream_token_sizes = [self.per_stream_tokens, self.per_stream_tokens, self.per_stream_tokens]
        self.stream_scales = list(stream_scales or [1.0, 1.0, 1.0])
        self.stream_step_scales = [1.0 for _ in self.stream_scales]
        self.stream_phase = "front"
        self.register_buffer(
            "_lowpass_kernel_3x3",
            torch.tensor(
                [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            / 16.0,
            persistent=False,
        )

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self._warned_projection_fallback = False
        self._warned_non_finite_ip = False

    def set_stream_step_scales(self, step_scales):
        if step_scales is None:
            self.stream_step_scales = [1.0 for _ in self.stream_scales]
            return
        self.stream_step_scales = list(step_scales)

    def set_stream_phase(self, phase_name):
        self.stream_phase = phase_name or "front"

    def _get_phase_mix(self):
        phase = self.stream_phase if self.stream_phase in set(PHASE_NAMES) else "front"
        phase_mix = get_phase_mix_table(self.save_name)
        return phase, phase_mix[phase]

    def _token_lowpass_3x3(self, x):
        # x: [batch, tokens, channels]
        b, t, c = x.shape
        side = int(math.sqrt(t))
        if side * side != t:
            return x
        x2 = x.transpose(1, 2).contiguous().reshape(b * c, 1, side, side)
        kernel = self._lowpass_kernel_3x3
        if kernel.device != x.device or kernel.dtype != x.dtype:
            kernel = kernel.to(device=x.device, dtype=x.dtype)
        y = F.conv2d(x2, kernel, padding=1)
        return y.reshape(b, c, side, side).reshape(b, c, t).transpose(1, 2)

    def _sanitize_ip_tensor(self, x, label):
        if x is None:
            return None
        if not torch.isfinite(x).all():
            if not self._warned_non_finite_ip:
                print(
                    "[WARN] Non-finite IP tensor detected in {0} ({1}); sanitizing before projection".format(
                        self.save_name,
                        label,
                    )
                )
                self._warned_non_finite_ip = True
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        if x.dtype == torch.float16:
            x = x.clamp(min=-6e4, max=6e4)
        return x.contiguous()

    def _project_ip(self, proj, hidden_states, label):
        hidden_states = self._sanitize_ip_tensor(hidden_states, label)
        try:
            projected = proj(hidden_states)
        except RuntimeError as exc:
            message = str(exc).lower()
            should_retry_fp32 = (
                hidden_states.is_cuda
                and hidden_states.dtype == torch.float16
                and ("cublas" in message or "cuda error" in message)
            )
            if not should_retry_fp32:
                raise
            if not self._warned_projection_fallback:
                print(
                    "[WARN] Retrying unstable IP projection in float32 for {0}".format(self.save_name)
                )
                self._warned_projection_fallback = True
            weight = proj.weight.float()
            bias = proj.bias.float() if proj.bias is not None else None
            projected = F.linear(hidden_states.float(), weight, bias).to(hidden_states.dtype)
        return self._sanitize_ip_tensor(projected, "{0}_projected".format(label))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        end_pos = encoder_hidden_states.shape[1] - self.total_ip_tokens
        if end_pos <= 0:
            text_embeds = encoder_hidden_states
            geo_ip = tex_ip = clr_ip = None
        else:
            text_embeds = encoder_hidden_states[:, :end_pos, :]
            all_ip_embeds = encoder_hidden_states[:, end_pos:, :]
            tk0 = self.stream_token_sizes[0] if len(self.stream_token_sizes) > 0 else 0
            tk1 = self.stream_token_sizes[1] if len(self.stream_token_sizes) > 1 else 0
            tk2 = self.stream_token_sizes[2] if len(self.stream_token_sizes) > 2 else 0
            # Must match ip_adapter.py concat order: [color, geometry, texture]
            clr_ip = all_ip_embeds[:, 0:tk0, :]
            geo_ip = all_ip_embeds[:, tk0 : tk0 + tk1, :]
            tex_ip = all_ip_embeds[:, tk0 + tk1 : tk0 + tk1 + tk2, :]

        if attn.norm_cross:
            text_embeds = attn.norm_encoder_hidden_states(text_embeds)

        key = attn.to_k(text_embeds)
        value = attn.to_v(text_embeds)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        q_res = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k_res = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v_res = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Text attention
        if is_sdpa_available():
            hidden_states = F.scaled_dot_product_attention(
                q_res, k_res, v_res, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            attn_scores = torch.matmul(q_res, k_res.transpose(-1, -2)) * (head_dim**-0.5)
            attn_probs = attn_scores.softmax(dim=-1)
            hidden_states = torch.matmul(attn_probs, v_res)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # IP attention (phase-aware three-stream routing)
        if not self.skip and any(stream is not None for stream in (clr_ip, geo_ip, tex_ip)):
            phase_name, phase_mix = self._get_phase_mix()
            color_weight, geometry_weight, texture_weight = phase_mix
            color_base = self.stream_scales[min(0, len(self.stream_scales) - 1)]
            geometry_base = self.stream_scales[min(1, len(self.stream_scales) - 1)]
            texture_base = self.stream_scales[min(2, len(self.stream_scales) - 1)]
            color_step = self.stream_step_scales[min(0, len(self.stream_step_scales) - 1)]
            geometry_step = self.stream_step_scales[min(1, len(self.stream_step_scales) - 1)]
            texture_step = self.stream_step_scales[min(2, len(self.stream_step_scales) - 1)]

            effective_base = (
                color_weight * color_base
                + geometry_weight * geometry_base
                + texture_weight * texture_base
            )
            effective_step = (
                color_weight * color_step
                + geometry_weight * geometry_step
                + texture_weight * texture_step
            )

            if effective_base > 0 and effective_step > 0:
                geo_source = self._token_lowpass_3x3(geo_ip) if "down_blocks" in self.save_name else geo_ip
                active_ip_hidden = self._sanitize_ip_tensor((
                    color_weight * clr_ip
                    + geometry_weight * geo_source
                    + texture_weight * tex_ip
                ), "active_ip_hidden")
                current_ip_scale = self.scale * effective_base * effective_step

                ip_k = self._project_ip(self.to_k_ip, active_ip_hidden, "ip_k").view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
                ip_v = self._project_ip(self.to_v_ip, active_ip_hidden, "ip_v").view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)

                # Keep the geometry confidence gate only in structural down blocks.
                if "down_blocks" in self.save_name and geometry_weight > 0 and geometry_step > 0:
                    geom_gate = torch.ones((batch_size, 1, 1), device=query.device, dtype=query.dtype)
                    with torch.no_grad():
                        geo_only_hidden = self._sanitize_ip_tensor(
                            geometry_weight * geo_source,
                            "geo_only_hidden",
                        )
                        if torch.isfinite(q_res).all() and torch.isfinite(geo_only_hidden).all():
                            try:
                                geo_k = self._project_ip(self.to_k_ip, geo_only_hidden, "geo_k").view(
                                    batch_size, -1, attn.heads, head_dim
                                ).transpose(1, 2)
                                q_gate = q_res.float()
                                geo_k = geo_k.float()
                                geo_scores = torch.matmul(q_gate, geo_k.transpose(-1, -2)) * (head_dim**-0.5)
                                geo_scores = geo_scores - geo_scores.amax(dim=-1, keepdim=True)
                                geo_probs = geo_scores.softmax(dim=-1)
                                if torch.isfinite(geo_probs).all():
                                    token_count = max(int(geo_probs.shape[-1]), 2)
                                    geo_entropy = -(geo_probs * torch.log(geo_probs.clamp_min(1e-6))).sum(dim=-1)
                                    geo_entropy = geo_entropy.mean(dim=(1, 2), keepdim=True)
                                    entropy_norm = geo_entropy / math.log(float(token_count))
                                    geom_gate = (1.25 - entropy_norm).clamp(0.85, 1.15).to(query.dtype)
                            except RuntimeError:
                                # The gate is auxiliary only; fall back to neutral scaling if this
                                # numerical side path becomes unstable on the current CUDA runtime.
                                geom_gate = torch.ones((batch_size, 1, 1), device=query.device, dtype=query.dtype)
                    current_ip_scale = current_ip_scale * geom_gate.view(batch_size, 1, 1)

                if is_sdpa_available():
                    ip_res = F.scaled_dot_product_attention(q_res, ip_k, ip_v, attn_mask=None, dropout_p=0.0)
                else:
                    ip_scores = torch.matmul(q_res, ip_k.transpose(-1, -2)) * (head_dim**-0.5)
                    ip_res = torch.matmul(ip_scores.softmax(dim=-1), ip_v)

                ip_res = ip_res.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states + current_ip_scale * ip_res.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class CNAttnProcessor2_0(nn.Module):
    def __init__(self, num_tokens=4):
        super().__init__()
        # Caller passes total IP tokens; we strip those from text context.
        self.total_ip_tokens = int(num_tokens) if num_tokens is not None else 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.total_ip_tokens
            if end_pos > 0:
                encoder_hidden_states = encoder_hidden_states[:, :end_pos, :]

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if is_sdpa_available():
            hidden_states = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        else:
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim**-0.5)
            hidden_states = torch.matmul(attn_scores.softmax(dim=-1), v)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


# Compatibility aliases used by current ip_adapter.py imports
AttnProcessor2_0 = AttnProcessor
IPAttnProcessor = IPAttnProcessor2_0
CNAttnProcessor = CNAttnProcessor2_0



