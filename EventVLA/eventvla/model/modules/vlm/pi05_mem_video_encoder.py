import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling


class TemporalPositionEncoding(nn.Module):
    """Fixed sinusoidal temporal encoding with zero embedding on the current frame."""

    def __init__(self, embed_dim: int, max_frames: int = 32) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")

        # Reverse the positions so the newest frame uses the last row.
        positions = torch.arange(max_frames - 1, -1, -1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(max_frames, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        # Keep the current frame identical to single-image initialization.
        pe[-1].zero_()
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 4:
            raise ValueError(f"Expected [B, T, N, D], got {tuple(hidden_states.shape)}")

        num_frames = hidden_states.shape[1]
        if num_frames > self.pe.shape[0]:
            raise ValueError(f"num_frames={num_frames} exceeds max_frames={self.pe.shape[0]}")
        if num_frames == 1:
            return hidden_states

        temporal_pe = self.pe[-num_frames:].view(1, num_frames, 1, -1).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        return hidden_states + temporal_pe


def causal_temporal_attention(
    hidden_states: torch.Tensor,
    *,
    layer_norm: nn.LayerNorm,
    attention: nn.Module,
) -> torch.Tensor:
    """Apply causal attention over time for the same spatial patch using reused ViT attention weights."""

    if hidden_states.ndim != 4:
        raise ValueError(f"Expected [B, T, N, D], got {tuple(hidden_states.shape)}")

    batch_size, num_frames, num_patches, embed_dim = hidden_states.shape
    if num_frames == 1:
        return hidden_states

    residual = hidden_states
    # Group tokens by spatial patch so temporal attention only mixes the same patch
    # across frames, instead of attending over all space-time tokens jointly.
    temporal_states = hidden_states.transpose(1, 2).reshape(batch_size * num_patches, num_frames, embed_dim)
    temporal_states = layer_norm(temporal_states)

    queries = attention.q_proj(temporal_states)
    keys = attention.k_proj(temporal_states)
    values = attention.v_proj(temporal_states)

    queries = queries.view(batch_size * num_patches, num_frames, attention.num_heads, attention.head_dim).transpose(1, 2)
    keys = keys.view(batch_size * num_patches, num_frames, attention.num_heads, attention.head_dim).transpose(1, 2)
    values = values.view(batch_size * num_patches, num_frames, attention.num_heads, attention.head_dim).transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(
        queries,
        keys,
        values,
        attn_mask=None,
        dropout_p=attention.dropout if attention.training else 0.0,
        is_causal=True,
    )
    attn_output = attn_output.transpose(1, 2).reshape(batch_size * num_patches, num_frames, embed_dim).contiguous()
    attn_output = attention.out_proj(attn_output)
    attn_output = attn_output.reshape(batch_size, num_patches, num_frames, embed_dim).transpose(1, 2).contiguous()
    return residual + attn_output


class Pi05MemVisionTower(nn.Module):
    """SigLIP vision wrapper with MEM-style temporal attention injected every N layers."""

    def __init__(self, base_vision_tower: nn.Module, *, max_frames: int = 32, temporal_attn_every_n_layers: int = 4) -> None:
        super().__init__()
        if temporal_attn_every_n_layers <= 0:
            raise ValueError("temporal_attn_every_n_layers must be positive")

        self.config = base_vision_tower.config
        self.vision_model = base_vision_tower.vision_model
        self.temporal_position_encoding = TemporalPositionEncoding(
            embed_dim=self.vision_model.config.hidden_size,
            max_frames=max_frames,
        )
        self.temporal_attn_every_n_layers = temporal_attn_every_n_layers
        self.temporal_layer_indices = {
            idx for idx in range(len(self.vision_model.encoder.layers)) if (idx + 1) % temporal_attn_every_n_layers == 0
        }

    @property
    def gradient_checkpointing(self) -> bool:
        return bool(self.vision_model.encoder.gradient_checkpointing)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, enabled: bool) -> None:
        self.vision_model.encoder.gradient_checkpointing = enabled

    def _forward_single_image(
        self,
        pixel_values: torch.Tensor,
        *,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        if pixel_values.ndim == 4:
            # Pure single-image path: preserve the original SigLIP behavior exactly.
            return self._forward_single_image(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected [B, C, H, W] or [B, T, C, H, W], got {tuple(pixel_values.shape)}")

        batch_size, num_frames, channels, height, width = pixel_values.shape
        if num_frames == 1:
            # Even if callers keep a temporal dimension, T=1 should still behave as
            # the pretrained single-frame encoder.
            return self._forward_single_image(
                pixel_values[:, 0],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

        # Run patch embedding frame-by-frame, then restore [B, T, N, D] so we can
        # alternate spatial attention with temporal attention.
        flat_pixel_values = pixel_values.reshape(batch_size * num_frames, channels, height, width)
        hidden_states = self.vision_model.embeddings(
            flat_pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )
        _, num_patches, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, num_frames, num_patches, -1)
        hidden_states = self.temporal_position_encoding(hidden_states)

        if self.vision_model.encoder.layers and self.vision_model.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)

        encoder_states = () if output_hidden_states else None
        spatial_attentions = () if output_attentions else None

        for layer_idx, layer in enumerate(self.vision_model.encoder.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states[:, -1].contiguous(),)

            # Apply the original ViT block independently on each frame.
            spatial_states = hidden_states.view(batch_size * num_frames, num_patches, -1)
            layer_outputs = layer(
                spatial_states,
                attention_mask=None,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0].view(batch_size, num_frames, num_patches, -1)

            if layer_idx in self.temporal_layer_indices:
                # Reuse the pretrained attention projections to inject causal temporal
                # mixing every N layers without introducing extra attention weights.
                hidden_states = causal_temporal_attention(
                    hidden_states,
                    layer_norm=layer.layer_norm1,
                    attention=layer.self_attn,
                )

            if output_attentions:
                spatial_attentions = spatial_attentions + (layer_outputs[1],)

        # Match the original PI05 token interface: only expose the current frame to
        # downstream modules, while historical frames act as internal memory.
        current_frame_features = self.vision_model.post_layernorm(hidden_states[:, -1].contiguous())
        if output_hidden_states:
            encoder_states = encoder_states + (current_frame_features,)

        pooler_output = self.vision_model.head(current_frame_features) if self.vision_model.use_head else None
        return BaseModelOutputWithPooling(
            last_hidden_state=current_frame_features,
            pooler_output=pooler_output,
            hidden_states=encoder_states,
            attentions=spatial_attentions,
        )
