from __future__ import annotations

import torch
import torch.nn as nn

try:
    from diffusers import AudioLDM2Pipeline
except Exception:
    AudioLDM2Pipeline = None


class AudioLDMUNetWrapper(nn.Module):
    def __init__(
        self,
        *,
        model_id: str,
        device: torch.device | str,
        dtype: torch.dtype,
        cache_pipeline: bool = True,
        audioldm_unet_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if audioldm_unet_kwargs is not None:
            raise ValueError(
                "model.audioldm_unet_kwargs is deprecated. "
                "This repo now loads the pretrained U-Net directly from AudioLDM2Pipeline."
            )
        if AudioLDM2Pipeline is None:
            raise ImportError(
                "diffusers is required to load the pretrained AudioLDM2 U-Net."
            )

        self.device = torch.device(device)
        self.dtype = dtype
        self.model_id = str(model_id)

        pipe = AudioLDM2Pipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        if not hasattr(pipe, "unet") or pipe.unet is None:
            raise RuntimeError(
                f"AudioLDM2Pipeline({self.model_id!r}) did not expose a pretrained U-Net."
            )

        self.backbone = pipe.unet.to(device=self.device, dtype=self.dtype)
        self.backbone.eval()
        self.pipeline = pipe if cache_pipeline else None
        if not cache_pipeline:
            del pipe

        self.config = getattr(self.backbone, "config", None)
        if self.config is None:
            raise RuntimeError("Loaded pretrained U-Net is missing a config object.")

        self._input_block_channels = self._infer_residual_channels()
        self._middle_block_channel = self._infer_mid_block_channel()

    @property
    def backend_name(self) -> str:
        return "diffusers_pretrained_unet"

    def get_model_channels(self) -> int:
        block_out_channels = getattr(self.config, "block_out_channels", None)
        if not block_out_channels:
            raise RuntimeError("Pretrained U-Net config is missing block_out_channels.")
        return int(block_out_channels[0])

    def get_time_proj(self) -> nn.Module:
        return self.backbone.time_proj

    def get_time_embedding(self) -> nn.Module:
        return self.backbone.time_embedding

    def get_control_modules(self) -> dict[str, nn.Module]:
        return {
            "conv_in": self.backbone.conv_in,
            "down_blocks": self.backbone.down_blocks,
            "mid_block": self.backbone.mid_block,
        }

    @property
    def control_specs(self) -> dict[str, object]:
        return {
            "input_block_channels": list(self._input_block_channels),
            "middle_block_channel": int(self._middle_block_channel),
        }

    @staticmethod
    def _flatten_values(value) -> list[object]:
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.extend(AudioLDMUNetWrapper._flatten_values(item))
            return out
        return [value]

    def get_cross_attention_dims(self) -> tuple[int, int | None]:
        candidates = []
        field_value = getattr(self.config, "cross_attention_dim", None)
        candidates.extend(self._flatten_values(field_value))
        dims = [int(candidate) for candidate in candidates if candidate is not None]
        unique_dims = []
        for dim in dims:
            if dim not in unique_dims:
                unique_dims.append(dim)
        if len(unique_dims) >= 2:
            return unique_dims[0], unique_dims[1]
        if len(unique_dims) == 1:
            return unique_dims[0], None

        discovered = []
        for block in self.backbone.down_blocks:
            attentions = getattr(block, "attentions", None)
            if attentions is None:
                continue
            for attention in attentions:
                transformer_blocks = getattr(attention, "transformer_blocks", None)
                if transformer_blocks is None:
                    continue
                for transformer in transformer_blocks:
                    attn2 = getattr(transformer, "attn2", None)
                    to_k = getattr(attn2, "to_k", None)
                    in_features = getattr(to_k, "in_features", None)
                    if in_features is not None and int(in_features) not in discovered:
                        discovered.append(int(in_features))
        if len(discovered) >= 2:
            return discovered[0], discovered[1]
        if len(discovered) == 1:
            return discovered[0], None
        raise RuntimeError(
            "Could not infer valid cross-attention dimensions from the pretrained U-Net config or modules."
        )

    def prepare_encoder_hidden_states(
        self,
        batch_size: int,
        *,
        encoder_hidden_states: torch.Tensor | None,
        encoder_hidden_states_1: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor | None]:
        primary_dim, secondary_dim = self.get_cross_attention_dims()
        if encoder_hidden_states is None:
            encoder_hidden_states = torch.zeros(
                batch_size,
                1,
                primary_dim,
                device=device,
                dtype=dtype,
            )
        else:
            encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=dtype)

        if secondary_dim is None:
            encoder_hidden_states_1 = None
        elif encoder_hidden_states_1 is None:
            encoder_hidden_states_1 = torch.zeros(
                batch_size,
                1,
                secondary_dim,
                device=device,
                dtype=dtype,
            )
        else:
            encoder_hidden_states_1 = encoder_hidden_states_1.to(device=device, dtype=dtype)

        return {
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_1": encoder_hidden_states_1,
        }

    def _infer_residual_channels(self) -> list[int]:
        channels = [int(self.backbone.conv_in.out_channels)]
        for block in self.backbone.down_blocks:
            if hasattr(block, "resnets"):
                for resnet in block.resnets:
                    out_channels = getattr(resnet, "out_channels", None)
                    if out_channels is None and hasattr(resnet, "conv2"):
                        out_channels = getattr(resnet.conv2, "out_channels", None)
                    if out_channels is None:
                        raise RuntimeError(
                            "Could not infer residual channel count from pretrained down block."
                        )
                    channels.append(int(out_channels))
            if hasattr(block, "downsamplers") and block.downsamplers is not None:
                for downsampler in block.downsamplers:
                    out_channels = getattr(downsampler, "out_channels", None)
                    if out_channels is None and hasattr(downsampler, "conv"):
                        out_channels = getattr(downsampler.conv, "out_channels", None)
                    if out_channels is None:
                        raise RuntimeError(
                            "Could not infer downsampler channel count from pretrained U-Net."
                        )
                    channels.append(int(out_channels))
        return channels

    def _infer_mid_block_channel(self) -> int:
        out_channels = None
        if hasattr(self.backbone.mid_block, "resnets") and len(self.backbone.mid_block.resnets) > 0:
            resnet = self.backbone.mid_block.resnets[-1]
            out_channels = getattr(resnet, "out_channels", None)
            if out_channels is None and hasattr(resnet, "conv2"):
                out_channels = getattr(resnet.conv2, "out_channels", None)
        if out_channels is None:
            raise RuntimeError("Could not infer middle block channel count from pretrained U-Net.")
        return int(out_channels)

    def _get_unet_spatial_factor(self) -> int:
        num_upsamplers = getattr(self.backbone, "num_upsamplers", None)
        if num_upsamplers is not None:
            return 2 ** int(num_upsamplers)
        block_out_channels = getattr(self.config, "block_out_channels", ())
        return 2 ** max(0, len(block_out_channels) - 1)

    @staticmethod
    def _pad_for_unet(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[int, int]]:
        if factor <= 1:
            return x, (x.shape[-2], x.shape[-1])
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h == 0 and pad_w == 0:
            return x, (h, w)
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (h, w)

    @staticmethod
    def _crop_to_hw(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        h, w = hw
        return x[..., :h, :w]

    def _forward_with_control(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_1: torch.Tensor | None,
        control_residuals: dict[str, object] | None,
    ) -> torch.Tensor:
        attention_mask = None
        encoder_attention_mask = None
        encoder_attention_mask_1 = None
        cross_attention_kwargs = None
        timestep_cond = None
        class_labels = None
        return_dict = True

        default_overall_up_factor = 2 ** self.backbone.num_upsamplers
        forward_upsample_size = any(s % default_overall_up_factor != 0 for s in sample.shape[-2:])
        upsample_size = None

        timesteps = timesteps
        if not torch.is_tensor(timesteps):
            dtype = torch.float64 if isinstance(timesteps, float) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.backbone.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.backbone.time_embedding(t_emb, timestep_cond)

        if self.backbone.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

        sample = self.backbone.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.backbone.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_hidden_states_1=encoder_hidden_states_1,
                    encoder_attention_mask_1=encoder_attention_mask_1,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if control_residuals is not None:
            extra_down = control_residuals.get("down_block_residuals", None)
            if extra_down is not None:
                if len(extra_down) != len(down_block_res_samples):
                    raise RuntimeError(
                        f"Control residual count mismatch: got {len(extra_down)} residuals for "
                        f"{len(down_block_res_samples)} U-Net skip states."
                    )
                down_block_res_samples = tuple(
                    base
                    + (
                        residual.to(device=base.device, dtype=base.dtype)
                        if residual.shape[-2:] == base.shape[-2:]
                        else torch.nn.functional.interpolate(
                            residual.to(device=base.device, dtype=base.dtype),
                            size=base.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                    for base, residual in zip(down_block_res_samples, extra_down)
                )

        if self.backbone.mid_block is not None:
            sample = self.backbone.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states_1=encoder_hidden_states_1,
                encoder_attention_mask_1=encoder_attention_mask_1,
            )
        if control_residuals is not None:
            mid = control_residuals.get("mid_block_residual", None)
            if mid is not None:
                mid = mid.to(device=sample.device, dtype=sample.dtype)
                if mid.shape[-2:] != sample.shape[-2:]:
                    mid = torch.nn.functional.interpolate(
                        mid,
                        size=sample.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                sample = sample + mid

        for i, upsample_block in enumerate(self.backbone.up_blocks):
            is_final_block = i == len(self.backbone.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    encoder_hidden_states_1=encoder_hidden_states_1,
                    encoder_attention_mask_1=encoder_attention_mask_1,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        if self.backbone.conv_norm_out:
            sample = self.backbone.conv_norm_out(sample)
            sample = self.backbone.conv_act(sample)
        sample = self.backbone.conv_out(sample)

        if return_dict:
            return sample
        return sample

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_1: torch.Tensor | None = None,
        control_residuals: dict[str, object] | None = None,
        control_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if timesteps is None:
            raise ValueError("timesteps is required")

        spatial_factor = self._get_unet_spatial_factor()
        x_in, orig_hw = self._pad_for_unet(x, factor=spatial_factor)
        x_in = x_in.to(device=self.device, dtype=self.dtype)
        timesteps = timesteps.to(device=self.device)
        encoder_state_dict = self.prepare_encoder_hidden_states(
            batch_size=x_in.shape[0],
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_1=encoder_hidden_states_1,
            device=self.device,
            dtype=self.dtype,
        )

        down_block_additional_residuals = None
        mid_block_additional_residual = None
        if control_residuals is not None:
            down = control_residuals.get("down_block_residuals", None)
            if down is not None:
                down_block_additional_residuals = tuple(
                    float(control_scale) * residual.to(device=self.device, dtype=self.dtype)
                    for residual in down
                )
            mid = control_residuals.get("mid_block_residual", None)
            if mid is not None:
                mid_block_additional_residual = (
                    float(control_scale) * mid.to(device=self.device, dtype=self.dtype)
                )

        merged_control = None
        if down_block_additional_residuals is not None or mid_block_additional_residual is not None:
            merged_control = {
                "down_block_residuals": down_block_additional_residuals,
                "mid_block_residual": mid_block_additional_residual,
            }
        if merged_control is None:
            out = self.backbone(
                sample=x_in,
                timestep=timesteps,
                encoder_hidden_states=encoder_state_dict["encoder_hidden_states"],
                encoder_hidden_states_1=encoder_state_dict["encoder_hidden_states_1"],
            )
            sample = out.sample if hasattr(out, "sample") else out
        else:
            sample = self._forward_with_control(
                sample=x_in,
                timesteps=timesteps,
                encoder_hidden_states=encoder_state_dict["encoder_hidden_states"],
                encoder_hidden_states_1=encoder_state_dict["encoder_hidden_states_1"],
                control_residuals=merged_control,
            )
        sample = sample.to(dtype=x.dtype)
        return self._crop_to_hw(sample, orig_hw)
