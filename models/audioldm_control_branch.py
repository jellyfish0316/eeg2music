from __future__ import annotations

import copy

import torch
import torch.nn as nn


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class AudioLDMControlBranch(nn.Module):
    def __init__(
        self,
        *,
        conv_in: nn.Module,
        down_blocks: nn.ModuleList,
        mid_block: nn.Module,
        time_proj: nn.Module,
        time_embedding: nn.Module,
        latent_channels: int,
        latent_hw: tuple[int, int],
        cross_attention_dims: tuple[int, int | None],
        input_block_channels: list[int],
        middle_block_channel: int,
        zero_init: bool = True,
        inject_middle_block: bool = True,
    ) -> None:
        super().__init__()
        self.conv_in = copy.deepcopy(conv_in)
        self.down_blocks = copy.deepcopy(down_blocks)
        self.mid_block = copy.deepcopy(mid_block)
        self.time_proj = copy.deepcopy(time_proj)
        self.time_embedding = copy.deepcopy(time_embedding)
        self.latent_channels = int(latent_channels)
        self.latent_hw = (int(latent_hw[0]), int(latent_hw[1]))
        self.cross_attention_dims = (
            int(cross_attention_dims[0]),
            None if cross_attention_dims[1] is None else int(cross_attention_dims[1]),
        )
        self.inject_middle_block = bool(inject_middle_block)

        conv_builder = zero_module if zero_init else (lambda m: m)
        self.down_zero_convs = nn.ModuleList(
            [conv_builder(nn.Conv2d(int(ch), int(ch), kernel_size=1)) for ch in input_block_channels]
        )
        self.mid_zero_conv = conv_builder(
            nn.Conv2d(int(middle_block_channel), int(middle_block_channel), kernel_size=1)
        )

        inferred_count = len(self._infer_residual_shapes())
        if inferred_count != len(self.down_zero_convs):
            raise RuntimeError(
                "Control branch residual count does not match pretrained U-Net injection sites "
                f"(got {inferred_count}, expected {len(self.down_zero_convs)})."
            )

        ref_param = next(self.conv_in.parameters())
        self.to(device=ref_param.device)
        self.float()

    def _default_encoder_hidden_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        width: int,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            1,
            width,
            device=device,
            dtype=dtype,
        )

    def _compute_temporal_embedding(
        self,
        timesteps: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        timestep_proj = self.time_proj(timesteps)
        time_embed_dtype = next(self.time_embedding.parameters()).dtype
        timestep_proj = timestep_proj.to(dtype=time_embed_dtype)
        temb = self.time_embedding(timestep_proj)
        return temb.to(dtype=dtype)

    def _run_down_block(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_1: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if bool(getattr(block, "has_cross_attention", False)):
            return block(
                hidden_states=hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_1=encoder_hidden_states_1,
            )
        return block(hidden_states=hidden_states, temb=temb)

    def _run_mid_block(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_1: torch.Tensor | None,
    ) -> torch.Tensor:
        if bool(getattr(self.mid_block, "has_cross_attention", False)):
            return self.mid_block(
                hidden_states=hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_1=encoder_hidden_states_1,
            )
        return self.mid_block(hidden_states=hidden_states, temb=temb)

    def _infer_residual_shapes(self) -> list[tuple[int, int, int]]:
        device = next(self.conv_in.parameters()).device
        dtype = next(self.conv_in.parameters()).dtype
        hidden_states = torch.zeros(
            1,
            self.latent_channels,
            self.latent_hw[0],
            self.latent_hw[1],
            device=device,
            dtype=dtype,
        )
        timesteps = torch.zeros(1, device=device, dtype=torch.long)
        encoder_hidden_states = self._default_encoder_hidden_state(
            batch_size=1,
            device=device,
            dtype=dtype,
            width=self.cross_attention_dims[0],
        )
        encoder_hidden_states_1 = None
        if self.cross_attention_dims[1] is not None:
            encoder_hidden_states_1 = self._default_encoder_hidden_state(
                batch_size=1,
                device=device,
                dtype=dtype,
                width=self.cross_attention_dims[1],
            )
        temb = self._compute_temporal_embedding(timesteps, dtype=dtype)

        hidden_states = self.conv_in(hidden_states)
        residuals = [tuple(int(v) for v in hidden_states.shape[1:])]

        for block in self.down_blocks:
            hidden_states, res_samples = self._run_down_block(
                block=block,
                hidden_states=hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_1=encoder_hidden_states_1,
            )
            residuals.extend(tuple(int(v) for v in sample.shape[1:]) for sample in res_samples)

        return residuals

    def forward(
        self,
        zt: torch.Tensor,
        projected_latent: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_1: torch.Tensor | None = None,
    ) -> dict[str, object]:
        branch_dtype = next(self.conv_in.parameters()).dtype
        hidden_states = (zt + projected_latent).to(
            device=next(self.conv_in.parameters()).device,
            dtype=branch_dtype,
        )
        if encoder_hidden_states is None:
            encoder_hidden_states = self._default_encoder_hidden_state(
                batch_size=hidden_states.shape[0],
                device=hidden_states.device,
                dtype=branch_dtype,
                width=self.cross_attention_dims[0],
            )
        else:
            encoder_hidden_states = encoder_hidden_states.to(
                device=hidden_states.device,
                dtype=branch_dtype,
            )
        if self.cross_attention_dims[1] is None:
            encoder_hidden_states_1 = None
        elif encoder_hidden_states_1 is None:
            encoder_hidden_states_1 = self._default_encoder_hidden_state(
                batch_size=hidden_states.shape[0],
                device=hidden_states.device,
                dtype=branch_dtype,
                width=self.cross_attention_dims[1],
            )
        else:
            encoder_hidden_states_1 = encoder_hidden_states_1.to(
                device=hidden_states.device,
                dtype=branch_dtype,
            )

        temb = self._compute_temporal_embedding(
            timesteps.to(device=hidden_states.device),
            dtype=branch_dtype,
        )

        down_block_residuals = []
        hidden_states = self.conv_in(hidden_states)
        down_block_residuals.append(self.down_zero_convs[0](hidden_states))

        residual_index = 1
        for block in self.down_blocks:
            hidden_states, res_samples = self._run_down_block(
                block=block,
                hidden_states=hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_1=encoder_hidden_states_1,
            )
            for sample in res_samples:
                down_block_residuals.append(self.down_zero_convs[residual_index](sample))
                residual_index += 1

        hidden_states = self._run_mid_block(
            hidden_states=hidden_states,
            temb=temb,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_1=encoder_hidden_states_1,
        )
        mid_block_residual = (
            self.mid_zero_conv(hidden_states) if self.inject_middle_block else None
        )
        return {
            "down_block_residuals": tuple(down_block_residuals),
            "mid_block_residual": mid_block_residual,
        }
