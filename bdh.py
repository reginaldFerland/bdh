import dataclasses
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    tokenizer_type: str = "byte"
    tokenizer_vocab_size: int = 256
    tokenizer_path: Optional[str] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    unk_token_id: Optional[int] = None


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig, tokenizer: Optional[object] = None):
        super().__init__()
        assert config.vocab_size is not None

        if tokenizer is not None:
            vocab_size = getattr(tokenizer, "vocab_size", config.vocab_size)
            config.vocab_size = int(vocab_size)
            config.tokenizer_vocab_size = int(vocab_size)
            config.tokenizer_type = getattr(tokenizer, "tokenizer_type", config.tokenizer_type)
            config.bos_token_id = getattr(tokenizer, "bos_token_id", config.bos_token_id)
            config.eos_token_id = getattr(tokenizer, "eos_token_id", config.eos_token_id)
            config.pad_token_id = getattr(tokenizer, "pad_token_id", config.pad_token_id)
            config.unk_token_id = getattr(tokenizer, "unk_token_id", config.unk_token_id)

        self.config = config
        self.tokenizer = tokenizer

        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config

        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)  # B, 1, T, D

        for _ in range(C.n_layer):
            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        logits = x.view(B, T, D) @ self.lm_head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> str:
        if self.tokenizer is None or not hasattr(self.tokenizer, "encode"):
            raise ValueError("A tokenizer is required to generate text prompts.")

        self.eval()
        device = next(self.parameters()).device
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tensor = torch.tensor(
            prompt_tokens,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        generated = self.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        full_tokens = generated[0].tolist()
        continuation_tokens = full_tokens[len(prompt_tokens) :]
        continuation = self.tokenizer.decode(continuation_tokens)
        return prompt + continuation

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        tokenizer: Optional[object] = None,
    ) -> Path:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        tokenizer_to_save = tokenizer or self.tokenizer
        if tokenizer_to_save is not None and hasattr(tokenizer_to_save, "save"):
            tokenizer_dir = save_path / "tokenizer"
            tokenizer_to_save.save(tokenizer_dir)
            self.config.tokenizer_path = str(tokenizer_dir)

        config_path = save_path / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, sort_keys=True)

        model_path = save_path / "model.pt"
        torch.save(self.state_dict(), model_path)
        return save_path

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        device: Optional[torch.device] = None,
    ) -> "BDH":
        directory = Path(directory)
        config_path = directory / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config at {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = BDHConfig(**config_data)

        tokenizer = None
        tokenizer_dir = directory / "tokenizer"
        if tokenizer_dir.exists():
            from tokenizer_utils import TokenizerManager

            tokenizer = TokenizerManager.from_directory(tokenizer_dir)

        model = cls(config=config, tokenizer=tokenizer)
        state_path = directory / "model.pt"
        map_location = device if device is not None else "cpu"
        state_dict = torch.load(state_path, map_location=map_location)
        model.load_state_dict(state_dict)
        if device is not None:
            model.to(device)
        return model
