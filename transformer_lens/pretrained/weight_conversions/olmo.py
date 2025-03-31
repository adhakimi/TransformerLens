import logging

import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_olmo_weights(olmo, cfg: HookedTransformerConfig):
    state_dict = {}
    
    logging.debug(f"Official model name found: {olmo}")
    
    state_dict["embed.W_E"] = olmo.model.embed_tokens.weight
   
    for l in range(cfg.n_layers):
        W_Q = olmo.model.layers[l].self_attn.q_proj.weight
        W_K = olmo.model.layers[l].self_attn.k_proj.weight
        W_V = olmo.model.layers[l].self_attn.v_proj.weight

        # in case of quantization,
        # parameters should stay as bitsandbytes.nn.modules.Params4bit
        if not cfg.load_in_4bit:
            W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
            W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_heads)
            W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(
            cfg.n_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(
            cfg.n_heads,
            cfg.d_head,
            dtype=cfg.dtype,
            device=cfg.device,
        )

        W_O = olmo.model.layers[l].self_attn.o_proj.weight

        if not cfg.load_in_4bit:
            W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_O"] = W_O.to(device=cfg.device)

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

        if not cfg.load_in_4bit:
            state_dict[f"blocks.{l}.mlp.W_in"] = olmo.model.layers[l].mlp.up_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_gate"] = olmo.model.layers[l].mlp.gate_proj.weight.T
            state_dict[f"blocks.{l}.mlp.W_out"] = olmo.model.layers[l].mlp.down_proj.weight.T
        else:
            state_dict[f"blocks.{l}.mlp.W_in"] = olmo.model.layers[l].mlp.up_proj.weight
            state_dict[f"blocks.{l}.mlp.W_gate"] = olmo.model.layers[l].mlp.gate_proj.weight
            state_dict[f"blocks.{l}.mlp.W_out"] = olmo.model.layers[l].mlp.down_proj.weight

        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=cfg.device
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=cfg.device
        )

    state_dict["unembed.W_U"] = olmo.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=cfg.device)

    return state_dict