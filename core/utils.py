import torch
def _grad_norm(p: torch.Tensor) -> float:
    if p is None or p.grad is None:
        return 0.0
    # use fp32 for stable logging
    return float(p.grad.detach().float().norm(2).item())

@torch.no_grad()
def _param_norm(p: torch.Tensor) -> float:
    return float(p.detach().float().norm(2).item())

def log_grads(logger, model, rowcol_loss=None, every=100, step=0):
    """
    Log gradient and parameter norms for a few representative tensors.
    Call this right after backward() (and after scaler.unscale_ if using AMP).
    """
    if step % every != 0:
        return

    items = []

    # --- model: classifier head ---
    if hasattr(model, "head") and hasattr(model.head, "weight"):
        items.append(("model.head.weight", _grad_norm(model.head.weight), _param_norm(model.head.weight)))

    # --- model: patch embed conv ---
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        pe_w = model.patch_embed.proj.weight
        items.append(("patch_embed.proj.weight", _grad_norm(pe_w), _param_norm(pe_w)))

    # --- model: one early block + one late block (works for timm ViTs/EVA) ---
    # Adjust indices if your depth differs
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        b0 = model.blocks[0]
        blast = model.blocks[-1]
        # pick a stable tensor to track (qkv or proj)
        if hasattr(b0, "attn") and hasattr(b0.attn, "qkv"):
            items.append(("blocks[0].attn.qkv.weight", _grad_norm(b0.attn.qkv.weight), _param_norm(b0.attn.qkv.weight)))
        if hasattr(blast, "attn") and hasattr(blast.attn, "qkv"):
            items.append(("blocks[-1].attn.qkv.weight", _grad_norm(blast.attn.qkv.weight), _param_norm(blast.attn.qkv.weight)))

    # --- aux head ---
    if rowcol_loss is not None:
        # your MLP is Sequential: Linear(0) -> ReLU(1) -> Linear(2)
        try:
            w0 = rowcol_loss.mlp[0].weight
            w2 = rowcol_loss.mlp[2].weight
            items.append(("rowcol_loss.mlp[0].weight", _grad_norm(w0), _param_norm(w0)))
            items.append(("rowcol_loss.mlp[2].weight", _grad_norm(w2), _param_norm(w2)))
        except Exception as e:
            items.append((f"rowcol_loss.inspect_error:{type(e).__name__}", 0.0, 0.0))

    msg = " | ".join([f"{n}: g={gn:.3e}, p={pn:.3e}" for (n, gn, pn) in items])
    logger.info(f"[gradcheck step={step}] {msg}")
