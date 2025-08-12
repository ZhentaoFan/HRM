from typing import Optional, Any, Sequence, List, Dict, Tuple
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
# from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


# =========================
# Configs
# =========================

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Model/Arch
    arch: ArchConfig

    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []

    # ====== NEW: T5 warm-start options ======
    # HuggingFace hub id, e.g. "google-t5/t5-base"
    init_from_t5: Optional[str] = None
    # If attention/MLP shapes mismatch, directly use native T5 structure via wrapper
    use_t5_arch_if_mismatch: bool = False
    # Freeze first N encoder/decoder blocks after loading (stabilize re-pretrain)
    freeze_t5_prefix_layers: int = 0
    # Whether to copy Relative Position Bias from T5; if HRM uses RoPE, set False
    t5_use_rpb: bool = True


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int


# =========================
# Dataloader
# =========================

def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


# =========================
# T5 Warm Start Helpers
# =========================

class HRMT5Wrapper(nn.Module):
    """Fallback: Use native T5ForConditionalGeneration but expose HRM-like interface."""
    def __init__(self, t5_model: nn.Module):
        super().__init__()
        self.t5 = t5_model

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        return None

    def forward(self, *, carry=None, batch: Dict[str, torch.Tensor], return_keys: List[str] = None):
        # Expect batch to contain standard keys; you may adapt here for your dataset
        input_ids = batch.get("input_ids", None)
        attention_mask = batch.get("attention_mask", None)
        labels = batch.get("labels", None)

        if input_ids is None:
            # Heuristic fallbacks (adjust to your dataset fields if needed)
            input_ids = batch.get("x") or batch.get("tokens") or batch.get("src_ids")
        if labels is None:
            labels = batch.get("y") or batch.get("tgt_ids")

        out = self.t5(input_ids=input_ids,
                      attention_mask=attention_mask,
                      labels=labels,
                      return_dict=True)

        loss = out.loss if hasattr(out, "loss") and out.loss is not None else torch.tensor(0.0, device=input_ids.device)
        metrics = {
            "loss": loss.detach(),
            "count": torch.tensor(input_ids.size(0), dtype=torch.float32, device=input_ids.device)
        }

        preds = {}
        all_finish = True  # single pass per batch
        return None, loss, metrics, preds, all_finish


def _maybe_import_transformers():
    try:
        from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoConfig  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "You set init_from_t5 but `transformers` is not available. "
            "Please `pip install transformers`."
        ) from e


def load_t5_model(t5_id: str):
    from transformers import T5ForConditionalGeneration
    t5 = T5ForConditionalGeneration.from_pretrained(t5_id)
    return t5

def peek_hrm_core(model: nn.Module, dump_file: str = "hrm_keys.txt"):
    import io, itertools

    def unwrap_core(m: nn.Module) -> nn.Module:
        for attr in ["model", "module", "backbone", "net"]:
            if hasattr(m, attr) and isinstance(getattr(m, attr), nn.Module):
                return unwrap_core(getattr(m, attr))
        return m

    core = unwrap_core(model)
    sd = core.state_dict()
    keys = list(sd.keys())

    # 层级前缀统计（看 encoder/decoder 的真实命名）
    prefixes = set(".".join(k.split(".")[:3]) for k in keys if k.startswith(("encoder", "decoder")))
    # 常见关键词分布
    hits = {tag: [k for k in keys if tag in k.lower()] for tag in ["embed", "token", "puzzle_emb", "q_proj", "k_proj", "v_proj", "o_proj", "qkv", "mlp", "ff", "layer_norm", "rpb", "relative_attention_bias"]}

    print("\n[PEEK] core type =", core.__class__.__name__)
    print("[PEEK] total params =", len(keys))
    print("[PEEK] first 40 keys:", keys[:40])
    print("[PEEK] top-level encoder/decoder prefixes:", sorted(prefixes)[:20])
    for tag, lst in hits.items():
        print(f"[PEEK] {tag}: {len(lst)} hits")

    # 写到文件里，方便你发我看
    with open(dump_file, "w") as f:
        f.write("\n".join(keys))

    # 打几个关键张量的形状（有就打）
    for probe in [
        "embed_tokens.weight", "tokens_embedding.weight", "input_embed.weight", "puzzle_emb.weight",
        "encoder.layers.0.self_attn.q_proj.weight", "encoder.layers.0.self_attn.k_proj.weight",
        "encoder.layers.0.self_attn.v_proj.weight", "encoder.layers.0.self_attn.o_proj.weight",
        "encoder.layers.0.attn.qkv.weight",
        "encoder.blocks.0.self_attn.q_proj.weight", "encoder.blocks.0.self_attn.qkv.weight",
        "encoder.layers.0.mlp.fc1.weight", "encoder.layers.0.mlp.fc2.weight",
        "encoder.layers.0.ffn.wi.weight", "encoder.layers.0.ffn.wo.weight",
        "encoder.layers.0.final_layer_norm.weight",
        "decoder.layers.0.self_attn.q_proj.weight",
    ]:
        if probe in sd:
            print("[PEEK] ", probe, sd[probe].shape)


@torch.no_grad()
def load_t5_into_hrm(hrm_model: nn.Module, t5_model: nn.Module, use_rpb: bool = False):
    """
    Tailored for HierarchicalReasoningModel_ACTV1 key patterns:
      - inner.embed_tokens.embedding_weight
      - inner.(H_level|L_level).layers.{i}.self_attn.{qkv_proj|o_proj}.weight
      - inner.(H_level|L_level).layers.{i}.mlp.{gate_up_proj|down_proj}.weight
      - inner.lm_head.weight (optional)
    """
    import re
    t5_sd = t5_model.state_dict()

    # ---- unwrap to .model.inner ----
    core = hrm_model
    for attr in ["model", "module", "backbone", "net"]:
        if hasattr(core, attr) and isinstance(getattr(core, attr), nn.Module):
            core = getattr(core, attr)
    if hasattr(core, "inner") and isinstance(core.inner, nn.Module):
        core = core.inner

    hrm_sd = core.state_dict()
    loaded, skipped, mismatched = [], [], []

    def cp(dst, src, note=None):
        if src not in t5_sd or dst not in hrm_sd:
            skipped.append((dst, src))
            return
        if hrm_sd[dst].shape == t5_sd[src].shape:
            hrm_sd[dst].copy_(t5_sd[src]); loaded.append((dst, src if note is None else f"{src}({note})"))
        else:
            mismatched.append((dst, tuple(hrm_sd[dst].shape), src, tuple(t5_sd[src].shape)))

    # ---- 0) embed & lm_head ----
    # embed
    if "shared.weight" in t5_sd and "embed_tokens.embedding_weight" in hrm_sd:
        if hrm_sd["embed_tokens.embedding_weight"].shape == t5_sd["shared.weight"].shape:
            hrm_sd["embed_tokens.embedding_weight"].copy_(t5_sd["shared.weight"])
            loaded.append(("embed_tokens.embedding_weight", "shared.weight"))
        else:
            mismatched.append(("embed_tokens.embedding_weight", tuple(hrm_sd["embed_tokens.embedding_weight"].shape),
                               "shared.weight", tuple(t5_sd["shared.weight"].shape)))
    # lm_head（可选）
    if "lm_head.weight" in hrm_sd:
        src = "lm_head.weight" if "lm_head.weight" in t5_sd else "shared.weight"
        if hrm_sd["lm_head.weight"].shape == t5_sd[src].shape:
            hrm_sd["lm_head.weight"].copy_(t5_sd[src]); loaded.append(("lm_head.weight", src))
        else:
            skipped.append(("lm_head.weight", src))

    # ---- 1) 找到 H/L 两个 level 和层数 ----
    def find_layers(level_name: str):
        pat = re.compile(rf"^{level_name}\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$")
        idxs = []
        for k in hrm_sd.keys():
            m = pat.match(k)
            if m: idxs.append(int(m.group(1)))
        return sorted(set(idxs))

    levels = []
    for lv in ["H_level", "L_level"]:
        if any(k.startswith(f"{lv}.layers.") for k in hrm_sd.keys()):
            levels.append(lv)

    # ---- 2) per-layer: attn qkv/o, mlp gate_up/down ----
    # T5 keys
    def t5_qkv_keys(i, side):  # side: "encoder"/"decoder"
        pref = f"{side}.block.{i}.layer.0.SelfAttention"
        return f"{pref}.q.weight", f"{pref}.k.weight", f"{pref}.v.weight", f"{pref}.o.weight"

    def t5_ffn_keys(i, side):
        """
        Return (wi, wo, wi_0, wi_1) keys for T5 FFN.
        Encoder FFN is at layer.1; Decoder FFN is at layer.2 (layer.1 is cross-attn).
        """
        ffn_layer_idx = 1 if side == "encoder" else 2
        pref = f"{side}.block.{i}.layer.{ffn_layer_idx}"
        wi = f"{pref}.DenseReluDense.wi.weight"
        wo = f"{pref}.DenseReluDense.wo.weight"
        wi_0 = f"{pref}.DenseReluDense.wi_0.weight"
        wi_1 = f"{pref}.DenseReluDense.wi_1.weight"
        return wi, wo, wi_0, wi_1


    # 猜测 side：H_level 对应 encoder，L_level 对应 decoder（不重要，只是取对应索引）
    side_map = {"H_level": "decoder", "L_level": "encoder"}

    for lv in levels:
        side = side_map.get(lv, "encoder")
        layer_ids = find_layers(lv)
        for i in layer_ids:
            # ---- self-attn qkv ----
            dst_qkv = f"{lv}.layers.{i}.self_attn.qkv_proj.weight"
            dst_o   = f"{lv}.layers.{i}.self_attn.o_proj.weight"
            if dst_qkv in hrm_sd:
                tq, tk, tv, to = t5_qkv_keys(i, side)
                if all(k in t5_sd for k in (tq, tk, tv)):
                    Wq, Wk, Wv = t5_sd[tq], t5_sd[tk], t5_sd[tv]
                    Wqkv = None
                    # 判定拼接维：Linear.weight 是 [out, in]
                    h_out, h_in = hrm_sd[dst_qkv].shape
                    q_out, q_in = Wq.shape
                    # 优先沿 out 维拼接
                    if 3 * q_out == h_out and q_in == h_in:
                        Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
                    # 退而沿 in 维拼接
                    elif q_out == h_out and 3 * q_in == h_in:
                        Wqkv = torch.cat([Wq, Wk, Wv], dim=1)
                    if Wqkv is not None and Wqkv.shape == hrm_sd[dst_qkv].shape:
                        hrm_sd[dst_qkv].copy_(Wqkv); loaded.append((dst_qkv, f"{tq}|{tk}|{tv}"))
                    else:
                        mismatched.append((dst_qkv, tuple(hrm_sd[dst_qkv].shape),
                                           "q|k|v", (tuple(Wq.shape), tuple(Wk.shape), tuple(Wv.shape))))
                else:
                    skipped.append((dst_qkv, "t5 q/k/v"))
            if dst_o in hrm_sd:
                _, _, _, to = t5_qkv_keys(i, side)
                cp(dst_o, to)

            # ---- MLP ----
            dst_gate_up = f"{lv}.layers.{i}.mlp.gate_up_proj.weight"
            dst_down    = f"{lv}.layers.{i}.mlp.down_proj.weight"
            wi, wo, wi0, wi1 = t5_ffn_keys(i, side)

            # down_proj ~ wo
            if dst_down in hrm_sd and wo in t5_sd:
                if hrm_sd[dst_down].shape == t5_sd[wo].shape:
                    hrm_sd[dst_down].copy_(t5_sd[wo]); loaded.append((dst_down, wo))
                else:
                    mismatched.append((dst_down, tuple(hrm_sd[dst_down].shape), wo, tuple(t5_sd[wo].shape)))
            else:
                skipped.append((dst_down, wo))

            # gate_up_proj ~ wi / (wi_0, wi_1)
            if dst_gate_up in hrm_sd:
                H_out, H_in = hrm_sd[dst_gate_up].shape
                if wi0 in t5_sd and wi1 in t5_sd:
                    W = torch.cat([t5_sd[wi0], t5_sd[wi1]], dim=0)  # [2*d_ff, d_model]
                    if W.shape == hrm_sd[dst_gate_up].shape:
                        hrm_sd[dst_gate_up].copy_(W); loaded.append((dst_gate_up, f"{wi0}|{wi1}"))
                    else:
                        mismatched.append((dst_gate_up, tuple(hrm_sd[dst_gate_up].shape),
                                           "wi_0|wi_1", (tuple(t5_sd[wi0].shape), tuple(t5_sd[wi1].shape))))
                elif wi in t5_sd:
                    W = t5_sd[wi]
                    if W.shape == hrm_sd[dst_gate_up].shape:
                        hrm_sd[dst_gate_up].copy_(W); loaded.append((dst_gate_up, wi))
                    elif (2 * W.shape[0] == H_out) and (W.shape[1] == H_in):
                        W2 = torch.cat([W, W], dim=0)
                        hrm_sd[dst_gate_up].copy_(W2); loaded.append((dst_gate_up, f"{wi}x2"))
                    else:
                        mismatched.append((dst_gate_up, tuple(hrm_sd[dst_gate_up].shape), wi, tuple(W.shape)))
                else:
                    skipped.append((dst_gate_up, "wi/wi_0,wi_1"))

    # ---- 写回 ----
    core.load_state_dict(hrm_sd, strict=False)
    return {"loaded": loaded, "skipped": skipped, "mismatched": mismatched}


def freeze_t5_prefix_layers_inplace(model: nn.Module, n_prefix: int):
    if n_prefix <= 0:
        return
    enc = getattr(model, "t5", None)
    if enc is None:
        return
    # encoder
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "block"):
        for i, blk in enumerate(enc.encoder.block):
            if i < n_prefix:
                for p in blk.parameters():
                    p.requires_grad = False
    # decoder
    if hasattr(enc, "decoder") and hasattr(enc.decoder, "block"):
        for i, blk in enumerate(enc.decoder.block):
            if i < n_prefix:
                for p in blk.parameters():
                    p.requires_grad = False


# =========================
# Model & Optimizers
# =========================

def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    """
    Build HRM(+loss) as before, but if init_from_t5 is set:
      - Try to load T5 weights into HRM before torch.compile
      - If mismatch and use_t5_arch_if_mismatch=True, replace with HRMT5Wrapper(T5)
    """
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=32128, #train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    # Build HRM w/ loss head (uncompiled yet; we want to load weights first)
    with torch.device("cuda"):
        hrm_core: nn.Module = model_cls(model_cfg)
        model: nn.Module = loss_head_cls(hrm_core, **config.arch.loss.__pydantic_extra__)  # type: ignore

    used_wrapper = False

    # ---- T5 warm-start ----
    if config.init_from_t5:
        _maybe_import_transformers()
        t5 = load_t5_model(config.init_from_t5)

        # Try to load into HRM (embedding + optional RPB + layers if shapes align)
        try:
            report = load_t5_into_hrm(model, t5, use_rpb=config.t5_use_rpb)
            loaded_n = len(report.get("loaded", []))
            mism_n   = len(report.get("mismatched", []))
            skip_n   = len(report.get("skipped", []))

            def rank0_print(msg: str):
                if world_size <= 1 or dist.get_rank() == 0:
                    print(msg)

            rank0_print(f"[T5→HRM] load summary: loaded={loaded_n}, mismatched={mism_n}, skipped={skip_n}")

            # 可选：展示前 5 条成功映射，便于确认参数名是否对齐
            for i, (dst, src) in enumerate(report.get("loaded", [])[:]):
                rank0_print(f"  [loaded {i+1}] {src} -> {dst}")

            if report["mismatched"]:
                if config.use_t5_arch_if_mismatch:
                    # Fall back to wrapper
                    print(f"[WARN] T5-HRM mismatch detected (showing up to 3): {report['mismatched'][:3]} ... "
                          f"Switching to native T5 wrapper.")
                    with torch.device("cuda"):
                        model = HRMT5Wrapper(t5)
                    used_wrapper = True
                else:
                    print(f"[WARN] Partial T5 loading done with mismatches (kept HRM). "
                          f"First mismatch: {report['mismatched'][0]}")
        except Exception as e:
            if config.use_t5_arch_if_mismatch:
                print(f"[WARN] load_t5_into_hrm failed: {e}. Falling back to T5 wrapper.")
                with torch.device("cuda"):
                    model = HRMT5Wrapper(t5)
                used_wrapper = True
            else:
                raise

        # Optional: freeze prefix layers (only effective for wrapper/native T5)
        if used_wrapper and config.freeze_t5_prefix_layers > 0:
            freeze_t5_prefix_layers_inplace(model, config.freeze_t5_prefix_layers)

    # ---- torch.compile (skip for wrapper to keep things simple) ----
    if (not used_wrapper) and "DISABLE_COMPILE" not in os.environ:
        with torch.device("cuda"):
            model = torch.compile(model, dynamic=False)  # type: ignore

    # ---- Broadcast parameters from rank 0 ----
    if world_size > 1:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)

    # ---- Optimizers & LR ----
    if used_wrapper:
        # No puzzle_emb in wrapper; use a single AdamW
        optimizers = [
            torch.optim.AdamW(
                model.parameters(),
                lr=0,  # to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2),
            )
        ]
        optimizer_lrs = [config.lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                # NOTE: adjust attr path if different in your codebase
                hrm_core.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            # AdamATan2(
            #     model.parameters(),
            #     lr=0,
            #     weight_decay=config.weight_decay,
            #     betas=(config.beta1, config.beta2)
            # )
            torch.optim.AdamW(
                model.parameters(),
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


# =========================
# LR schedule & Train state
# =========================

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


# =========================
# Train / Eval
# =========================

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    # To device
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    # Apply optimizer
    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))  # same order across ranks
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            count = max(reduced_metrics.get("count", 1), 1)  # avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
                               for k, v in reduced_metrics.items()}
            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        all_preds: Dict[str, List[torch.Tensor]] = {}
        metric_keys: List[str] = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]

        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward (loop until all_finish=True)
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=config.eval_save_outputs)
                if all_finish:
                    break

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs and torch.is_tensor(v):
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())

            del carry, preds, batch, all_finish

            # Aggregate
            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics_np = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {metric_name: reduced_metrics_np[set_id, metric_id]
                               for metric_id, metric_name in enumerate(metric_keys)}
                    for set_id, set_name in enumerate(set_ids)
                }
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count", 1.0)
                    reduced_metrics[set_name] = {k: v / max(count, 1.0) for k, v in m.items()}
                return reduced_metrics


# =========================
# Logging
# =========================

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)
            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code to W&B
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


# =========================
# Main
# =========================

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter
    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(
        config, "train",
        test_set_mode=False, epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE
    )
    eval_loader,  eval_metadata  = create_dataloader(
        config, "test",
        test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE
    )

    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(),
                   settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Training Loop
    for _iter_id in range(total_iters):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        # Train
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

        # Eval
        train_state.model.eval()
        metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)

        # Checkpoint
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()