# =================================================================================
# DINOv3 segmentation training (TPU v5e-8, Kaggle-friendly)
# =================================================================================
import gc
import glob
import logging
import math
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

SCRIPT_REV = "tpu-seg-20260127-1"

_IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"))


def _download_with_retries(url, dst, retries=3, timeout=30):
    tmp_path = dst + ".part"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/zip",
    }

    def _cleanup_tmp():
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    def _finalize():
        if not zipfile.is_zipfile(tmp_path):
            raise RuntimeError("Downloaded file is not a zip.")
        os.replace(tmp_path, dst)

    def _try_curl():
        curl = shutil.which("curl")
        if not curl:
            print("download: curl not found", flush=True)
            return False
        print("download: trying curl", flush=True)
        cmd = [
            curl,
            "-L",
            "--retry",
            "3",
            "--retry-all-errors",
            "--max-redirs",
            "20",
            "--connect-timeout",
            str(timeout),
            "-H",
            f"User-Agent: {headers['User-Agent']}",
            "-H",
            f"Accept: {headers['Accept']}",
            "-o",
            tmp_path,
            url,
        ]
        subprocess.check_call(cmd)
        _finalize()
        print("download: curl ok", flush=True)
        return True

    def _try_requests():
        try:
            import requests  # type: ignore
        except Exception:
            print("download: requests not available", flush=True)
            return False
        print("download: trying requests", flush=True)
        resp = requests.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        _finalize()
        print("download: requests ok", flush=True)
        return True

    def _try_urllib():
        print("download: trying urllib", flush=True)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(tmp_path, mode="wb") as f:
                shutil.copyfileobj(resp, f)
        _finalize()
        print("download: urllib ok", flush=True)
        return True

    methods = [_try_curl, _try_requests, _try_urllib]
    last_exc = None
    for attempt in range(1, retries + 1):
        for method in methods:
            try:
                ok = method()
                if ok:
                    return
            except Exception as exc:
                last_exc = exc
                print(f"download: method failed ({exc})", flush=True)
                _cleanup_tmp()
        if attempt < retries:
            time.sleep(2 * attempt)
    raise RuntimeError(f"Downloaded file is not a zip. Last error: {last_exc}")


def _preflight_kaggle_env():
    if not os.path.exists("/kaggle/working"):
        return
    if os.environ.get("TPU_UNINSTALL_TIMM_DONE") != "1":
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "timm"])
            os.environ["TPU_UNINSTALL_TIMM_DONE"] = "1"
        except Exception as exc:
            print(f"WARNING: timm uninstall failed ({exc}); continuing.")
    if os.environ.get("TPU_FIX_TF_DONE") != "1":
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflow-cpu"])
            os.environ["TPU_FIX_TF_DONE"] = "1"
        except Exception as exc:
            print(f"WARNING: tensorflow-cpu setup failed ({exc}); continuing.")

    def _find_repo_root():
        if os.path.isdir("/kaggle/working/pos"):
            return "/kaggle/working/pos"
        for name in os.listdir("/kaggle/working"):
            cand = os.path.join("/kaggle/working", name)
            if os.path.isdir(os.path.join(cand, "core")) and os.path.isdir(os.path.join(cand, "data")):
                return cand
        return None

    url = "https://github.com/lc126eml/pos/archive/refs/heads/master.zip"
    zip_path = "/kaggle/working/pos.zip"
    try:
        if os.path.exists(zip_path) and not zipfile.is_zipfile(zip_path):
            os.remove(zip_path)
        if not os.path.exists(zip_path):
            _download_with_retries(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall("/kaggle/working")
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to download pos repo: {exc}") from exc

    repo_root = _find_repo_root()
    if not repo_root:
        raise RuntimeError("POS repo not found after unzip; expected /kaggle/working/pos or a repo with core/ and data/.")
    os.environ["POS_REPO_ROOT"] = repo_root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    print(f"POS repo ready: {repo_root}", flush=True)


def _tpu_worker(index):
    os.environ["RUN_TPU_WORKER"] = "1"
    return main()


def _spawn_tpu(main_fn):
    print(f"SCRIPT_REV={SCRIPT_REV}", flush=True)
    if os.environ.get("RUN_TPU_WORKER") == "1":
        main_fn()
        return
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if os.path.exists("/kaggle/working"):
        os.environ.pop("TPU_PROCESS_ADDRESSES", None)
        os.environ.pop("TPU_PROCESS_COUNT", None)
        print("Cleared TPU_PROCESS_ADDRESSES/TPU_PROCESS_COUNT for PJRT auto-config.", flush=True)
        print(f"TPU_PROCESS_ADDRESSES={os.environ.get('TPU_PROCESS_ADDRESSES')}", flush=True)
        print(f"TPU_PROCESS_COUNT={os.environ.get('TPU_PROCESS_COUNT')}", flush=True)
        print(f"TPU_RUNTIME_METRICS_PORTS={os.environ.get('TPU_RUNTIME_METRICS_PORTS')}", flush=True)
    if os.environ.get("TPU_ENV_DUMPED") != "1":
        tpu_env = {k: v for k, v in os.environ.items() if k.startswith("TPU") or k.startswith("XLA")}
        print("TPU_ENV:", tpu_env, flush=True)
        os.environ["TPU_ENV_DUMPED"] = "1"
    try:
        import torch_xla.distributed.xla_multiprocessing as _xmp
    except Exception:
        _xmp = None
    if _xmp is None:
        main_fn()
        return
    try:
        _xmp.spawn(_tpu_worker, nprocs=None)
    except Exception as exc:
        raise RuntimeError(f"TPU spawn failed: {exc}") from exc


def _import_xla():
    global xm, pl, xr, txla, _xla_available
    if _xla_available:
        return
    try:
        import torch_xla.core.xla_model as _xm
        import torch_xla.distributed.parallel_loader as _pl
        import torch_xla.runtime as _xr
        import torch_xla as _txla
    except Exception:
        return
    xm = _xm
    pl = _pl
    xr = _xr
    txla = _txla
    _xla_available = True


def _select_device():
    _import_xla()
    xla_devices = []
    if _xla_available:
        try:
            xla_devices = xm.get_xla_supported_devices()
        except Exception:
            xla_devices = []
    if xla_devices:
        if txla is not None and hasattr(txla, "device"):
            return txla.device(), xla_devices
        return xm.xla_device(), xla_devices
    raise RuntimeError("TPU/XLA is required but not available.")


def main():
    global xm, pl, xr, txla, _xla_available
    xm = None
    pl = None
    xr = None
    txla = None
    _xla_available = False

    os.environ.setdefault("PJRT_DEVICE", "TPU")

    pos_root = os.environ.get("POS_REPO_ROOT")
    if pos_root and pos_root not in sys.path:
        sys.path.insert(0, pos_root)

    LOCAL_TIMM = "/kaggle/input/timm-repos/pytorch-image-models"
    if LOCAL_TIMM not in sys.path:
        sys.path.insert(0, LOCAL_TIMM)

    import timm
    from core.patch_pos import PatchRowColRegressionCriterion
    from seg.seg_aug import TrainSegAug, EvalSegPreprocess, EvalSegPreprocessMSFlip
    from seg.seg_head import PPMliteFCNHead, UPerNetTokenHead, FCNSegHead, LinearSegHead
    from seg.seg_loss import MMSegCrossEntropyLoss

    DEVICE, _xla_devices = _select_device()
    if xr is not None and hasattr(xr, "world_size"):
        WORLD_SIZE = xr.world_size()
        RANK = xr.global_ordinal()
    else:
        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
        RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    IS_MASTER = RANK == 0
    xm.master_print(f"✅ Success! TPU Rank {RANK} initialized. World size: {WORLD_SIZE}", flush=True)

    tpu_mem_warned = False
    tpu_info_checked = False
    tpu_info_available = False
    tpu_info_last_ts = 0.0
    tpu_info_last_vals = None
    tpu_info_peak_mb = None

    def _cpu_peak_mb():
        try:
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except Exception:
            return None
        if rss_kb is None:
            return None
        return float(rss_kb) / 1024.0

    def _cpu_total_mb():
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (ValueError, AttributeError, OSError):
            return None
        if pages is None or page_size is None:
            return None
        return (pages * page_size) / (1024.0 * 1024.0)
    cpu_total_mb = _cpu_total_mb()

    def _tpu_info_mem():
        nonlocal tpu_info_checked, tpu_info_available, tpu_info_last_ts, tpu_info_last_vals, tpu_info_peak_mb
        if RANK != 0:
            return None
        if not tpu_info_checked:
            tpu_info_available = shutil.which("tpu-info") is not None
            tpu_info_checked = True
        if not tpu_info_available:
            return None
        now = time.time()
        if tpu_info_last_vals is not None and (now - tpu_info_last_ts) < 10.0:
            return tpu_info_last_vals
        try:
            proc = subprocess.run(
                ["tpu-info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
        except Exception:
            return tpu_info_last_vals
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        pattern = re.compile(r"([0-9.]+)\s*([KMGTP]i?B)\s*/\s*([0-9.]+)\s*([KMGTP]i?B)")
        matches = pattern.findall(output)
        if not matches:
            return tpu_info_last_vals
        def _to_mb(val, unit):
            val = float(val)
            unit = unit.upper()
            if unit in ("KIB", "KB"):
                return val / 1024.0
            if unit in ("MIB", "MB"):
                return val
            if unit in ("GIB", "GB"):
                return val * 1024.0
            if unit in ("TIB", "TB"):
                return val * 1024.0 * 1024.0
            return val
        used_mbs = []
        total_mbs = []
        for used_val, used_unit, total_val, total_unit in matches:
            used_mbs.append(_to_mb(used_val, used_unit))
            total_mbs.append(_to_mb(total_val, total_unit))
        if not used_mbs or not total_mbs:
            return tpu_info_last_vals
        used_mb = max(used_mbs)
        total_mb = max(total_mbs)
        if tpu_info_peak_mb is None or used_mb > tpu_info_peak_mb:
            tpu_info_peak_mb = used_mb
        tpu_info_last_vals = (used_mb, total_mb, tpu_info_peak_mb)
        tpu_info_last_ts = now
        return tpu_info_last_vals

    use_amp = True
    use_bf16 = True
    autocast_dtype = torch.bfloat16

    def _autocast():
        if use_amp:
            return torch.autocast("xla", dtype=autocast_dtype)
        return nullcontext()

    def _xla_sync():
        if txla is not None and hasattr(txla, "sync"):
            txla.sync()
        elif xm is not None:
            xm.mark_step()

    # =============================================================================
    # Configuration
    # =============================================================================
    root_dir = "/kaggle/working" if _IS_KAGGLE else os.getcwd()
    base_path_default = "/kaggle/input/ade20k-dataset/ADEChallengeData2016"
    args = SimpleNamespace(
        model_type="dinov3",
        use_abs_pos_emb=False,
        use_rot_pos_emb=False,
        model_size='base',
        num_classes=150,
        batch_size=16,
        val_drop_last=False,
        val_pad_to_full_batch=False,
        train_img_size=336,
        eval_img_size=368,
        use_ms_flip_eval=False,
        scale_jitter=(1.0, 1.3),
        use_cat_max_ratio=True,
        cat_max_ratio=0.70,
        cat_max_ratio_tries=10,
        ms_scales=(0.90, 1.0, 1.15),
        eval_crop_mode="crop_or_pad",
        final_ms_flip_eval=True,
        lr=5e-05,
        lr_aux=1e-5,
        eta_min=1e-7,
        composite_lr=True,
        warmup_steps=500,
        weight_decay=0.01,
        epochs=130,
        overlap=0,
        start_epoch=0,
        seed=50,
        use_rc_loss=False,
        warmup_steps_for_aux=1,
        alpha_min=10,
        huber_beta=0.1,
        rc_alpha=70.0,
        seg_head="upernet",
        feature_layers=[2, 5, 8, 11],
        workers=2 if _IS_KAGGLE else 5,
        tpu_workers=0,
        tpu_threads=1,
        color_jitter={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05},
        color_jitter_prob=0.1,
        train=True,
        val=True,
        ckpt_path=None,
        lock=False if _IS_KAGGLE else True,
        clip_value=1.0,
        output_dir=root_dir,
        log_interval=150,
        csv_interval=3,
        show_peak_gpu_mem=True,
        compile_model=False,
        save_full_ckpt=False,
        save_best_ckpt=False,
        resume_full_ckpt=False,
        resume_ckpt_path=None,
        resume_scheduler=True,
        resume_optimizer=True,
        resume_bs=True,
        total_run_time_hr=9.0,
        base_path=base_path_default,
        pos_type=None,
        log_all_ranks=False,
        debug_xla=False,
    )

    base_global_batch = 16
    global_batch = args.batch_size * WORLD_SIZE
    lr_scale = global_batch / base_global_batch
    args.lr *= lr_scale
    args.lr_aux *= lr_scale

    tpu_workers = int(getattr(args, "tpu_workers", 0) or 0)
    args.workers = min(args.workers, tpu_workers)
    tpu_threads = int(getattr(args, "tpu_threads", 1) or 1)
    if tpu_threads < 1:
        tpu_threads = 1
    torch.set_num_threads(tpu_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(tpu_threads)

    seed = args.seed + RANK
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    def _seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    data_rng = torch.Generator()
    data_rng.manual_seed(seed)
    run_tag = time.strftime("%Y%m%d_%H%M%S")

    if args.use_abs_pos_emb or args.use_rot_pos_emb:
        args.overlap = 0
        args.use_rc_loss = False

    MODEL_NAME = f"vit_{args.model_size}_patch16_{args.model_type}"
    TRAIN_IMAGE_PATH = os.path.join(args.base_path, "images", "training")
    TRAIN_ANNOTATION_PATH = os.path.join(args.base_path, "annotations", "training")
    VALID_IMAGE_PATH = os.path.join(args.base_path, "images", "validation")
    VALID_ANNOTATION_PATH = os.path.join(args.base_path, "annotations", "validation")

    subdir_name = (
        f"{args.model_size}"
        f"{'_abs_pos' if args.use_abs_pos_emb else ''}"
        f"{'_rot_pos' if args.use_rot_pos_emb else ''}_rc_{args.use_rc_loss}_lr{int(args.lr/1e-5)}_s{args.seed}"
    )
    if args.use_rc_loss:
        subdir_name += f"_overlap_{args.overlap}_alpha_{int(args.rc_alpha)}"

    output_dir = args.output_dir
    ckpt_output_dir = os.path.join(args.output_dir, "ckpt")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_output_dir, exist_ok=True)
    last_ckpt_path = os.path.join(ckpt_output_dir, "last.pth")
    if args.resume_full_ckpt and args.resume_ckpt_path is None:
        args.resume_ckpt_path = last_ckpt_path

    log_file_path = os.path.join(output_dir, f"{subdir_name}.log")
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if IS_MASTER:
        log_handlers.insert(0, logging.FileHandler(log_file_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=log_handlers,
        force=True,
    )
    logger = logging.getLogger()
    if not IS_MASTER and not getattr(args, "log_all_ranks", False):
        logger.setLevel(logging.WARNING)

    def _master_print(msg):
        try:
            xm.master_print(msg, flush=True)
        except Exception:
            if IS_MASTER:
                print(msg, flush=True)

    if IS_MASTER and getattr(args, "debug_xla", False):
        _master_print("DEBUG: XLA debug logging enabled")

    logger.info("Using device: %s (xla=True)", DEVICE)
    logger.info("Using mixed precision: bf16=%s", use_bf16)
    logger.info("Arguments: %s", args)
    logger.info("Global batch: %s (lr_scale=%.3f)", global_batch, lr_scale)
    logger.info("Output dir: %s", output_dir)
    logger.info("Subdir name: %s", subdir_name)

    if not os.path.isdir(TRAIN_IMAGE_PATH):
        logger.error("Missing training images at %s", TRAIN_IMAGE_PATH)
        if _IS_KAGGLE and os.path.isdir("/kaggle/input"):
            logger.error("Available /kaggle/input entries: %s", os.listdir("/kaggle/input"))
        raise FileNotFoundError(f"Training images not found: {TRAIN_IMAGE_PATH}")

    # =============================================================================
    # Dataset and dataloaders
    # =============================================================================
    ImageLike = Union[Image.Image, np.ndarray, torch.Tensor]
    MaskLike = Union[Image.Image, np.ndarray, torch.Tensor]

    class SegmentationDataset(Dataset):
        def __init__(self, image_dir, annotation_dir, pair_transform):
            self.image_dir = image_dir
            self.annotation_dir = annotation_dir
            self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
            self.pair_transform = pair_transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name = self.images[idx]
            img_path = os.path.join(self.image_dir, img_name)
            ann_path = os.path.join(self.annotation_dir, img_name.replace(".jpg", ".png"))

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(ann_path).convert("L")

            out = self.pair_transform(image, mask)
            if isinstance(out, tuple) and len(out) == 3:
                image_t, mask_t, _ = out
            else:
                image_t, mask_t = out
            mask_t = mask_t.long() - 1
            return image_t, mask_t

    train_dataset = SegmentationDataset(
        TRAIN_IMAGE_PATH,
        TRAIN_ANNOTATION_PATH,
        pair_transform=TrainSegAug(
            target_size=(args.train_img_size, args.train_img_size),
            scale_jitter=args.scale_jitter,
            cat_max_ratio=(args.cat_max_ratio if args.use_cat_max_ratio else None),
            cat_max_ratio_tries=args.cat_max_ratio_tries,
            ignore_index=0 if args.use_cat_max_ratio else None,
            color_jitter=args.color_jitter,
            color_jitter_prob=args.color_jitter_prob,
            normalize=True,
        ),
    )
    valid_dataset = SegmentationDataset(
        VALID_IMAGE_PATH,
        VALID_ANNOTATION_PATH,
        pair_transform=EvalSegPreprocess(
            target_size=(args.eval_img_size, args.eval_img_size),
            target_by="shorter",
            eval_crop_mode=args.eval_crop_mode,
            normalize=True,
        ),
    )

    if getattr(args, "val_drop_last", False) and getattr(args, "val_pad_to_full_batch", False):
        if IS_MASTER:
            logger.warning("val_drop_last and val_pad_to_full_batch are both True; disabling val_pad_to_full_batch.")
        args.val_pad_to_full_batch = False

    pin_memory = False
    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=data_rng,
        persistent_workers=(args.workers > 0),
    )
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=True,
        drop_last=True,
    )
    eval_drop_last = bool(getattr(args, "val_drop_last", False))
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=False,
        drop_last=eval_drop_last,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=valid_sampler,
        drop_last=eval_drop_last,
        **loader_kwargs,
    )

    train_loader = pl.MpDeviceLoader(train_loader, DEVICE)
    valid_loader = pl.MpDeviceLoader(valid_loader, DEVICE)

    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = steps_per_epoch
    logger.info("DataLoaders created: train=%s, val=%s", len(train_dataset), len(valid_dataset))
    logger.info("steps_per_epoch=%s, val_steps=%s", steps_per_epoch, len(valid_loader))

    # =============================================================================
    # Model, head, optimizer
    # =============================================================================
    logger.info("Initializing model: %s for %s classes", MODEL_NAME, args.num_classes)
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        use_abs_pos_emb=args.use_abs_pos_emb,
        use_rot_pos_emb=args.use_rot_pos_emb,
        num_classes=0,
        dynamic_img_size=True,
        img_size=args.train_img_size,
    ).to(DEVICE)

    grid_h, grid_w = model.patch_embed.grid_size
    decoder_type = getattr(args, "seg_head", "ppmlite").lower()
    if decoder_type == "ppmlite":
        decoder = PPMliteFCNHead(
            embed_dim=model.embed_dim,
            num_classes=args.num_classes,
            grid_size=(grid_h, grid_w),
            out_size=(args.train_img_size, args.train_img_size),
            mid_channels=256,
            ppm_bins=(1, 2, 3),
            ppm_channels=64,
            dropout=0.1,
            norm="gn",
        ).to(DEVICE)
    elif decoder_type == "upernet":
        embed_dims = [model.embed_dim] * len(args.feature_layers)
        decoder = UPerNetTokenHead(
            embed_dims=embed_dims,
            num_classes=args.num_classes,
            grid_size=(grid_h, grid_w),
            out_size=(args.train_img_size, args.train_img_size),
            fpn_channels=256,
            ppm_bins=(1, 2, 3, 6),
            dropout=0.1,
            norm="gn",
        ).to(DEVICE)
    elif decoder_type == "fcn":
        decoder = FCNSegHead(
            embed_dim=model.embed_dim,
            num_classes=args.num_classes,
            grid_size=(grid_h, grid_w),
            out_size=(args.train_img_size, args.train_img_size),
            mid_channels=256,
            dropout=0.1,
            norm="gn",
        ).to(DEVICE)
    elif decoder_type == "linear":
        decoder = LinearSegHead(
            embed_dim=model.embed_dim,
            num_classes=args.num_classes,
            grid_size=(grid_h, grid_w),
            out_size=(args.train_img_size, args.train_img_size),
            dropout=0.1,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unsupported seg_head='{decoder_type}'. Use 'ppmlite', 'upernet', 'fcn', or 'linear'.")

    if args.overlap > 0:
        original_patch_size = model.patch_embed.proj.kernel_size[0]
        new_patch_size = original_patch_size + args.overlap
        stride = original_patch_size
        original_grid_size = args.train_img_size // stride
        padding = ((original_grid_size - 1) * stride + new_patch_size - args.train_img_size + 1) // 2
        in_chans = model.patch_embed.proj.in_channels
        embed_dim = model.patch_embed.proj.out_channels
        model.patch_embed.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(new_patch_size, new_patch_size),
            stride=(stride, stride),
            padding=padding,
        ).to(DEVICE)

    if args.compile_model:
        logger.warning("torch.compile is not recommended on TPU; skipping.")

    dynamic = True
    training_parameters = list(model.parameters()) + list(decoder.parameters())
    param_groups = []
    lr_aux = getattr(args, "lr_aux", args.lr)
    if args.use_rc_loss:
        grid_h, grid_w = model.patch_embed.grid_size
        dynamic = False
        rowcol_loss = PatchRowColRegressionCriterion(
            feat_dim=model.embed_dim,
            grid_h=grid_h,
            grid_w=grid_w,
            huber_beta=args.huber_beta,
        ).to(DEVICE)
        training_parameters += list(rowcol_loss.parameters())
        param_groups.append({"params": rowcol_loss.parameters(), "weight_decay": 0.0, "lr": lr_aux})

    decay_params = []
    no_decay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or ("norm" in n.lower()):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    for n, p in decoder.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or ("norm" in n.lower()):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups.append({"params": decay_params, "lr": args.lr, "weight_decay": args.weight_decay})
    param_groups.append({"params": no_decay_params, "lr": args.lr, "weight_decay": 0.0})

    ce_criterion = MMSegCrossEntropyLoss(ignore_index=-1, avg_non_ignore=True)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * optimizer_steps_per_epoch
    if args.composite_lr:
        warmup_steps = min(args.warmup_steps, max(1, total_steps - 1))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-7 / args.lr,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.eta_min
        )
    logger.info("Initialized loss, optimizer, and scheduler.")

    # =============================================================================
    # Helpers
    # =============================================================================
    def _infer_grid_hw(model, inputs):
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            ph, pw = patch_size
        else:
            ph = pw = patch_size
        return (inputs.shape[-2] // ph, inputs.shape[-1] // pw)

    def _round_to_multiple(x: int, m: int) -> int:
        return max(m, int(round(x / m) * m))

    def _strip_prefix_tokens(features, grid_hw, num_prefix_tokens):
        if num_prefix_tokens <= 0:
            return features
        tokens_needed = grid_hw[0] * grid_hw[1]
        stripped = []
        for feat in features:
            if feat.dim() == 3 and feat.shape[1] == tokens_needed + num_prefix_tokens:
                stripped.append(feat[:, num_prefix_tokens:, :])
            else:
                stripped.append(feat)
        return stripped

    def _forward_upernet(model, decoder, inputs, feature_layers):
        if not feature_layers:
            raise ValueError("feature_layers must be set when using seg_head='upernet'.")
        grid_hw = _infer_grid_hw(model, inputs)
        features = model.forward_intermediates(
            inputs,
            indices=feature_layers,
            norm=False,
            intermediates_only=True,
            output_fmt="NLC",
        )
        features = _strip_prefix_tokens(features, grid_hw, model.num_prefix_tokens)
        outputs = decoder(features, grid_sizes=[grid_hw] * len(features), out_size=inputs.shape[-2:])
        return outputs, features, grid_hw

    def _ms_flip_predict(model, decoder, inputs, num_classes, scales, flip, patch_size, *, feature_layers=None):
        if isinstance(patch_size, tuple):
            ph, pw = patch_size
        else:
            ph = pw = patch_size
        b, _, h0, w0 = inputs.shape
        logits_sum = torch.zeros((b, num_classes, h0, w0), device=inputs.device, dtype=inputs.dtype)
        count = 0
        for s in scales:
            hs = _round_to_multiple(int(round(h0 * s)), ph)
            ws = _round_to_multiple(int(round(w0 * s)), pw)
            x_s = F.interpolate(inputs, size=(hs, ws), mode="bilinear", align_corners=False)
            grid_hw = _infer_grid_hw(model, x_s)
            if args.seg_head == "upernet":
                if not feature_layers:
                    raise ValueError("feature_layers must be set when using seg_head='upernet'.")
                feats = model.forward_intermediates(
                    x_s,
                    indices=feature_layers,
                    norm=False,
                    intermediates_only=True,
                    output_fmt="NLC",
                )
                feats = _strip_prefix_tokens(feats, grid_hw, model.num_prefix_tokens)
                logits = decoder(feats, grid_sizes=[grid_hw] * len(feats), out_size=(hs, ws))
            else:
                feats = model.forward_features(x_s)
                logits = decoder(feats[:, model.num_prefix_tokens:, :], grid_size=grid_hw, out_size=(hs, ws))
            logits = F.interpolate(logits, size=(h0, w0), mode="bilinear", align_corners=False)
            logits_sum += logits
            count += 1
            if flip:
                x_f = torch.flip(x_s, dims=[3])
                if args.seg_head == "upernet":
                    if not feature_layers:
                        raise ValueError("feature_layers must be set when using seg_head='upernet'.")
                    feats_f = model.forward_intermediates(
                        x_f,
                        indices=feature_layers,
                        norm=False,
                        intermediates_only=True,
                        output_fmt="NLC",
                    )
                    feats_f = _strip_prefix_tokens(feats_f, grid_hw, model.num_prefix_tokens)
                    logits_f = decoder(feats_f, grid_sizes=[grid_hw] * len(feats_f), out_size=(hs, ws))
                else:
                    feats_f = model.forward_features(x_f)
                    logits_f = decoder(feats_f[:, model.num_prefix_tokens:, :], grid_size=grid_hw, out_size=(hs, ws))
                logits_f = torch.flip(logits_f, dims=[3])
                logits_f = F.interpolate(logits_f, size=(h0, w0), mode="bilinear", align_corners=False)
                logits_sum += logits_f
                count += 1
        return logits_sum / max(count, 1)

    @torch.no_grad()
    def fast_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = -1):
        pred = pred.view(-1).to(torch.int64)
        target = target.view(-1).to(torch.int64)
        valid = target != ignore_index
        pred = pred[valid]
        target = target[valid]
        idx = target * num_classes + pred
        conf = torch.bincount(idx, minlength=num_classes * num_classes)
        return conf.view(num_classes, num_classes)

    # =============================================================================
    # Train / validation
    # =============================================================================
    train_start_time = time.time()
    ckpt = None
    if args.resume_full_ckpt and args.resume_ckpt_path:
        if not os.path.exists(args.resume_ckpt_path):
            resume_dir = os.path.dirname(args.resume_ckpt_path)
            parts = os.path.normpath(resume_dir).split(os.sep)
            if os.path.isabs(resume_dir):
                prefix_parts = parts[1:4]
                search_root = os.path.join(os.sep, *prefix_parts)
            else:
                prefix_parts = parts[:3]
                search_root = os.path.join(*prefix_parts)
            candidates = sorted(glob.glob(os.path.join(search_root, "**", "last.pth"), recursive=True))
            if candidates:
                args.resume_ckpt_path = candidates[0]
        skip_keys = [
            "resume_full_ckpt",
            "resume_ckpt_path",
            "resume_bs",
            "resume_scheduler",
            "resume_optimizer",
            "total_run_time_hr",
            "output_dir",
        ]
        if not args.resume_scheduler:
            skip_keys.extend([
                "epochs",
                "warmup_steps",
                "eta_min",
                "composite_lr",
            ])
        if not args.resume_bs:
            skip_keys.extend(["batch_size"])
        ckpt = torch.load(args.resume_ckpt_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", None)
        if ckpt_args is not None:
            for k, v in vars(ckpt_args).items():
                if k not in skip_keys:
                    setattr(args, k, v)

    if args.train:
        logger.info("Starting training for %s", MODEL_NAME)
        start_epoch = 0
        training_history = None
        if args.resume_full_ckpt and args.resume_ckpt_path and ckpt is not None:
            model.load_state_dict(ckpt.get("model", {}), strict=False)
            decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
            if args.resume_optimizer:
                if "optimizer" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer"])
            else:
                logger.info("Skipping optimizer state load (resume_optimizer=False).")
            if args.resume_scheduler:
                start_epoch = int(ckpt.get("epoch", 0))
                if "scheduler" in ckpt and ckpt["scheduler"] is not None:
                    scheduler.load_state_dict(ckpt["scheduler"])
            else:
                logger.info("Skipping scheduler state load (resume_scheduler=False).")
            if args.use_rc_loss and "rowcol_loss" in ckpt and ckpt["rowcol_loss"] is not None:
                for k in ["row_targets", "col_targets", "row_index_full", "col_index_full"]:
                    if k in ckpt["rowcol_loss"]:
                        ckpt["rowcol_loss"].pop(k)
                rowcol_loss.load_state_dict(ckpt["rowcol_loss"])
            logger.info("Resumed full checkpoint from %s at epoch %s", args.resume_ckpt_path, start_epoch)
            training_history = ckpt.get("training_history", None)

        if not isinstance(training_history, dict):
            if args.use_rc_loss:
                training_history = {
                    "train_loss": [],
                    "train_acc": [],
                    "valid_acc": [],
                    "valid_miou": [],
                    "train_time": [],
                    "val_time": [],
                    "epoch": [],
                    "step": [],
                    "base_loss": [],
                    "aux_loss": [],
                }
            else:
                training_history = {
                    "train_loss": [],
                    "train_acc": [],
                    "valid_acc": [],
                    "valid_miou": [],
                    "train_time": [],
                    "val_time": [],
                    "epoch": [],
                    "step": [],
                }
        training_history.setdefault("train_time", [])
        training_history.setdefault("val_time", [])
        training_history.setdefault("train_loss", [])
        training_history.setdefault("train_acc", [])
        training_history.setdefault("valid_acc", [])
        training_history.setdefault("valid_miou", [])
        training_history.setdefault("epoch", [])
        training_history.setdefault("step", [])
        if args.use_rc_loss:
            training_history.setdefault("base_loss", [])
            training_history.setdefault("aux_loss", [])

        def _pad_history(hist, fill_value=None):
            keys = [k for k, v in hist.items() if isinstance(v, list)]
            if not keys:
                return
            max_len = max(len(hist[k]) for k in keys)
            for k in keys:
                if len(hist[k]) < max_len:
                    hist[k].extend([fill_value] * (max_len - len(hist[k])))

        def _append_eval_log(log_path, row):
            if not row:
                return
            df = pd.DataFrame([row])
            header = not os.path.exists(log_path)
            df.to_csv(log_path, mode="a", header=header, index=False)

        if args.resume_full_ckpt:
            _pad_history(training_history)

        step = int(training_history.get("step", [0])[-1]) if training_history.get("step") else 0
        best_acc = 0.0
        best_miou = 0.0
        best_ckpt_path = os.path.join(ckpt_output_dir, "best.pth")
        if training_history.get("valid_miou"):
            best_miou = max(training_history.get("valid_miou", [0.0]) or [0.0])
        if training_history.get("valid_acc"):
            best_acc = max(training_history.get("valid_acc", [0.0]) or [0.0])
        log_interval = getattr(args, "log_interval", 50)
        csv_interval = getattr(args, "csv_interval", 1)
        last_trained_epoch = int(start_epoch)

        for epoch in range(start_epoch, args.epochs):
            epoch_train_start = time.time()
            model.train()
            decoder.train()

            running_loss_t = torch.zeros((), device=DEVICE)
            base_loss_t = torch.zeros((), device=DEVICE)
            aux_loss_sum_t = torch.zeros((), device=DEVICE)
            train_correct_t = torch.zeros((), device=DEVICE)
            train_total_t = torch.zeros((), device=DEVICE)
            train_samples_t = torch.zeros((), device=DEVICE)

            train_sampler.set_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)

            for step_in_epoch, (inputs, labels) in enumerate(train_loader):
                step += 1
                if getattr(args, "debug_xla", False):
                    _master_print(f"[rank {RANK}] step{step}: batch moved")
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                bs = inputs.size(0)
                aux_loss = None

                with _autocast():
                    if args.seg_head == "upernet":
                        outputs, features, grid_hw = _forward_upernet(model, decoder, inputs, args.feature_layers)
                        last_tokens = features[-1]
                    else:
                        feats = model.forward_features(inputs)
                        grid_hw = _infer_grid_hw(model, inputs)
                        outputs = decoder(
                            feats[:, model.num_prefix_tokens:, :],
                            grid_size=grid_hw,
                            out_size=inputs.shape[-2:],
                        )
                        last_tokens = feats[:, model.num_prefix_tokens:, :]
                    loss = ce_criterion(outputs, labels)
                    base_loss = loss
                    if args.use_rc_loss:
                        aux_loss = rowcol_loss(last_tokens)
                        aux_loss_sum_t += aux_loss.detach() * bs
                        t = min(1.0, (step + 1) / args.warmup_steps_for_aux)
                        alpha_t = args.alpha_min + (args.rc_alpha - args.alpha_min) * t
                        loss = base_loss + alpha_t * aux_loss

                loss.backward()
                grad_norm = None
                if args.clip_value is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(training_parameters, max_norm=args.clip_value)
                xm.optimizer_step(optimizer, barrier=True)
                # if getattr(args, "debug_xla", False):
                #     _master_print(f"[rank {RANK}] step{step}: optimizer step done")
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if step % log_interval == 0:
                    _xla_sync()

                with torch.no_grad():
                    pred = outputs.detach().argmax(dim=1)
                    mask = labels >= 0
                    valid_pixels = mask.sum()
                    train_correct_t += ((pred == labels) & mask).sum()
                    train_total_t += valid_pixels
                    train_samples_t += bs

                running_loss_t += loss.detach() * valid_pixels
                if args.use_rc_loss:
                    base_loss_t += base_loss.detach() * valid_pixels
                train_total_t = train_total_t.clamp_min(1)
                train_samples_t = train_samples_t.clamp_min(1)
                if step % log_interval == 0 and IS_MASTER:
                    avg_loss = (running_loss_t / train_total_t).float().item()
                    avg_acc = (train_correct_t / train_total_t).float().item()
                    msg = f"Epoch {epoch+1}/{args.epochs} step {step}: loss={avg_loss:.4f} acc={avg_acc:.3f}"
                    if aux_loss is not None:
                        avg_aux = (aux_loss_sum_t / train_samples_t).float().item()
                        msg += f" aux={avg_aux:.4f}"
                    if grad_norm is not None:
                        try:
                            msg += f" grad_norm={float(grad_norm):.4g}"
                        except Exception:
                            msg += " grad_norm=ERR"
                    if args.show_peak_gpu_mem:
                        info = _tpu_info_mem()
                        if info is not None:
                            used_mb, total_mb, peak_mb = info
                            msg += f" tpu_mem={used_mb:.0f}/{peak_mb:.0f}/{total_mb:.0f}MB"
                        elif not tpu_mem_warned:
                            tpu_mem_warned = True
                            print("TPU memory info unavailable; skipping TPU mem logging.", flush=True)
                        cpu_peak = _cpu_peak_mb()
                        if cpu_peak is not None:
                            if cpu_total_mb is not None:
                                msg += f" cpu_peak={cpu_peak:.0f}/{cpu_total_mb:.0f}MB"
                            else:
                                msg += f" cpu_peak={cpu_peak:.0f}MB"
                    logger.info(msg)

            train_time = time.time() - epoch_train_start
            model.eval()
            decoder.eval()
            val_correct_t = torch.zeros((), device=DEVICE)
            val_total_t = torch.zeros((), device=DEVICE)
            confmat = torch.zeros((args.num_classes, args.num_classes), device=DEVICE, dtype=torch.int64)
            val_start = time.time()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    real_bs = labels.shape[0]
                    if args.val_pad_to_full_batch and real_bs < args.batch_size:
                        pad_n = args.batch_size - real_bs
                        pad_inputs = torch.zeros((pad_n,) + inputs.shape[1:], device=inputs.device, dtype=inputs.dtype)
                        pad_labels = torch.full(
                            (pad_n,) + labels.shape[1:], -1, device=labels.device, dtype=labels.dtype
                        )
                        inputs = torch.cat([inputs, pad_inputs], dim=0)
                        labels = torch.cat([labels, pad_labels], dim=0)
                    with _autocast():
                        if args.use_ms_flip_eval:
                            outputs = _ms_flip_predict(
                                model,
                                decoder,
                                inputs,
                                args.num_classes,
                                args.ms_scales,
                                True,
                                model.patch_embed.patch_size,
                                feature_layers=args.feature_layers,
                            )
                        else:
                            if args.seg_head == "upernet":
                                outputs, _, _ = _forward_upernet(model, decoder, inputs, args.feature_layers)
                            else:
                                feats = model.forward_features(inputs)
                                grid_hw = _infer_grid_hw(model, inputs)
                                outputs = decoder(
                                    feats[:, model.num_prefix_tokens:, :],
                                    grid_size=grid_hw,
                                    out_size=inputs.shape[-2:],
                                )
                          
                    pred = outputs.argmax(dim=1)
                    mask = labels >= 0
                    if args.val_pad_to_full_batch and real_bs < args.batch_size:
                        mask[real_bs:] = False
                    val_correct_t += ((pred == labels) & mask).sum()
                    val_total_t += mask.sum()
                    confmat += fast_confusion_matrix(pred, labels, args.num_classes, ignore_index=-1)

            val_time = time.time() - val_start
            if WORLD_SIZE > 1:
                xm.all_reduce(
                    xm.REDUCE_SUM,
                    [
                        running_loss_t,
                        train_correct_t,
                        train_total_t,
                        train_samples_t,
                        val_correct_t,
                        val_total_t,
                        aux_loss_sum_t,
                        base_loss_t,
                    ],
                )
                xm.all_reduce(xm.REDUCE_SUM, [confmat])
            
            confmat_f = confmat.to(torch.float32)
            intersection = torch.diag(confmat_f)
            union = confmat_f.sum(dim=1) + confmat_f.sum(dim=0) - intersection
            valid = union > 0
            epoch_val_miou = (intersection[valid] / union[valid]).mean().item() if valid.any() else 0.0

            epoch_val_acc = (val_correct_t / val_total_t.clamp_min(1)).float().item()
            epoch_train_acc = (train_correct_t / train_total_t).float().item()
            epoch_train_loss = (running_loss_t / train_total_t).float().item()

            improved_miou = epoch_val_miou > best_miou
            if improved_miou:
                best_miou = epoch_val_miou
            if best_acc < epoch_val_acc:
                best_acc = epoch_val_acc

            if IS_MASTER:
                logger.info("Epoch %s/%s Summary:", epoch + 1 + args.start_epoch, args.epochs)
                logger.info("Step %s Summary:", step)

                if args.use_rc_loss:
                    epoch_aux_loss = (aux_loss_sum_t / train_samples_t).float().item()
                    epoch_base_loss = (base_loss_t / train_total_t).float().item()
                    logger.info(
                        "  Train Loss: %.4f | Aux Loss: %.4f | Base Loss: %.4f | Train Acc: %.4f | "
                        "Valid Acc: %.4f | Valid mIoU: %.4f | train_time: %.1fs | val_time: %.1fs",
                        epoch_train_loss,
                        epoch_aux_loss,
                        epoch_base_loss,
                        epoch_train_acc,
                        epoch_val_acc,
                        epoch_val_miou,
                        train_time,
                        val_time,
                    )
                    training_history["aux_loss"].append(epoch_aux_loss)
                    training_history["base_loss"].append(epoch_base_loss)
                else:
                    logger.info(
                        "  Train Loss: %.4f | Train Acc: %.4f | Valid Acc: %.4f | Valid mIoU: %.4f | "
                        "train_time: %.1fs | val_time: %.1fs",
                        epoch_train_loss,
                        epoch_train_acc,
                        epoch_val_acc,
                        epoch_val_miou,
                        train_time,
                        val_time,
                    )

                training_history["train_loss"].append(epoch_train_loss)
                training_history["train_acc"].append(epoch_train_acc)
                training_history["valid_acc"].append(epoch_val_acc)
                training_history["valid_miou"].append(epoch_val_miou)
                training_history["train_time"].append(train_time)
                training_history["val_time"].append(val_time)
                training_history["epoch"].append(epoch + 1)
                training_history["step"].append(step)
                last_trained_epoch = epoch + 1

                if (epoch + 1) % csv_interval == 0:
                    pd.DataFrame(training_history).to_csv(os.path.join(output_dir, f"{subdir_name}.csv"), index=False)

            save_best = improved_miou and bool(getattr(args, "save_best_ckpt", True))
            if save_best:
                best_ckpt = {}
                if IS_MASTER:
                    best_ckpt = {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "decoder": decoder.state_dict(),
                        "metric": {"valid_miou": epoch_val_miou, "valid_acc": epoch_val_acc},
                    }
                _xla_sync()
                xm.save(best_ckpt, best_ckpt_path, master_only=True)
                if IS_MASTER:
                    logger.info("Saved best checkpoint (weights only) to %s", best_ckpt_path)

            if args.save_full_ckpt and not save_best:
                ckpt_payload = {}
                if IS_MASTER:
                    ckpt_payload = {
                        "epoch": epoch + 1,
                        "step": step,
                        "model": model.state_dict(),
                        "decoder": decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "rowcol_loss": rowcol_loss.state_dict() if args.use_rc_loss else None,
                        "training_history": training_history,
                        "args": args,
                    }
                _xla_sync()
                xm.save(ckpt_payload, last_ckpt_path, master_only=True)
                if IS_MASTER:
                    logger.info("Saved full checkpoint to %s", last_ckpt_path)

            if args.total_run_time_hr is not None:
                elapsed = time.time() - train_start_time
                max_run_time_sec = args.total_run_time_hr * 3600
                if elapsed + (train_time + val_time) + 900 >= max_run_time_sec:
                    logger.info(
                        "Stopping training: elapsed %.0fs reached limit %.2fh.",
                        elapsed,
                        args.total_run_time_hr,
                    )
                    break
            gc.collect()

        if IS_MASTER:
            logger.info("Training complete.")
            logger.info("Best Accuracy: %.4f", best_acc)
            logger.info(output_dir)

        if args.final_ms_flip_eval and not args.use_ms_flip_eval:
            def _run_ms_flip_eval(tag: str):
                model.eval()
                decoder.eval()
                val_correct_t = torch.zeros((), device=DEVICE)
                val_total_t = torch.zeros((), device=DEVICE)
                confmat = torch.zeros((args.num_classes, args.num_classes), device=DEVICE, dtype=torch.int64)
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs = inputs.to(DEVICE, non_blocking=True)
                        labels = labels.to(DEVICE, non_blocking=True)
                        real_bs = labels.shape[0]
                        if args.val_pad_to_full_batch and real_bs < args.batch_size:
                            pad_n = args.batch_size - real_bs
                            pad_inputs = torch.zeros((pad_n,) + inputs.shape[1:], device=inputs.device, dtype=inputs.dtype)
                            pad_labels = torch.full(
                                (pad_n,) + labels.shape[1:], -1, device=labels.device, dtype=labels.dtype
                            )
                            inputs = torch.cat([inputs, pad_inputs], dim=0)
                            labels = torch.cat([labels, pad_labels], dim=0)
                        with _autocast():
                            outputs = _ms_flip_predict(
                                model,
                                decoder,
                                inputs,
                                args.num_classes,
                                args.ms_scales,
                                True,
                                model.patch_embed.patch_size,
                                feature_layers=args.feature_layers,
                            )
                        pred = outputs.argmax(dim=1)
                        mask = labels >= 0
                        if args.val_pad_to_full_batch and real_bs < args.batch_size:
                            mask[real_bs:] = False
                        val_correct_t += ((pred == labels) & mask).sum()
                        val_total_t += mask.sum()
                        confmat += fast_confusion_matrix(pred, labels, args.num_classes, ignore_index=-1)

                if WORLD_SIZE > 1:
                    xm.all_reduce(xm.REDUCE_SUM, [val_correct_t, val_total_t])
                    xm.all_reduce(xm.REDUCE_SUM, [confmat])

                confmat_f = confmat.to(torch.float32)
                intersection = torch.diag(confmat_f)
                union = confmat_f.sum(dim=1) + confmat_f.sum(dim=0) - intersection
                valid = union > 0
                ms_miou = (intersection[valid] / union[valid]).mean().item() if valid.any() else 0.0
                ms_acc = (val_correct_t / val_total_t.clamp_min(1)).float().item()
                if IS_MASTER:
                    logger.info("%s MS+Flip Acc: %.4f | %s MS+Flip mIoU: %.4f", tag, ms_acc, tag, ms_miou)
                return ms_acc, ms_miou

            if IS_MASTER:
                logger.info("Running final multi-scale + flip evaluation (final checkpoint)...")
            final_eval_row = {
                "run_tag": run_tag,
                "subdir_name": subdir_name,
                "output_dir": output_dir,
                "epoch": int(last_trained_epoch),
            }
            final_ms_acc, final_ms_miou = _run_ms_flip_eval("Final")
            if IS_MASTER:
                final_eval_row["final_ms_flip_acc"] = final_ms_acc
                final_eval_row["final_ms_flip_miou"] = final_ms_miou

                if os.path.exists(best_ckpt_path):
                    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(best_ckpt.get("model", {}), strict=False)
                    decoder.load_state_dict(best_ckpt.get("decoder", {}), strict=False)
                    logger.info("Loaded best checkpoint for final MS+Flip evaluation.")
                    best_ms_acc, best_ms_miou = _run_ms_flip_eval("Best")
                    final_eval_row["best_ms_flip_acc"] = best_ms_acc
                    final_eval_row["best_ms_flip_miou"] = best_ms_miou
                else:
                    logger.info("Best checkpoint not found; skipping best MS+Flip evaluation.")

                final_eval_log = os.path.join(output_dir, f"{subdir_name}_final_eval.csv")
                _append_eval_log(final_eval_log, final_eval_row)

            if IS_MASTER:
                history_df = pd.DataFrame(training_history)
                history_df.to_csv(os.path.join(output_dir, f"{subdir_name}.csv"), index=False)
                best_miou = history_df["valid_miou"].max()
                best_epoch = history_df.loc[history_df["valid_miou"].idxmax(), "epoch"]
                logger.info(f"Best miou: {best_miou:.4f} at epoch {best_epoch}")

                best_miou_row = history_df.loc[history_df["valid_miou"].idxmax()]
                best_miou_epoch = int(best_miou_row["epoch"])
                best_miou_val = best_miou_row["valid_miou"]

                best_acc_row = history_df.loc[history_df["valid_acc"].idxmax()]
                best_acc_epoch = int(best_acc_row["epoch"])
                best_acc_val = best_acc_row["valid_acc"]

                logger.info("\n--- Best Validation Metrics from History ---")
                logger.info(f"  Best miou:      {best_miou_val:.4f} (Epoch {best_miou_epoch})")
                logger.info(f"  Best acc:  {best_acc_val:.4f} (Epoch {best_acc_epoch})")
                logger.info("------------------------------------------")


if __name__ == "__main__":
    _preflight_kaggle_env()
    _spawn_tpu(main)
