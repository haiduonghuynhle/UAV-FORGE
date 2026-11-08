"""
UltraVPR Global Retrieval Module.

Implements the Phase-1 coarse localization using UltraVPR (Chen et al., 2025):
  - E2ResNet rotation-equivariant backbone (via escnn / e2cnn)
  - GroupPooling + GeM aggregation → 256-D rotation-invariant descriptor
  - ONNX Runtime (GPU) inference for edge deployment
  - FAISS-backed database index for nearest-neighbour tile retrieval

The module outputs only a Candidate Index (k) and geodetic anchor.
It does NOT forward spatial tensors — Phase 2 is strictly independent.

Reference:
    Chen et al., "UltraVPR: Unsupervised Lightweight Rotation-Invariant
    Aerial Visual Place Recognition", IEEE RAL 2025.
    
    Yuan et al., "CAMP: Cross-view Alignment for Mutual Perception" (Native Port)
"""

from __future__ import annotations

import logging
import os
import time
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _torch_expected_device(device_str: str):
    """Canonical torch.device for UltraVPR inference."""
    import torch

    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device_str)


def _audit_module_devices(module, expected) -> list[tuple[str, str]]:
    """Return list of (name, actual_device_str) for params/buffers not on expected device."""
    import torch

    bad: list[tuple[str, str]] = []
    exp = torch.device(expected)

    def _device_mismatch(actual) -> bool:
        act = torch.device(actual)
        if exp.type != act.type:
            return True
        if exp.type == "cpu":
            return False
        ei = 0 if exp.index is None else exp.index
        ai = 0 if act.index is None else act.index
        return ei != ai

    for name, p in module.named_parameters(recurse=True):
        if p is not None and _device_mismatch(p.device):
            bad.append((f"param:{name}", str(p.device)))
    for name, b in module.named_buffers(recurse=True):
        if b is not None and _device_mismatch(b.device):
            bad.append((f"buffer:{name}", str(b.device)))
    return bad


def _force_module_to_device(module, device) -> None:
    """Move module and all parameters/buffers to device (handles stale CPU tensors)."""
    import torch

    dev = torch.device(device)
    module.to(dev)
    for p in module.parameters():
        if p.data.device != dev:
            p.data = p.data.to(dev)
    for b in module.buffers():
        if b.device != dev:
            b.data = b.data.to(dev)
    module.to(dev)


def _force_unregistered_tensors_to_device(module, device) -> None:
    """
    Move module-owned tensor attributes that are not registered parameters/buffers.

    Some third-party stacks (escnn) may keep runtime tensors in plain attributes.
    If those stay on CPU while parameters are on CUDA, matrix ops can fail with
    wrapper_CUDA_bmm mixed-device errors.
    """
    import torch

    dev = torch.device(device)
    def _move_any(x):
        if isinstance(x, torch.Tensor):
            return x.to(dev) if x.device != dev else x
        if isinstance(x, list):
            return [_move_any(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_move_any(v) for v in x)
        if isinstance(x, dict):
            return {k: _move_any(v) for k, v in x.items()}
        return x

    for sub in module.modules():
        for key, value in list(vars(sub).items()):
            moved = _move_any(value)
            if moved is not value:
                try:
                    setattr(sub, key, moved)
                except Exception:
                    # Best-effort: skip readonly attrs.
                    continue


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ULTRAVPR_DESCRIPTOR_DIM = 256   # Half of MixVPR's 512 — 50% DB size reduction
ULTRAVPR_INPUT_SIZE = 320       # Canonical input resolution
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_ultravpr_preprocess_contract(
    input_hw: Tuple[int, int],
    *,
    resize_interpolation: str = "cv2.INTER_LINEAR",
) -> Dict[str, object]:
    """Document the fixed Phase-1 preprocessing contract for audit/reproducibility."""
    return {
        "input_hw": [int(input_hw[0]), int(input_hw[1])],
        "channel_order": "RGB",
        "resize": resize_interpolation,
        "value_range": "[0, 1] float32",
        "normalize": "imagenet",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "descriptor_l2_normalize": True,
    }
ULTRAVPR_NUM_ROTATIONS = 8      # E2ResNet discrete rotation angles (N)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Phase-1 output contract."""
    candidate_index: int
    similarity_score: float
    geodetic_anchor: Dict[str, float]
    descriptor: Optional[np.ndarray] = None
    timing_ms: float = 0.0


@dataclass
class DatabaseEntry:
    """Single tile in the reference database."""
    index: int
    descriptor: np.ndarray          # (256,) float32
    geodetic_anchor: Dict[str, float]  # {lat, lon}


# ---------------------------------------------------------------------------
# UltraVPR Model (PyTorch — for training & ONNX export)
# ---------------------------------------------------------------------------

class UltraVPRModel:
    """
    UltraVPR inference wrapper.

    Supports two backends:
      1. PyTorch (for training / desktop evaluation)
      2. ONNX Runtime GPU (for Jetson edge deployment)

    Architecture:
      E2ResNet50 backbone (N=8 rotation channels, D=256 feature depth)
        → GroupPooling (N*D, H, W) → (D, H, W)
        → GeM pooling → FC → 256-D rotation-invariant descriptor
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        onnx_path: Optional[Path] = None,
        device: str = "cpu",
        descriptor_dim: int = ULTRAVPR_DESCRIPTOR_DIM,
        input_size: int | Tuple[int, int] = ULTRAVPR_INPUT_SIZE,
    ):
        """
        Initialize UltraVPR model.

        Args:
            weights_path: Path to PyTorch checkpoint (.pth)
            onnx_path: Path to ONNX model (.onnx) — takes priority on Jetson
            device: 'cpu' or 'cuda'
            descriptor_dim: Output descriptor dimensionality (default 256)
            input_size: Input image size (int -> square, or (H, W))
        """
        self._device = device
        self._descriptor_dim = descriptor_dim
        if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
            self._input_hw = (int(input_size[0]), int(input_size[1]))
        else:
            s = int(input_size)
            self._input_hw = (s, s)
        self._weights_path = weights_path
        self._onnx_path = onnx_path

        self._torch_model = None
        self._onnx_session = None
        self._backend = None   # 'torch' or 'onnx'
        self._torch_device = None

        self._loaded = False

    @property
    def descriptor_dim(self) -> int:
        return self._descriptor_dim

    @property
    def input_size(self) -> int:
        return self._input_hw

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def get_preprocess_contract(self) -> Dict[str, object]:
        """Return the active preprocessing contract for this model instance."""
        return get_ultravpr_preprocess_contract(self._input_hw)

    def load(self) -> None:
        """Load model weights. Prefers ONNX if path is provided."""
        if self._loaded:
            return

        if self._onnx_path is not None and Path(self._onnx_path).exists():
            self._load_onnx()
        elif self._weights_path is not None and Path(self._weights_path).exists():
            self._load_torch()
        else:
            logger.warning(
                "No UltraVPR weights found. Using random projection fallback "
                "(descriptor quality will be meaningless)."
            )
            self._backend = "random_projection"
            self._projection_matrix = np.random.randn(
                3 * self._input_hw[0] * self._input_hw[1] // 64,
                self._descriptor_dim,
            ).astype(np.float32)
            self._loaded = True

    def _load_onnx(self) -> None:
        """Load ONNX Runtime session (GPU provider preferred)."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX backend. "
                "Install: pip install onnxruntime-gpu  (Jetson: from NVIDIA)"
            )

        providers = []
        if self._device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self._onnx_session = ort.InferenceSession(
            str(self._onnx_path),
            providers=providers,
        )
        self._backend = "onnx"
        self._loaded = True
        logger.info(
            "UltraVPR loaded via ONNX Runtime (providers=%s, dim=%d, preprocess=%s)",
            self._onnx_session.get_providers(),
            self._descriptor_dim,
            self.get_preprocess_contract(),
        )

    def _load_torch(self) -> None:
        """Load PyTorch model."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for the torch backend.")

        # Build architecture
        model = self._build_torch_model()

        # Load checkpoint
        ckpt = torch.load(
            str(self._weights_path),
            map_location=self._device,
            weights_only=False,
        )
        # Handle Lightning-style checkpoints
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip 'model.' prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            key = k.replace("model.", "").replace("module.", "")
            cleaned[key] = v

        load_result = model.load_state_dict(cleaned, strict=False)
        if len(load_result.missing_keys) > 10 or len(load_result.unexpected_keys) > 10:
            raise RuntimeError(
                "UltraVPR checkpoint is incompatible with the runtime architecture: "
                f"missing={len(load_result.missing_keys)}, "
                f"unexpected={len(load_result.unexpected_keys)}. "
                "This would produce near-random descriptors."
            )
        torch_dev = _torch_expected_device(self._device)
        if str(torch_dev).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("UltraVPR torch backend requested CUDA but CUDA is not available")
        if torch_dev.type == "cuda":
            torch.cuda.set_device(torch_dev.index if torch_dev.index is not None else 0)
        _force_module_to_device(model, torch_dev)
        _force_unregistered_tensors_to_device(model, torch_dev)
        bad = _audit_module_devices(model, torch_dev)
        if bad:
            logger.warning(
                "UltraVPR device audit: %d tensors off-device after .to(); retrying hard move. Sample: %s",
                len(bad),
                bad[:3],
            )
            _force_module_to_device(model, torch_dev)
            _force_unregistered_tensors_to_device(model, torch_dev)
            bad = _audit_module_devices(model, torch_dev)
            if bad:
                raise RuntimeError(
                    "UltraVPR escnn model has parameters/buffers not on "
                    f"{torch_dev} after forced move: {bad[:8]}"
                )
        # escnn expands basis tensors on eval(); keep this after device alignment.
        model.eval()
        _force_unregistered_tensors_to_device(model, torch_dev)
        bad = _audit_module_devices(model, torch_dev)
        if bad:
            raise RuntimeError(
                "UltraVPR escnn model has off-device tensors after eval(): "
                f"{bad[:8]}"
            )

        self._torch_model = model
        self._torch_device = torch_dev
        self._backend = "torch"
        self._loaded = True
        logger.info(
            "UltraVPR loaded via PyTorch (device=%s, dim=%d, preprocess=%s)",
            torch_dev,
            self._descriptor_dim,
            self.get_preprocess_contract(),
        )
        self._warmup_torch()

    def _build_torch_model(self):
        """
        Build UltraVPR PyTorch model.

        Architecture:
            E2ResNet50 (escnn-based) → GroupPooling → GeM → FC(256)

        Note: Requires `escnn` package for E2-equivariant convolutions.
        Falls back to a standard ResNet50 + GeM if escnn is unavailable.
        """
        import torch
        import torch.nn as nn
        # Prefer the official UltraVPR architecture from the reference implementation.
        try:
            return self._build_reference_ultravpr_model()
        except Exception as exc:
            logger.warning("Reference UltraVPR model unavailable (%s); using local fallback.", exc)

        try:
            from escnn import gspaces, nn as e2nn
            return self._build_e2resnet_model(gspaces, e2nn)
        except ImportError:
            logger.warning(
                "escnn not installed. Falling back to ResNet50 + GeM "
                "(no rotation equivariance). "
                "Install: pip install escnn"
            )
            return self._build_resnet_gem_fallback()

    def _build_reference_ultravpr_model(self):
        """
        Build UltraVPR exactly as in reference/UltraVPR (checkpoint-compatible).
        """
        import importlib
        import torch.nn as nn
        import torch.nn.init as init

        base_roots: list[Path] = []
        env_root = (os.environ.get("HD4_REPO_ROOT") or os.environ.get("FORGE_CODE_ROOT") or "").strip()
        if env_root:
            base_roots.append(Path(env_root).expanduser().resolve())
        here_root = Path(__file__).resolve().parents[3]
        base_roots.append(here_root)
        sibling = here_root.parent / "code"
        if sibling not in base_roots:
            base_roots.append(sibling)

        ref_root = None
        for root in base_roots:
            cand = root / "reference" / "UltraVPR"
            if cand.exists():
                ref_root = cand
                break
        if ref_root is None:
            raise FileNotFoundError(
                "Reference UltraVPR repo not found under any candidate root: "
                + ", ".join(str((r / "reference" / "UltraVPR")) for r in base_roots)
            )

        # e2resnet.py depends on mmcv for tiny utilities. Provide a minimal shim
        # so runtime does not require full mmcv installation.
        if "mmcv" not in sys.modules:
            mmcv_mod = types.ModuleType("mmcv")
            mmcv_cnn_mod = types.ModuleType("mmcv.cnn")
            mmcv_utils_mod = types.ModuleType("mmcv.utils")
            mmcv_runner_mod = types.ModuleType("mmcv.runner")

            def constant_init(module, val, bias=0):
                if hasattr(module, "weight") and module.weight is not None:
                    init.constant_(module.weight, val)
                if hasattr(module, "bias") and module.bias is not None:
                    init.constant_(module.bias, bias)

            def kaiming_init(module):
                if hasattr(module, "weight") and module.weight is not None:
                    init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if hasattr(module, "bias") and module.bias is not None:
                    init.constant_(module.bias, 0)

            class Registry:
                def __init__(self, _name):
                    self._module_dict = {}

                def register_module(self, module=None, name=None):
                    def _register(cls):
                        key = name or cls.__name__
                        self._module_dict[key] = cls
                        return cls
                    if module is None:
                        return _register
                    return _register(module)

            mmcv_cnn_mod.constant_init = constant_init
            mmcv_cnn_mod.kaiming_init = kaiming_init
            mmcv_utils_mod.Registry = Registry
            def load_checkpoint(model, filename, strict=False, logger=None):
                import torch
                ckpt = torch.load(filename, map_location="cpu")
                state = ckpt.get("state_dict", ckpt)
                return model.load_state_dict(state, strict=strict)
            mmcv_runner_mod.load_checkpoint = load_checkpoint
            mmcv_mod.cnn = mmcv_cnn_mod
            mmcv_mod.utils = mmcv_utils_mod
            mmcv_mod.runner = mmcv_runner_mod
            sys.modules["mmcv"] = mmcv_mod
            sys.modules["mmcv.cnn"] = mmcv_cnn_mod
            sys.modules["mmcv.utils"] = mmcv_utils_mod
            sys.modules["mmcv.runner"] = mmcv_runner_mod

        if str(ref_root) not in sys.path:
            sys.path.insert(0, str(ref_root))

        e2resnet_mod = importlib.import_module("models.backbones.e2resnet")
        se2gem_mod = importlib.import_module("models.aggregators.se2gem")
        E2ResNet = getattr(e2resnet_mod, "E2ResNet")
        se2gem = getattr(se2gem_mod, "se2gem")

        class UltraVPRRef(nn.Module):
            def __init__(self, out_dim: int):
                super().__init__()
                self.backbone = E2ResNet(
                    depth=50,
                    out_indices=(3,),
                    with_geotensor=True,
                    orientation=8,
                    middle_channels=2048,
                )
                self.aggregator = se2gem(in_dim=256, out_dim=out_dim)

            def forward(self, x):
                return self.aggregator(self.backbone(x))

        return UltraVPRRef(self._descriptor_dim)

    def _build_e2resnet_model(self, gspaces, e2nn):
        """Build the full E2ResNet + GroupPooling + GeM model."""
        import torch
        import torch.nn as nn

        class GeM(nn.Module):
            """Generalized Mean Pooling."""
            def __init__(self, p: float = 3.0, eps: float = 1e-6):
                super().__init__()
                self.p = nn.Parameter(torch.ones(1) * p)
                self.eps = eps

            def forward(self, x):
                return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)

        class UltraVPRNet(nn.Module):
            """
            UltraVPR: E2ResNet50 + GroupPooling + GeM + FC.

            The E2ResNet backbone is built using escnn with C_N cyclic group
            (N=8 discrete rotations). GroupPooling reduces (N*D, H, W) to
            (D, H, W) while maintaining rotation equivariance. GeM produces
            a rotation-invariant global descriptor.
            """
            def __init__(self, descriptor_dim: int = 256, num_rotations: int = 8):
                super().__init__()
                self.descriptor_dim = descriptor_dim

                # Rotation group: cyclic C_N
                r2_act = gspaces.rot2dOnR2(N=num_rotations)

                # Input type: 3 channels (RGB), trivial representation
                in_type = e2nn.FieldType(r2_act, 3 * [r2_act.trivial_repr])

                # Build E2ResNet layers using regular representations
                feat_type_64 = e2nn.FieldType(
                    r2_act, 64 // num_rotations * [r2_act.regular_repr]
                )
                feat_type_128 = e2nn.FieldType(
                    r2_act, 128 // num_rotations * [r2_act.regular_repr]
                )
                feat_type_256 = e2nn.FieldType(
                    r2_act, 256 // num_rotations * [r2_act.regular_repr]
                )
                feat_type_512 = e2nn.FieldType(
                    r2_act, 512 // num_rotations * [r2_act.regular_repr]
                )

                self.input_type = in_type

                # Stem
                self.stem = e2nn.SequentialModule(
                    e2nn.R2Conv(in_type, feat_type_64, kernel_size=7,
                                stride=2, padding=3, bias=False),
                    e2nn.InnerBatchNorm(feat_type_64),
                    e2nn.ReLU(feat_type_64, inplace=True),
                    e2nn.PointwiseMaxPool(feat_type_64, kernel_size=3,
                                          stride=2, padding=1),
                )

                # Simplified residual blocks
                self.layer1 = self._make_layer(r2_act, feat_type_64, feat_type_64)
                self.layer2 = self._make_layer(r2_act, feat_type_64, feat_type_128, stride=2)
                self.layer3 = self._make_layer(r2_act, feat_type_128, feat_type_256, stride=2)
                self.layer4 = self._make_layer(r2_act, feat_type_256, feat_type_512, stride=2)

                # GroupPooling: (N*D, H, W) → (D, H, W)
                self.group_pool = e2nn.GroupPooling(feat_type_512)

                # GeM + FC head (standard PyTorch, post-equivariance)
                pool_out_dim = 512 // num_rotations
                self.gem = GeM()
                self.fc = nn.Linear(pool_out_dim, descriptor_dim)

            def _make_layer(self, r2_act, in_type, out_type, stride=1):
                layers = [
                    e2nn.R2Conv(in_type, out_type, kernel_size=3,
                                stride=stride, padding=1, bias=False),
                    e2nn.InnerBatchNorm(out_type),
                    e2nn.ReLU(out_type, inplace=True),
                    e2nn.R2Conv(out_type, out_type, kernel_size=3,
                                stride=1, padding=1, bias=False),
                    e2nn.InnerBatchNorm(out_type),
                    e2nn.ReLU(out_type, inplace=True),
                ]
                return e2nn.SequentialModule(*layers)

            def forward(self, x):
                import torch
                # Wrap input as geometric tensor
                x = e2nn.GeometricTensor(x, self.input_type)
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.group_pool(x)
                # Convert to standard tensor for GeM + FC
                x = x.tensor
                x = self.gem(x)
                x = self.fc(x)
                # L2-normalize
                x = torch.nn.functional.normalize(x, p=2, dim=-1)
                return x

        return UltraVPRNet(
            descriptor_dim=self._descriptor_dim,
            num_rotations=ULTRAVPR_NUM_ROTATIONS,
        )

    def _build_resnet_gem_fallback(self):
        """Fallback: standard ResNet50 + GeM (no rotation equivariance)."""
        import torch
        import torch.nn as nn
        import torchvision.models as models

        class GeM(nn.Module):
            def __init__(self, p=3.0, eps=1e-6):
                super().__init__()
                self.p = nn.Parameter(torch.ones(1) * p)
                self.eps = eps

            def forward(self, x):
                return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)

        class ResNetGeM(nn.Module):
            def __init__(self, descriptor_dim=256):
                super().__init__()
                backbone = models.resnet50(weights=None)
                self.features = nn.Sequential(*list(backbone.children())[:-2])
                self.gem = GeM()
                self.fc = nn.Linear(2048, descriptor_dim)

            def forward(self, x):
                x = self.features(x)
                x = self.gem(x)
                x = self.fc(x)
                x = torch.nn.functional.normalize(x, p=2, dim=-1)
                return x

        return ResNetGeM(descriptor_dim=self._descriptor_dim)

    def extract_descriptor(self, image: np.ndarray | str) -> np.ndarray:
        """
        Extract a 256-D rotation-invariant descriptor from an image.

        Args:
            image: (H, W, 3) uint8 RGB image or absolute path string to image file

        Returns:
            (256,) float32 L2-normalized descriptor
        """
        if not self._loaded:
            self.load()

        if isinstance(image, str):
            import cv2
            img_bgr = cv2.imread(image, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read lazy tile image from cache: {image}")
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        preprocessed = self._preprocess(image)

        if self._backend == "onnx":
            return self._infer_onnx(preprocessed)
        elif self._backend == "torch":
            return self._infer_torch(preprocessed)
        else:
            # Random projection fallback
            flat = preprocessed.flatten()
            n = min(len(flat), self._projection_matrix.shape[0])
            desc = flat[:n] @ self._projection_matrix[:n]
            norm = np.linalg.norm(desc)
            if norm > 1e-8:
                desc /= norm
            return desc.astype(np.float32)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for UltraVPR inference.

        Returns:
            (1, 3, H, W) float32 normalized tensor
        """
        import cv2
        # Resize to canonical input
        target_h, target_w = self._input_hw
        resized = cv2.resize(image, (target_w, target_h))
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        # HWC → CHW, normalize to [0, 1]
        tensor = resized.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)  # (3, H, W)
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor[np.newaxis]  # (1, 3, H, W)

    def _infer_onnx(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run ONNX Runtime inference."""
        input_name = self._onnx_session.get_inputs()[0].name
        output = self._onnx_session.run(None, {input_name: preprocessed})
        desc = output[0].flatten().astype(np.float32)
        # L2-normalize
        norm = np.linalg.norm(desc)
        if norm > 1e-8:
            desc /= norm
        return desc

    def _warmup_torch(self) -> None:
        """One forward pass to surface device/shape issues early (escnn + CUDA)."""
        if self._torch_model is None or self._backend != "torch":
            return
        import torch

        dev = getattr(self, "_torch_device", None) or _torch_expected_device(self._device)
        h, w = self._input_hw
        try:
            with torch.no_grad():
                if dev.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.set_device(dev.index if dev.index is not None else 0)
                blank = np.zeros((h, w, 3), dtype=np.uint8)
                dummy_np = self._preprocess(blank)
                dummy = torch.from_numpy(dummy_np).to(dev, dtype=torch.float32)
                _ = self._torch_model(dummy)
        except Exception as exc:
            bad = _audit_module_devices(self._torch_model, dev)
            raise RuntimeError(
                f"UltraVPR CUDA warmup failed (device={dev}): {exc}. "
                f"Off-device sample: {bad[:6]}"
            ) from exc

    def _infer_torch(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run PyTorch inference."""
        import torch

        dev = getattr(self, "_torch_device", None) or _torch_expected_device(self._device)
        try:
            with torch.no_grad():
                if dev.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.set_device(dev.index if dev.index is not None else 0)
                tensor = torch.from_numpy(preprocessed).to(dev, dtype=torch.float32)
                desc = self._torch_model(tensor)
                return desc.detach().cpu().numpy().flatten().astype(np.float32)
        except RuntimeError as exc:
            if "cuda" in str(exc).lower() or "device" in str(exc).lower():
                bad = _audit_module_devices(self._torch_model, dev)
                raise RuntimeError(
                    f"UltraVPR inference failed on {dev}: {exc}. "
                    f"Device audit (first mismatches): {bad[:10]}"
                ) from exc
            raise

# ---------------------------------------------------------------------------
# CAMP Model (PyTorch)
# ---------------------------------------------------------------------------

class CAMPModel:
    """
    CAMP (Cross-view Alignment for Mutual Perception) inference wrapper.
    Used by the AnyVisLoc E2E baseline.
    """
    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: str = "cpu",
        descriptor_dim: int = 768,
        input_size: int | Tuple[int, int] = 384,
    ):
        self._device = device
        self._weights_path = weights_path
        self._descriptor_dim = descriptor_dim
        if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
            self._input_hw = (int(input_size[0]), int(input_size[1]))
        else:
            s = int(input_size)
            self._input_hw = (s, s)
        
        self._torch_model = None
        self._torch_device = None
        self._loaded = False
        self._transform = None

    @property
    def descriptor_dim(self) -> int:
        return self._descriptor_dim

    @property
    def input_size(self) -> int:
        return self._input_hw

    def get_preprocess_contract(self) -> Dict[str, object]:
        return {
            "input_hw": [self._input_hw[0], self._input_hw[1]],
            "channel_order": "RGB",
            "resize": "torchvision.transforms.Resize (PIL)",
            "value_range": "[0, 1] float32",
            "normalize": "imagenet",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "descriptor_l2_normalize": True,
        }

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        import torchvision.transforms as T
        import sys
        
        # Add the repo root to sys path if not present so Retrieval_Models can be imported
        here_root = Path(__file__).resolve().parents[3]
        if str(here_root) not in sys.path:
            sys.path.insert(0, str(here_root))

        try:
            from Retrieval_Models.CAMP.get_CAMP import get_CAMP_model
        except ImportError as e:
            raise ImportError(f"Failed to import CAMP model. Ensure Retrieval_Models/CAMP is in the repo root. Error: {e}")

        self._torch_device = _torch_expected_device(self._device)
        self._torch_model = get_CAMP_model().to(self._torch_device).eval()

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self._input_hw),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._loaded = True
        logger.info(
            "CAMP loaded via PyTorch (device=%s, dim=%d, preprocess=%s)",
            self._torch_device,
            self._descriptor_dim,
            self.get_preprocess_contract(),
        )

    def extract_descriptor(self, image: np.ndarray | str) -> np.ndarray:
        if not self._loaded:
            self.load()

        if isinstance(image, str):
            import cv2
            img_bgr = cv2.imread(image, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read lazy tile image from cache: {image}")
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Image is expected to be RGB uint8 here
        import torch
        with torch.no_grad():
            tensor = self._transform(image).unsqueeze(0).to(self._torch_device)
            out = self._torch_model(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = torch.nn.functional.normalize(out, dim=1)
            desc = out[0].detach().cpu().numpy().astype(np.float32)
        return desc


class MegaLocModel:
    """
    MegaLoc (Berton et al. 2025) foundation-VPR wrapper, loaded via torch.hub
    ("gmberton/MegaLoc"). Zero-shot retrieval-only baseline; no local weights
    file - torch.hub caches the checkpoint under ~/.cache/torch/hub.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,  # unused; kept for factory symmetry
        device: str = "cpu",
        input_size: int | Tuple[int, int] = 322,
    ):
        self._device = device
        self._weights_path = weights_path
        self._descriptor_dim = 0  # inferred on load
        if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
            self._input_hw = (int(input_size[0]), int(input_size[1]))
        else:
            s = int(input_size)
            self._input_hw = (s, s)
        # DINOv2 backbone needs H/W multiples of 14
        self._input_hw = tuple((max(14, (d // 14) * 14)) for d in self._input_hw)

        self._torch_model = None
        self._torch_device = None
        self._loaded = False
        self._transform = None

    @property
    def descriptor_dim(self) -> int:
        return self._descriptor_dim

    @property
    def input_size(self) -> Tuple[int, int]:
        return self._input_hw

    def get_preprocess_contract(self) -> Dict[str, object]:
        return {
            "input_hw": [self._input_hw[0], self._input_hw[1]],
            "channel_order": "RGB",
            "resize": "torchvision.transforms.Resize (PIL)",
            "value_range": "[0, 1] float32",
            "normalize": "imagenet",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "descriptor_l2_normalize": True,
            "hub_repo": "gmberton/MegaLoc",
        }

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        import torchvision.transforms as T

        self._torch_device = _torch_expected_device(self._device)
        self._torch_model = torch.hub.load(
            "gmberton/MegaLoc", "get_trained_model"
        ).to(self._torch_device).eval()

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self._input_hw),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self._input_hw[0], self._input_hw[1], device=self._torch_device)
            out = self._torch_model(dummy)
            if isinstance(out, (tuple, list)):
                out = out[0]
            self._descriptor_dim = int(out.shape[-1])

        self._loaded = True
        logger.info(
            "MegaLoc loaded via torch.hub (device=%s, dim=%d, input_hw=%s)",
            self._torch_device,
            self._descriptor_dim,
            self._input_hw,
        )

    def extract_descriptor(self, image: np.ndarray | str) -> np.ndarray:
        if not self._loaded:
            self.load()

        if isinstance(image, str):
            import cv2
            img_bgr = cv2.imread(image, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read lazy tile image from cache: {image}")
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        import torch
        with torch.no_grad():
            tensor = self._transform(image).unsqueeze(0).to(self._torch_device)
            out = self._torch_model(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = torch.nn.functional.normalize(out, dim=1)
            desc = out[0].detach().cpu().numpy().astype(np.float32)
        return desc


# ---------------------------------------------------------------------------
# Database Index
# ---------------------------------------------------------------------------

class RetrievalDatabase:
    """
    Reference tile database with FAISS index for fast retrieval.

    Storage: 256-D float32 descriptors → 8 MB for 7440 tiles
    (50% smaller than MixVPR's 512-D → 16 MB).
    """

    def __init__(self):
        self._descriptors: List[np.ndarray] = []
        self._anchors: List[Dict[str, float]] = []
        self._index = None
        self._built = False

    @property
    def size(self) -> int:
        return len(self._descriptors)

    @property
    def memory_mb(self) -> float:
        """Estimated database memory in MB."""
        if not self._descriptors:
            return 0.0
        return len(self._descriptors) * self._descriptors[0].nbytes / (1024 * 1024)

    def add_tile(self, descriptor: np.ndarray, anchor: Dict[str, float]) -> None:
        """Add a tile descriptor + geodetic anchor to the database."""
        self._descriptors.append(descriptor.astype(np.float32))
        self._anchors.append(anchor)
        self._built = False

    def build_index(self) -> None:
        """Build FAISS index for fast nearest-neighbour search."""
        if not self._descriptors:
            logger.warning("Cannot build index: database is empty")
            return

        desc_matrix = np.vstack(self._descriptors).astype(np.float32)

        try:
            import faiss
            self._index = faiss.IndexFlatL2(desc_matrix.shape[1])
            self._index.add(desc_matrix)
            self._built = True
            logger.info(
                "FAISS index built: %d tiles, %d-D, %.2f MB",
                len(self._descriptors),
                desc_matrix.shape[1],
                self.memory_mb,
            )
        except ImportError:
            # Fallback: brute-force NumPy search
            logger.warning(
                "FAISS not available. Using brute-force NumPy search. "
                "Install: pip install faiss-cpu"
            )
            self._desc_matrix = desc_matrix
            self._built = True

    def search(self, query_desc: np.ndarray, top_k: int = 1) -> List[RetrievalResult]:
        """
        Search database for nearest tiles.

        Args:
            query_desc: (256,) float32 query descriptor
            top_k: Number of top candidates to return

        Returns:
            List of RetrievalResult sorted by similarity (best first)
        """
        if not self._built:
            self.build_index()

        query = query_desc.reshape(1, -1).astype(np.float32)

        if self._index is not None:
            # FAISS search
            distances, indices = self._index.search(query, top_k)
            results = []
            for i in range(top_k):
                idx = int(indices[0, i])
                if idx < 0:
                    continue
                results.append(RetrievalResult(
                    candidate_index=idx,
                    similarity_score=float(1.0 / (1.0 + distances[0, i])),
                    geodetic_anchor=self._anchors[idx],
                ))
            return results
        else:
            # NumPy brute-force fallback
            diffs = self._desc_matrix - query
            dists = np.linalg.norm(diffs, axis=1)
            indices = np.argsort(dists)[:top_k]
            results = []
            for idx in indices:
                results.append(RetrievalResult(
                    candidate_index=int(idx),
                    similarity_score=float(1.0 / (1.0 + dists[idx])),
                    geodetic_anchor=self._anchors[idx],
                ))
            return results

    def save(self, path: Path) -> None:
        """Save database to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        desc_matrix = np.vstack(self._descriptors).astype(np.float32)
        np.savez_compressed(
            path,
            descriptors=desc_matrix,
            anchors=np.array([
                [a.get("lat", 0.0), a.get("lon", 0.0)]
                for a in self._anchors
            ], dtype=np.float64),
        )
        logger.info("Database saved: %s (%.2f MB)", path, self.memory_mb)

    def load(self, path: Path) -> None:
        """Load database from disk."""
        path = Path(path)
        data = np.load(path)
        descs = data["descriptors"]
        anchors = data["anchors"]
        self._descriptors = [descs[i] for i in range(len(descs))]
        self._anchors = [
            {"lat": float(anchors[i, 0]), "lon": float(anchors[i, 1])}
            for i in range(len(anchors))
        ]
        self._built = False
        self.build_index()
        logger.info("Database loaded: %s (%d tiles)", path, self.size)


# ---------------------------------------------------------------------------
# Retriever — combines model + database
# ---------------------------------------------------------------------------

class RetrievalModule:
    """
    Phase-1: FAISS-backed Tile Retrieval.
    """

    def __init__(
        self,
        backend: str = "ultravpr",
        top_k: int = 5,
        verify_top_k: bool = False,
        weights_path: Optional[Path] = None,
        onnx_path: Optional[Path] = None,
        device: str = "cpu",
        input_size: Optional[Tuple[int, int]] = None,
    ):
        self._top_k = top_k
        self._verify_top_k = verify_top_k
        self._backend = backend
        
        self._model = self._build_model(
            backend, weights_path, onnx_path, device, input_size
        )
        self._database = RetrievalDatabase()

    def _build_model(
        self, backend: str, weights_path: Optional[Path], onnx_path: Optional[Path],
        device: str, input_size: Optional[Tuple[int, int]]
    ):
        if backend == "ultravpr":
            kwargs = {
                "device": device,
                "weights_path": weights_path,
                "onnx_path": onnx_path,
            }
            if input_size is not None:
                kwargs["input_size"] = input_size
            return UltraVPRModel(**kwargs)
        elif backend == "camp":
            kwargs = {
                "device": device,
                "weights_path": weights_path,
            }
            if input_size is not None:
                kwargs["input_size"] = input_size
            return CAMPModel(**kwargs)
        elif backend == "megaloc":
            kwargs = {
                "device": device,
                "weights_path": weights_path,
            }
            if input_size is not None:
                kwargs["input_size"] = input_size
            return MegaLocModel(**kwargs)
        else:
            raise ValueError(f"Unknown retrieval backend: {backend}")

    @property
    def model(self):
        return self._model

    @property
    def database(self) -> RetrievalDatabase:
        return self._database

    def encode_database(
        self,
        tile_images: List[np.ndarray],
        tile_anchors: List[Dict[str, float]],
    ) -> None:
        """
        Encode all reference tiles into the database.

        Args:
            tile_images: List of (H, W, 3) uint8 images
            tile_anchors: Corresponding geodetic anchors [{lat, lon}, ...]
        """
        self._model.load()

        from tqdm import tqdm
        for img, anchor in tqdm(
            zip(tile_images, tile_anchors),
            total=len(tile_images),
            desc="Encoding tiles",
        ):
            desc = self._model.extract_descriptor(img)
            self._database.add_tile(desc, anchor)

        self._database.build_index()
        logger.info(
            "Database encoded: %d tiles, %.2f MB",
            self._database.size,
            self._database.memory_mb,
        )

    def retrieve(self, query_image: np.ndarray) -> RetrievalResult:
        """
        Run Phase-1 retrieval.

        Args:
            query_image: (H, W, 3) uint8 RGB query frame

        Returns:
            RetrievalResult with candidate_index and geodetic_anchor
        """
        self._model.load()

        t0 = time.perf_counter()
        desc = self._model.extract_descriptor(query_image)
        results = self._database.search(desc, top_k=self._top_k)
        elapsed = (time.perf_counter() - t0) * 1000.0

        if not results:
            return RetrievalResult(
                candidate_index=-1,
                similarity_score=0.0,
                geodetic_anchor={"lat": 0.0, "lon": 0.0},
                timing_ms=elapsed,
            )

        result = results[0]
        result.descriptor = desc
        result.timing_ms = elapsed
        return result

    def retrieve_top_k(self, query_image: np.ndarray, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve multiple candidates for geometric verification."""
        self._model.load()
        t0 = time.perf_counter()
        desc = self._model.extract_descriptor(query_image)
        try:
            k = int(top_k if top_k is not None else self._top_k)
        except (TypeError, ValueError):
            k = 1
        results = self._database.search(desc, top_k=max(1, k))
        elapsed = (time.perf_counter() - t0) * 1000.0
        for r in results:
            r.descriptor = desc
            r.timing_ms = elapsed
        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_retriever(
    backend: str = "ultravpr",
    weights_path: Optional[Path] = None,
    onnx_path: Optional[Path] = None,
    device: str = "cpu",
    top_k: int = 1,
    verify_top_k: bool = False,
    input_size: int | Tuple[int, int] | None = None,
) -> RetrievalModule:
    """
    Factory for Phase-1 retrieval engine.
    """
    if input_size is None and backend == "ultravpr":
        input_size = ULTRAVPR_INPUT_SIZE
    elif input_size is None and backend == "camp":
        input_size = 384
    elif input_size is None and backend == "megaloc":
        input_size = 322
        
    return RetrievalModule(
        backend=backend,
        top_k=top_k,
        verify_top_k=verify_top_k,
        weights_path=weights_path,
        onnx_path=onnx_path,
        device=device,
        input_size=input_size,
    )
