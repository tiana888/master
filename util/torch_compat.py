import re
from typing import Optional, Set, Tuple

import torch


def _parse_cuda_version(version: Optional[str]) -> Tuple[int, int]:
    if not version:
        return (0, 0)
    parts = version.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return (major, minor)


def _parse_sm_tag(tag: str) -> Optional[Tuple[int, int]]:
    match = re.fullmatch(r"sm_(\d+)(\d)", tag)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)))


def _min_cuda_for_capability(capability: Tuple[int, int]) -> Tuple[int, int]:
    if capability >= (8, 9):
        # Ada / Hopper class GPUs need a newer CUDA runtime even if PyTorch's
        # arch list only exposes lower same-major cubins such as sm_86.
        return (11, 8)
    return (0, 0)


def ensure_supported_cuda_runtime(device: str) -> None:
    device_str = str(device)
    if not device_str.startswith("cuda"):
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested ({0}), but torch.cuda.is_available() is False.".format(device_str)
        )

    torch_device = torch.device(device_str)
    device_index = torch_device.index
    if device_index is None:
        try:
            device_index = torch.cuda.current_device()
        except Exception:
            device_index = 0

    major, minor = torch.cuda.get_device_capability(device_index)
    required_tags = {"sm_{0}{1}".format(major, minor), "compute_{0}{1}".format(major, minor)}

    try:
        supported_tags: Set[str] = set(torch.cuda.get_arch_list())
    except Exception:
        supported_tags = set()

    if required_tags & supported_tags:
        return

    cuda_version = _parse_cuda_version(torch.version.cuda)
    min_cuda = _min_cuda_for_capability((major, minor))
    supported_sms = [item for item in (_parse_sm_tag(tag) for tag in supported_tags) if item is not None]

    # Exact sm_XY is not always listed for newer GPUs. For example, an Ada GPU
    # may still run correctly with a modern CUDA runtime and lower same-major
    # kernels such as sm_86.
    has_same_major_fallback = any(sm_major == major and sm_minor <= minor for sm_major, sm_minor in supported_sms)
    if has_same_major_fallback and cuda_version >= min_cuda:
        return

    gpu_name = torch.cuda.get_device_name(device_index)
    supported_arches = ", ".join(sorted(supported_tags)) if supported_tags else "unknown"
    raise RuntimeError(
        "Installed PyTorch CUDA runtime is not compatible with {0} ({1}). "
        "Current runtime: torch {2}, CUDA {3}, supported arch list [{4}]. "
        "This often surfaces as opaque failures such as "
        "`CUBLAS_STATUS_INTERNAL_ERROR` during SDXL/IP-Adapter inference. "
        "Please upgrade to a newer PyTorch build with CUDA {5}.{6} or newer "
        "for this GPU, or switch to a supported GPU/runtime combination.".format(
            gpu_name,
            "sm_{0}{1}".format(major, minor),
            torch.__version__,
            torch.version.cuda,
            supported_arches,
            min_cuda[0],
            min_cuda[1],
        )
    )
