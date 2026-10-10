"""Check the installed training backend with a real forward/backward operation."""

import argparse
import json
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        parser.error("PyTorch is missing. Follow docs/NEURAL_BASELINES.md.")
    available = torch.cuda.is_available()
    if args.require_gpu and not available:
        parser.error("No CUDA/ROCm GPU is visible. A CPU wheel cannot use the RX 7800 XT; see the WSL setup guide.")
    device = torch.device("cuda" if available else "cpu")
    matrix = torch.randn(64, 64, device=device, requires_grad=True)
    loss = (matrix @ matrix).square().mean()
    loss.backward()
    if not torch.isfinite(matrix.grad).all():
        parser.error("Backend produced non-finite gradients")
    if available:
        torch.cuda.synchronize()
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
                      "gpu_available": available,
                      "device": torch.cuda.get_device_name(0) if available else "CPU",
                      "forward_backward": "passed"}, indent=2))


if __name__ == "__main__":
    main()
