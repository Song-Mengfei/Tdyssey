"""Report a target machine's environment; never install packages or download assets."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="Run tiny torch/Pymunk checks")
    parser.add_argument("--require-cuda", action="store_true", help="Check the cu124 profile and GPU")
    parser.add_argument("--gpc", action="store_true", help="Also check optional GPC package versions")
    parser.add_argument("--dp", action="store_true", help="Also check official Diffusion Policy package versions")
    parser.add_argument("--backend", choices=("cpu", "cu124", "cu128"),
                        help="Locked torch profile; CUDA profiles require --runtime --require-cuda")
    parser.add_argument("--tracking", action="store_true", help="Check optional W&B package version")
    args = parser.parse_args()
    if args.require_cuda and not args.runtime:
        parser.error("--require-cuda requires --runtime")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = list(project["dependencies"])
    profile = args.backend or ("cu124" if args.require_cuda else "cpu")
    if (profile != "cpu") != args.require_cuda:
        parser.error("CUDA backend requires --require-cuda; cpu cannot use --require-cuda")
    requirements.extend(project["optional-dependencies"][profile])
    if args.gpc:
        requirements.extend(project["optional-dependencies"]["gpc"])
    if args.dp:
        requirements.extend(project["optional-dependencies"]["dp"])
    if args.tracking:
        requirements.extend(project["optional-dependencies"]["tracking"])
    report: dict = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "profile_checked": profile,
        "runtime_requested": args.runtime,
        "packages": {},
        "errors": [],
    }
    expected_python = (ROOT / ".python-version").read_text().strip()
    if report["python"] != expected_python:
        report["errors"].append(f"Expected Python {expected_python}")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        report["errors"].append("Reference runtime platform is Linux x86_64")
    for requirement in requirements:
        name, expected = requirement.split("==", 1)
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        report["packages"][name] = {"expected": expected, "actual": actual}
        if actual is None or actual.split("+")[0] != expected:
            report["errors"].append(f"Package mismatch: {name} expected {expected}, got {actual}")

    if args.runtime:
        try:
            import torch

            report["torch_cuda_runtime"] = torch.version.cuda
            report["cuda_available"] = torch.cuda.is_available()
            if args.require_cuda:
                expected_cuda = {"cu124": "12.4", "cu128": "12.8"}[profile]
                if torch.version.cuda != expected_cuda:
                    report["errors"].append(f"Expected the locked CUDA {expected_cuda} torch build")
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable")
                report["gpu"] = torch.cuda.get_device_name(0)
                report["compute_capability"] = list(torch.cuda.get_device_capability(0))
                report["torch_architectures"] = torch.cuda.get_arch_list()
                if report["compute_capability"][0] >= 10 and profile == "cu124":
                    report["errors"].append("Blackwell needs the cu128 profile and actual model smoke checks")
                device = "cuda"
            else:
                if torch.version.cuda is not None:
                    report["errors"].append("Expected the locked CPU-only torch build")
                device = "cpu"
            values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, requires_grad=True)
            loss = (values * values).sum()
            result = float(loss.item())
            report["torch_calculation"] = result
            if result != 30.0:
                raise RuntimeError("Unexpected tensor calculation result")
            loss.backward()
            report["gradient_device"] = str(values.grad.device)
            report["torch_gradient"] = values.grad.tolist()
            if values.grad.tolist() != [2.0, 4.0, 6.0, 8.0]:
                raise RuntimeError("Unexpected tensor gradient result")
            # Load torchvision operators as a basic binary-compatibility check.
            from torchvision.ops import nms

            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)
            scores = torch.tensor([1.0], device=device)
            if nms(boxes, scores, 0.5).tolist() != [0]:
                raise RuntimeError("Unexpected torchvision operator result")
            convolution = torch.nn.Conv1d(4, 8, 3, padding=1).to(device)
            convolution(torch.ones(2, 4, 16, device=device)).square().mean().backward()
            report["convolution_gradient_device"] = str(convolution.weight.grad.device)
        except Exception as exc:
            report["errors"].append(f"Torch/torchvision runtime: {type(exc).__name__}: {exc}")

        try:
            import pymunk

            space = pymunk.Space()
            body = pymunk.Body(1.0, 1.0)
            body.velocity = (1.0, 0.0)
            space.add(body)
            space.step(0.1)
            report["pymunk_body_x"] = body.position.x
            if abs(body.position.x - 0.1) > 1e-6:
                raise RuntimeError("Unexpected Pymunk integration result")
        except Exception as exc:
            report["errors"].append(f"Pymunk runtime: {type(exc).__name__}: {exc}")

    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
