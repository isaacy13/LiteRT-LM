#!/usr/bin/env python3
"""Reject missing LFS objects and incompatible Apple runtime binaries early."""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = (
    "libLiteRt.dylib",
    "libGemmaModelConstraintProvider.dylib",
    "libLiteRtMetalAccelerator.dylib",
    "libLiteRtTopKMetalSampler.dylib",
)
PLATFORMS = {"ios_arm64": "IOS", "ios_sim_arm64": "IOSSIMULATOR"}


def output(*arguments: str) -> str:
    return subprocess.check_output(arguments, text=True).strip()


def verify() -> list[dict[str, str]]:
    results = []
    for directory, platform in PLATFORMS.items():
        for name in LIBRARIES:
            path = ROOT / "prebuilt" / directory / name
            with path.open("rb") as binary:
                if binary.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
                    raise RuntimeError(f"Unfetched LFS binary: {path}")
            architectures = output("lipo", "-archs", str(path)).split()
            if architectures != ["arm64"]:
                raise RuntimeError(f"Expected arm64 at {path}; got {architectures}")
            build = output("vtool", "-show-build", str(path))
            recorded = [line.split()[1] for line in build.splitlines()
                        if line.strip().startswith("platform ")]
            if recorded != [platform]:
                raise RuntimeError(f"Expected {platform} at {path}; got {recorded}")
            if name == "libLiteRt.dylib":
                symbols = output("nm", "-gU", str(path)).split()
                if "_kLiteRtRuntimeBuiltin" not in symbols:
                    raise RuntimeError(f"Missing LiteRT runtime ABI at {path}")
            results.append({"library": str(path.relative_to(ROOT)),
                            "architecture": "arm64", "platform": platform})
    return results


if __name__ == "__main__":
    print(json.dumps({"appleRuntimeSlices": verify()}, indent=2))
