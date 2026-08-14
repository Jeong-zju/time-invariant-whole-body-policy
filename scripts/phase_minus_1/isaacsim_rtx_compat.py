"""Experimental process-local Isaac Sim 5.1 diagnostic for driver 595.x.

This module enables a Khronos Vulkan Profiles layer before Isaac Sim imports
Vulkan. It overrides only ``maxMemoryAllocationSize``, whose UINT64_MAX value
on recent drivers can crash the Isaac Sim 5.1 RTX scene database at startup.

The workaround follows the diagnosis and profile layout posted in NVIDIA's
IsaacSim issue #568. It did not make the Phase -1 RTX 5090 / driver 595.84 host
pass an OmniGibson smoke test, so it must not be treated as a supported runtime
fix. Prefer a host with NVIDIA driver 580.65.06. Import this module before
importing OmniGibson or Isaac Sim.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from ctypes import CDLL
from urllib.request import urlretrieve


PACKAGE_URL = (
    "https://archive.archlinux.org/packages/v/vulkan-profiles/"
    "vulkan-profiles-1.4.341.0-1-x86_64.pkg.tar.zst"
)
PACKAGE_SHA256 = "d52d72259b6c8911c9e3b41985e014cb76954f013b87af45f5ef1fcff1126131"
LAYER_SHA256 = "99f20d7af6e073eac74a04308be01a3a81505629cff6dc066a7d3282b4dce533"
PROFILE_NAME = "VP_ISAAC_SIM_5_1_RTX_COMPAT"
MAX_ALLOCATION_BYTES = 4 * 1024 * 1024 * 1024 - 2 * 1024 * 1024
CACHE_DIR = Path(
    os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
) / "isaacsim-rtx-compat"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepend_env(name: str, path: Path) -> None:
    value = str(path)
    current = os.environ.get(name, "")
    parts = current.split(":") if current else []
    if value not in parts:
        os.environ[name] = f"{value}:{current}" if current else value


def install_layer() -> Path:
    override = os.environ.get("RTX_VULKAN_COMPAT_LAYER_PATH")
    if override:
        layer_path = Path(override).expanduser().resolve()
        if not layer_path.is_file():
            raise RuntimeError(
                f"RTX_VULKAN_COMPAT_LAYER_PATH does not exist: {layer_path}"
            )
        try:
            CDLL(str(layer_path))
        except OSError as error:
            raise RuntimeError(
                f"Vulkan Profiles layer has unresolved runtime dependencies: {error}"
            ) from error
        return layer_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    package_path = CACHE_DIR / "vulkan-profiles.pkg.tar.zst"
    layer_path = CACHE_DIR / "libVkLayer_khronos_profiles.so"

    if not layer_path.exists() or sha256sum(layer_path) != LAYER_SHA256:
        if not package_path.exists() or sha256sum(package_path) != PACKAGE_SHA256:
            download_path = CACHE_DIR / "vulkan-profiles.download"
            urlretrieve(PACKAGE_URL, download_path)
            if sha256sum(download_path) != PACKAGE_SHA256:
                download_path.unlink(missing_ok=True)
                raise RuntimeError("Vulkan Profiles package checksum mismatch")
            download_path.replace(package_path)

        layer_bytes = subprocess.run(
            [
                "tar",
                "--zstd",
                "-xOf",
                str(package_path),
                "usr/lib/libVkLayer_khronos_profiles.so",
            ],
            check=True,
            capture_output=True,
        ).stdout
        layer_path.write_bytes(layer_bytes)
        layer_path.chmod(0o755)
        if sha256sum(layer_path) != LAYER_SHA256:
            layer_path.unlink(missing_ok=True)
            raise RuntimeError("Vulkan Profiles layer checksum mismatch")

    try:
        CDLL(str(layer_path))
    except OSError as error:
        raise RuntimeError(
            f"Vulkan Profiles layer has unresolved runtime dependencies: {error}"
        ) from error
    return layer_path


def enable() -> None:
    layer_path = install_layer()
    profile_dir = CACHE_DIR / "profiles"
    data_root = CACHE_DIR / "xdg-data"
    manifest_dir = data_root / "vulkan" / "implicit_layer.d"
    profile_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "$schema": "https://schema.khronos.org/vulkan/profiles-0.8-latest.json#",
        "capabilities": {
            "ISAAC_SIM_RTX_COMPAT": {
                "properties": {
                    "VkPhysicalDeviceMaintenance3Properties": {
                        "maxMemoryAllocationSize": MAX_ALLOCATION_BYTES
                    }
                }
            }
        },
        "profiles": {
            PROFILE_NAME: {
                "version": 1,
                "api-version": "1.1.0",
                "label": "Isaac Sim 5.1 RTX driver compatibility",
                "description": "Cap the Vulkan maintenance3 allocation-size property.",
                "contributors": {"local": {"github": "https://github.com/yushijinhun"}},
                "history": [
                    {
                        "revision": 1,
                        "date": "2026-08-13",
                        "author": "local",
                        "comment": "Generated for Phase -1 deployment.",
                    }
                ],
                "capabilities": ["ISAAC_SIM_RTX_COMPAT"],
            }
        },
    }
    (profile_dir / f"{PROFILE_NAME}.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "file_format_version": "1.2.1",
        "layer": {
            "name": "VK_LAYER_KHRONOS_profiles",
            "type": "GLOBAL",
            "library_path": str(layer_path),
            "api_version": "1.4.341",
            "implementation_version": "1",
            "description": "Khronos Profiles layer",
            "enable_environment": {"RTX_VULKAN_COMPAT_ENABLE_LAYER": "1"},
            "disable_environment": {"RTX_VULKAN_COMPAT_DISABLE_LAYER": ""},
        },
    }
    (manifest_dir / "VkLayer_KHRONOS_profiles.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    os.environ["RTX_VULKAN_COMPAT_ENABLE_LAYER"] = "1"
    os.environ["VK_KHRONOS_PROFILES_PROFILE_NAME"] = PROFILE_NAME
    os.environ["VK_KHRONOS_PROFILES_SIMULATE_CAPABILITIES"] = "SIMULATE_PROPERTIES_BIT"
    os.environ["VK_KHRONOS_PROFILES_DEBUG_REPORTS"] = "DEBUG_REPORT_ERROR_BIT"
    os.environ["RTX_VULKAN_COMPAT_ACTIVE"] = "1"
    prepend_env("VK_KHRONOS_PROFILES_PROFILE_DIRS", profile_dir)
    prepend_env("VK_ADD_IMPLICIT_LAYER_PATH", manifest_dir)
    prepend_env("XDG_DATA_DIRS", data_root)
    print(
        "Isaac Sim RTX compatibility profile enabled: "
        f"maxMemoryAllocationSize={MAX_ALLOCATION_BYTES}, layer={layer_path}",
        flush=True,
    )


enable()
