#!/usr/bin/env python3
"""Probe local TT host for model-bringup routing.

Distinguishes:
  - runtime_chip_count  : xr.global_runtime_device_count() — SPMD / mesh size (e.g. 8)
  - visible_board_count : TT_VISIBLE_DEVICES valid IDs (e.g. 4 on n300 llmbox → 0,1,2,3)

On n300 llmbox: 4 boards × 2 chips/board = 8 runtime chips, but TT_VISIBLE_DEVICES
must enumerate BOARDS (0-3), not chips (0-7).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SINGLE_CHIP_BRINGUP_ARCHES = frozenset({"n150", "p150"})
MULTICHIP_BRINGUP_ARCHES = frozenset(
    {"n300", "n300-llmbox", "galaxy-wh-6u", "lb-blackhole", "qb2-blackhole"}
)

# Observed silicon family -> single-chip bringup arch. Attribution comes from the
# SILICON, never the host name: one chip of a qb2-blackhole is a Blackhole, so it
# reports p150 — never n150.
SILICON_TO_SINGLE_CHIP_ARCH = {"blackhole": "p150", "wormhole": "n150"}

SINGLE_CHIP_HOST_NOTE = (
    "A component runs on whatever mesh its graph compiles to; the host's chip count "
    "does not gate it. Run the unsharded test on the box you have — if it passes, it "
    "passed on the mesh it compiled to. Do not set TT_VISIBLE_DEVICES for single-device "
    "runs. Record the arch from the observed silicon (blackhole->p150, wormhole->n150), "
    "not from the host name."
)
MULTICHIP_HOST_NOTE = (
    "Multichip: mesh uses runtime_chip_count (XLA). TT_VISIBLE_DEVICES uses visible_board_count "
    "(tt-smi resettable boards). On n300 llmbox often chip_count=8, TT_VISIBLE_DEVICES=0,1,2,3."
)
TT_SMI_INSTALL_HINT = (
    "tt-smi not found — install: git clone https://github.com/tenstorrent/tt-smi && "
    "cd tt-smi && pip install .  Then: tt-smi -ls (boards), tt-smi -r (reset)."
)


def visible_devices_for_count(n: int) -> str:
    if n <= 0:
        return ""
    return ",".join(str(i) for i in range(n))


def parse_visible_device_indices(vis: str) -> list[int] | None:
    vis = (vis or "").strip()
    if not vis:
        return None
    if "-" in vis and "," not in vis:
        parts = vis.split("-", 1)
        try:
            lo, hi = int(parts[0]), int(parts[1])
            return list(range(lo, hi + 1))
        except ValueError:
            return None
    out: list[int] = []
    for part in vis.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out or None


def probe_tt_smi_boards() -> dict:
    """Parse `tt-smi -ls` for resettable board / UMD IDs (TT_VISIBLE_DEVICES range)."""
    result: dict = {
        "available": False,
        "visible_board_count": None,
        "resettable_umd_ids": [],
        "all_umd_ids": [],
        "error": None,
        "source": "tt-smi -ls",
    }
    try:
        proc = subprocess.run(
            ["tt-smi", "-ls"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        result["error"] = "tt-smi not found on PATH"
        result["install_hint"] = TT_SMI_INSTALL_HINT
        return result
    except subprocess.TimeoutExpired:
        result["error"] = "tt-smi -ls timed out"
        return result

    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout or "tt-smi failed").strip()[:500]
        return result

    result["available"] = True
    text = proc.stdout or ""

    # Parse each table independently — never merge "all boards" UMD rows with
    # "resettable" board rows (n300 llmbox: 8 chip rows + 4 board rows → 12 bug).
    section: str | None = None
    reset_ids: list[int] = []
    all_ids: list[int] = []
    id_re = re.compile(r"^\s*[│|]\s*(\d+)\s+[│|]")

    for line in text.splitlines():
        if "All available boards" in line:
            section = "all"
            continue
        if "Boards that can be reset" in line:
            section = "reset"
            continue
        m = id_re.match(line.replace("┃", "│"))
        if not m or section is None:
            continue
        uid = int(m.group(1))
        if section == "reset":
            reset_ids.append(uid)
        else:
            all_ids.append(uid)

    reset_ids = list(dict.fromkeys(reset_ids))
    all_ids = list(dict.fromkeys(all_ids))
    result["resettable_umd_ids"] = reset_ids
    result["all_umd_ids"] = all_ids

    if reset_ids:
        result["visible_board_count"] = len(reset_ids)
        result["note"] = "resettable board IDs from tt-smi -ls"
    elif all_ids:
        n = len(all_ids)
        if n == 8:
            result["visible_board_count"] = 4
            result["note"] = "8 UMD chip IDs → 4 boards (n300 L+R); no reset section found"
        else:
            result["visible_board_count"] = n
            result["note"] = f"{n} UMD IDs listed; no reset section found"

    return result


def probe_runtime() -> dict:
    out: dict = {
        "tt_xla_arch_env": os.environ.get("TT_XLA_ARCH", ""),
        "tt_visible_devices_env": os.environ.get("TT_VISIBLE_DEVICES", ""),
        "runtime_chip_count": 0,
        "device_arch": "",
        "probe_error": None,
    }
    try:
        import torch_xla.runtime as xr

        out["runtime_chip_count"] = int(xr.global_runtime_device_count())
        attrs = xr.global_runtime_device_attributes()
        if attrs:
            out["device_arch"] = str(attrs[0].get("device_arch", ""))
    except Exception as exc:  # noqa: BLE001
        out["probe_error"] = str(exc)
    # Back-compat alias used by older skill text
    out["device_count"] = out["runtime_chip_count"]
    return out


def resolve_visible_board_count(runtime: dict, tt_smi: dict) -> tuple[int, str, str]:
    """Return (visible_board_count, tt_visible_devices string, provenance note)."""
    chip_count = int(runtime.get("runtime_chip_count") or 0)
    tt_smi_n = tt_smi.get("visible_board_count")
    raw_n = tt_smi_n

    if tt_smi_n is not None and tt_smi_n > 0:
        board_count = int(tt_smi_n)
        prov = f"tt-smi -ls resettable boards ({board_count})"
    elif chip_count == 1:
        board_count = 1
        prov = "runtime_chip_count==1 → single board"
    elif chip_count > 1 and chip_count % 2 == 0 and "wormhole" in (
        runtime.get("device_arch") or ""
    ).lower():
        board_count = chip_count // 2
        prov = (
            f"heuristic: runtime_chip_count={chip_count}, 2 chips/board → "
            f"{board_count} boards (run tt-smi -ls to confirm)"
        )
    else:
        board_count = chip_count
        prov = f"fallback: visible_board_count=runtime_chip_count={chip_count}"

    # Sanity: board count cannot exceed chip count (catches merged-table double-count).
    if chip_count > 0 and board_count > chip_count:
        corrected = chip_count // 2 if chip_count % 2 == 0 else chip_count
        prov = (
            f"corrected visible_board_count {board_count}→{corrected} "
            f"(boards cannot exceed runtime_chip_count={chip_count}; "
            f"tt-smi raw={raw_n})"
        )
        board_count = corrected

    return board_count, visible_devices_for_count(board_count), prov


def resolve_single_chip_arch(host: dict) -> tuple[str, str]:
    """Map the OBSERVED silicon to a single-chip bringup arch.

    Returns (arch, provenance). Empty arch means the silicon family could not be
    identified, so a single-chip result could not be attributed honestly.
    """
    dev_arch = (host.get("device_arch") or "").lower()
    for silicon, arch in SILICON_TO_SINGLE_CHIP_ARCH.items():
        if silicon in dev_arch:
            return arch, f"device_arch={dev_arch!r} -> {arch}"
    env_arch = (host.get("tt_xla_arch_env") or "").strip()
    if env_arch in SINGLE_CHIP_BRINGUP_ARCHES:
        return env_arch, f"TT_XLA_ARCH={env_arch!r} (device_arch unavailable)"
    return "", (
        f"silicon family unknown (device_arch={dev_arch!r}, "
        f"TT_XLA_ARCH={env_arch!r} is not a single-chip arch)"
    )


def single_chip_bringup_eligible(host: dict) -> tuple[bool, str]:
    """Host chip count does not gate a single-device component.

    The component runs on whatever mesh its graph compiles to, so there is
    nothing here to decide: run it on the box you have. The old rule refused
    runtime_chip_count != 1, which blocked components that fit one chip (a
    0.34 GB VAE on a 4-chip quiet box) purely because the host had spare silicon.
    Only "no TT device at all" is a real blocker.
    """
    if host.get("probe_error"):
        return False, f"host probe failed: {host['probe_error']}"
    chips = int(host.get("runtime_chip_count") or 0)
    if chips < 1:
        return False, (
            f"single-chip bringup requires at least one TT device "
            f"(runtime_chip_count={chips})."
        )
    return True, ""


def multichip_bringup_eligible(host: dict, min_chips: int = 2) -> tuple[bool, str]:
    if host.get("probe_error"):
        return False, f"host probe failed: {host['probe_error']}"
    chips = int(host.get("runtime_chip_count") or 0)
    if chips < min_chips:
        return False, (
            f"multichip bringup requires runtime_chip_count>={min_chips} (got {chips}). "
            "Switch to a multichip host and re-probe."
        )
    if chips == 1:
        return False, "multichip bringup cannot run when runtime_chip_count==1."
    return True, ""


def check_tt_visible_devices_env(host: dict) -> tuple[bool, str]:
    vis = (host.get("tt_visible_devices_env") or "").strip()
    if not vis:
        return True, ""
    indices = parse_visible_device_indices(vis)
    board_count = int(host.get("visible_board_count") or 0)
    chips = int(host.get("runtime_chip_count") or 0)
    if indices is None or board_count <= 0:
        return True, ""
    max_idx = max(indices)
    if max_idx >= board_count:
        return False, (
            f"TT_VISIBLE_DEVICES={vis!r} uses index {max_idx} but visible_board_count="
            f"{board_count} (valid IDs 0..{board_count - 1}). "
            f"runtime_chip_count={chips} is for mesh/SPSD — not TT_VISIBLE_DEVICES. "
            f"Use {host.get('recommended_tt_visible_devices', {}).get('multichip_bringup', '0,1,2,3')}."
        )
    return True, ""


def valid_tp_degrees(chip_count: int) -> list[int]:
    """Powers of 2 that divide chip_count and are >= 2 (e.g. 8 → [2, 4, 8])."""
    if chip_count < 2:
        return []
    out: list[int] = []
    d = 2
    while d <= chip_count:
        if chip_count % d == 0:
            out.append(d)
        d *= 2
    return out


def can_run_component(
    host: dict,
    parallelism_mode: str,
    expected_mesh_chips: int | None = None,
) -> tuple[bool, str]:
    mode = (parallelism_mode or "").strip().lower()
    if mode in ("single_device", "single-device", "single"):
        ok, reason = single_chip_bringup_eligible(host)
        if not ok:
            return False, reason
        return True, ""

    if mode in ("tensor_parallel", "tensor-parallel", "tp", "multichip"):
        if not host.get("tt_smi", {}).get("available"):
            err = host.get("tt_smi", {}).get("error") or "tt-smi unavailable"
            hint = host.get("tt_smi", {}).get("install_hint") or TT_SMI_INSTALL_HINT
            return False, f"{err}. {hint}"
        mc_ok, mc_reason = multichip_bringup_eligible(host)
        if not mc_ok:
            return False, mc_reason
        if not host.get("tt_visible_devices_env_valid", True):
            return False, host.get("tt_visible_devices_env_skip_reason") or (
                "invalid TT_VISIBLE_DEVICES for visible_board_count"
            )
        if expected_mesh_chips and expected_mesh_chips > 0:
            chips = int(host.get("runtime_chip_count") or 0)
            degrees = valid_tp_degrees(chips)
            if expected_mesh_chips not in degrees:
                return False, (
                    f"mesh_chip_count={expected_mesh_chips} not in valid_tp_degrees={degrees} "
                    f"for runtime_chip_count={chips}. Valid degrees: {degrees}. "
                    "Bringup ladder uses {2,4,8,32} — change host or update scaffold mesh_shape."
                )
            exp_ok, exp_reason = check_expected_mesh_chips(host, expected_mesh_chips)
            if not exp_ok:
                return False, exp_reason
        return True, ""

    return True, ""


def check_expected_mesh_chips(host: dict, expected: int | None) -> tuple[bool, str]:
    if expected is None or expected <= 0:
        return True, ""
    chips = int(host.get("runtime_chip_count") or 0)
    if chips == expected:
        return True, ""
    return False, (
        f"mesh chip mismatch: runtime_chip_count={chips} but scaffold/promotion "
        f"expected mesh_chip_count={expected}. Update mesh_shape / change host."
    )


def classify(runtime: dict, tt_smi: dict) -> dict:
    chips = int(runtime.get("runtime_chip_count") or 0)
    env_arch = (runtime.get("tt_xla_arch_env") or "").strip()
    dev_arch = (runtime.get("device_arch") or "").lower()

    board_count, multichip_vis, vis_prov = resolve_visible_board_count(runtime, tt_smi)
    chips_per_board = (
        chips // board_count if board_count > 0 and chips >= board_count else None
    )

    if chips <= 1:
        host_tier = "dedicated_single_chip"
        if "blackhole" in dev_arch:
            inferred = "p150"
        elif "wormhole" in dev_arch:
            inferred = "n150"
        elif env_arch in SINGLE_CHIP_BRINGUP_ARCHES:
            inferred = env_arch
        else:
            inferred = env_arch or "unknown"
    else:
        host_tier = "multichip_fabric"
        if env_arch in MULTICHIP_BRINGUP_ARCHES | SINGLE_CHIP_BRINGUP_ARCHES:
            inferred = env_arch
        elif env_arch:
            inferred = env_arch
        elif chips >= 8:
            inferred = "n300-llmbox" if "wormhole" in dev_arch else "lb-blackhole"
        elif chips >= 4:
            inferred = "n300-llmbox" if "wormhole" in dev_arch else "lb-blackhole"
        else:
            inferred = "n300" if "wormhole" in dev_arch else "lb-blackhole"

    host = {
        **runtime,
        "tt_smi": tt_smi,
        "host_tier": host_tier,
        "inferred_bringup_arch": inferred,
        "visible_board_count": board_count,
        "mesh_chip_count": chips,
        "chips_per_board": chips_per_board,
        "board_vs_chip_note": (
            f"runtime_chip_count={chips} (XLA mesh / SPMD). "
            f"visible_board_count={board_count} → TT_VISIBLE_DEVICES={multichip_vis}. "
            "These differ on n300 llmbox (e.g. 8 chips, 4 boards). "
            f"Provenance: {vis_prov}."
        ),
        "recommended_tt_visible_devices": {
            "single_chip_bringup": "0",
            "multichip_bringup": multichip_vis,
        },
        "policy": {
            "single_chip": SINGLE_CHIP_HOST_NOTE,
            "multichip": MULTICHIP_HOST_NOTE,
        },
    }

    sc_ok, sc_reason = single_chip_bringup_eligible(host)
    mc_ok, mc_reason = multichip_bringup_eligible(host)
    vis_ok, vis_reason = check_tt_visible_devices_env(host)

    host["can_run_single_chip_bringup"] = sc_ok
    host["single_chip_skip_reason"] = sc_reason if not sc_ok else ""
    sc_arch, sc_arch_prov = resolve_single_chip_arch(host)
    host["single_chip_arch"] = sc_arch
    host["single_chip_arch_provenance"] = sc_arch_prov
    host["can_run_multichip_bringup"] = mc_ok and vis_ok
    host["multichip_skip_reason"] = mc_reason if not mc_ok else (vis_reason if not vis_ok else "")
    host["tt_visible_devices_env_valid"] = vis_ok
    host["tt_visible_devices_env_skip_reason"] = vis_reason if not vis_ok else ""
    host["valid_tp_degrees"] = valid_tp_degrees(chips)
    if tt_smi.get("visible_board_count") != host.get("visible_board_count"):
        host["visible_board_count_tt_smi_raw"] = tt_smi.get("visible_board_count")
    if not tt_smi.get("available"):
        host["tt_smi_missing"] = True
        host["tt_smi_install_hint"] = tt_smi.get("install_hint") or TT_SMI_INSTALL_HINT

    return host


def filter_arch_queue(
    arch_queue: list[str],
    host: dict,
    *,
    p150_only: bool = False,
) -> tuple[list[str], list[dict]]:
    skipped: list[dict] = []
    if not host.get("can_run_single_chip_bringup"):
        reason = host.get("single_chip_skip_reason") or "single-chip bringup not allowed"
        for arch in arch_queue:
            skipped.append({"arch": arch, "reason": reason})
        return [], skipped

    runnable: list[str] = []
    dev_arch = (host.get("device_arch") or "").lower()
    for arch in arch_queue:
        if p150_only and arch == "n150":
            skipped.append({"arch": arch, "reason": "component is p150_only"})
            continue
        if arch == "p150" and "wormhole" in dev_arch and "blackhole" not in dev_arch:
            skipped.append({"arch": arch, "reason": "wormhole host; p150 needs Blackhole"})
            continue
        if arch == "n150" and "blackhole" in dev_arch:
            skipped.append({"arch": arch, "reason": "blackhole host; n150 needs Wormhole"})
            continue
        runnable.append(arch)
    return runnable, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch-queue", default="")
    parser.add_argument("--p150-only", action="store_true")
    parser.add_argument(
        "--min-multichip-devices",
        type=int,
        default=2,
        help="Minimum runtime_chip_count for multichip",
    )
    parser.add_argument(
        "--expected-mesh-chips",
        type=int,
        default=0,
        help="Expected runtime_chip_count for mesh (from scaffold_multichip.json)",
    )
    parser.add_argument(
        "--parallelism-mode",
        default="",
        help="Component mode: single_device | tensor_parallel (from weight_fit.json)",
    )
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    runtime = probe_runtime()
    tt_smi = probe_tt_smi_boards()
    host = classify(runtime, tt_smi)

    mc_ok, mc_reason = multichip_bringup_eligible(host, args.min_multichip_devices)
    host["can_run_multichip_bringup"] = mc_ok and host.get("tt_visible_devices_env_valid", True)
    if not mc_ok:
        host["multichip_skip_reason"] = mc_reason

    if args.expected_mesh_chips:
        exp_ok, exp_reason = check_expected_mesh_chips(host, args.expected_mesh_chips)
        host["expected_mesh_chip_count"] = args.expected_mesh_chips
        host["mesh_chip_count_matches_expected"] = exp_ok
        if not exp_ok:
            host["can_run_multichip_bringup"] = False
            host["multichip_skip_reason"] = exp_reason

    if args.parallelism_mode.strip():
        mode = args.parallelism_mode.strip()
        exp = args.expected_mesh_chips or None
        comp_ok, comp_reason = can_run_component(host, mode, exp)
        host["parallelism_mode_checked"] = mode
        host["can_run_component"] = comp_ok
        host["component_skip_reason"] = comp_reason if not comp_ok else ""

    if args.arch_queue.strip():
        queue = [a.strip() for a in args.arch_queue.split(",") if a.strip()]
        runnable, skipped = filter_arch_queue(queue, host, p150_only=args.p150_only)
        host["planned_arch_queue"] = queue
        host["runnable_arch_queue"] = runnable
        host["skipped_archs"] = skipped

    text = json.dumps(host, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
