#!/usr/bin/env python3
"""Deterministic runtime fault injections for telemetry freshness/version gates."""
from __future__ import annotations

import json
import hashlib
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.state_manager import NetworkStateManager


def fresh_manager(ttl_s: float = 60.0) -> NetworkStateManager:
    manager = NetworkStateManager(topology_name="vietnam", max_telemetry_age_s=ttl_s)
    for node in range(manager.num_nodes):
        manager.update_node_telemetry(node, cpu_util=0, ram_util=0, msd_used=0)
    return manager


def main() -> None:
    results = []

    stale = fresh_manager()
    old_snapshot = stale.snapshot()
    expired_at = time.monotonic() - stale.max_telemetry_age_s - 1.0
    with stale._lock:
        stale._resource_telemetry_at = [expired_at] * stale.num_nodes
        stale._resource_field_at = [[expired_at, expired_at, expired_at] for _ in range(stale.num_nodes)]
    stale_snapshot = stale.snapshot()
    accepted, reason = stale.try_reserve(
        0, 1, 10, 5, 1, expected_state_version=old_snapshot.state_version,
    )
    results.append({
        "fault": "telemetry_older_than_ttl",
        "max_age_s": stale.max_telemetry_age_s,
        "stale_nodes": len(stale_snapshot.stale_telemetry_nodes),
        "nodes": stale.num_nodes,
        "reserve_accepted": accepted,
        "reserve_rejection": reason,
        "expected": "stale_telemetry",
    })

    changed = fresh_manager()
    snapshot = changed.snapshot()
    changed.update_node_telemetry(0, cpu_util=99, ram_util=0, msd_used=0)
    accepted, reason = changed.try_reserve(
        0, 1, 10, 5, 1, expected_state_version=snapshot.state_version,
    )
    results.append({
        "fault": "reported_resource_change_after_snapshot",
        "snapshot_version": snapshot.state_version,
        "current_version": changed.snapshot().state_version,
        "reserve_accepted": accepted,
        "reserve_rejection": reason,
        "expected": "stale_state_version",
    })

    between_validation_and_actuation = fresh_manager()
    snapshot = between_validation_and_actuation.snapshot()
    reserved, reserve_error = between_validation_and_actuation.try_reserve(
        0, 1, 10, 5, 1, vnf_name="fault-window",
        expected_state_version=snapshot.state_version,
    )
    admission_version = snapshot.state_version + 1
    between_validation_and_actuation.update_node_telemetry(0, cpu_util=95, ram_util=0, msd_used=0)
    actuation_ready, actuation_error = between_validation_and_actuation.mark_actuation_requested(
        "fault-window", expected_state_version=admission_version,
    )
    released, release_error = between_validation_and_actuation.release_unactuated_reservation("fault-window")
    results.append({
        "fault": "reported_resource_change_after_reservation_before_actuation",
        "reserve_accepted": reserved,
        "reserve_rejection": reserve_error,
        "actuation_marked": actuation_ready,
        "actuation_rejection": actuation_error,
        "reservation_released": released,
        "release_rejection": release_error,
        "expected": "stale_state_version_and_release_before_actuation",
    })

    unreported = fresh_manager()
    snapshot = unreported.snapshot()
    # Model an external resource mutation after local reservation that is not
    # delivered to telemetry; the local manager therefore sees no version change.
    reserved, reserve_error = unreported.try_reserve(
        0, 1, 10, 5, 1, vnf_name="unreported-change",
        expected_state_version=snapshot.state_version,
    )
    actuation_ready, actuation_error = unreported.mark_actuation_requested(
        "unreported-change", expected_state_version=snapshot.state_version + 1,
    )
    results.append({
        "fault": "external_resource_change_unreported_to_manager",
        "reserve_accepted": reserved,
        "reserve_rejection": reserve_error,
        "actuation_marked": actuation_ready,
        "actuation_rejection": actuation_error,
        "expected": "not_detectable_from_local_state",
    })

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/core/state_manager.py",
                ROOT / "src/portal/backend/app/routers/orchestration_router.py",
                ROOT / "tests/test_orchestration_api.py",
            )
        },
        "topology": "vietnam",
        "fault_injection_count": len(results),
        "results": results,
        "scope": "process-local admission and pre-actuation version gate; the unreported-change case documents an observability gap, not a hardware-side experiment",
    }
    output_dir = ROOT / "results/rev1/stale_telemetry_fault_injection"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
