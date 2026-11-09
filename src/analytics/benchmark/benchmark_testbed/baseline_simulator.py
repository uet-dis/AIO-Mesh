from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

from .exporters import write_records_csv
from .latency_model import modeled_service_latency_ms
from .metrics import summarize_records
from .models import RequestRecord, RequestSpec, ScenarioDefinition
from .runner import stable_name_offset


BASELINE_ALGORITHMS = [
    "jo_vppm",
    "saf_h",
    "heuristic",
    "greedy",
    "decoupled",
    "greedy_protected",
    "decoupled_protected",
]


class BaselineSimulator:
    """Local baseline runner using the same request stream as runtime scenarios."""

    def __init__(self, seed: int, steps: int = 0) -> None:
        self.seed = seed
        self.steps = steps

    def run(
        self,
        scenario: ScenarioDefinition,
        algorithm: str,
        output_dir: Path,
    ) -> Tuple[List[RequestRecord], object]:
        if algorithm not in BASELINE_ALGORITHMS:
            raise ValueError(f"Unsupported baseline algorithm: {algorithm}")

        import random
        from src.ai.heuristic import (
            HardConstraintError,
            get_decoupled_action,
            get_resilience_safe_action,
        )
        from src.ai.dgrl_agent import get_dgrl_agent
        from src.core.state_manager import NetworkStateManager

        rng = random.Random(self.seed + stable_name_offset(scenario.name))
        state_manager = NetworkStateManager()
        dgrl_agent = get_dgrl_agent() if algorithm == "jo_vppm" else None
        if dgrl_agent is not None:
            warmup_req = scenario.request_factory(0, self.steps or scenario.default_steps, rng)
            if warmup_req is not None:
                dgrl_agent.get_action(state_manager, warmup_req)
            else:
                dgrl_agent.get_action(
                    state_manager,
                    RequestSpec(service_type="Data", cpu_req=8, ram_req=4, msd_req=1),
                )
            rng = random.Random(self.seed + stable_name_offset(scenario.name))
        records: List[RequestRecord] = []
        active: List[Dict[str, object]] = []
        steps = self.steps or scenario.default_steps

        for step in range(steps):
            remaining: List[Dict[str, object]] = []
            for reservation in active:
                if int(reservation["expire_step"]) <= step:
                    state_manager.free_resources(
                        int(reservation["v_place"]),
                        int(reservation["v_route"]),
                        float(reservation["cpu_req"]),
                        float(reservation["ram_req"]),
                        int(reservation["msd_req"]),
                    )
                else:
                    remaining.append(reservation)
            active = remaining

            spec = scenario.request_factory(step, steps, rng)
            if spec is None:
                continue

            start = time.perf_counter()
            accepted = False
            reject_reason = ""
            msd_violation = False
            v_place = None
            v_route = None
            method_used = algorithm
            failure = ""
            admission_reason = ""

            if spec.alert_flag:
                state_manager.set_forecast_alert(True)

            try:
                snapshot = state_manager.snapshot()
                if algorithm == "jo_vppm":
                    decision = dgrl_agent.get_action(state_manager, spec)
                    v_place = int(decision.choice.v_place)
                    v_route = int(decision.choice.v_route)
                    method_used = "JO-VPPM" if decision.model_loaded else "JO-VPPM fallback"
                elif algorithm == "saf_h":
                    v_place, v_route = self._saf_h_choice(snapshot, spec)
                    method_used = "SAF-H (shared admission)"
                elif algorithm == "heuristic":
                    decision = get_resilience_safe_action(snapshot, spec)
                    v_place = int(decision.v_place)
                    v_route = int(decision.v_route)
                    method_used = "Resilience Heuristic"
                elif algorithm in {"decoupled", "decoupled_protected"}:
                    v_place, v_route = self._unsafe_decoupled_choice(snapshot, spec)
                    method_used = "Decoupled Heuristic"
                    if algorithm.endswith("_protected"):
                        method_used += "+Admission"
                elif algorithm in {"greedy", "greedy_protected"}:
                    v_place, v_route = self._greedy_choice(snapshot, spec)
                    method_used = "Greedy"
                    if algorithm.endswith("_protected"):
                        method_used += "+Admission"

                sid_depth = state_manager.required_sid_depth(v_place, v_route, spec.msd_req)
                failures = self._constraint_failures(snapshot, v_place, v_route, spec, sid_depth)
                failure = failures[0] if failures else ""
                if algorithm in {"greedy", "decoupled"}:
                    if failure:
                        reject_reason = failure
                        admission_reason = f"candidate_infeasible:{'|'.join(failures)}"
                    else:
                        accepted, error = state_manager.try_reserve(
                            v_place,
                            v_route,
                            spec.cpu_req,
                            spec.ram_req,
                            sid_depth,
                            vnf_name=f"{algorithm}-{scenario.name}-{step}",
                        )
                        if not accepted:
                            reject_reason = error or "reserve_failed"
                            admission_reason = reject_reason
                else:
                    accepted, error = state_manager.try_reserve(
                        v_place,
                        v_route,
                        spec.cpu_req,
                        spec.ram_req,
                        sid_depth,
                        vnf_name=f"{algorithm}-{scenario.name}-{step}",
                    )
                    if not accepted:
                        reject_reason = error or "reserve_failed"
                        admission_reason = reject_reason
            except HardConstraintError as exc:
                reject_reason = str(exc)
                admission_reason = reject_reason
            except Exception as exc:
                reject_reason = f"error:{exc}"
                admission_reason = reject_reason

            latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
            service_latency_ms = 0.0
            if accepted and v_place is not None and v_route is not None:
                snap_after = state_manager.snapshot()
                service_latency_ms = modeled_service_latency_ms(
                    v_place,
                    v_route,
                    state_manager.required_sid_depth(v_place, v_route, spec.msd_req),
                    float(snap_after.state[int(v_place), 0]) / 100.0,
                    float(snap_after.state[int(v_route), 0]) / 100.0,
                )
            if accepted and v_place is not None and v_route is not None:
                active.append(
                    {
                        "expire_step": step + spec.ttl_steps,
                        "v_place": v_place,
                        "v_route": v_route,
                        "cpu_req": spec.cpu_req,
                        "ram_req": spec.ram_req,
                        "msd_req": state_manager.required_sid_depth(v_place, v_route, spec.msd_req),
                    }
                )

            records.append(
                RequestRecord(
                    algorithm=algorithm,
                    scenario=scenario.name,
                    step=step,
                    request_id=f"{scenario.name}:{self.seed}:{step}",
                    seed=self.seed,
                    accepted=accepted,
                    status_code=200 if accepted else 409,
                    decision_latency_ms=latency_ms,
                    service_latency_ms=round(service_latency_ms, 3),
                    service_type=spec.service_type,
                    cpu_req=spec.cpu_req,
                    ram_req=spec.ram_req,
                    msd_req=spec.msd_req,
                    source_node=spec.source_node,
                    destination_node=spec.destination_node,
                    alert_flag=spec.alert_flag or spec.is_ddos_spike,
                    method_used=method_used,
                    hybrid_branch="",
                    vnf_name=f"{algorithm}-{scenario.name}-{step}" if accepted else "",
                    placement_node_id=v_place,
                    routing_node_id=v_route,
                    sid_count=state_manager.required_sid_depth(v_place, v_route, spec.msd_req) if accepted else 0,
                    msd_violation="msd_violation" in failures,
                    admitted_msd_violation=msd_violation,
                    candidate_hard_infeasible=bool(failure),
                    candidate_cpu_infeasible="cpu_capacity" in failures,
                    candidate_ram_infeasible="ram_capacity" in failures,
                    candidate_msd_infeasible="msd_violation" in failures,
                    candidate_reject_reason=failure,
                    admission_reason=admission_reason,
                    deployment_status="simulation_only" if accepted else "not_attempted",
                    reject_reason="" if accepted else reject_reason,
                    tags=dict(spec.tags),
                )
            )

        summary = summarize_records(
            algorithm=algorithm,
            scenario=scenario.name,
            title=scenario.title,
            records=records,
            msd_drop_delta=0,
            final_active_vnfs=len(state_manager.as_public_dict().get("active_vnfs") or []),
        )
        write_records_csv(output_dir / f"{scenario.name}_{algorithm}_records.csv", records)
        return records, summary

    @staticmethod
    def _greedy_choice(snapshot, spec) -> Tuple[int, int]:
        state = snapshot.state
        best_node = None
        best_free_cpu = -1.0
        for node in range(len(snapshot.node_names)):
            free_cpu = 100.0 - float(state[node, 0])
            if free_cpu >= spec.cpu_req and free_cpu > best_free_cpu:
                best_node = node
                best_free_cpu = free_cpu
        if best_node is None:
            from src.ai.heuristic import HardConstraintError

            raise HardConstraintError("greedy_no_feasible_node")
        return int(best_node), int(best_node)

    @staticmethod
    def _saf_h_choice(snapshot, spec) -> Tuple[int, int]:
        """Feasible-pair search using the SAF-H cost used by algorithm benchmarks."""
        from src.ai.heuristic import HardConstraintError

        state = snapshot.state
        best_pair = None
        best_cost = float("inf")
        for place in range(len(snapshot.node_names)):
            for route in range(len(snapshot.node_names)):
                sid_depth = int(spec.msd_req) + int(snapshot.hop_matrix[place, route])
                if BaselineSimulator._constraint_failures(
                    snapshot, place, route, spec, sid_depth
                ):
                    continue
                cpu_util = (float(state[place, 0]) + spec.cpu_req) / 100.0
                cost = (
                    float(snapshot.latency_matrix[place, route]) * 0.5
                    + cpu_util * 100.0
                    + int(snapshot.hop_matrix[place, route]) * 5.0
                )
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (place, route)
        if best_pair is None:
            raise HardConstraintError("saf_h_no_feasible_pair")
        return int(best_pair[0]), int(best_pair[1])

    @staticmethod
    def _unsafe_decoupled_choice(snapshot, spec) -> Tuple[int, int]:
        state = snapshot.state
        place_candidates = [
            node for node in range(len(snapshot.node_names))
            if 100.0 - float(state[node, 0]) >= spec.cpu_req
        ]
        if not place_candidates:
            from src.ai.heuristic import HardConstraintError

            raise HardConstraintError("decoupled_no_feasible_cpu_placement")
        v_place = min(place_candidates, key=lambda node: float(state[node, 0]))
        if spec.destination_node is not None:
            v_route = int(spec.destination_node)
        else:
            v_route = min(
                [node for node in range(len(snapshot.node_names)) if node != v_place],
                key=lambda node: float(snapshot.latency_matrix[v_place][node]),
            )
        return int(v_place), int(v_route)

    @staticmethod
    def _constraint_failures(snapshot, v_place: int, v_route: int, spec, sid_depth: int) -> List[str]:
        state = snapshot.state
        place, route = int(v_place), int(v_route)
        failures: List[str] = []
        if float(state[place, 0]) + spec.cpu_req > 100.0:
            failures.append("cpu_capacity")
        if float(state[place, 1]) + spec.ram_req > 100.0:
            failures.append("ram_capacity")
        if float(state[route, 2]) + sid_depth > float(snapshot.msd_limits[route]):
            failures.append("msd_violation")
        return failures
