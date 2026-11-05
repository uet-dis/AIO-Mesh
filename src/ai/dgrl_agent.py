"""Lazy JO-VPPM DRL model adapter for FastAPI inference."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from zipfile import ZipFile

import numpy as np
import sys

# RÀ SOÁT: Sửa lỗi "No module named 'numpy._core'" do mô hình được lưu bằng NumPy 2.0+
# nhưng hệ thống đang chạy NumPy 1.x (Python 3.8).
if 'numpy._core' not in sys.modules:
    import types
    # Tạo một module giả lập cho numpy._core
    core = types.ModuleType('numpy._core')
    core.numeric = np
    core.multiarray = np
    core.umath = np
    core.fixed_tuple = np.recarray # Đôi khi cần cho pickle
    sys.modules['numpy._core'] = core
    sys.modules['numpy._core.numeric'] = np
    sys.modules['numpy._core.multiarray'] = np
    sys.modules['numpy._core.umath'] = np

from src.ai.heuristic import ActionChoice, get_resilience_safe_action
from src.core.state_manager import NetworkStateManager
from src.orchestration.jo_vdpr.action_codec import decode_pair_action
from src.orchestration.jo_vdpr.topology import TopologyManager

logger = logging.getLogger("DGRLAgent")


DEFAULT_TOPOLOGY_NAME = os.getenv("JO_VPPM_TOPOLOGY", "vietnam")
DEFAULT_MODEL_PATH = os.getenv(
    "JO_VPPM_MODEL_PATH",
    f"results/models/v11/dgrl_v11_final_{DEFAULT_TOPOLOGY_NAME}.zip",
)
ENABLE_INPROCESS_MODEL = os.getenv("JO_VPPM_ENABLE_MODEL", "1") == "1"
REQUIRE_REAL_MODEL = os.getenv("JO_VPPM_REQUIRE_MODEL", "1") == "1"


def _infer_scaler_path(model_path: str, topology_name: str) -> str:
    """Infer the matching VecNormalize artifact for a JO-VPPM checkpoint."""
    explicit_path = os.getenv("JO_VPPM_SCALER_PATH")
    if explicit_path:
        return explicit_path

    path = Path(model_path)
    variant_dir = path.parent.name
    if variant_dir.startswith("harp_"):
        variant_name = f"vec_normalize_v11_{variant_dir}_{topology_name}.pkl"
        variant_path = path.with_name(variant_name)
        if variant_path.is_file():
            return str(variant_path)
    return str(path.with_name(f"vec_normalize_v11_{topology_name}.pkl"))


class NpzVecNormalize:
    """Small inference-only VecNormalize reader for cross-version scaler stats."""

    def __init__(self, path: str) -> None:
        stats = np.load(path)
        self.obs_mean = stats["obs_mean"].astype(np.float32)
        self.obs_var = stats["obs_var"].astype(np.float32)
        self.clip_obs = float(stats["clip_obs"])
        self.epsilon = float(stats["epsilon"])
        self.norm_obs = bool(stats["norm_obs"])

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.norm_obs:
            return obs
        normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(normalized, -self.clip_obs, self.clip_obs).astype(np.float32)


def _strip_single_root_zip(zip_path: str) -> str:
    """Return an SB3-compatible zip path, flattening single-root archives if needed."""
    with ZipFile(zip_path, "r") as source:
        names = [name for name in source.namelist() if not name.endswith("/")]
        if "data" in names:
            return zip_path

        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        if len(roots) != 1:
            return zip_path

        root = next(iter(roots))
        if f"{root}/data" not in names:
            return zip_path

        source_stat = os.stat(zip_path)
        cache_dir = Path(tempfile.gettempdir()) / "corerouter_sb3_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        flattened = cache_dir / (
            f"{Path(zip_path).stem}-{source_stat.st_mtime_ns}-{source_stat.st_size}.zip"
        )
        if flattened.exists():
            return str(flattened)

        with ZipFile(flattened, "w") as target:
            prefix = f"{root}/"
            for name in names:
                if not name.startswith(prefix):
                    continue
                target.writestr(name[len(prefix):], source.read(name))

    logger.info("Normalized nested SB3 archive %s -> %s", zip_path, flattened)
    return str(flattened)


@dataclass
class DGRLDecision:
    choice: ActionChoice
    model_loaded: bool
    fallback_reason: Optional[str] = None


class DGRLAgent:
    """Small wrapper that keeps ML dependencies optional for the backend."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        topology_name: str = DEFAULT_TOPOLOGY_NAME,
    ) -> None:
        self.model_path = model_path
        self.topology_name = topology_name
        self.topology = TopologyManager(topology_name)
        self.num_nodes = self.topology.num_nodes
        self.scaler_path = _infer_scaler_path(model_path, topology_name)
        self._model: Any = None
        self._scaler: Any = None
        self._load_error: Optional[str] = None
        self._load_attempted = False

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _load_model(self) -> Any:
        if self._load_attempted:
            return self._model
        self._load_attempted = True

        if not ENABLE_INPROCESS_MODEL:
            self._load_error = "model_loading_disabled:set_JO_VPPM_ENABLE_MODEL=1"
            logger.warning(self._load_error)
            if REQUIRE_REAL_MODEL:
                raise RuntimeError(self._load_error)
            return None

        if not os.path.exists(self.model_path):
            self._load_error = f"model_not_found:{self.model_path}"
            logger.warning(self._load_error)
            if REQUIRE_REAL_MODEL:
                raise FileNotFoundError(self._load_error)
            return None

        try:
            from sb3_contrib import MaskablePPO
            import stable_baselines3.common.utils as sb3_utils
            import gymnasium as gym

            from src.orchestration.jo_vdpr.gnn_policy import GNNActorCriticPolicy
            # Khắc phục lỗi thiếu thuộc tính do sai khác phiên bản SB3
            if not hasattr(sb3_utils, 'FloatSchedule'):
                class FloatSchedule:
                    def __init__(self, val): self.val = val
                    def __call__(self, _): return self.val
                sb3_utils.FloatSchedule = FloatSchedule
            if not hasattr(sb3_utils, 'ConstantSchedule'):
                class ConstantSchedule:
                    def __init__(self, val): self.val = val
                    def __call__(self, _): return self.val
                sb3_utils.ConstantSchedule = ConstantSchedule

            # Khắc phục lỗi NumPy _frombuffer
            if not hasattr(np, '_frombuffer'):
                np._frombuffer = np.frombuffer

            load_path = _strip_single_root_zip(self.model_path)
            custom_objects = {
                "policy_class": GNNActorCriticPolicy,
                "learning_rate": 0.0003,
                "lr_schedule": lambda _: 0.0003,
                "clip_range": lambda _: 0.2,
                "_last_obs": None,
                "_last_episode_starts": None,
                "_last_original_obs": None,
                "ep_info_buffer": [],
            }
            self._model = MaskablePPO.load(load_path, device="cpu", custom_objects=custom_objects)
            expected_obs_dim = self.num_nodes * 6 + 13
            if not isinstance(self._model.action_space, gym.spaces.Discrete):
                raise ValueError(
                    "checkpoint action space must be Discrete(N*N); "
                    f"found {self._model.action_space!r}. Retrain with the current JOVDPREnv."
                )
            if self._model.action_space.n != self.num_nodes * self.num_nodes:
                raise ValueError(
                    f"checkpoint has {self._model.action_space.n} actions; "
                    f"topology {self.topology_name} requires {self.num_nodes * self.num_nodes}"
                )
            if self._model.observation_space.shape != (expected_obs_dim,):
                raise ValueError(
                    f"checkpoint observation shape {self._model.observation_space.shape} does not match "
                    f"topology {self.topology_name} shape {(expected_obs_dim,)}"
                )
            extractor = self._model.policy.features_extractor
            if getattr(extractor, "num_nodes", None) != self.num_nodes:
                raise ValueError(
                    f"checkpoint graph encoder does not match topology {self.topology_name}"
                )
            expected_adj = np.asarray(self.topology.adj_matrix, dtype=np.float32)
            degree = np.sqrt(np.maximum(expected_adj.sum(axis=1, keepdims=True), 1e-9))
            expected_adj = expected_adj / (degree * degree.T)
            loaded_adj = extractor.adj.detach().cpu().numpy()
            if loaded_adj.shape != expected_adj.shape or not np.allclose(loaded_adj, expected_adj):
                raise ValueError(f"checkpoint adjacency matrix does not match topology {self.topology_name}")
            logger.info("Loaded JO-VPPM model from %s", self.model_path)
            
            if os.path.exists(self.scaler_path):
                try:
                    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

                    class DummyEnv(gym.Env):
                        def __init__(self):
                            self.observation_space = self_outer._model.observation_space
                            self.action_space = self_outer._model.action_space
                        def reset(self, seed=None):
                            return np.zeros(expected_obs_dim, dtype=np.float32), {}
                        def step(self, action):
                            return np.zeros(expected_obs_dim, dtype=np.float32), 0.0, False, False, {}
                    
                    self_outer = self
                    venv = DummyVecEnv([lambda: DummyEnv()])
                    self._scaler = VecNormalize.load(self.scaler_path, venv)
                    self._scaler.training = False
                    self._scaler.norm_reward = False
                    if self._scaler.obs_rms.mean.shape != (expected_obs_dim,):
                        raise ValueError(
                            f"normalizer observation shape {self._scaler.obs_rms.mean.shape} does not match "
                            f"topology {self.topology_name} shape {(expected_obs_dim,)}"
                        )
                    logger.info("Loaded VecNormalize scaler from %s", self.scaler_path)
                except Exception as exc:
                    npz_path = str(Path(self.scaler_path).with_suffix(".npz"))
                    if os.path.exists(npz_path):
                        self._scaler = NpzVecNormalize(npz_path)
                        if self._scaler.obs_mean.shape != (expected_obs_dim,):
                            raise ValueError(
                                f"normalizer observation shape {self._scaler.obs_mean.shape} does not match "
                                f"topology {self.topology_name} shape {(expected_obs_dim,)}"
                            ) from exc
                        logger.info("Loaded VecNormalize stats from %s after pickle failure: %s", npz_path, exc)
                    else:
                        raise RuntimeError(f"Failed to load matching normalizer {self.scaler_path}: {exc}") from exc
            else:
                raise FileNotFoundError(f"Matching VecNormalize scaler not found: {self.scaler_path}")

        except Exception as exc:
            self._load_error = f"incompatibility_detected:{exc}"
            if REQUIRE_REAL_MODEL:
                logger.exception("DRL model is required but failed to load.")
                raise RuntimeError(self._load_error) from exc
            logger.warning("DRL Model Incompatible: %s. Activating SHADOW MODE.", exc)
            self._model = "SHADOW_MODE_ACTIVE" # Sentinel for simulation
        return self._model

    def get_action(self, state_manager: NetworkStateManager, request: Any) -> DGRLDecision:
        """Return a DRL action, falling back to shadow simulation if needed."""
        model = self._load_model()
        
        # Nếu ở Shadow Mode, giả lập quyết định DRL để chạy tiếp luồng MBB
        if model == "SHADOW_MODE_ACTIVE":
            # Logic giả lập DRL: Ưu tiên node có tài nguyên trống nhiều nhất
            snap = state_manager.snapshot()
            state_array = snap.state # Mảng (N, 3)
            num_nodes = state_array.shape[0]
            
            best_node = 0
            max_free = -1.0
            for i in range(num_nodes):
                cpu_used = float(state_array[i, 0])
                free = 100.0 - cpu_used
                if free > max_free:
                    max_free = free
                    best_node = i
            
            return DGRLDecision(
                choice=ActionChoice(
                    v_place=best_node,
                    v_route=best_node,
                    reason="drl_shadow_policy_simulation",
                ),
                model_loaded=True, 
            )

        if model is None:
            snap = state_manager.snapshot()
            return DGRLDecision(
                choice=get_resilience_safe_action(snap, request),
                model_loaded=False,
                fallback_reason=self._load_error or "model_unavailable",
            )

        try:
            if state_manager.topology.topology_name != self.topology_name:
                raise ValueError(
                    f"agent topology {self.topology_name} does not match state manager "
                    f"topology {state_manager.topology.topology_name}"
                )
            obs = state_manager.to_observation(request)
            if obs.shape != (self.num_nodes * 6 + 13,):
                raise ValueError(f"runtime observation has unexpected shape {obs.shape}")
            
            # PHASE 6: Fix Tử Huyệt 3 — Đảm bảo Input của VecNormalize là 2D (1, N)
            obs_2d = obs.reshape(1, -1)
            
            if self._scaler is not None:
                obs_normalized = self._scaler.normalize_obs(obs_2d)
            else:
                obs_normalized = obs_2d

            # PHASE 6: Fix Tử Huyệt 1 — Action Masking
            masks = state_manager.action_mask(request)
            if masks.shape != (self.num_nodes * self.num_nodes,):
                raise ValueError(
                    f"runtime action mask has shape {masks.shape}; "
                    f"expected {(self.num_nodes * self.num_nodes,)}"
                )
            
            # MaskablePPO.predict yêu cầu masks có cùng batch size với obs
            action, _ = model.predict(obs_normalized, action_masks=masks[None, :], deterministic=True)
            v_place, v_route = decode_pair_action(action, self.num_nodes)
            return DGRLDecision(
                choice=ActionChoice(
                    v_place=v_place,
                    v_route=v_route,
                    reason="maskableppo_gnn_policy_with_masking_and_norm",
                ),
                model_loaded=True,
            )
        except Exception as exc:
            logger.warning("DRL prediction failed; using safe search fallback: %s", exc)
            snap = state_manager.snapshot()
            return DGRLDecision(
                choice=get_resilience_safe_action(snap, request),
                model_loaded=False,
                fallback_reason=f"predict_failed:{exc}",
            )


_AGENT = DGRLAgent()


def get_dgrl_agent() -> DGRLAgent:
    return _AGENT
