"""Constants derived from config.yaml."""

from __future__ import annotations

from privacy_integration.settings import settings

_s = settings()

EPSILON_MAX: float = _s.privacy.epsilon_total_max
EPSILON_CALIBRATION_MAX: float = _s.privacy.epsilon_calibration_max
DELTA: float = _s.privacy.delta
CLIPPING_NORM: float = _s.privacy.clipping_norm
RANDOM_SEED: int = _s.privacy.random_seed
BATCH_SIZE: int = _s.dataset.batch_size
FEATURE_DIM: int = _s.dataset.feature_dim
MIA_SUCCESS_CEILING: float = _s.attack.success_rate_ceiling
REID_SUCCESS_CEILING: float = _s.attack.reid_ceiling
