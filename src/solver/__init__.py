# RegionBGC adaptation for BGC region prediction; upstream attribution is retained.
"""by lyuwenyu
"""

from .solver import BaseSolver
from .region_solver import RegionBGCSolver


from typing import Dict

TASKS :Dict[str, BaseSolver] = {
    'region_prediction': RegionBGCSolver,
}
