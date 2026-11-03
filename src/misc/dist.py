"""Distributed helpers required by the shared interval loss implementation."""
from .distributed import get_world_size, is_initialized as is_dist_available_and_initialized
