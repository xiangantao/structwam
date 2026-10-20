"""Import-time names used by Cosmos utilities (CUDA-graph RNG tracking): not available here."""


def graph_safe_rng_available() -> bool:
    return False


def get_all_rng_states() -> dict:
    return {}
