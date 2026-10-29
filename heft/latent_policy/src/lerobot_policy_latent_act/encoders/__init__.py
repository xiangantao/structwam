"""Local-only frozen encoder adapters, deliberately outside the trainable module tree."""


def make_encoder(config, device):
    if config.encoder == "distilled":
        import os

        python = getattr(config, "distilled_runtime_python", "") or os.environ.get(
            "LATENT_DISTILLED_PYTHON", ""
        )
        if python:
            from .distilled_process import DistilledProcessEncoder

            return DistilledProcessEncoder(config, device, python)
        from .distilled import DistilledEncoder

        return DistilledEncoder(config, device)
    if config.encoder == "wan":
        import os

        python = config.wan_runtime_python or os.environ.get("LATENT_WAN_PYTHON", "")
        if python:
            from .wan_process import WanProcessEncoder

            return WanProcessEncoder(config, device, python)
        from .wan import WanEncoder

        return WanEncoder(config, device)
    if config.encoder == "wan_jepa":
        import os

        python = config.wan_runtime_python or os.environ.get("LATENT_WAN_PYTHON", "")
        if not python:
            raise ValueError("wan_jepa requires wan_runtime_python or LATENT_WAN_PYTHON")
        if getattr(config, "feature_seed_mode", "fixed") == "per_clip":
            from .wan_jepa_process_seeded import WanJepaProcessEncoderSeeded

            return WanJepaProcessEncoderSeeded(config, device, python)
        from .wan_jepa_process import WanJepaProcessEncoder

        return WanJepaProcessEncoder(config, device, python)
    if config.encoder == "vjepa2":
        from .vjepa2 import VJepa2Encoder

        return VJepa2Encoder(config, device)
    raise ValueError(f"Unknown encoder: {config.encoder}")
