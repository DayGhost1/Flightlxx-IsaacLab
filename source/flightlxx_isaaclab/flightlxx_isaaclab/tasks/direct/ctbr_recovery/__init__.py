import gymnasium as gym

from .ctbr_recovery_env import CTBRPreflightEnvCfg, CTBRRecoveryEnv, CTBRRecoveryEnvCfg

gym.register(
    id="Isaac-FlightLxx-CTBR-Recovery-Direct-v0",
    entry_point="flightlxx_isaaclab.tasks.direct.ctbr_recovery:CTBRRecoveryEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": CTBRRecoveryEnvCfg},
)

gym.register(
    id="Isaac-FlightLxx-CTBR-Preflight-Direct-v0",
    entry_point="flightlxx_isaaclab.tasks.direct.ctbr_recovery:CTBRRecoveryEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": CTBRPreflightEnvCfg},
)

__all__ = ["CTBRPreflightEnvCfg", "CTBRRecoveryEnv", "CTBRRecoveryEnvCfg"]
