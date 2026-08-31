from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ImpactCurriculumCfg:
    """Configuration for the bidirectional five-band impact curriculum."""

    no_impact_fraction: float = 0.15
    easy_fraction: float = 0.15
    middle_fraction: float = 0.25
    current_fraction: float = 0.35
    probe_fraction: float = 0.10

    no_impact_window_episodes: int = 512
    current_window_episodes: int = 2048
    probe_window_episodes: int = 512

    promote_no_impact_success_rate: float = 0.90
    promote_current_success_rate: float = 0.70
    promote_current_crash_rate: float = 0.05
    promote_probe_success_rate: float = 0.50
    demote_current_success_rate: float = 0.35
    demote_current_crash_rate: float = 0.15

    promote_step: float = 0.05
    demote_step: float = 0.025
    minimum_difficulty: float = 0.10
    probe_band_width: float = 0.10
    consecutive_windows: int = 3
    cooldown_steps: int = 5000


@dataclass
class CurriculumSample:
    impacted: torch.Tensor
    difficulty: torch.Tensor
    band: torch.Tensor


@dataclass(frozen=True)
class TwoStageCurriculumCfg:
    """Interleave handoff and easy-impact exposure without a deadlock gate."""

    initial_step: float = 0.05
    disturbance_step: float = 0.025
    impact_probability_step: float = 0.05
    initial_disturbance_difficulty: float = 0.05
    initial_impact_probability: float = 0.05
    minimum_initial_difficulty: float = 0.05
    promote_success_rate: float = 0.70
    promote_crash_rate: float = 0.05
    consecutive_windows: int = 3
    stagnation_steps: int = 10_000


@dataclass(frozen=True)
class TwoStageCurriculumSample:
    initial_difficulty: float
    disturbance_difficulty: float
    impact_probability: float


class TwoStageCurriculum:
    """Interleaved outer course; retained name preserves checkpoint compatibility."""

    def __init__(self, cfg: TwoStageCurriculumCfg | None = None, *, initial_difficulty: float = 0.10):
        self.cfg = cfg or TwoStageCurriculumCfg()
        if self.cfg.initial_step <= 0.0 or self.cfg.disturbance_step <= 0.0:
            raise ValueError("curriculum steps must be positive")
        if self.cfg.impact_probability_step <= 0.0:
            raise ValueError("impact_probability_step must be positive")
        if self.cfg.consecutive_windows <= 0:
            raise ValueError("consecutive_windows must be positive")
        if self.cfg.stagnation_steps <= 0:
            raise ValueError("stagnation_steps must be positive")
        self.initial_difficulty = float(
            min(1.0, max(self.cfg.minimum_initial_difficulty, initial_difficulty))
        )
        self.disturbance_difficulty = float(
            min(1.0, max(0.0, self.cfg.initial_disturbance_difficulty))
        )
        self.impact_probability = float(
            min(1.0, max(0.0, self.cfg.initial_impact_probability))
        )
        self._promotion_streak = 0
        self._last_progress_step = 0

    def sample(self) -> TwoStageCurriculumSample:
        return TwoStageCurriculumSample(
            self.initial_difficulty,
            self.disturbance_difficulty,
            self.impact_probability,
        )

    def record_window(
        self,
        *,
        success_rate: float,
        crash_rate: float,
        global_step: int | None = None,
    ) -> TwoStageCurriculumSample:
        if not 0.0 <= success_rate <= 1.0 or not 0.0 <= crash_rate <= 1.0:
            raise ValueError("success_rate and crash_rate must lie in [0, 1]")
        mastered = success_rate >= self.cfg.promote_success_rate and crash_rate <= self.cfg.promote_crash_rate
        self._promotion_streak = self._promotion_streak + 1 if mastered else 0
        if self._promotion_streak >= self.cfg.consecutive_windows:
            self.initial_difficulty = min(1.0, self.initial_difficulty + self.cfg.initial_step)
            self.disturbance_difficulty = min(
                1.0, self.disturbance_difficulty + self.cfg.disturbance_step
            )
            self.impact_probability = min(
                1.0, self.impact_probability + self.cfg.impact_probability_step
            )
            self._promotion_streak = 0
            if global_step is not None:
                self._last_progress_step = int(global_step)
        elif (
            global_step is not None
            and int(global_step) - self._last_progress_step >= self.cfg.stagnation_steps
            and self.initial_difficulty > self.cfg.minimum_initial_difficulty
        ):
            self.initial_difficulty = max(
                self.cfg.minimum_initial_difficulty,
                self.initial_difficulty - self.cfg.initial_step,
            )
            self._last_progress_step = int(global_step)
        return self.sample()


@dataclass
class _BandStats:
    episodes: int = 0
    successes: int = 0
    crashes: int = 0

    def add(self, recovered: torch.Tensor, crashed: torch.Tensor, mask: torch.Tensor) -> None:
        self.episodes += int(mask.sum().item())
        self.successes += int((recovered & mask).sum().item())
        self.crashes += int((crashed & mask).sum().item())

    def reset(self) -> None:
        self.episodes = self.successes = self.crashes = 0

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    def load_state_dict(self, state: dict | None) -> None:
        state = state or {}
        self.episodes = int(state.get("episodes", 0))
        self.successes = int(state.get("successes", 0))
        self.crashes = int(state.get("crashes", 0))


class ImpactCurriculum:
    """Bidirectional curriculum with five persistent difficulty bands.

    Band IDs are 0=no impact, 1=easy, 2=middle, 3=current and 4=probe.
    Easy and middle episodes preserve coverage but do not vote on a level
    change.
    """

    STATE_VERSION = 3

    def __init__(self, cfg: ImpactCurriculumCfg | None = None, initial_difficulty: float = 0.10):
        self.cfg = cfg or ImpactCurriculumCfg()
        fractions = (
            self.cfg.no_impact_fraction,
            self.cfg.easy_fraction,
            self.cfg.middle_fraction,
            self.cfg.current_fraction,
            self.cfg.probe_fraction,
        )
        if any(value < 0.0 for value in fractions) or abs(sum(fractions) - 1.0) > 1.0e-6:
            raise ValueError("curriculum band fractions must be non-negative and sum to one")
        windows = (
            self.cfg.no_impact_window_episodes,
            self.cfg.current_window_episodes,
            self.cfg.probe_window_episodes,
        )
        if any(value <= 0 for value in windows):
            raise ValueError("curriculum window sizes must be positive")
        if self.cfg.consecutive_windows <= 0 or self.cfg.cooldown_steps < 0:
            raise ValueError("consecutive_windows must be positive and cooldown_steps non-negative")
        if self.cfg.promote_step <= 0.0 or self.cfg.demote_step <= 0.0:
            raise ValueError("curriculum step sizes must be positive")
        if not 0.0 <= self.cfg.minimum_difficulty <= 1.0:
            raise ValueError("minimum_difficulty must lie in [0, 1]")

        self.difficulty = float(
            min(1.0, max(self.cfg.minimum_difficulty, initial_difficulty))
        )
        self.mastered_difficulty = 0.0
        self.promotion_count = 0
        self.demotion_count = 0
        self.promotion_streak = 0
        self.demotion_streak = 0
        self.next_change_step = 0
        self.last_action = "none"

        self._no_impact = _BandStats()
        self._current = _BandStats()
        self._probe = _BandStats()
        self.last_no_impact_success_rate = 0.0
        self.last_no_impact_crash_rate = 0.0
        self.last_current_success_rate = 0.0
        self.last_current_crash_rate = 0.0
        self.last_probe_success_rate = 0.0
        self.last_probe_crash_rate = 0.0

    @property
    def last_success_rate(self) -> float:
        """Backward-compatible dashboard alias."""

        return self.last_current_success_rate

    @property
    def last_crash_rate(self) -> float:
        """Backward-compatible dashboard alias."""

        return self.last_current_crash_rate

    def sample(self, num_envs, device, seed=None) -> CurriculumSample:
        device = torch.device(device)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        draw = torch.rand(num_envs, device=device, generator=generator)
        boundaries = torch.tensor(
            [
                self.cfg.no_impact_fraction,
                self.cfg.no_impact_fraction + self.cfg.easy_fraction,
                self.cfg.no_impact_fraction + self.cfg.easy_fraction + self.cfg.middle_fraction,
                self.cfg.no_impact_fraction
                + self.cfg.easy_fraction
                + self.cfg.middle_fraction
                + self.cfg.current_fraction,
            ],
            device=device,
        )
        band = torch.bucketize(draw, boundaries).to(torch.uint8)
        impacted = band > 0

        scale = torch.rand(num_envs, device=device, generator=generator)
        difficulty = torch.zeros(num_envs, device=device)
        level = self.difficulty
        ranges = (
            (0.0, 0.0),
            (0.0, 0.30 * level),
            (0.30 * level, 0.70 * level),
            (0.70 * level, level),
        )
        for band_id, (low, high) in enumerate(ranges):
            mask = band == band_id
            difficulty[mask] = low + scale[mask] * (high - low)

        probe = band == 4
        probe_low = level if level < 1.0 else max(0.0, 1.0 - self.cfg.probe_band_width)
        probe_high = min(1.0, level + self.cfg.probe_band_width)
        difficulty[probe] = probe_low + scale[probe] * (probe_high - probe_low)
        return CurriculumSample(impacted=impacted, difficulty=difficulty, band=band)

    def record_batch(self, recovered, crashed, band, global_step: int = 0) -> bool:
        recovered = recovered.bool()
        crashed = crashed.bool()
        band = band.to(torch.uint8)
        self._no_impact.add(recovered, crashed, band == 0)
        self._current.add(recovered, crashed, band == 3)
        self._probe.add(recovered, crashed, band == 4)
        if (
            self._no_impact.episodes < self.cfg.no_impact_window_episodes
            or self._current.episodes < self.cfg.current_window_episodes
            or self._probe.episodes < self.cfg.probe_window_episodes
        ):
            return False

        self.last_no_impact_success_rate = self._no_impact.successes / self._no_impact.episodes
        self.last_no_impact_crash_rate = self._no_impact.crashes / self._no_impact.episodes
        self.last_current_success_rate = self._current.successes / self._current.episodes
        self.last_current_crash_rate = self._current.crashes / self._current.episodes
        self.last_probe_success_rate = self._probe.successes / self._probe.episodes
        self.last_probe_crash_rate = self._probe.crashes / self._probe.episodes

        promote = (
            self.last_no_impact_success_rate >= self.cfg.promote_no_impact_success_rate
            and self.last_current_success_rate >= self.cfg.promote_current_success_rate
            and self.last_current_crash_rate <= self.cfg.promote_current_crash_rate
            and self.last_probe_success_rate >= self.cfg.promote_probe_success_rate
        )
        demote = (
            self.last_current_success_rate < self.cfg.demote_current_success_rate
            or self.last_current_crash_rate > self.cfg.demote_current_crash_rate
        )

        self.last_action = "none"
        if int(global_step) < self.next_change_step:
            self.promotion_streak = 0
            self.demotion_streak = 0
        else:
            self.promotion_streak = self.promotion_streak + 1 if promote else 0
            self.demotion_streak = self.demotion_streak + 1 if demote else 0

            if self.promotion_streak >= self.cfg.consecutive_windows and self.difficulty < 1.0:
                old_difficulty = self.difficulty
                self.mastered_difficulty = max(self.mastered_difficulty, old_difficulty)
                self.difficulty = min(1.0, old_difficulty + self.cfg.promote_step)
                self.promotion_count += 1
                self.last_action = "promote"
                self.next_change_step = int(global_step) + self.cfg.cooldown_steps
                self.promotion_streak = 0
                self.demotion_streak = 0
            elif (
                self.demotion_streak >= self.cfg.consecutive_windows
                and self.difficulty > self.cfg.minimum_difficulty
            ):
                self.difficulty = max(
                    self.cfg.minimum_difficulty,
                    self.difficulty - self.cfg.demote_step,
                )
                self.demotion_count += 1
                self.last_action = "demote"
                self.next_change_step = int(global_step) + self.cfg.cooldown_steps
                self.promotion_streak = 0
                self.demotion_streak = 0

        self._no_impact.reset()
        self._current.reset()
        self._probe.reset()
        return True

    def state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "cfg": asdict(self.cfg),
            "difficulty": self.difficulty,
            "mastered_difficulty": self.mastered_difficulty,
            "promotion_count": self.promotion_count,
            "demotion_count": self.demotion_count,
            "promotion_streak": self.promotion_streak,
            "demotion_streak": self.demotion_streak,
            "next_change_step": self.next_change_step,
            "last_action": self.last_action,
            "no_impact": self._no_impact.state_dict(),
            "current": self._current.state_dict(),
            "probe": self._probe.state_dict(),
            "last_rates": {
                "no_impact_success": self.last_no_impact_success_rate,
                "no_impact_crash": self.last_no_impact_crash_rate,
                "current_success": self.last_current_success_rate,
                "current_crash": self.last_current_crash_rate,
                "probe_success": self.last_probe_success_rate,
                "probe_crash": self.last_probe_crash_rate,
            },
        }

    def load_state_dict(self, state):
        if int(state.get("version", 1)) < self.STATE_VERSION:
            self.difficulty = float(
                min(1.0, max(self.cfg.minimum_difficulty, state.get("difficulty", 0.10)))
            )
            self.mastered_difficulty = max(
                0.0,
                min(self.difficulty, state.get("mastered_difficulty", 0.0)),
            )
            self.promotion_count = int(state.get("promotion_count", 0))
            self.demotion_count = 0
            self.promotion_streak = 0
            self.demotion_streak = 0
            self.next_change_step = 0
            self.last_action = "legacy_reset"
            self._no_impact.reset()
            self._current.reset()
            self._probe.reset()
            self.last_no_impact_success_rate = 0.0
            self.last_no_impact_crash_rate = 0.0
            self.last_current_success_rate = float(state.get("last_success_rate", 0.0))
            self.last_current_crash_rate = float(state.get("last_crash_rate", 0.0))
            self.last_probe_success_rate = 0.0
            self.last_probe_crash_rate = 0.0
            return

        self.difficulty = float(
            min(1.0, max(self.cfg.minimum_difficulty, state["difficulty"]))
        )
        self.mastered_difficulty = float(
            min(self.difficulty, max(0.0, state.get("mastered_difficulty", 0.0)))
        )
        self.promotion_count = int(state.get("promotion_count", 0))
        self.demotion_count = int(state.get("demotion_count", 0))
        self.promotion_streak = int(state.get("promotion_streak", 0))
        self.demotion_streak = int(state.get("demotion_streak", 0))
        self.next_change_step = int(state.get("next_change_step", 0))
        self.last_action = str(state.get("last_action", "none"))
        self._no_impact.load_state_dict(state.get("no_impact"))
        self._current.load_state_dict(state.get("current"))
        self._probe.load_state_dict(state.get("probe"))
        rates = state.get("last_rates", {})
        self.last_no_impact_success_rate = float(rates.get("no_impact_success", 0.0))
        self.last_no_impact_crash_rate = float(rates.get("no_impact_crash", 0.0))
        self.last_current_success_rate = float(rates.get("current_success", 0.0))
        self.last_current_crash_rate = float(rates.get("current_crash", 0.0))
        self.last_probe_success_rate = float(rates.get("probe_success", 0.0))
        self.last_probe_crash_rate = float(rates.get("probe_crash", 0.0))
