import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TheoremState:
    current_level: int = 0
    # Rolling window of recent successes (1 for success, 0 for failure)
    history: deque = field(default_factory=lambda: deque(maxlen=8))
    # Count of successes at the current level on DIFFERENT hole placements
    # (Simplified: we just count total successes in window for now,
    # as tracking unique hole placements is complex without hole hashes)
    consecutive_successes: int = 0


class CurriculumManager:
    def __init__(
        self,
        levels: List[float] = None,
        window_size: int = 8,
        promotion_threshold: int = 5,
    ):
        """
        levels: List of mask ratios, e.g. [0.1, 0.2, 0.3, ... 1.0]
        """
        # Default levels if none provided
        if levels is None:
            self.levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
        else:
            self.levels = levels

        self.window_size = window_size
        self.promotion_threshold = promotion_threshold
        self.states: Dict[str, TheoremState] = defaultdict(TheoremState)

        # Sampling probabilities
        self.prob_current = 0.70
        self.prob_review = 0.20
        self.prob_challenge = 0.10

    def get_mask_ratio(self, theorem_id: str) -> float:
        """
        Decide the mask ratio for the next sample of this theorem.
        Uses the '70% current, 20% review, 10% challenge' policy.
        """
        state = self.states[theorem_id]
        current_idx = state.current_level
        max_idx = len(self.levels) - 1

        # Sampling logic
        rand = random.random()

        if rand < self.prob_current:
            # 70%: Current level
            idx = current_idx
        elif rand < self.prob_current + self.prob_review:
            # 20%: Review (any level below current)
            if current_idx > 0:
                idx = random.randint(0, current_idx - 1)
            else:
                idx = 0
        else:
            # 10%: Challenge (next level, if available)
            idx = min(current_idx + 1, max_idx)

        return self.levels[idx]

    def update_outcome(self, theorem_id: str, success: bool):
        """
        Update the state based on the verification outcome.
        """
        state = self.states[theorem_id]

        # Add to history
        state.history.append(1 if success else 0)

        # Check for promotion condition
        # Logic: If we have >= m successes in the last W attempts at CURRENT level
        # Note: Ideally we only count attempts made AT the current level.
        # But for simplicity, we look at recent history.
        # A more robust impl would store (ratio, outcome) in history.

        recent_successes = sum(state.history)
        if recent_successes >= self.promotion_threshold:
            # Promote!
            if state.current_level < len(self.levels) - 1:
                state.current_level += 1
                state.history.clear()  # Clear history on promotion (fresh start)
                # print(f"Theorem {theorem_id} promoted to level {self.levels[state.current_level]}")

    def get_current_level_index(self, theorem_id: str) -> int:
        return self.states[theorem_id].current_level
