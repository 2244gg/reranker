"""Base classes and unified data structures for dataset adapters."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class InteractionItem:
    """One user-item interaction (e.g., a rating or a play event)."""

    item_id: str
    title: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    rating: Optional[float] = None
    timestamp: Optional[int] = None


@dataclass
class UserRecord:
    """Standardized per-user record produced by every dataset adapter."""

    user_id: str
    demographics: Dict[str, str] = field(default_factory=dict)
    history: List[InteractionItem] = field(default_factory=list)
    test_set: List[InteractionItem] = field(default_factory=list)

    def demographic_value(self, dim: str) -> str:
        return str(self.demographics.get(dim, "unknown"))


class BaseDatasetAdapter(ABC):
    """Abstract dataset adapter.

    Concrete subclasses materialize a list of ``UserRecord`` from raw files
    and expose a stable ``demographic_schema`` plus standard split/sampling.
    """

    name: str = "base"

    def __init__(
        self,
        raw_dir: Optional[str] = None,
        rating_threshold: Optional[float] = None,
        min_interactions: int = 10,
    ) -> None:
        self.raw_dir = raw_dir
        self.rating_threshold = rating_threshold
        self.min_interactions = min_interactions
        self._users: Optional[List[UserRecord]] = None  # cached after load()

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    @abstractmethod
    def domain(self) -> str:
        """Return one of: ``movie``, ``book``, ``music``, ``video``."""

    @abstractmethod
    def demographic_schema(self) -> List[str]:
        """Ordered list of demographic dimension names."""

    @abstractmethod
    def _load_users(self) -> List[UserRecord]:
        """Load and return all ``UserRecord``s with ``history`` populated.

        ``test_set`` should be empty here; ``split()`` will populate it.
        """

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def positive_feedback_predicate(self, item: InteractionItem) -> bool:
        """Decide whether an item counts as positive feedback.

        Default: rating > rating_threshold (if both set), else any rating.
        """
        if self.rating_threshold is not None and item.rating is not None:
            return item.rating > self.rating_threshold
        return True

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load(self) -> List[UserRecord]:
        """Load all users (cached)."""
        if self._users is None:
            users = self._load_users()
            # Filter by min interactions on positive feedback only.
            users = [
                u for u in users
                if sum(1 for it in u.history if self.positive_feedback_predicate(it))
                >= self.min_interactions
            ]
            self._users = users
        return self._users

    def split(
        self,
        seed: int = 42,
        test_size: float = 0.7,
    ) -> List[UserRecord]:
        """Per-user random split into history/test_set.

        Returns the same UserRecord objects with ``test_set`` populated.
        Only positive-feedback items are split; non-positive go to history
        as context but are excluded from test relevance judgments.

        After splitting, ``u.history`` is sorted by ``rating`` (or whatever
        signal the adapter stores in that slot) descending so that the
        prompt builder's ``history[:history_limit]`` slice gives the user's
        TOP interactions, not a random shuffle. This is important for
        downstream "mainstream-ness" computation which uses the same
        top-N items as the prompt.
        """
        users = self.load()
        rng = random.Random(seed)
        for u in users:
            positives = [it for it in u.history if self.positive_feedback_predicate(it)]
            negatives = [it for it in u.history if not self.positive_feedback_predicate(it)]
            if len(positives) < 2:
                u.test_set = []
                continue
            shuffled = list(positives)
            rng.shuffle(shuffled)
            n_test = max(1, int(round(len(shuffled) * test_size)))
            n_test = min(n_test, len(shuffled) - 1)
            test = shuffled[:n_test]
            train = shuffled[n_test:]
            # Sort training history by rating signal (high first) so that
            # `history[:N]` represents the user's top-rated/top-played items.
            train_sorted = sorted(
                train,
                key=lambda it: (it.rating if it.rating is not None else 0.0),
                reverse=True,
            )
            negatives_sorted = sorted(
                negatives,
                key=lambda it: (it.rating if it.rating is not None else 0.0),
                reverse=True,
            )
            u.history = train_sorted + negatives_sorted
            u.test_set = test
        return users

    def stratified_sample(
        self,
        n_users: int,
        seed: int = 42,
        users: Optional[List[UserRecord]] = None,
    ) -> List[UserRecord]:
        """Draw ``n_users`` stratified across the demographic Cartesian product.

        Each non-empty stratum receives at least one user (subject to
        availability). Remaining slots are filled proportionally.
        """
        users = users if users is not None else self.load()
        dims = self.demographic_schema()
        rng = random.Random(seed)

        # Bucket by demographic combination.
        buckets: Dict[Tuple[str, ...], List[UserRecord]] = defaultdict(list)
        for u in users:
            key = tuple(u.demographic_value(d) for d in dims)
            buckets[key].append(u)
        for k in buckets:
            rng.shuffle(buckets[k])

        if not buckets:
            return []
        if n_users >= len(users):
            return list(users)

        # Round-robin draw across buckets to keep slices balanced.
        keys = list(buckets.keys())
        rng.shuffle(keys)
        sampled: List[UserRecord] = []
        idx_per_bucket = {k: 0 for k in keys}
        while len(sampled) < n_users:
            progressed = False
            for k in keys:
                if len(sampled) >= n_users:
                    break
                idx = idx_per_bucket[k]
                if idx < len(buckets[k]):
                    sampled.append(buckets[k][idx])
                    idx_per_bucket[k] = idx + 1
                    progressed = True
            if not progressed:
                break
        return sampled

    def scenario_combinations(
        self,
        max_dims: Optional[int] = None,
    ) -> List[Tuple[str, ...]]:
        """Enumerate scenarios = subsets of demographic dimensions, plus 'neutral'.

        Returns a list of tuples; the empty tuple denotes the neutral scenario.
        """
        dims = self.demographic_schema()
        upper = max_dims if max_dims is not None else len(dims)
        upper = min(upper, len(dims))
        scenarios: List[Tuple[str, ...]] = [()]  # neutral
        for r in range(1, upper + 1):
            for combo in _combinations(dims, r):
                scenarios.append(tuple(combo))
        return scenarios

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"


def _combinations(seq: Iterable[str], r: int) -> Iterable[Tuple[str, ...]]:
    from itertools import combinations as _c
    return _c(list(seq), r)
