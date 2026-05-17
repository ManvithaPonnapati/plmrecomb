from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from plmrecomb.score import SchemaScorer, clean_seq


@dataclass
class BlockScheme:
    parents: Dict[str, str]
    block_boundaries: List[Tuple[int, int]]
    parent_order: List[str] = field(init=False)

    def __post_init__(self):
        self.parent_order = sorted(self.parents)
        L = len(next(iter(self.parents.values())))
        for s in self.parents.values():
            if len(s) != L:
                raise ValueError("All parents must be the same length (use aligned sequences)")

    @property
    def n_blocks(self) -> int:
        return len(self.block_boundaries)

    @property
    def n_parents(self) -> int:
        return len(self.parent_order)

    def chimera_from_assignment(self, assignment: Sequence[int]) -> str:
        if len(assignment) != self.n_blocks:
            raise ValueError("assignment length must equal number of blocks")
        out = []
        for b_idx, (start, end) in enumerate(self.block_boundaries):
            pname = self.parent_order[assignment[b_idx]]
            out.append(self.parents[pname][start:end])
        return "".join(out)

    def parent_per_position(self, assignment: Sequence[int]) -> str:
        out = []
        for b_idx, (start, end) in enumerate(self.block_boundaries):
            out.append(str(assignment[b_idx]) * (end - start))
        return "".join(out)

    def random_assignment(self, rng: np.random.Generator) -> List[int]:
        return [int(rng.integers(0, self.n_parents)) for _ in range(self.n_blocks)]


@dataclass
class EnergyTerms:
    schema_E: float = 0.0
    neg_pll: float = 0.0
    neg_stability: float = 0.0
    total: float = 0.0


@dataclass
class Energy:
    scorer: Optional[SchemaScorer] = None
    pll_fn: Optional[Callable[[str], float]] = None
    stability_fn: Optional[Callable[[str], float]] = None
    schema_weight: float = 1.0
    pll_weight: float = 0.0
    stability_weight: float = 0.0

    def __call__(self, sequence: str, block_string: Optional[str] = None) -> EnergyTerms:
        terms = EnergyTerms()
        if self.scorer is not None and self.schema_weight != 0.0:
            if block_string is not None:
                r = self.scorer.score_with_blocks(sequence, block_string)
            else:
                r = self.scorer.score(sequence)
            terms.schema_E = float(r["E"])
        if self.pll_fn is not None and self.pll_weight != 0.0:
            terms.neg_pll = -float(self.pll_fn(sequence))
        if self.stability_fn is not None and self.stability_weight != 0.0:
            terms.neg_stability = -float(self.stability_fn(sequence))
        terms.total = (
            self.schema_weight * terms.schema_E
            + self.pll_weight * terms.neg_pll
            + self.stability_weight * terms.neg_stability
        )
        return terms


@dataclass
class MCMCTrace:
    step: List[int] = field(default_factory=list)
    accepted: List[bool] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    schema_E: List[float] = field(default_factory=list)
    neg_pll: List[float] = field(default_factory=list)
    neg_stability: List[float] = field(default_factory=list)
    sequence: List[str] = field(default_factory=list)
    assignment: List[List[int]] = field(default_factory=list)

    def record(self, step: int, accepted: bool, seq: str,
               assignment: Sequence[int], terms: EnergyTerms) -> None:
        self.step.append(step)
        self.accepted.append(accepted)
        self.energy.append(terms.total)
        self.schema_E.append(terms.schema_E)
        self.neg_pll.append(terms.neg_pll)
        self.neg_stability.append(terms.neg_stability)
        self.sequence.append(seq)
        self.assignment.append(list(assignment))


@dataclass
class MCMCSampler:
    scheme: BlockScheme
    energy: Energy
    temperature: float = 1.0
    seed: int = 0
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def propose(self, assignment: List[int]) -> List[int]:
        new = list(assignment)
        b = int(self.rng.integers(0, self.scheme.n_blocks))
        choices = [p for p in range(self.scheme.n_parents) if p != new[b]]
        new[b] = int(self.rng.choice(choices))
        return new

    def evaluate(self, assignment: List[int]) -> Tuple[str, EnergyTerms]:
        seq = self.scheme.chimera_from_assignment(assignment)
        block_str = self.scheme.parent_per_position(assignment)
        terms = self.energy(seq, block_string=block_str)
        return seq, terms

    def step(self, assignment: List[int], current_terms: EnergyTerms
             ) -> Tuple[List[int], EnergyTerms, bool, str]:
        proposal = self.propose(assignment)
        new_seq, new_terms = self.evaluate(proposal)
        delta = new_terms.total - current_terms.total
        if delta <= 0:
            accept = True
        else:
            accept = bool(self.rng.random() < np.exp(-delta / max(self.temperature, 1e-12)))
        if accept:
            return proposal, new_terms, True, new_seq
        cur_seq = self.scheme.chimera_from_assignment(assignment)
        return assignment, current_terms, False, cur_seq

    def run(self, n_steps: int, start: Optional[List[int]] = None,
            record_every: int = 1) -> MCMCTrace:
        assignment = list(start) if start is not None else self.scheme.random_assignment(self.rng)
        seq, terms = self.evaluate(assignment)
        trace = MCMCTrace()
        trace.record(0, True, seq, assignment, terms)
        for s in range(1, n_steps + 1):
            assignment, terms, accepted, seq = self.step(assignment, terms)
            if s % record_every == 0:
                trace.record(s, accepted, seq, assignment, terms)
        return trace


def random_walk_distances(parent: str, n_samples: int, max_mutations: int,
                          alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
                          seed: int = 0) -> List[Tuple[int, str]]:
    rng = np.random.default_rng(seed)
    parent = clean_seq(parent)
    out: List[Tuple[int, str]] = []
    for _ in range(n_samples):
        m = int(rng.integers(0, max_mutations + 1))
        positions = rng.choice(len(parent), size=m, replace=False)
        chars = list(parent)
        for p in positions:
            choices = [a for a in alphabet if a != chars[p]]
            chars[p] = str(rng.choice(choices))
        out.append((m, "".join(chars)))
    return out


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("sequences must be equal length for hamming distance")
    return sum(1 for x, y in zip(a, b) if x != y)


def pairwise_identity(seqs: Sequence[str]) -> np.ndarray:
    n = len(seqs)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            if len(seqs[i]) != len(seqs[j]):
                out[i, j] = out[j, i] = float("nan")
                continue
            d = hamming(seqs[i], seqs[j])
            ident = 1.0 - d / len(seqs[i])
            out[i, j] = out[j, i] = ident
    return out
