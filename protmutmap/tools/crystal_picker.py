"""Pick the "nearest" crystal node for a FASPR-built intermediate.

When a node in the mutation graph has no crystal of its own, we still need a
starting PDB for FASPR — we pick the crystal whose mutation set has
the smallest symmetric difference with the target node. Ties are broken
deterministically by alphabetical order on the canonical ``mutation_str``.

Both *addition* and *removal* mutations are admissible: FASPR can
replace any side chain with any other amino acid, so the picker treats the
two operations as equally costly.
"""

from __future__ import annotations

from typing import Iterable, Optional

from .mutations import MutationList

def _mutset(mut_str: str) -> "frozenset[tuple]":
    """Convert a canonical mutation_str into a frozenset of ``(chain, resid, icode, after_res)``."""
    if not mut_str or mut_str.strip() in {"", "WT", "wt"}:
        return frozenset()
    ml = MutationList.from_string(mut_str)
    return frozenset(
        (m.chain, m.resid, m.icode, m.after_res) for m in ml
    )

def symmetric_diff_distance(a: str, b: str) -> int:
    """Symmetric set difference distance between two mutation_str values."""
    return len(_mutset(a) ^ _mutset(b))

def pick_nearest_crystal(
    target: str,
    crystals: Iterable[str],
) -> Optional[str]:
    """Return the crystal mutation_str nearest to ``target``.

    Distance is the symmetric set difference of canonical mutations. Ties
    are broken alphabetically (so picks are deterministic across runs).

    Parameters
    ----------
    target : canonical mutation_str of the node needing a starting PDB.
    crystals : iterable of canonical mutation_str values that have a
        registered crystal PDB.

    Returns
    -------
    The chosen crystal mutation_str, or ``None`` if ``crystals`` is empty.
    """
    crystals_list = list(crystals)
    if not crystals_list:
        return None
    best: Optional[str] = None
    best_distance: Optional[int] = None
    for crystal in sorted(crystals_list):  # alphabetical tie-break
        d = symmetric_diff_distance(target, crystal)
        if best_distance is None or d < best_distance:
            best = crystal
            best_distance = d
    return best
