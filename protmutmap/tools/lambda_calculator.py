"""Choose replica counts from mutation type and charge change."""

from typing import Sequence, Tuple
from .mutation import Mutation

DEFAULT_NREP = 32
DIFFICULT_NREP = 48
CHARGE_NREP = 64
EXTREME_NREP = 96
MULTI_MUTATION_MULTIPLIER = 1.5

def calculate_charge(residue: str) -> int:
    """
    Calculate the charge of a residue.

    Args:
        residue: Single letter amino acid code

    Returns:
        Charge value: -1 for DE, +1 for RK, 0 for others
    """
    if residue in "DE":
        return -1
    elif residue in "RK":
        return 1
    else:
        return 0

def assess_mutation_difficulty(mutations: Sequence[Mutation]) -> Tuple[bool, bool, bool]:
    """
    Assess the difficulty of a set of mutations.

    A mutation is considered difficult if:
    - It involves bulky/aromatic residues (P, F, Y, W)
    - Exception: F<->Y mutations are not considered difficult
    - The mutation changes the total charge

    A mutation is considered extreme if:
    - It is both difficult (bulky/aromatic) AND charge-changing

    Args:
        mutations: Sequence of Mutation objects

    Returns:
        Tuple of (is_difficult, has_charge_change, is_extreme)
        - is_difficult: True if mutation involves bulky/aromatic residues
        - has_charge_change: True if mutation changes total charge
        - is_extreme: True if mutation is both difficult and charge-changing
    """
    difficult = False
    total_charge = 0

    for mutation in mutations:
        mut_from = mutation.before_res
        mut_to = mutation.after_res

        # Track charge changes
        total_charge -= calculate_charge(mut_from)
        total_charge += calculate_charge(mut_to)

        # Check for bulky/aromatic residues
        if mut_to in "PFYW" or mut_from in "PFYW":
            difficult = True
            # Exception: F<->Y mutations are not difficult
            if mut_to in "FY" and mut_from in "FY":
                difficult = False

    has_charge_change = (total_charge != 0)
    is_extreme = difficult and has_charge_change

    return (difficult, has_charge_change, is_extreme)

def calculate_nrep(mutations: Sequence[Mutation],
                   default_nrep: int = DEFAULT_NREP,
                   difficult_nrep: int = DIFFICULT_NREP,
                   charge_nrep: int = CHARGE_NREP,
                   extreme_nrep: int = EXTREME_NREP,
                   multi_mutation_multiplier: float = MULTI_MUTATION_MULTIPLIER) -> int:
    """
    Calculate the number of lambda replicas needed for a mutation.

    4-tier system:
      - default: easy mutations
      - difficult: bulky/aromatic (F/Y/W/P) only
      - charge: charge-changing only
      - extreme: both bulky/aromatic AND charge-changing

    For multi-mutation edges, the base NREP is scaled by
    multi_mutation_multiplier^(n-1) where n is the number of mutations.

    Args:
        mutations: Sequence of Mutation objects
        default_nrep: Number of replicas for easy mutations (default: DEFAULT_NREP)
        difficult_nrep: Number of replicas for difficult/bulky mutations (default: DIFFICULT_NREP)
        charge_nrep: Number of replicas for charge-changing mutations (default: CHARGE_NREP)
        extreme_nrep: Number of replicas for extreme mutations (default: EXTREME_NREP)
                     Extreme mutations are both bulky/aromatic AND charge-changing
        multi_mutation_multiplier: Multiplier per additional mutation beyond the first
                                   (default: MULTI_MUTATION_MULTIPLIER)

    Returns:
        Number of lambda replicas to use
    """
    num_mutations = len(mutations)

    if num_mutations > 1:
        individual_nreps = [
            calculate_nrep([m], default_nrep, difficult_nrep, charge_nrep,
                           extreme_nrep, multi_mutation_multiplier)
            for m in mutations
        ]
        avg_nrep = sum(individual_nreps) / num_mutations
        multiplier = multi_mutation_multiplier ** (num_mutations - 1)
        return int(avg_nrep * multiplier)

    is_difficult, has_charge_change, is_extreme = assess_mutation_difficulty(mutations)

    if is_extreme:
        return extreme_nrep
    elif has_charge_change:
        return charge_nrep
    elif is_difficult:
        return difficult_nrep
    else:
        return default_nrep

def generate_para_conf_content(mutations: Sequence[Mutation],
                               default_nrep: int = DEFAULT_NREP,
                               difficult_nrep: int = DIFFICULT_NREP,
                               charge_nrep: int = CHARGE_NREP,
                               extreme_nrep: int = EXTREME_NREP,
                               multi_mutation_multiplier: float = MULTI_MUTATION_MULTIPLIER,
                               always_write: bool = True) -> str:
    """
    Generate the content for para_conf.zsh file.

    Args:
        mutations: Sequence of Mutation objects
        default_nrep: Number of replicas for easy mutations (default: DEFAULT_NREP)
        difficult_nrep: Number of replicas for difficult/bulky mutations (default: DIFFICULT_NREP)
        charge_nrep: Number of replicas for charge-changing mutations (default: CHARGE_NREP)
        extreme_nrep: Number of replicas for extreme mutations (default: EXTREME_NREP)
        multi_mutation_multiplier: Multiplier per additional mutation (default: MULTI_MUTATION_MULTIPLIER)
        always_write: If True, always write NREP even for easy mutations (default: True)

    Returns:
        Content string for para_conf.zsh
    """
    nrep = calculate_nrep(
        mutations,
        default_nrep=default_nrep,
        difficult_nrep=difficult_nrep,
        charge_nrep=charge_nrep,
        extreme_nrep=extreme_nrep,
        multi_mutation_multiplier=multi_mutation_multiplier,
    )

    if always_write or nrep != default_nrep:
        return f"NREP={nrep}\n"
    else:
        return ""

