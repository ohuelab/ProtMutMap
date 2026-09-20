"""Sampson/Rocklin net-charge and undersolvation corrections for mutation RBFE.

Adapted from FEPsuite abfe/charge_correction_generator.py.
Reference: Rocklin et al., J. Chem. Phys. 139, 184103 (2013).

The correction uses box size and endpoint charges, and returns kcal/mol.
It vanishes when both endpoints have zero total charge. Residual integrated
potential, empirical and discrete-solvent corrections are not implemented.
Box dimensions must represent the equilibrated simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Physical constants (kJ/mol unit system, then converted to kcal/mol on output)
_AVOGADRO = 6.02214076e23  # 1/mol
_EPS0 = 8.8541878128e-12 / 1.602176634e-19**2 / _AVOGADRO / 1e9 * 1e3  # e^2/(kJ/mol)/nm
_KE = 1.0 / (4.0 * math.pi * _EPS0)  # kJ/mol·nm/e^2
_KCAL_PER_KJ = 1.0 / 4.184

# Lattice-sum constant for cubic box (Rocklin Table I, "Cubic")
# ξ_LS = ζ_LS * (4π/3)^(1/3); ζ_LS for cubic ≈ -2.837297
_XI_LS_CUBIC = -2.837297

def amino_acid_net_charge(aa1: str) -> int:
    """Net charge of an amino acid at neutral pH (Asp/Glu = -1, Lys/Arg = +1, His ≈ 0).

    Used to compute ΔQL for a single-residue mutation.
    """
    aa1 = aa1.upper()
    if aa1 in ("D", "E"):  # Asp, Glu
        return -1
    if aa1 in ("K", "R"):  # Lys, Arg (His treated as neutral by default)
        return 1
    return 0

def parse_mutation_charge_change(mutation_fs: str) -> tuple[int, int]:
    """Return (Q_WT, Q_mutant) for a single fs-style mutation, e.g. 'E:99Q'.

    The fs format only has the target residue letter; the from-residue must be
    inferred from the surrounding context. Here we accept the standard 1-char-prefix
    form 'RE99Q' (R→Q at chain E pos 99) and the fs short form 'E:99Q' (the
    from-residue is unknown, caller must supply Q_WT separately).
    """
    s = mutation_fs.strip()
    # Long form: <fromAA><chain><resid><toAA>, e.g. RE99Q
    if len(s) >= 5 and s[0].isalpha() and not s[1].isdigit() and ":" not in s:
        from_aa = s[0]
        to_aa = s[-1]
        return amino_acid_net_charge(from_aa), amino_acid_net_charge(to_aa)
    # fs form: <chain>:<resid><toAA>, no from-AA encoded
    raise ValueError(
        f"Cannot infer Q_WT from fs-only form '{mutation_fs}'. "
        "Pass full mutation label like 'RE99Q' or supply (q_wt, q_mut) explicitly."
    )

def analytical_correction_kcal(
    delta_q: float,
    *,
    box_length_nm: float,
    epsS: float = 78.0,
    qp: float = 0.0,
) -> dict[str, float]:
    """Analytical PBC + undersolvation correction (Rocklin net + usv terms).

    Args:
        delta_q: mutant charge minus WT charge (= ΔQL).
        box_length_nm: cubic box edge length (Lref) in nm.
        epsS: solvent dielectric (TIP3P at 300 K ≈ 78; AMBER+TIP3P literature 78–100).
        qp: net charge of the "rest of system" (everything not mutating). When the
            box is neutralized by counter-ions per endpoint, qp+QL_A=qp+QL_B=0 and
            this reduces the correction to zero.

    Returns dict with keys ddG_net_kcal, ddG_usv_kcal, ddG_total_kcal and
    intermediate kJ values for transparency.
    """
    QL_A = 0.0  # WT charge of the ligand fragment is the reference; we work with delta
    QL_B = float(delta_q)
    L = float(box_length_nm)
    # ((QP + QL_B)^2 - (QP + QL_A)^2)
    factor = (qp + QL_B) ** 2 - (qp + QL_A) ** 2
    ddG_net_kJ = -0.5 * _KE * _XI_LS_CUBIC * factor / L
    ddG_usv_kJ = 0.5 * _KE * _XI_LS_CUBIC * (1.0 - 1.0 / epsS) * factor / L
    ddG_total_kJ = ddG_net_kJ + ddG_usv_kJ
    return {
        "ddG_net_kJ": ddG_net_kJ,
        "ddG_usv_kJ": ddG_usv_kJ,
        "ddG_net_kcal": ddG_net_kJ * _KCAL_PER_KJ,
        "ddG_usv_kcal": ddG_usv_kJ * _KCAL_PER_KJ,
        "ddG_total_kcal": ddG_total_kJ * _KCAL_PER_KJ,
        "delta_q": delta_q,
        "box_length_nm": L,
        "epsS": epsS,
        "qp": qp,
    }

@dataclass
class EdgeCorrection:
    """Per-edge correction record for one (from, to) directed mutation step."""
    from_mutation: str
    to_mutation: str
    delta_q: float
    correction_complex_kcal: float
    correction_partner1_kcal: float
    correction_partner2_kcal: float
    correction_ddG_kcal: float  # complex - partner1 - partner2

    @property
    def applies(self) -> bool:
        return abs(self.delta_q) > 1e-9

def _compute_edge_delta_q(from_mut: str, to_mut: str) -> float:
    """ΔQ_total for the directed edge from_mut → to_mut.

    Sums per-residue charge changes for residues that differ between endpoints.
    Uses long-form mutation labels like 'RE99Q' to infer from/to AA.
    """
    from_set = set() if from_mut == "WT" else set(from_mut.split(","))
    to_set = set() if to_mut == "WT" else set(to_mut.split(","))
    diff = (to_set - from_set) | (from_set - to_set)

    delta_q = 0.0
    for mut_str in diff:
        # mut_str like 'RE99Q'. Infer from/to charges.
        if len(mut_str) < 5 or not mut_str[0].isalpha():
            continue
        from_aa = mut_str[0]
        to_aa = mut_str[-1]
        if mut_str in to_set and mut_str not in from_set:
            # Going from from_aa to to_aa
            delta_q += amino_acid_net_charge(to_aa) - amino_acid_net_charge(from_aa)
        else:
            # Reversing: from to_aa back to from_aa (net charge change is opposite)
            delta_q += amino_acid_net_charge(from_aa) - amino_acid_net_charge(to_aa)
    return delta_q

def correction_for_edge(
    from_mut: str,
    to_mut: str,
    *,
    box_length_complex_nm: float,
    box_length_partner1_nm: float | None,
    box_length_partner2_nm: float | None,
    epsS: float = 78.0,
) -> EdgeCorrection:
    """Compute ΔΔG_corr for one RBFE edge.

    For each FEP leg (complex / partner1 / partner2), the analytical correction
    depends on the box size of that leg. Combined: ddG_corr = c_complex - c_p1 - c_p2.

    If a partner is None (e.g. ligand-side absent), its correction is 0 and the
    binding ddG correction reduces accordingly.
    """
    delta_q = _compute_edge_delta_q(from_mut, to_mut)

    def _maybe_corr(L: float | None) -> float:
        if L is None or L <= 0 or abs(delta_q) < 1e-9:
            return 0.0
        return analytical_correction_kcal(delta_q, box_length_nm=L, epsS=epsS)["ddG_total_kcal"]

    c_complex = _maybe_corr(box_length_complex_nm)
    c_p1 = _maybe_corr(box_length_partner1_nm)
    c_p2 = _maybe_corr(box_length_partner2_nm)
    ddG_corr = c_complex - c_p1 - c_p2

    return EdgeCorrection(
        from_mutation=from_mut,
        to_mutation=to_mut,
        delta_q=delta_q,
        correction_complex_kcal=c_complex,
        correction_partner1_kcal=c_p1,
        correction_partner2_kcal=c_p2,
        correction_ddG_kcal=ddG_corr,
    )

def read_box_length_from_gro(gro_path: Path) -> float | None:
    """Read cubic box edge from the last line of a GRO file. Returns None on error.

    Triclinic boxes (non-zero off-diagonal) get the smallest diagonal length;
    callers should validate roughly cubic geometry before trusting the value.
    """
    try:
        with open(gro_path) as fp:
            lines = fp.readlines()
        if not lines:
            return None
        # GRO box line is the LAST line; format: "  L_x L_y L_z [v_x_y v_x_z ...]"
        parts = lines[-1].strip().split()
        if len(parts) < 3:
            return None
        Lx, Ly, Lz = float(parts[0]), float(parts[1]), float(parts[2])
        return float(min(Lx, Ly, Lz))  # conservative
    except Exception:
        return None

def cell_volume_from_cryst1(a_A: float, b_A: float, c_A: float,
                             alpha_deg: float, beta_deg: float, gamma_deg: float) -> float:
    """Triclinic cell volume in nm^3 from CRYST1 record values (Å, degrees)."""
    a = a_A / 10.0
    b = b_A / 10.0
    c = c_A / 10.0
    al = math.radians(alpha_deg)
    be = math.radians(beta_deg)
    ga = math.radians(gamma_deg)
    factor = (1.0
              + 2.0 * math.cos(al) * math.cos(be) * math.cos(ga)
              - math.cos(al) ** 2 - math.cos(be) ** 2 - math.cos(ga) ** 2)
    if factor < 0:
        return 0.0
    return a * b * c * math.sqrt(factor)

def read_box_lref_from_pdb(pdb_path: Path) -> float | None:
    """Return Lref = V^(1/3) (nm) from a CRYST1 line in a PDB.

    Rocklin recommends using the cube-root of the cell volume as the effective
    box edge for non-cubic cells (rhombic dodecahedron, octahedron, etc.).
    Returns None if no CRYST1 line or the cell is degenerate.
    """
    try:
        with open(pdb_path) as fp:
            for line in fp:
                if line.startswith("CRYST1"):
                    # Fixed-width record per PDB spec (1-based 7-15 a, 16-24 b, 25-33 c, ...)
                    a = float(line[6:15].strip())
                    b = float(line[15:24].strip())
                    c = float(line[24:33].strip())
                    al = float(line[33:40].strip())
                    be = float(line[40:47].strip())
                    ga = float(line[47:54].strip())
                    V = cell_volume_from_cryst1(a, b, c, al, be, ga)
                    if V <= 0:
                        return None
                    return float(V ** (1.0 / 3.0))
    except Exception:
        return None
    return None

def find_box_length_for_edge(
    base_target_dir: Path,
    mode: str,
    from_fs: str,
    diff_fs: str,
) -> float | None:
    """Locate conf_ionized.pdb (or fallback) for a FEP edge and return Lref in nm.

    Mirrors gather_results._process_single_edge directory layout.
    """
    target_dir = base_target_dir / mode / from_fs / f"wt_{diff_fs}"
    for candidate in ("conf_ionized.pdb", "conf_solvated.pdb", "conf_box.pdb", "fepbase.pdb"):
        p = target_dir / candidate
        if p.exists():
            L = read_box_lref_from_pdb(p)
            if L is not None:
                return L
    return None

def apply_corrections_to_edges_df(
    edges_df: pd.DataFrame,
    box_length_lookup: dict[tuple[str, str, str], float] | None = None,
    *,
    default_box_nm: float = 8.0,
    epsS: float = 78.0,
    only_charge_edges: bool = True,
) -> pd.DataFrame:
    """Augment a calc_df with `correction_ddG_kcal` and `calc_ddG_corrected` columns.

    box_length_lookup: optional dict keyed by (from, to, partner_mode) → box_nm.
        If absent, uses default_box_nm everywhere. Caller is expected to populate
        this from .gro files in the FEP work directories before invoking the
        downstream gather_results pattern computation.

    only_charge_edges: when True (default), correction = 0 for has_charge_change=False
        edges (saves work and avoids spurious tiny corrections from rounding).
    """
    out = edges_df.copy()
    rows = []
    for _, row in out.iterrows():
        f, t = row["from_mutation"], row["to_mutation"]
        if only_charge_edges and "has_charge_change" in row.index and not bool(row.get("has_charge_change", False)):
            rows.append({"correction_ddG_kcal": 0.0, "delta_q": 0.0})
            continue

        def _lookup(mode: str) -> float | None:
            if box_length_lookup is None:
                return default_box_nm
            return box_length_lookup.get((f, t, mode), default_box_nm)

        edge_corr = correction_for_edge(
            f, t,
            box_length_complex_nm=_lookup("complex"),
            box_length_partner1_nm=_lookup("partner1"),
            box_length_partner2_nm=_lookup("partner2"),
            epsS=epsS,
        )
        rows.append({
            "correction_ddG_kcal": edge_corr.correction_ddG_kcal,
            "delta_q": edge_corr.delta_q,
        })

    correction_df = pd.DataFrame(rows, index=out.index)
    out["correction_ddG_kcal"] = correction_df["correction_ddG_kcal"]
    out["delta_q"] = correction_df["delta_q"]
    if "calc_ddG" in out.columns:
        out["calc_ddG_corrected"] = out["calc_ddG"].astype(float) + out["correction_ddG_kcal"].astype(float)
    return out
