"""
Tests for lambda_calculator module.
"""

from protmutmap.tools.lambda_calculator import (
    calculate_charge,
    assess_mutation_difficulty,
    calculate_nrep,
    generate_para_conf_content,
    DEFAULT_NREP,
    DIFFICULT_NREP,
    CHARGE_NREP,
    EXTREME_NREP,
    MULTI_MUTATION_MULTIPLIER,
)
from protmutmap.tools.mutation import Mutation


class TestCalculateCharge:
    """Tests for calculate_charge function."""

    def test_negative_charge_residues(self):
        """Test that D and E return -1."""
        assert calculate_charge("D") == -1
        assert calculate_charge("E") == -1

    def test_positive_charge_residues(self):
        """Test that R and K return +1."""
        assert calculate_charge("R") == +1
        assert calculate_charge("K") == +1

    def test_neutral_residues(self):
        """Test that other residues return 0."""
        neutral_residues = "ACFGHILMNPSQTVWY"
        for residue in neutral_residues:
            assert calculate_charge(residue) == 0, f"Residue {residue} should have charge 0"


class TestAssessMutationDifficulty:
    """Tests for assess_mutation_difficulty function."""

    def test_simple_mutation_not_difficult(self):
        """Test that simple mutations (e.g., A->V) are not difficult."""
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="V")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert difficult is False
        assert charge_change is False
        assert extreme is False

    def test_bulky_aromatic_residues_difficult(self):
        """Test that mutations involving P, F, Y, W are difficult."""
        # Phenylalanine
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="F")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert difficult is True
        assert charge_change is False
        assert extreme is False

        # Tyrosine
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="Y")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert difficult is True

        # Tryptophan
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="W")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert difficult is True

        # Proline
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="P")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert difficult is True

        # From bulky to non-bulky
        mutation = Mutation(chain="A", resid=123, before_res="F", after_res="A")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert difficult is True

    def test_fy_exception_not_difficult(self):
        """Test that F<->Y mutations are not considered difficult."""
        mutation = Mutation(chain="A", resid=123, before_res="F", after_res="Y")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert difficult is False

        mutation = Mutation(chain="A", resid=123, before_res="Y", after_res="F")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert difficult is False

    def test_charge_change_detection(self):
        """Test that charge changes are detected correctly."""
        # Neutral to positive
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="K")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert charge_change is True
        assert extreme is False

        # Neutral to negative
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="D")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert charge_change is True

        # Positive to neutral
        mutation = Mutation(chain="A", resid=123, before_res="K", after_res="A")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert charge_change is True

        # Negative to neutral
        mutation = Mutation(chain="A", resid=123, before_res="D", after_res="A")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert charge_change is True

        # Positive to negative (net change of -2)
        mutation = Mutation(chain="A", resid=123, before_res="K", after_res="D")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert charge_change is True

        # No charge change
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="V")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert charge_change is False

        # Same charge type (positive to positive)
        mutation = Mutation(chain="A", resid=123, before_res="K", after_res="R")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert charge_change is False

    def test_extreme_mutation(self):
        """Test that extreme mutations (bulky + charge change) are detected."""
        # Bulky aromatic with charge change
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="F")
        # First make it bulky, then add charge change
        mutation2 = Mutation(chain="A", resid=124, before_res="A", after_res="K")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation, mutation2])
        assert difficult is True
        assert charge_change is True
        assert extreme is True

        # Single mutation: bulky + charge change
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="K")
        # This is charge change but not bulky, so not extreme
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert extreme is False

        # Single mutation: bulky aromatic to charged
        mutation = Mutation(chain="A", resid=123, before_res="F", after_res="K")
        difficult, charge_change, extreme = assess_mutation_difficulty([mutation])
        assert difficult is True
        assert charge_change is True
        assert extreme is True

    def test_multiple_mutations(self):
        """Test assessment with multiple mutations."""
        mutations = [
            Mutation(chain="A", resid=123, before_res="A", after_res="V"),
            Mutation(chain="A", resid=124, before_res="A", after_res="L"),
        ]
        difficult, charge_change, extreme = assess_mutation_difficulty(mutations)
        assert difficult is False
        assert charge_change is False
        assert extreme is False

        # Multiple mutations with charge changes that cancel out
        mutations = [
            Mutation(chain="A", resid=123, before_res="A", after_res="K"),  # +1
            Mutation(chain="A", resid=124, before_res="A", after_res="D"),  # -1
        ]
        difficult, charge_change, extreme = assess_mutation_difficulty(mutations)
        assert charge_change is False  # Net charge change is 0

        # Multiple mutations with net charge change
        mutations = [
            Mutation(chain="A", resid=123, before_res="A", after_res="K"),  # +1
            Mutation(chain="A", resid=124, before_res="A", after_res="K"),  # +1
        ]
        difficult, charge_change, extreme = assess_mutation_difficulty(mutations)
        assert charge_change is True  # Net charge change is +2


class TestCalculateNrep:
    """Tests for calculate_nrep function."""

    def test_default_mutation(self):
        """Test that simple mutations use default_nrep."""
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="V")
        nrep = calculate_nrep([mutation])
        assert nrep == DEFAULT_NREP

    def test_difficult_mutation(self):
        """Test that difficult mutations use difficult_nrep."""
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="F")
        nrep = calculate_nrep([mutation])
        assert nrep == DIFFICULT_NREP

    def test_charge_change_mutation(self):
        """Test that charge-changing mutations use charge_nrep."""
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="K")
        nrep = calculate_nrep([mutation])
        assert nrep == CHARGE_NREP

    def test_extreme_mutation(self):
        """Test that extreme mutations use extreme_nrep."""
        # Single mutation that is both bulky and charge-changing
        mutation = Mutation(chain="A", resid=123, before_res="F", after_res="K")
        nrep = calculate_nrep([mutation])
        assert nrep == EXTREME_NREP

        # Two mutations: F(difficult=48) + K(charge=64) → avg(48,64)*1.5 = 84
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="F")
        mutation2 = Mutation(chain="A", resid=124, before_res="A", after_res="K")
        nrep = calculate_nrep([mutation, mutation2])
        expected = int((DIFFICULT_NREP + CHARGE_NREP) / 2 * MULTI_MUTATION_MULTIPLIER)
        assert nrep == expected

    def test_multiple_mutations_scaling(self):
        """Test that multiple mutations use avg NREP * multiplier^(n-1)."""
        # Two standard mutations: avg(32,32)*1.5 = 48
        mutations = [
            Mutation(chain="A", resid=123, before_res="A", after_res="V"),
            Mutation(chain="A", resid=124, before_res="A", after_res="L"),
        ]
        nrep = calculate_nrep(mutations)
        expected = int(DEFAULT_NREP * MULTI_MUTATION_MULTIPLIER)
        assert nrep == expected

        # Three standard mutations: avg(32,32,32)*1.5^2 = 72
        mutations = [
            Mutation(chain="A", resid=123, before_res="A", after_res="V"),
            Mutation(chain="A", resid=124, before_res="A", after_res="L"),
            Mutation(chain="A", resid=125, before_res="A", after_res="I"),
        ]
        nrep = calculate_nrep(mutations)
        expected = int(DEFAULT_NREP * (MULTI_MUTATION_MULTIPLIER ** 2))
        assert nrep == expected

        # Two difficult mutations: avg(48,48)*1.5 = 72
        mutations = [
            Mutation(chain="A", resid=123, before_res="A", after_res="F"),
            Mutation(chain="A", resid=124, before_res="A", after_res="Y"),
        ]
        nrep = calculate_nrep(mutations)
        expected = int(DIFFICULT_NREP * MULTI_MUTATION_MULTIPLIER)
        assert nrep == expected

        # Mixed std+dif: avg(32,48)*1.5 = 60
        mutations = [
            Mutation(chain="A", resid=123, before_res="A", after_res="V"),
            Mutation(chain="A", resid=124, before_res="A", after_res="F"),
        ]
        nrep = calculate_nrep(mutations)
        expected = int((DEFAULT_NREP + DIFFICULT_NREP) / 2 * MULTI_MUTATION_MULTIPLIER)
        assert nrep == expected

    def test_custom_parameters(self):
        """Test that custom parameters are used correctly."""
        # Easy → default_nrep
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="V")
        nrep = calculate_nrep([mutation], default_nrep=8, difficult_nrep=24, charge_nrep=32, extreme_nrep=48)
        assert nrep == 8

        # Bulky → difficult_nrep
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="F")
        nrep = calculate_nrep([mutation], default_nrep=8, difficult_nrep=24, charge_nrep=32, extreme_nrep=48)
        assert nrep == 24

        # Charge-only → charge_nrep
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="K")
        nrep = calculate_nrep([mutation], default_nrep=8, difficult_nrep=24, charge_nrep=32, extreme_nrep=48)
        assert nrep == 32

        # Extreme (bulky + charge) → extreme_nrep
        mutation = Mutation(chain="A", resid=123, before_res="F", after_res="K")
        nrep = calculate_nrep([mutation], default_nrep=8, difficult_nrep=24, charge_nrep=32, extreme_nrep=48)
        assert nrep == 48

        # Custom multiplier
        mutations = [
            Mutation(chain="A", resid=123, before_res="A", after_res="V"),
            Mutation(chain="A", resid=124, before_res="A", after_res="L"),
        ]
        nrep = calculate_nrep(mutations, multi_mutation_multiplier=1.5)
        expected = int(DEFAULT_NREP * 1.5)
        assert nrep == expected

    def test_charge_nrep_separate_from_difficult(self):
        """Test that charge-only and difficult-only mutations use different nrep values."""
        # Charge-only: A->K (not bulky)
        charge_mut = Mutation(chain="A", resid=123, before_res="A", after_res="K")
        nrep_charge = calculate_nrep([charge_mut])
        assert nrep_charge == CHARGE_NREP

        # Difficult-only: A->F (bulky, no charge change)
        difficult_mut = Mutation(chain="A", resid=123, before_res="A", after_res="F")
        nrep_difficult = calculate_nrep([difficult_mut])
        assert nrep_difficult == DIFFICULT_NREP

        assert CHARGE_NREP != DIFFICULT_NREP


class TestGenerateParaConfContent:
    """Tests for generate_para_conf_content function."""

    def test_default_mutation_always_write(self):
        """Test that default mutations generate content when always_write=True."""
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="V")
        content = generate_para_conf_content([mutation], always_write=True)
        assert content == f"NREP={DEFAULT_NREP}\n"

    def test_default_mutation_no_always_write(self):
        """Test that default mutations don't generate content when always_write=False."""
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="V")
        content = generate_para_conf_content([mutation], always_write=False)
        assert content == ""

    def test_difficult_mutation_always_write(self):
        """Test that difficult mutations always generate content."""
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="F")
        content = generate_para_conf_content([mutation], always_write=True)
        assert content == f"NREP={DIFFICULT_NREP}\n"

        content = generate_para_conf_content([mutation], always_write=False)
        assert content == f"NREP={DIFFICULT_NREP}\n"  # Still writes because != default

    def test_extreme_mutation(self):
        """Test that extreme mutations generate correct content."""
        mutation = Mutation(chain="A", resid=123, before_res="F", after_res="K")
        content = generate_para_conf_content([mutation], always_write=True)
        assert content == f"NREP={EXTREME_NREP}\n"

    def test_charge_mutation(self):
        """Test that charge-changing mutations generate correct NREP."""
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="K")
        content = generate_para_conf_content([mutation], always_write=True)
        assert content == f"NREP={CHARGE_NREP}\n"

        content = generate_para_conf_content([mutation], charge_nrep=80, always_write=True)
        assert content == "NREP=80\n"

    def test_custom_parameters(self):
        """Test that custom parameters are used correctly."""
        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="V")
        content = generate_para_conf_content(
            [mutation],
            default_nrep=8,
            difficult_nrep=24,
            charge_nrep=32,
            extreme_nrep=48,
            always_write=True
        )
        assert content == "NREP=8\n"

        mutation = Mutation(chain="A", resid=123, before_res="A", after_res="F")
        content = generate_para_conf_content(
            [mutation],
            default_nrep=8,
            difficult_nrep=24,
            charge_nrep=32,
            extreme_nrep=48,
            always_write=False
        )
        assert content == "NREP=24\n"

    def test_multiple_mutations(self):
        """Test content generation for multiple mutations."""
        mutations = [
            Mutation(chain="A", resid=123, before_res="A", after_res="V"),
            Mutation(chain="A", resid=124, before_res="A", after_res="L"),
        ]
        content = generate_para_conf_content(mutations, always_write=True)
        expected_nrep = int(DEFAULT_NREP * MULTI_MUTATION_MULTIPLIER)
        assert content == f"NREP={expected_nrep}\n"
