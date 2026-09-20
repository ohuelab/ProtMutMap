"""
Tests for mutations.py module.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from protmutmap.tools.mutations import MutationList
from protmutmap.tools.mutation import Mutation


class TestMutationListInitialization:
    """Test MutationList initialization with various input types."""

    def test_init_empty(self):
        """Test initialization with no arguments."""
        mut_list = MutationList()
        assert len(mut_list) == 0
        assert mut_list.to_list() == []

    def test_init_with_mutation_objects(self):
        """Test initialization with Mutation objects."""
        mut1 = Mutation(chain="A", resid=100, before_res="H", after_res="C", icode="")
        mut2 = Mutation(chain="B", resid=101, before_res="Y", after_res="A", icode="")
        mut_list = MutationList([mut1, mut2])
        assert len(mut_list) == 2
        assert mut_list[0] == mut1
        assert mut_list[1] == mut2

    def test_init_with_5_tuple(self):
        """Test initialization with 5-tuple format (wt_aa, chain, seq_idx, icode, mut_aa)."""
        mut_list = MutationList([("H", "A", 100, "", "C")])
        assert len(mut_list) == 1
        assert mut_list[0].chain == "A"
        assert mut_list[0].resid == 100
        assert mut_list[0].before_res == "H"
        assert mut_list[0].after_res == "C"
        assert mut_list[0].icode == ""

    def test_init_with_4_tuple(self):
        """Test initialization with 4-tuple format (chain, seq_idx, icode, mut_aa)."""
        mut_list = MutationList([("A", 100, "", "C")])
        assert len(mut_list) == 1
        assert mut_list[0].chain == "A"
        assert mut_list[0].resid == 100
        assert mut_list[0].before_res is None
        assert mut_list[0].after_res == "C"

    def test_init_with_icode(self):
        """Test initialization with insertion code."""
        mut_list = MutationList([("H", "A", 101, "a", "C")])
        assert len(mut_list) == 1
        assert mut_list[0].icode == "a"

    def test_init_with_invalid_tuple_length(self):
        """Test initialization with invalid tuple length."""
        with pytest.raises(ValueError, match="Invalid mutation tuple"):
            MutationList([("A", 100)])

    def test_init_with_invalid_seq_idx_type(self):
        """Test initialization with non-integer seq_idx."""
        with pytest.raises(ValueError, match="seq_idx must be integer"):
            MutationList([("A", "100", "", "C")])

    def test_init_with_invalid_aa_code(self):
        """Test initialization with invalid amino acid code."""
        with pytest.raises(ValueError, match="Invalid amino acid code"):
            MutationList([("A", 100, "", "Z")])

    def test_init_with_invalid_wt_aa_code(self):
        """Test initialization with invalid wild-type amino acid code."""
        with pytest.raises(ValueError, match="Invalid wild-type amino acid code"):
            MutationList([("Z", "A", 100, "", "C")])

    def test_init_with_invalid_type(self):
        """Test initialization with invalid type."""
        with pytest.raises(ValueError, match="Invalid mutation type"):
            MutationList(["invalid"])

    def test_init_with_sort(self):
        """Test initialization with sort=True."""
        mut1 = Mutation(chain="B", resid=101, before_res="Y", after_res="A", icode="")
        mut2 = Mutation(chain="A", resid=100, before_res="H", after_res="C", icode="")
        mut_list = MutationList([mut1, mut2], sort=True)
        assert mut_list[0].chain == "A"
        assert mut_list[0].resid == 100
        assert mut_list[1].chain == "B"
        assert mut_list[1].resid == 101


class TestMutationListFromString:
    """Test MutationList.from_string method."""

    def test_from_string_single(self):
        """Test from_string with single mutation."""
        mut_list = MutationList.from_string("HA100C")
        assert len(mut_list) == 1
        assert mut_list[0].chain == "A"
        assert mut_list[0].resid == 100
        assert mut_list[0].before_res == "H"
        assert mut_list[0].after_res == "C"

    def test_from_string_multiple_comma(self):
        """Test from_string with comma-separated mutations."""
        mut_list = MutationList.from_string("HA100C,YB101A")
        assert len(mut_list) == 2
        assert mut_list[0].chain == "A"
        assert mut_list[1].chain == "B"

    def test_from_string_multiple_semicolon(self):
        """Test from_string with semicolon-separated mutations."""
        mut_list = MutationList.from_string("HA100C;YB101A")
        assert len(mut_list) == 2

    def test_from_string_with_icode(self):
        """Test from_string with insertion code."""
        mut_list = MutationList.from_string("HA101aC")
        assert len(mut_list) == 1
        assert mut_list[0].resid == 101
        assert mut_list[0].icode == "a"

    def test_from_string_empty(self):
        """Test from_string with empty string."""
        mut_list = MutationList.from_string("")
        assert len(mut_list) == 0

    def test_from_string_whitespace(self):
        """Test from_string with whitespace."""
        mut_list = MutationList.from_string(" HA100C , YB101A ")
        assert len(mut_list) == 2

    def test_from_string_invalid(self):
        """Test from_string with invalid mutation string."""
        with pytest.raises(ValueError, match="Bad mutation token"):
            MutationList.from_string("INVALID")


class TestMutationListFromList:
    """Test MutationList.from_list method."""

    def test_from_list_single(self):
        """Test from_list with single mutation."""
        mut_list = MutationList.from_list(["HA100C"])
        assert len(mut_list) == 1

    def test_from_list_multiple(self):
        """Test from_list with multiple mutations."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        assert len(mut_list) == 2

    def test_from_list_invalid(self):
        """Test from_list with invalid mutation."""
        with pytest.raises(ValueError, match="Bad mutation"):
            MutationList.from_list(["INVALID"])


class TestMutationListFromFSMutations:
    """Test MutationList.from_fs_mutations method."""

    def test_from_fs_mutations_single(self):
        """Test from_fs_mutations with single mutation."""
        mut_list = MutationList.from_fs_mutations(["A:100C"])
        assert len(mut_list) == 1
        assert mut_list[0].chain == "A"
        assert mut_list[0].resid == 100
        assert mut_list[0].after_res == "C"
        assert mut_list[0].before_res is None

    def test_from_fs_mutations_multiple(self):
        """Test from_fs_mutations with multiple mutations."""
        mut_list = MutationList.from_fs_mutations(["A:100C", "B:101A"])
        assert len(mut_list) == 2

    def test_from_fs_mutations_with_icode(self):
        """Test from_fs_mutations with insertion code."""
        mut_list = MutationList.from_fs_mutations(["A:101aC"])
        assert len(mut_list) == 1
        assert mut_list[0].resid == 101
        assert mut_list[0].icode == "a"

    def test_from_fs_mutations_invalid(self):
        """Test from_fs_mutations with invalid format."""
        with pytest.raises(ValueError, match="Bad FEPSuite mutation"):
            MutationList.from_fs_mutations(["INVALID"])


class TestMutationListFromMutations:
    """Test MutationList.from_mutations method."""

    def test_from_mutations(self):
        """Test from_mutations with Mutation objects."""
        mut1 = Mutation(chain="A", resid=100, before_res="H", after_res="C", icode="")
        mut2 = Mutation(chain="B", resid=101, before_res="Y", after_res="A", icode="")
        mut_list = MutationList.from_mutations([mut1, mut2])
        assert len(mut_list) == 2


class TestMutationListFromTuples:
    """Test MutationList.from_tuples method."""

    def test_from_tuples_5_tuple(self):
        """Test from_tuples with 5-tuple format."""
        mut_list = MutationList.from_tuples([("H", "A", 100, "", "C")])
        assert len(mut_list) == 1

    def test_from_tuples_4_tuple(self):
        """Test from_tuples with 4-tuple format."""
        mut_list = MutationList.from_tuples([("A", 100, "", "C")])
        assert len(mut_list) == 1


class TestMutationListToString:
    """Test MutationList.to_string method."""

    def test_to_string_standard(self):
        """Test to_string with standard format."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        result = mut_list.to_string()
        assert result == "HA100C,YB101A"

    def test_to_string_fs(self):
        """Test to_string with FEPSuite format."""
        mut_list = MutationList.from_fs_mutations(["A:100C", "B:101A"])
        result = mut_list.to_string(fs=True)
        assert result == "A:100C_B:101A"

    def test_to_string_empty(self):
        """Test to_string with empty list."""
        mut_list = MutationList()
        assert mut_list.to_string() == ""


class TestMutationListToList:
    """Test MutationList.to_list method."""

    def test_to_list(self):
        """Test to_list method."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        result = mut_list.to_list()
        assert result == ["HA100C", "YB101A"]

    def test_to_list_without_wt(self):
        """Test to_list with mutations without wild-type AA."""
        mut_list = MutationList.from_fs_mutations(["A:100C"])
        result = mut_list.to_list()
        assert result == ["XA100C"]  # Uses 'X' as placeholder


class TestMutationListToFSMutations:
    """Test MutationList.to_fs_mutations method."""

    def test_to_fs_mutations(self):
        """Test to_fs_mutations method."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        result = mut_list.to_fs_mutations()
        assert result == ["A:100C", "B:101A"]

    def test_to_fs_mutations_with_icode(self):
        """Test to_fs_mutations with insertion code."""
        mut_list = MutationList.from_list(["HA101aC"])
        result = mut_list.to_fs_mutations()
        assert result == ["A:101aC"]


class TestMutationListToMutations:
    """Test MutationList.to_mutations method."""

    def test_to_mutations(self):
        """Test to_mutations method."""
        mut1 = Mutation(chain="A", resid=100, before_res="H", after_res="C", icode="")
        mut2 = Mutation(chain="B", resid=101, before_res="Y", after_res="A", icode="")
        mut_list = MutationList([mut1, mut2])
        result = mut_list.to_mutations()
        assert len(result) == 2
        assert result[0] == mut1
        assert result[1] == mut2


class TestMutationListToTuples:
    """Test MutationList.to_tuples method."""

    def test_to_tuples(self):
        """Test to_tuples method."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        result = mut_list.to_tuples()
        assert len(result) == 2
        assert result[0] == ("H", "A", 100, "", "C")
        assert result[1] == ("Y", "B", 101, "", "A")


class TestMutationListSort:
    """Test MutationList.sort method."""

    def test_sort(self):
        """Test sort method."""
        mut_list = MutationList.from_list(["YB101A", "HA100C"])
        mut_list.sort()
        assert mut_list[0].chain == "A"
        assert mut_list[0].resid == 100
        assert mut_list[1].chain == "B"
        assert mut_list[1].resid == 101

    def test_sort_reverse(self):
        """Test sort with reverse=True."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        mut_list.sort(reverse=True)
        assert mut_list[0].chain == "B"
        assert mut_list[1].chain == "A"


class TestMutationListGetMutationChains:
    """Test MutationList.get_mutation_chains method."""

    def test_get_mutation_chains(self):
        """Test get_mutation_chains method."""
        mut_list = MutationList.from_list(["HA100C", "YB101A", "HC102D"])
        chains = mut_list.get_mutation_chains()
        assert chains == {"A", "B", "C"}


class TestMutationListFilterByChains:
    """Test MutationList.filter_by_chains method."""

    def test_filter_by_chains(self):
        """Test filter_by_chains method."""
        mut_list = MutationList.from_list(["HA100C", "YB101A", "HC102D"])
        filtered = mut_list.filter_by_chains(["A", "C"])
        assert len(filtered) == 2
        assert filtered[0].chain == "A"
        assert filtered[1].chain == "C"


class TestMutationListGenerateCombinations:
    """Test MutationList.generate_combinations method."""

    def test_generate_combinations_mutationlist(self):
        """Test generate_combinations with MutationList return type."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        combinations = mut_list.generate_combinations('MutationList')
        assert len(combinations) == 4  # 2^2 = 4 combinations (including wild type)
        assert len(combinations[0]) == 0  # Wild type
        assert len(combinations[1]) == 1
        assert len(combinations[2]) == 1
        assert len(combinations[3]) == 2

    def test_generate_combinations_mutations(self):
        """Test generate_combinations with mutations return type."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        combinations = mut_list.generate_combinations('mutations')
        assert len(combinations) == 4
        assert combinations[0] == []  # Wild type
        assert combinations[1] == ["HA100C"]
        assert combinations[2] == ["YB101A"]
        assert combinations[3] == ["HA100C", "YB101A"]

    def test_generate_combinations_fs_mutations(self):
        """Test generate_combinations with fs_mutations return type."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        combinations = mut_list.generate_combinations('fs_mutations')
        assert len(combinations) == 4
        assert combinations[0] == []  # Wild type
        assert combinations[1] == ["A:100C"]
        assert combinations[2] == ["B:101A"]
        assert combinations[3] == ["A:100C", "B:101A"]

    def test_generate_combinations_strings(self):
        """Test generate_combinations with strings return type."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        combinations = mut_list.generate_combinations('strings')
        assert len(combinations) == 4
        assert combinations[0] == ""  # Wild type
        assert combinations[1] == "HA100C"
        assert combinations[2] == "YB101A"
        assert combinations[3] == "HA100C,YB101A"

    def test_generate_combinations_tuples(self):
        """Test generate_combinations with tuples return type."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        combinations = mut_list.generate_combinations('tuples')
        assert len(combinations) == 4
        assert combinations[0] == []  # Wild type
        assert combinations[1] == [("H", "A", 100, "", "C")]
        assert combinations[2] == [("Y", "B", 101, "", "A")]
        assert combinations[3] == [("H", "A", 100, "", "C"), ("Y", "B", 101, "", "A")]


class TestMutationListValidate:
    """Test MutationList.validate method."""

    def test_validate_valid(self):
        """Test validate with valid mutations."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        assert mut_list.validate() is True

    def test_validate_invalid_after_res(self):
        """Test validate with invalid after_res."""
        mut = Mutation(chain="A", resid=100, before_res="H", after_res="Z", icode="")
        mut_list = MutationList([mut])
        with pytest.raises(ValueError, match="Invalid amino acid code"):
            mut_list.validate()

    def test_validate_invalid_before_res(self):
        """Test validate with invalid before_res."""
        mut = Mutation(chain="A", resid=100, before_res="Z", after_res="C", icode="")
        mut_list = MutationList([mut])
        with pytest.raises(ValueError, match="Invalid wild-type amino acid code"):
            mut_list.validate()


class TestMutationListToThreeLetter:
    """Test MutationList.to_three_letter method."""

    def test_to_three_letter(self):
        """Test to_three_letter method."""
        mut_list = MutationList.from_list(["HA100C"])
        result = mut_list.to_three_letter()
        assert len(result) == 1
        assert result[0][0] == "HIS"  # H -> HIS
        assert result[0][4] == "CYS"  # C -> CYS


class TestMutationListDiff:
    """Test MutationList.diff method."""

    def test_diff_same_mutations(self):
        """Test diff with same mutations."""
        mut_list1 = MutationList.from_list(["HA100C", "YB101A"])
        mut_list2 = MutationList.from_list(["HA100C", "YB101A"])
        diff = mut_list1.diff(mut_list2)
        assert len(diff) == 0  # No difference

    def test_diff_different_mutations(self):
        """Test diff with different mutations."""
        mut_list1 = MutationList.from_list(["HA100C"])
        mut_list2 = MutationList.from_list(["HA100D"])
        diff = mut_list1.diff(mut_list2)
        assert len(diff) == 1
        assert diff[0].before_res == "D"  # From mut_list2
        assert diff[0].after_res == "C"    # From mut_list1

    def test_diff_unique_in_first(self):
        """Test diff with unique mutation in first list."""
        mut_list1 = MutationList.from_list(["HA100C", "YB101A"])
        mut_list2 = MutationList.from_list(["HA100C"])
        diff = mut_list1.diff(mut_list2)
        assert len(diff) == 1
        assert diff[0].chain == "B"
        assert diff[0].resid == 101

    def test_diff_unique_in_second(self):
        """Test diff with unique mutation in second list."""
        mut_list1 = MutationList.from_list(["HA100C"])
        mut_list2 = MutationList.from_list(["HA100C", "YB101A"])
        diff = mut_list1.diff(mut_list2)
        assert len(diff) == 1
        # Should be reverse mutation: A -> Y
        assert diff[0].chain == "B"
        assert diff[0].resid == 101
        assert diff[0].before_res == "A"
        assert diff[0].after_res == "Y"


class TestMutationListSubtract:
    """Test MutationList.__sub__ method."""

    def test_subtract(self):
        """Test subtraction operator."""
        mut_list1 = MutationList.from_list(["HA100C", "YB101A"])
        mut_list2 = MutationList.from_list(["HA100C"])
        diff = mut_list1 - mut_list2
        assert len(diff) == 1
        assert diff[0].chain == "B"


class TestMutationListNeg:
    """Test MutationList.__neg__ method."""

    def test_neg(self):
        """Test unary minus operator (reverse mutations)."""
        mut_list = MutationList.from_list(["HA100C"])
        reversed_mut_list = -mut_list
        assert len(reversed_mut_list) == 1
        assert reversed_mut_list[0].before_res == "C"
        assert reversed_mut_list[0].after_res == "H"

    def test_neg_without_before_res(self):
        """Test unary minus with mutation without before_res."""
        mut_list = MutationList.from_fs_mutations(["A:100C"])
        reversed_mut_list = -mut_list
        # Should not include reverse mutation if before_res is None
        assert len(reversed_mut_list) == 0


class TestMutationListListInterface:
    """Test MutationList list-like interface methods."""

    def test_len(self):
        """Test __len__ method."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        assert len(mut_list) == 2

    def test_iter(self):
        """Test __iter__ method."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        mutations = list(mut_list)
        assert len(mutations) == 2
        assert all(isinstance(m, Mutation) for m in mutations)

    def test_getitem(self):
        """Test __getitem__ method."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        assert mut_list[0].chain == "A"
        assert mut_list[1].chain == "B"

    def test_str(self):
        """Test __str__ method."""
        mut_list = MutationList.from_list(["HA100C", "YB101A"])
        result = str(mut_list)
        assert result == "A:100C,B:101A"  # FEPSuite format

    def test_repr(self):
        """Test __repr__ method."""
        mut_list = MutationList.from_list(["HA100C"])
        result = repr(mut_list)
        assert "MutationList" in result
        assert "A:100C" in result


class TestMutationListEquality:
    """Test MutationList.__eq__ method."""

    def test_eq_same_order(self):
        """Test equality with same order."""
        mut_list1 = MutationList.from_list(["HA100C", "YB101A"])
        mut_list2 = MutationList.from_list(["HA100C", "YB101A"])
        assert mut_list1 == mut_list2

    def test_eq_different_order(self):
        """Test equality with different order."""
        mut_list1 = MutationList.from_list(["HA100C", "YB101A"])
        mut_list2 = MutationList.from_list(["YB101A", "HA100C"])
        assert mut_list1 == mut_list2  # Order should not matter

    def test_eq_different(self):
        """Test equality with different mutations."""
        mut_list1 = MutationList.from_list(["HA100C"])
        mut_list2 = MutationList.from_list(["YB101A"])
        assert mut_list1 != mut_list2

    def test_eq_not_mutationlist(self):
        """Test equality with non-MutationList object."""
        mut_list = MutationList.from_list(["HA100C"])
        assert mut_list != "not a MutationList"


class TestMutationListHash:
    """Test MutationList.__hash__ method."""

    def test_hash_same_mutations(self):
        """Test hash with same mutations."""
        mut_list1 = MutationList.from_list(["HA100C", "YB101A"])
        mut_list2 = MutationList.from_list(["HA100C", "YB101A"])
        assert hash(mut_list1) == hash(mut_list2)

    def test_hash_different_order(self):
        """Test hash with different order."""
        mut_list1 = MutationList.from_list(["HA100C", "YB101A"])
        mut_list2 = MutationList.from_list(["YB101A", "HA100C"])
        assert hash(mut_list1) == hash(mut_list2)  # Order should not matter

    def test_hash_different_mutations(self):
        """Test hash with different mutations."""
        mut_list1 = MutationList.from_list(["HA100C"])
        mut_list2 = MutationList.from_list(["YB101A"])
        assert hash(mut_list1) != hash(mut_list2)

    def test_hash_in_set(self):
        """Test that MutationList can be used in sets."""
        mut_list1 = MutationList.from_list(["HA100C", "YB101A"])
        mut_list2 = MutationList.from_list(["YB101A", "HA100C"])
        mut_list3 = MutationList.from_list(["HA100C"])
        s = {mut_list1, mut_list2, mut_list3}
        assert len(s) == 2  # mut_list1 and mut_list2 should be the same


class TestParseSingleMutation:
    """Test _parse_single_mutation static method."""

    def test_parse_single_mutation_simple(self):
        """Test parsing simple mutation."""
        mut = MutationList._parse_single_mutation("HA100C")
        assert mut.chain == "A"
        assert mut.resid == 100
        assert mut.before_res == "H"
        assert mut.after_res == "C"

    def test_parse_single_mutation_with_icode(self):
        """Test parsing mutation with insertion code."""
        mut = MutationList._parse_single_mutation("HA101aC")
        assert mut.resid == 101
        assert mut.icode == "a"

    def test_parse_single_mutation_invalid_aa(self):
        """Test parsing mutation with invalid amino acid."""
        with pytest.raises(ValueError, match="Unknown AA code"):
            MutationList._parse_single_mutation("HA100Z")


class TestParseFSMutation:
    """Test _parse_fs_mutation static method."""

    def test_parse_fs_mutation_simple(self):
        """Test parsing simple FEPSuite mutation."""
        mut = MutationList._parse_fs_mutation("A:100C")
        assert mut.chain == "A"
        assert mut.resid == 100
        assert mut.after_res == "C"
        assert mut.before_res is None

    def test_parse_fs_mutation_with_icode(self):
        """Test parsing FEPSuite mutation with insertion code."""
        mut = MutationList._parse_fs_mutation("A:101aC")
        assert mut.resid == 101
        assert mut.icode == "a"

    def test_parse_fs_mutation_invalid_format(self):
        """Test parsing invalid FEPSuite format."""
        with pytest.raises(ValueError, match="Invalid FEPSuite format"):
            MutationList._parse_fs_mutation("INVALID")
