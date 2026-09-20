"""Parse and manipulate amino-acid substitutions."""

from typing import List, Optional, Union
from itertools import combinations

from .mutation import Mutation

AA_ONE_TO_THREE = {
    'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY',
    'H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER',
    'T':'THR','W':'TRP','Y':'TYR','V':'VAL'
}

class MutationList:
    """
    A class for handling collections of protein mutations with support for multiple formats.

    Uses List[Mutation] internally for compatibility with the tools.mutation module.

    Supported formats:
    - Standard mutation: {wt_aa}{chain_id}{seq_idx}{icode}{mut_aa} (e.g., "HB100C")
    - FEPSuite mutation: {chain}:{seq_id}{icode}{mut_aa} (e.g., "B:100C")
    """

    def __init__(self, mutations: Optional[Union[List[Mutation], List[tuple]]] = None, sort: bool = False):
        """
        Initialize MutationList object.

        Args:
            mutations: List of Mutation objects or tuples in tuple format
                      Legacy tuple format: (wt_aa, chain, seq_idx, icode, mut_aa) or (chain, seq_idx, icode, mut_aa)
            sort: If True, sort mutations by (chain, resid, icode)
        """
        if mutations is None:
            mutations = []

        self._mutations: List[Mutation] = []

        for mut in mutations:
            if isinstance(mut, Mutation):
                self._mutations.append(mut)
            elif isinstance(mut, tuple):
                if len(mut) == 4:
                    chain, seq_idx, icode, mut_aa = mut
                    wt_aa = None
                elif len(mut) == 5:
                    wt_aa, chain, seq_idx, icode, mut_aa = mut
                else:
                    raise ValueError(f"Invalid mutation tuple: {mut}. Expected (wt_aa, chain, seq_idx, icode, mut_aa) or (chain, seq_idx, icode, mut_aa)")

                if not isinstance(seq_idx, int):
                    raise ValueError(f"seq_idx must be integer, got {type(seq_idx)}: {seq_idx}")
                if mut_aa.upper() not in AA_ONE_TO_THREE:
                    raise ValueError(f"Invalid amino acid code: {mut_aa}")
                if wt_aa is not None and wt_aa.upper() not in AA_ONE_TO_THREE:
                    raise ValueError(f"Invalid wild-type amino acid code: {wt_aa}")

                mutation = Mutation(
                    chain=chain,
                    resid=seq_idx,
                    before_res=wt_aa.upper() if wt_aa else None,
                    after_res=mut_aa.upper(),
                    icode=icode if icode else ""
                )
                self._mutations.append(mutation)
            else:
                raise ValueError(f"Invalid mutation type: {type(mut)}. Expected Mutation object or tuple")

        if sort:
            self.sort()

    @classmethod
    def from_string(cls, mutation_str: str, sort: bool = False) -> 'MutationList':
        """
        Create MutationList object from comma/semicolon separated mutation string.

        Args:
            mutation_str: String like "AB100C,DB101A,DB101aA"
            sort: If True, sort mutations by (chain, resid, icode)

        Returns:
            MutationList object
        """
        if not mutation_str or not mutation_str.strip():
            return cls([])

        mutations = []
        for token in mutation_str.replace(';', ',').split(','):
            token = token.strip()
            if not token:
                continue
            try:
                mutations.append(cls._parse_single_mutation(token))
            except Exception as e:
                raise ValueError(f"Bad mutation token '{token}'. Expected format like 'AB100C' (wt_aa+chain+seq_idx+mut_aa).") from e

        return cls(mutations, sort=sort)

    @classmethod
    def from_list(cls, mutation_list: List[str], sort: bool = False) -> 'MutationList':
        """
        Create MutationList object from list of mutation strings.

        Args:
            mutation_list: List of strings like ["AB100C", "DB101A"]
            sort: If True, sort mutations by (chain, resid, icode)

        Returns:
            MutationList object
        """
        mutations = []
        for mutation in mutation_list:
            try:
                mutations.append(cls._parse_single_mutation(mutation))
            except Exception as e:
                raise ValueError(f"Bad mutation '{mutation}'. Expected format like 'AB100C' (wt_aa+chain+seq_idx+mut_aa).") from e

        return cls(mutations, sort=sort)

    @classmethod
    def from_fs_mutations(cls, fs_mutation_list: List[str], sort: bool = False) -> 'MutationList':
        """
        Create MutationList object from FEPSuite format mutation list.

        Args:
            fs_mutation_list: List of strings like ["B:100C", "B:101A"]
            sort: If True, sort mutations by (chain, resid, icode)

        Returns:
            MutationList object
        """
        mutations = []
        for fs_mut in fs_mutation_list:
            try:
                mutations.append(cls._parse_fs_mutation(fs_mut))
            except Exception as e:
                raise ValueError(f"Bad FEPSuite mutation '{fs_mut}'. Expected format like 'B:100C' (chain:seq_idx+icode+mut_aa).") from e

        return cls(mutations, sort=sort)

    @classmethod
    def from_mutations(cls, mutation_list: List[Mutation], sort: bool = False) -> 'MutationList':
        """
        Create MutationList object from list of Mutation objects.

        Args:
            mutation_list: List of Mutation objects
            sort: If True, sort mutations by (chain, resid, icode)

        Returns:
            MutationList object
        """
        return cls(mutation_list, sort=sort)

    @classmethod
    def from_tuples(cls, tuple_list: List[tuple], sort: bool = False) -> 'MutationList':
        """
        Create MutationList object from list of internal format tuples.

        Args:
            tuple_list: List of tuples like [("H", "B", 100, "", "C"), ("Y", "B", 101, "a", "A")]
            sort: If True, sort mutations by (chain, resid, icode)

        Returns:
            MutationList object
        """
        return cls(tuple_list, sort=sort)

    @staticmethod
    def _parse_single_mutation(mutation: str) -> Mutation:
        """Parse single mutation string to Mutation object."""
        wt_aa = mutation[0]
        chain = mutation[1]

        mut_aa = mutation[-1]

        seq_idx_str = mutation[2:-1]

        # Handle insertion codes (like 101a)
        if seq_idx_str and seq_idx_str[-1].islower():
            seq_idx = int(seq_idx_str[:-1])
            icode = seq_idx_str[-1]
        else:
            seq_idx = int(seq_idx_str)
            icode = ''

        if mut_aa.upper() not in AA_ONE_TO_THREE:
            raise ValueError(f"Unknown AA code: {mut_aa}")
        if wt_aa.upper() not in AA_ONE_TO_THREE:
            raise ValueError(f"Unknown wild-type AA code: {wt_aa}")

        return Mutation(
            chain=chain,
            resid=seq_idx,
            before_res=wt_aa.upper(),
            after_res=mut_aa.upper(),
            icode=icode
        )

    @staticmethod
    def _parse_fs_mutation(fs_mut: str) -> Mutation:
        """Parse FEPSuite format mutation to Mutation object."""
        parts = fs_mut.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid FEPSuite format: {fs_mut}")

        chain = parts[0]
        remainder = parts[1]  # seq_id + icode + mut_aa

        mut_aa = remainder[-1]

        seq_icode_str = remainder[:-1]

        # Handle insertion codes (like 101a)
        if seq_icode_str and seq_icode_str[-1].islower():
            seq_id = int(seq_icode_str[:-1])
            icode = seq_icode_str[-1]
        else:
            seq_id = int(seq_icode_str)
            icode = ''

        return Mutation(
            chain=chain,
            resid=seq_id,
            before_res=None,
            after_res=mut_aa.upper(),
            icode=icode
        )

    def to_string(self, fs: bool = False) -> str:
        """
        Convert to comma-separated standard format string.

        Args:
            fs: If True, use FEPSuite format with underscore separator

        Returns:
            String like "AB100C,DB101A,DB101aA" or "B:100C_B:101A_B:101aA"
        """
        if fs:
            return '_'.join(self.to_fs_mutations())
        else:
            return ','.join(self.to_list())

    def to_list(self) -> List[str]:
        """
        Convert to list of standard format strings.

        Returns:
            List like ["HB100C", "DB101A", "DB101aA"]
        """
        result = []
        for mut in self._mutations:
            # For standard format, we need wild-type AA
            # Use 'X' as placeholder if wild-type AA is not available
            wt_aa_str = mut.before_res if mut.before_res else 'X'
            mutation_str = f"{wt_aa_str}{mut.chain}{mut.resid}{mut.icode}{mut.after_res}"
            result.append(mutation_str)
        return result

    def to_fs_mutations(self) -> List[str]:
        """
        Convert to FEPSuite format list.

        Returns:
            List like ["B:100C", "B:101aA"]
        """
        result = []
        for mut in self._mutations:
            fs_mutation = f"{mut.chain}:{mut.resid}{mut.icode}{mut.after_res}"
            result.append(fs_mutation)
        return result

    def to_mutations(self) -> List[Mutation]:
        """
        Convert to list of Mutation objects.

        Returns:
            List of Mutation objects
        """
        return list(self._mutations)

    def to_tuples(self) -> List[tuple]:
        """
        Return mutations in tuple format.

        Returns:
            List of tuples like [("H", "B", 100, "", "C"), ("Y", "B", 101, "a", "A")]
        """
        result = []
        for mut in self._mutations:
            result.append((mut.before_res, mut.chain, mut.resid, mut.icode, mut.after_res))
        return result

    def sort(self, reverse: bool = False):
        """
        Sort mutations by (chain, resid, icode) in place.

        Args:
            reverse: If True, sort in descending order
        """
        self._mutations.sort(key=lambda x: (x.chain, x.resid, x.icode), reverse=reverse)

    def get_mutation_chains(self):
        return set([mut.chain for mut in self._mutations if mut.chain is not None])

    def filter_by_chains(self, chains: List[str]) -> 'MutationList':
        """
        Filter mutations to only include those affecting the specified chains.

        Args:
            chains: List of chain IDs to include

        Returns:
            New MutationList object containing only mutations from the specified chains
        """
        filtered_mutations = [
            mut for mut in self._mutations
            if mut.chain in chains
        ]
        return MutationList(filtered_mutations)

    def generate_combinations(self, return_type: str = 'MutationList'):
        """
        Generate all possible combinations of mutations including wild type.

        Args:
            return_type: Format of returned combinations
                - 'mutations': List of lists with standard format strings like ['HH101Y', 'YH103W']
                - 'fs_mutations': List of lists with FEPSuite format strings like ['H:101Y', 'H:103W']
                - 'MutationList': List of MutationList objects (default)
                - 'tuples': List of lists with internal tuple format
                - 'strings': List of comma-separated standard format strings

        Returns:
            List of all possible mutation combinations in specified format, including wild type (empty list)
        """
        all_mutation_combinations = []

        all_mutation_combinations.append([])

        for r in range(1, len(self._mutations) + 1):
            for combo in combinations(self._mutations, r):
                all_mutation_combinations.append(list(combo))

        if return_type == 'fs_mutations':
            result = []
            for combo in all_mutation_combinations:
                combo_mutations = MutationList(combo)
                result.append(combo_mutations.to_fs_mutations())
            return result
        elif return_type == 'mutations':
            result = []
            for combo in all_mutation_combinations:
                combo_mutations = MutationList(combo)
                result.append(combo_mutations.to_list())
            return result
        elif return_type == 'tuples':
            result = []
            for combo in all_mutation_combinations:
                combo_mutations = MutationList(combo)
                result.append(combo_mutations.to_tuples())
            return result
        elif return_type == 'strings':
            result = []
            for combo in all_mutation_combinations:
                combo_mutations = MutationList(combo)
                result.append(combo_mutations.to_string())
            return result
        else:  # return_type == 'MutationList' (default)
            result = []
            for combo in all_mutation_combinations:
                combo_mutations = MutationList(combo)
                result.append(combo_mutations)
            return result

    def validate(self) -> bool:
        """
        Validate all mutations have valid amino acid codes.

        Returns:
            True if all mutations are valid

        Raises:
            ValueError: If any mutation has invalid amino acid code
        """
        for mut in self._mutations:
            if mut.after_res.upper() not in AA_ONE_TO_THREE:
                raise ValueError(f"Invalid amino acid code: {mut.after_res} in mutation {mut.chain}:{mut.resid}{mut.icode}{mut.after_res}")
            if mut.before_res is not None and mut.before_res.upper() not in AA_ONE_TO_THREE:
                raise ValueError(f"Invalid wild-type amino acid code: {mut.before_res} in mutation {mut.chain}:{mut.resid}{mut.icode}{mut.after_res}")
        return True

    def to_three_letter(self) -> List[tuple]:
        """
        Convert mutations to three-letter amino acid codes.

        Returns:
            List of tuples with three-letter codes: [("HIS", "B", 100, "", "CYS"), ...]
        """
        result = []
        for mut in self._mutations:
            three_letter_mut = AA_ONE_TO_THREE[mut.after_res.upper()]
            three_letter_wt = AA_ONE_TO_THREE[mut.before_res.upper()] if mut.before_res else None
            result.append((three_letter_wt, mut.chain, mut.resid, mut.icode, three_letter_mut))
        return result

    def diff(self, other: 'MutationList') -> 'MutationList':
        """
        Find mutations that are in this MutationList but not in the other MutationList.

        For positions that exist in both lists, creates a mutation where:
        - The wild-type AA comes from the mutation in the other list
        - The mutant AA comes from the mutation in this list

        For positions that exist only in this list, includes the mutation as-is.

        For positions that exist only in the other list, creates reverse mutations
        (mutant AA -> wild-type AA) of those mutations.

        Args:
            other: MutationList to compare against

        Returns:
            New MutationList containing the difference mutations
        """
        other_mutations_map = {(mut.chain, mut.resid, mut.icode): mut for mut in other._mutations}
        self_mutations_map = {(mut.chain, mut.resid, mut.icode): mut for mut in self._mutations}

        diff_mutations = []
        for mut in self._mutations:
            position = (mut.chain, mut.resid, mut.icode)

            if position in other_mutations_map:
                # Position exists in both lists - create a new mutation
                # where wild-type comes from other and mutant comes from self
                other_mut = other_mutations_map[position]
                new_mutation = Mutation(
                    chain=mut.chain,
                    resid=mut.resid,
                    before_res=other_mut.after_res,  # Use mutant AA from other as wild-type
                    after_res=mut.after_res,        # Use mutant AA from self
                    icode=mut.icode
                )
                if new_mutation.before_res != new_mutation.after_res:
                    diff_mutations.append(new_mutation)
            else:
                diff_mutations.append(mut)

        # Process mutations in other that are not in self (add reverse mutations)
        for mut in other._mutations:
            position = (mut.chain, mut.resid, mut.icode)
            if position not in self_mutations_map:
                if mut.before_res is not None and mut.after_res is not None:
                    reverse_mutation = Mutation(
                        chain=mut.chain,
                        resid=mut.resid,
                        before_res=mut.after_res,  # Use mutant AA from other as wild-type
                        after_res=mut.before_res,  # Use wild-type AA from other as mutant
                        icode=mut.icode
                    )
                    if reverse_mutation.before_res != reverse_mutation.after_res:
                        diff_mutations.append(reverse_mutation)

        return MutationList(diff_mutations)

    def __sub__(self, other: 'MutationList') -> 'MutationList':
        """
        Subtract operator for MutationList.

        Returns mutations that are in this MutationList but not in the other MutationList.
        This is equivalent to calling self.diff(other).

        Args:
            other: MutationList to subtract

        Returns:
            New MutationList containing the difference
        """
        return self.diff(other)

    def __neg__(self) -> 'MutationList':
        """
        Unary minus operator for MutationList.

        Returns reverse mutations (mutant AA -> wild-type AA) of this MutationList.

        Returns:
            New MutationList containing reverse mutations
        """
        reverse_mutations = []
        for mut in self._mutations:
            if mut.before_res is not None and mut.after_res is not None:
                reverse_mutation = Mutation(
                    chain=mut.chain,
                    resid=mut.resid,
                    before_res=mut.after_res,  # Use mutant AA as wild-type
                    after_res=mut.before_res,  # Use wild-type AA as mutant
                    icode=mut.icode
                )
                if reverse_mutation.before_res != reverse_mutation.after_res:
                    reverse_mutations.append(reverse_mutation)
        return MutationList(reverse_mutations)

    def __len__(self) -> int:
        """Return number of mutations."""
        return len(self._mutations)

    def __iter__(self):
        """Iterate over mutations."""
        return iter(self._mutations)

    def __getitem__(self, index) -> Mutation:
        """Get mutation by index."""
        return self._mutations[index]

    def __str__(self) -> str:
        """String representation using FEPSuite format."""
        return ','.join(self.to_fs_mutations())

    def __repr__(self) -> str:
        """String representation using FEPSuite format."""
        return 'MutationList({})'.format(','.join(self.to_fs_mutations()))

    def __eq__(self, other) -> bool:
        """
        Check equality of MutationList objects.

        Two MutationList objects are equal if they contain the same set of mutations,
        regardless of order.

        Args:
            other: Another MutationList object to compare

        Returns:
            True if both lists contain the same mutations, False otherwise
        """
        if not isinstance(other, MutationList):
            return False
        return set(self._mutations) == set(other._mutations)

    def __hash__(self) -> int:
        """
        Compute hash value for MutationList.

        Hash is computed from the set of mutations, so order does not matter.

        Returns:
            Hash value of the MutationList
        """
        return hash(frozenset(self._mutations))

Mutations = MutationList
