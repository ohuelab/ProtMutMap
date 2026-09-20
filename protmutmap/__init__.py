"""Mutation networks and binding free-energy estimation."""

# Core functionality - always available
from .tools.make_mutant_pdb import make_mutant_pdb
from .mutant_preparer import MutationPreparer
from .calculation_files_generator import (
    generate_calculation_files,
    get_unique_mutations_from_links
)
from .fepsuite_preparer import FEPSuitePreparer
from .core import ProtMutMap
from .tools.lambda_calculator import (
    calculate_nrep,
    assess_mutation_difficulty,
    calculate_charge,
    generate_para_conf_content
)

# Graph and analysis functions (optional - requires matplotlib, networkx)
try:
    from .mutation_map import (
        generate_digraph_from_df,
        generate_dataframe_from_digraph,
        draw_mutation_graph,
        topological_layout,
        safe_layout
    )
    _HAS_GRAPH = True
except ImportError:
    _HAS_GRAPH = False
    generate_digraph_from_df = None
    generate_dataframe_from_digraph = None
    draw_mutation_graph = None
    topological_layout = None
    safe_layout = None

# Note: gather_results is available as a script but not imported by default
# due to external dependencies (wcc_main). Use: from protmutmap.gather_results import gather_results
_HAS_GATHER = False

# Tools subpackage
from . import tools

# Import commonly used classes from tools
from .tools import (
    Mutation,
    MutationList,
    MutationKey,
    HashableMutations,
    parse_mutations,
    parse_single_mutation,
    make_mutant_pdb as tools_make_mutant_pdb,
    is_nan_or_none,
    PDBInfoSummary,
    parse_pdb,
    update_mutinfo,
    DefaultProteinMutationGenerator
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    'ProtMutMap',
    'MutationPreparer',
    'FEPSuitePreparer',
    'MutationList',
    'Mutation',

    # Main functions
    'make_mutant_pdb',
    'generate_calculation_files',
    'get_unique_mutations_from_links',

    # Lambda calculator
    'calculate_nrep',
    'assess_mutation_difficulty',
    'calculate_charge',
    'generate_para_conf_content',

    # Graph functions
    'generate_digraph_from_df',
    'generate_dataframe_from_digraph',
    'draw_mutation_graph',
    'topological_layout',
    'safe_layout',

    # Mutation utilities
    'parse_mutations',
    'parse_single_mutation',
    'MutationKey',
    'HashableMutations',

    # PDB utilities
    'PDBInfoSummary',
    'parse_pdb',
    'update_mutinfo',
    'DefaultProteinMutationGenerator',

    # Utilities
    'is_nan_or_none',

    # Subpackages
    'tools',
]
