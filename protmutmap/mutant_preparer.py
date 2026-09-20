import shutil
from pathlib import Path
import subprocess
from typing import Union, Optional, List
import logging
import sys

from .tools.make_mutant_pdb import make_mutant_pdb
from .tools.mutations import MutationList

class MutationPreparer:
    """
    A class for preparing protein mutations using FEP suite tools.

    This class encapsulates the functionality for setting up directories,
    preparing source PDB files, and processing mutations in parallel.
    """

    def __init__(self,
                 input_pdb_file: Union[str, Path],
                 work_dir: Union[str, Path],
                 faspr_bin: Union[str, Path],
                 override: bool = False,
                 log_level: str = "INFO",
                 log_file: Optional[Union[str, Path]] = None):
        """
        Initialize the MutationPreparer.

        Args:
            input_pdb_file: Path to the input PDB file
            work_dir: Working directory for mutation processing
            faspr_bin: Path to the FASPR binary
            override: If True, override existing files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file. If None, logs to console only.
        """
        self.input_pdb_file = Path(input_pdb_file)
        self.target = self.input_pdb_file.stem
        self.work_dir = Path(work_dir)
        self.faspr_bin = Path(faspr_bin)
        self.override = override

        self.logger = self._setup_logger(log_level, log_file)

        self.source_pdb_dir = self.work_dir / "pdbs"
        self.source_pdb_file = self.source_pdb_dir / f"{self.target}.fixed.pdb"
        self.mutant_output_dir = self.work_dir / "mutated_pdbs"

        self.setup_directories()
        if not self.source_pdb_file.exists():
            self.logger.info("Source PDB file not found, preparing...")
            self.prepare_source_pdb()
        else:
            self.logger.info("Source PDB file found, skipping preparation")

    def _setup_logger(self, log_level: str, log_file: Optional[Union[str, Path]]) -> logging.Logger:
        """
        Set up logger with specified level and optional file output.

        Args:
            log_level: Logging level string
            log_file: Optional path to log file

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"MutationPreparer_{self.target}")
        logger.setLevel(getattr(logging, log_level.upper()))

        logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def execute_shell_command(self, shell_cmd: List[str], error_message: str) -> None:
        """
        Execute a shell command and handle errors.

        Args:
            shell_cmd: List of command arguments
            error_message: Error message to display if command fails

        Raises:
            ValueError: If the shell command fails
        """
        cmd_str = ' '.join(shell_cmd)
        self.logger.debug(f"Executing command: {cmd_str}")

        try:
            result = subprocess.run(shell_cmd, check=True, capture_output=True, text=True)
            self.logger.debug(f"Command completed successfully: {cmd_str}")
            if result.stdout:
                self.logger.debug(f"Command stdout: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {cmd_str}")
            if e.stdout:
                self.logger.error(f"Command stdout: {e.stdout.strip()}")
            if e.stderr:
                self.logger.error(f"Command stderr: {e.stderr.strip()}")
            raise ValueError(f"{error_message}: {e}")

    def setup_directories(self) -> None:
        """
        Set up necessary directories for mutation processing.
        """
        self.logger.info("Setting up directories...")

        directories = [
            (self.work_dir, "work directory"),
            (self.source_pdb_dir, "source PDB directory"),
            (self.mutant_output_dir, "mutant output directory")
        ]

        for directory, desc in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created {desc}: {directory}")

    def prepare_source_pdb(self) -> None:
        """
        Prepare the source PDB file using pdb2pqr.

        This method converts the original PDB file to a fixed format
        suitable for mutation processing.
        """
        self.logger.info("Preparing source PDB file...")
        input_pdb_file = self.input_pdb_file
        if not input_pdb_file.exists():
            error_msg = f"Input PDB file not found: {input_pdb_file}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        source_pqr_output = self.source_pdb_dir / f"{self.target}.pqr"

        shell_cmd = [
            "pdb2pqr",
            "--ff=AMBER",
            "--ffout=AMBER",
            "--with-ph=7.4",
            "--keep-chain",
            "--pdb-output",
            str(self.source_pdb_file),
            str(input_pdb_file),
            str(source_pqr_output)
        ]

        self.execute_shell_command(shell_cmd, "pdb2pqr command failed")
        self.logger.info("Source PDB preparation completed")

    def process_single_mutation(self, fs_mutations: tuple) -> None:
        """
        Process a single mutation or set of mutations.

        Args:
            fs_mutations: Tuple of mutation strings (e.g., ('H:101Y', 'H:103W'))
        """
        fs_mutations_str = "_".join(fs_mutations)
        self.logger.info(f"Processing mutation: {fs_mutations_str}")

        if fs_mutations:
            tmp_dir = self.mutant_output_dir / fs_mutations_str
            output_file = self.mutant_output_dir / f"{self.target}.{fs_mutations_str}.pdb"
        else:
            tmp_dir = self.mutant_output_dir / "WT"
            output_file = self.mutant_output_dir / f"{self.target}.WT.pdb"


        if output_file.exists() and not self.override:
            self.logger.info(f"Output file already exists, skipping: {output_file}")
            return

        self.logger.debug(f"Temporary directory: {tmp_dir}")
        self.logger.debug(f"Output file: {output_file}")

        try:
            make_mutant_pdb(
                pdb_path=str(self.source_pdb_file),
                mutation_str=fs_mutations_str,
                faspr_bin=str(self.faspr_bin),
                output_path=str(output_file),
                work_dir=str(tmp_dir),
                seed=None  # You can add seed parameter to __init__ if needed
            )

            if output_file.exists():
                self.logger.debug(f"Successfully created mutant: {output_file}")
            else:
                self.logger.warning(f"Output file not found after processing: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to process mutation {fs_mutations_str}: {e}")
            raise
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
                self.logger.debug(f"Cleaned up temporary directory: {tmp_dir}")

        self.logger.info(f"Completed processing mutation: {fs_mutations_str}")

    def run(self, muts: MutationList) -> None:
        """
        Process a single mutation using MutationList format.

        Args:
            muts: MutationList object containing mutations to process
        """
        fs_mutations = muts.to_fs_mutations()
        fs_mutations_tuple = tuple(fs_mutations)

        self.logger.info(f"Processing MutationList: {muts}")

        self.process_single_mutation(fs_mutations_tuple)
