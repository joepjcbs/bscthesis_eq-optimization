import logging
from datetime import datetime
import time
import secrets
from pathlib import Path
import numpy as np

from auditory_aphasia.bscthesis_eq.eq import EQ
from auditory_aphasia.bscthesis_eq.random_search import RandomSearch

class Session():
    """
    Manages experiment sessions, including directory structure, logging,
    equalizer setup, random search strategy, and result persistence.

    Attributes:
        subject_name (str): Name of the current participant.
        eq (EQ): The initialized equalizer instance.
        search (RandomSearch): The initialized random search instance.
        run_name (str): Name of the current run (with index and timestamp).
        run_folder_path (Path): Path to the current run directory.
        logger (logging.Logger): Logger for saving experiment events.
        random_seed (int): Seed used for reproducible search sampling.
    """

    def __init__(self, subject_name, eq: EQ, search: RandomSearch):
        self.subject_name = subject_name
        self.subject_main_path = Path.cwd() / 'session_data' / subject_name

        # Create subject directory if it doesn't exist
        if not self.subject_main_path.exists():
            self.subject_main_path.mkdir(parents=True)

        # Initialize run-specific folder and logger
        self.run_name, self.run_folder_path = self.setup_run_folder()
        self.logger = self.setup_logger()

        # Store EQ and export path
        self.eq = eq
        self.eq_export_path = self.run_folder_path / 'eq_export'
        self.eq_export_path.mkdir(exist_ok=True)

        # Store search strategy
        self.search = search
        self.optimal_gains = None

        # Set a reproducible random seed
        self.random_seed = secrets.randbelow(2 ** 32)
        self.search.set_random_seed(self.random_seed)

        self.logger.info(f"Subject: {self.subject_name}")
        self.logger.info(f"Random seed used: {self.random_seed}")

        # Prepare folder for plots
        self.plots_path = self.run_folder_path / 'plots'
        self.plots_path.mkdir(exist_ok=True)
        
    def setup_run_folder(self):
        """
        Set up the run folder for storing logs and outputs.
        Checks for existing runs and allows reuse of the last one.

        Returns:
            tuple: (run_name, run_folder_path)
        """

        existing_runs = [
            p for p in self.subject_main_path.iterdir()
            if p.is_dir() and p.name.startswith("run_")
        ]      

        # Count runs and increment
        run_indices = []
        for run in existing_runs:
            parts = run.name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                run_indices.append(int(parts[1]))
        next_run_index = max(run_indices, default=0) + 1

        # Format folder name with timestamp
        timestamp = datetime.now().strftime("%Y%m%dT%H%M")
        run_name = f"run_{next_run_index:02d}_{timestamp}"
        run_folder_path = self.subject_main_path / run_name

        if len(existing_runs) > 0:
            last_run = existing_runs[-1]
            response = input(
                f"Found previous run folder: {last_run.name}. Use it? [y/n]: "
            ).strip().lower()

            if response == "y":
                print(f"Reusing existing folder: {last_run}")
                run_folder_path = last_run
            else:
                run_folder_path.mkdir(exist_ok=True)
                print(f"Created new folder: {run_folder_path}")
        else:
            run_folder_path.mkdir(exist_ok=True)
            print(f"Created new folder: {run_folder_path}")

        return run_name, run_folder_path

    def setup_logger(self):
        """
        Create and configure a logger that logs to a run-specific file.

        Returns:
            logging.Logger: The configured logger instance.
        """

        logger = logging.getLogger(f"logger_{self.subject_name}")
        logger.setLevel(logging.DEBUG)
        file_name = self.run_name + '_log.log'

        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh = logging.FileHandler(self.run_folder_path / file_name)
            fh.setLevel(logging.DEBUG)
            old_factory = logging.getLogRecordFactory()

            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                record.epoch_time = f"{record.created:.6f}"  # record.created is the exact float timestamp
                return record

            logging.setLogRecordFactory(record_factory)

            formatter = logging.Formatter('%(asctime)s - %(epoch_time)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger
    
    def save_history(self):
        """
        Save the search configuration history to a .npy file, naming it by block index.
        """

        config_path = self.run_folder_path / "configs"
        config_path.mkdir(exist_ok=True)

        i = 1
        while True:
            path = config_path / f"configs_block_{i}.npy"
            if not path.exists():
                break
            i += 1

        self.logger.info(f"Saving config history to {path}")
        np.save(path, self.search.history)
