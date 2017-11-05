"""Project config file. Modified from config.py.template file."""
import os
from utils import py_utils


class Config:
    """
    Config class with global project variables.

    ...

    Attributes
    ----------
    data_root : string
        Directory to root of dataset.
    project_directory : string
        Directory to root of project.
    tf_records : string
        Directory to tr_records.
    checkpoints : string
        Directory to checkpoints.
    summaries : string
        Directory to summaries.
    experiment_evaluations : string
        Directory to experiment evaluations.
    condition_evaluations : string
        Directory to condition evaluations.
    visualizations : string
        Directory to visualizations.
    plots : string
        Directory to plots.
    log_dir : string
        Directory to log files.
    dataset_info : string
        Local dataset classes.
    db_ssh_forward : bool
        Whether or not to use ssh forwarding to access the databset.
    """

    def __init__(self, **kwargs):
        """Global config file for normalization experiments."""
        self.data_root = '/media/data_cifs/image_datasets/'
        self.project_directory = '/media/data_cifs/contextual_circuit/'
        self.tf_records = os.path.join(
            self.project_directory,
            'tf_records')
        self.checkpoints = os.path.join(
            self.project_directory,
            'checkpoints')
        self.summaries = os.path.join(
            self.project_directory,
            'summaries')
        self.experiment_evaluations = os.path.join(
            self.project_directory,
            'experiment_evaluations')
        self.condition_evaluations = os.path.join(
            self.project_directory,
            'condition_evaluations')
        self.visualizations = os.path.join(
            self.project_directory,
            'visualizations')
        self.plots = os.path.join(
            self.project_directory,
            'plots')
        self.log_dir = os.path.join(self.project_directory, 'logs')
        self.dataset_info = 'dataset_processing'  # local dataset classes

        # DB
        self.db_ssh_forward = False
        machine_name = os.uname()[1]
        if len(machine_name) == 12:  # Docker container
            self.db_ssh_forward = True

        # Create directories if they do not exist
        check_dirs = [
            self.tf_records,
            self.checkpoints,
            self.experiment_evaluations,
            self.condition_evaluations,
            self.visualizations,
            self.plots,
            self.log_dir,
            self.dataset_info
        ]
        [py_utils.make_dir(x) for x in check_dirs]

    def __getitem__(self, name):
        """Get item from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains field."""
        return hasattr(self, name)
