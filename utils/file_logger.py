import os
import json
import csv
import pandas as pd
from typing import Dict, Any, Optional
import time
from datetime import datetime

class FileLogger:
    """File-based logger that mimics wandb functionality."""

    def __init__(self, project: str, config: Dict[str, Any], name: str, log_dir: str = "logs"):
        self.project = project
        self.config = config
        self.name = name
        self.log_dir = log_dir

        # Create timestamp for unique run identification
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, project, f"{name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Save config
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Initialize metrics file
        self.metrics_file = os.path.join(self.run_dir, "metrics.csv")
        self.metrics_written = False

        # Store for step tracking
        self.step_metrics = {}

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to CSV file."""
        if step is not None:
            data["step"] = step

        # Add timestamp
        data["timestamp"] = time.time()

        # Write to CSV
        df = pd.DataFrame([data])
        if not self.metrics_written:
            df.to_csv(self.metrics_file, index=False)
            self.metrics_written = True
        else:
            df.to_csv(self.metrics_file, mode='a', header=False, index=False)

    def define_metric(self, metric_name: str, step_metric: str):
        """Store metric definition (for compatibility with wandb)."""
        self.step_metrics[metric_name] = step_metric

    def finish(self):
        """Finish logging."""
        pass

class Table:
    """File-based table that mimics wandb.Table."""

    def __init__(self, columns):
        self.columns = columns
        self.data = []

    def add_data(self, *args):
        """Add a row of data."""
        self.data.append(args)

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        return pd.DataFrame(self.data, columns=self.columns)

# Global logger instance
_logger = None

def init(project: str, config: Dict[str, Any], name: str, log_dir: str = "logs"):
    """Initialize file logger."""
    global _logger
    _logger = FileLogger(project, config, name, log_dir)
    return _logger

def log(data: Dict[str, Any], step: Optional[int] = None):
    """Log data."""
    if _logger:
        _logger.log(data, step)

def define_metric(metric_name: str, step_metric: str):
    """Define metric."""
    if _logger:
        _logger.define_metric(metric_name, step_metric)

def finish():
    """Finish logging."""
    if _logger:
        _logger.finish()
