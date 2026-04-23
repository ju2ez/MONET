class BaseTaskConfig:
    """Base class for task configurations."""

    def __init__(self, task_id: int, task_vec):
        self.task_id = task_id
        self.task_vec = task_vec

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'task_vec': self.task_vec
        }

    def __repr__(self):
        return f"TaskConfig(id={self.task_id}, task_vec={self.task_vec})"