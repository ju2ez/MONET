class BaseEnv:
    """Base class for environments."""

    def __init__(self, task):
        self.task = task

    def evaluate_solution(self, solution):
        """Evaluate a solution. This should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement `evaluate_solution`.")