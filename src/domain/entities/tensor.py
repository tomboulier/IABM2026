"""Tensor entity - framework-independent abstraction."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tensor(Protocol):
    """Framework-independent tensor/array abstraction."""
    
    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the tensor's shape.
        
        Returns:
            shape (tuple[int, ...]): Tuple of integers representing the size of each dimension.
        """
        ...