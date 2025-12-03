"""Dataset entity - framework-independent abstraction."""
from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class Dataset(Protocol):
    """Framework-independent dataset abstraction."""
    
    def __len__(self) -> int:
        """
        Get the number of items in the dataset.
        
        Returns:
            int: The total number of items.
        """
        ...
    
    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """
        Retrieve the item and its label at the given index.
        
        Parameters:
            idx (int): Index of the dataset entry to retrieve.
        
        Returns:
            tuple[Any, Any]: A tuple (item, label) for the specified index.
        """
        ...