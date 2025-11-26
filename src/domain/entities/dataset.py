"""Dataset entity - framework-independent abstraction."""
from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class Dataset(Protocol):
    """Framework-independent dataset abstraction."""
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        ...
    
    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Get an item and its label from the dataset."""
        ...

    @property
    def num_channels(self) -> int:
        """Return the number of channels in the dataset images."""
        ...

    @property
    def image_size(self) -> int:
        """Return the size of the images in the dataset."""
        ...
