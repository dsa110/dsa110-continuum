"""
Measurement Set writing strategies for DSA-110 Continuum Imaging Pipeline.

Production writers for converting UVH5 subband files to Measurement Sets.

For testing-only writers (e.g., PyuvdataMonolithicWriter), see:
    backend/tests/fixtures/writers.py
"""

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyuvdata import UVData


class MSWriter(abc.ABC):
    """Abstract base class for a Measurement Set writer strategy."""

    def __init__(self, uv: "UVData", ms_path: str, **kwargs: Any) -> None:
        """Initialize the writer.

        Parameters
        ----------
        uv : UVData
            The UVData object containing the visibilities to write.
        ms_path : str
            The full path to the output Measurement Set.
            **kwargs
            Writer-specific options.
        """
        self.uv = uv
        self.ms_path = ms_path
        self.kwargs = kwargs

    @abc.abstractmethod
    def write(self) -> str:
        """Execute the writing strategy."""
        ...

    def get_files_to_process(self) -> list[str] | None:
        """ """
        return None


# Import the full implementation from direct_subband
# This avoids code duplication and circular imports
from .direct_subband import DirectSubbandWriter


def get_writer(writer_type: str) -> type:
    """Get a writer class by type name.

    Parameters
    ----------
    writer_type : str
        Writer type ('direct-subband').
        For testing writers, import from backend/tests/fixtures/writers.py.

    """
    writers = {
        "direct-subband": DirectSubbandWriter,
    }

    if writer_type == "pyuvdata":
        raise ValueError(
            "PyuvdataWriter is for testing only. "
            "Import from backend/tests/fixtures/writers.py instead."
        )

    if writer_type == "parallel-subband":
        raise ValueError("'parallel-subband' has been removed; use 'direct-subband' instead.")

    if writer_type == "auto":
        raise ValueError(
            "'auto' writer selection has been removed; use 'direct-subband' explicitly."
        )

    if writer_type not in writers:
        raise ValueError(f"Unknown writer type: {writer_type}. Available: {list(writers.keys())}")

    return writers[writer_type]
