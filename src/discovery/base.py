"""Search provider abstraction.

Everything downstream consumes SearchCandidate objects. No module outside
src/discovery/ may reference a provider's raw JSON structure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import SearchResult


class SearchError(RuntimeError):
    """Base class for discovery failures. Live mode surfaces these; it never
    substitutes cached or fabricated results for them."""


class SearchAuthError(SearchError):
    """Rejected credentials."""


class SearchRateLimitError(SearchError):
    """Provider quota or rate limit reached."""


class SearchUploadError(SearchError):
    """The image could not be handed to the provider."""


class SearchTransportError(SearchError):
    """Network-level failure talking to the provider."""


class SearchProviderError(SearchError):
    """The provider answered, but reported an error."""


class SearchProvider(ABC):
    """A reverse-image search backend."""

    name: str = "abstract"

    @abstractmethod
    def search(self, image_path: Path) -> SearchResult:
        """Run a live reverse-image search for a local image.

        Must raise a SearchError on failure. Returning an empty candidate
        list means the search genuinely succeeded with zero results - that is
        a legitimate outcome, not an error.
        """

    @abstractmethod
    def normalize(self, raw: dict) -> list:
        """Turn a raw provider response into SearchCandidate objects."""
