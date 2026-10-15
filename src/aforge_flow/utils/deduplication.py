"""Deduplication utilities to prevent processing duplicate assets."""

import hashlib
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class DuplicateDetector:
    """Detects and prevents processing of duplicate files."""

    def __init__(self, hash_cache_dir: Path | None = None):
        """
        Initialize duplicate detector.

        Args:
            hash_cache_dir: Optional directory to cache file hashes
        """
        self.hash_cache_dir = hash_cache_dir
        self.hash_map: dict[str, Path] = {}

        if self.hash_cache_dir:
            self.hash_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """
        Calculate hash of file.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)

        Returns:
            Hex digest of file hash
        """
        hash_obj = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def is_duplicate(self, file_path: Path) -> bool:
        """
        Check if file is duplicate.

        Args:
            file_path: Path to file to check

        Returns:
            True if file is duplicate, False otherwise
        """
        file_hash = self.get_file_hash(file_path)

        if file_hash in self.hash_map:
            logger.info(
                "duplicate_detected",
                current_file=str(file_path),
                original_file=str(self.hash_map[file_hash]),
                file_hash=file_hash,
            )
            return True

        self.hash_map[file_hash] = file_path
        return False

    def register_file(self, file_path: Path) -> str:
        """
        Register file hash.

        Args:
            file_path: Path to file

        Returns:
            File hash
        """
        file_hash = self.get_file_hash(file_path)
        self.hash_map[file_hash] = file_path
        return file_hash
