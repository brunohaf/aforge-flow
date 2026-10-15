"""Main asset processing pipeline."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import structlog
from tqdm import tqdm

from aforge_flow.config import Config
from aforge_flow.processors import (
    BackgroundRemovalProcessor,
    CaptionRenameProcessor,
    CreatureClassificationProcessor,
    MediaOptimizationProcessor,
    ProcessorResult,
)
from aforge_flow.utils import DuplicateDetector, get_files_recursively, validate_file

logger = structlog.get_logger(__name__)


def result_event(
    result: ProcessorResult,
    index: int | None = None,
    total: int | None = None,
) -> dict:
    """Build a JSON-serializable progress event from a :class:`ProcessorResult`.

    Shared by :meth:`AssetPipeline.process_directory` and the web JobManager so
    directory and single-file modes emit the same per-file event shape.
    """
    return {
        "type": "file_done",
        "index": index,
        "total": total,
        "input_path": str(result.input_path),
        "output_path": str(result.output_path) if result.output_path else None,
        "success": result.success,
        "error": result.error,
        "input_size": result.metadata.get("input_size"),
        "output_size": result.metadata.get("output_size"),
    }


class AssetPipeline:
    """Main asset processing pipeline orchestrator."""

    def __init__(self, config: Config):
        """
        Initialize pipeline.

        Args:
            config: Asset optimizer configuration
        """
        self.config = config
        self.logger = structlog.get_logger("AssetPipeline")

        self.background_removal_processor: BackgroundRemovalProcessor | None = None
        self.media_optimization_processor: MediaOptimizationProcessor | None = None
        self.caption_rename_processor: CaptionRenameProcessor | None = None
        self.creature_classification_processor: CreatureClassificationProcessor | None = None

        self.duplicate_detector = DuplicateDetector(self.config.cache_dir / "hashes")

        self.processed_files: list[ProcessorResult] = []
        self.failed_files: list[ProcessorResult] = []

        self.logger.info(
            "pipeline_initialized",
            input_dir=str(self.config.input_dir),
            output_dir=str(self.config.output_dir),
            use_gpu=self.config.use_gpu,
        )

    def setup(self) -> None:
        """Setup processors."""
        if self.config.rembg_enabled:
            self.background_removal_processor = BackgroundRemovalProcessor(
                self.config,
                category=self.config.rembg_category,
            )
            self.logger.info("background_removal_processor_initialized")

        if self.config.caption_rename_enabled:
            self.caption_rename_processor = CaptionRenameProcessor(self.config)
            self.logger.info("caption_rename_processor_initialized")

        if getattr(self.config, "classification_enabled", False):
            self.creature_classification_processor = CreatureClassificationProcessor(self.config)
            self.logger.info("creature_classification_processor_initialized")

        if self.config.media_optimization_enabled:
            self.media_optimization_processor = MediaOptimizationProcessor(
                self.config,
                image_format=self.config.image_format,
                video_format=self.config.video_format,
            )
            self.logger.info("media_optimization_processor_initialized")

    def process_asset(
        self,
        input_path: Path,
        remove_background: bool = True,
        optimize_media: bool = True,
    ) -> ProcessorResult:
        """
        Process single asset through pipeline.

        Args:
            input_path: Input file path
            remove_background: Run background removal
            optimize_media: Run media optimization

        Returns:
            ProcessorResult from last processor in chain
        """
        is_valid, error_msg = validate_file(input_path)
        if not is_valid:
            result = ProcessorResult(input_path, success=False, error=error_msg)
            self.failed_files.append(result)
            return result

        if self.duplicate_detector.is_duplicate(input_path):
            result = ProcessorResult(
                input_path,
                success=False,
                error="Duplicate file (hash already processed)",
            )
            self.failed_files.append(result)
            return result

        current_path = input_path
        result = ProcessorResult(input_path, success=True)

        if self.caption_rename_processor:
            if self.caption_rename_processor.can_process(current_path):
                result = self.caption_rename_processor.process(current_path)
                self.caption_rename_processor.log_result(result)

            if not result.success:
                self.failed_files.append(result)
                return result

            if result.output_path:
                current_path = result.output_path

        if self.creature_classification_processor:
            if self.creature_classification_processor.can_process(current_path):
                class_result = self.creature_classification_processor.process(current_path)
                self.creature_classification_processor.log_result(class_result)
                # Carry classification metadata forward; never blocks the chain.
                result.metadata.update(class_result.metadata)

        if remove_background and self.background_removal_processor:
            if self.background_removal_processor.can_process(current_path):
                result = self.background_removal_processor.process(current_path)
                self.background_removal_processor.log_result(result)

                if not result.success:
                    self.failed_files.append(result)
                    return result

                if result.output_path:
                    current_path = result.output_path

        if optimize_media and self.media_optimization_processor:
            if self.media_optimization_processor.can_process(current_path):
                result = self.media_optimization_processor.process(current_path)
                self.media_optimization_processor.log_result(result)

                if not result.success:
                    self.failed_files.append(result)
                    return result

        self.processed_files.append(result)
        return result

    def process_directory(
        self,
        remove_background: bool = True,
        optimize_media: bool = True,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> None:
        """
        Process all assets in input directory.

        Args:
            remove_background: Run background removal on all files
            optimize_media: Run media optimization on all files
            progress_callback: Optional callable invoked with an event dict as
                each file finishes (and once at start/end). Used by the web UI
                to stream live progress; ``None`` keeps the CLI behaviour
                unchanged. See :func:`result_event` for the event shape.
        """
        files = get_files_recursively(self.config.input_dir, supported_only=True)

        if not files:
            self.logger.warning("no_files_found", directory=str(self.config.input_dir))
            if progress_callback:
                progress_callback({"type": "start", "total": 0})
                progress_callback({"type": "done", "summary": self.get_summary()})
            return

        self.logger.info(
            "processing_directory",
            total_files=len(files),
            input_dir=str(self.config.input_dir),
        )

        # Pre-pass: batch-embed every image once on the GPU so the per-file
        # classification step below is a pure cache lookup (no GPU work).
        if self.creature_classification_processor:
            self.logger.info("precomputing_embeddings", total=len(files))
            computed = self.creature_classification_processor.precompute_embeddings(files)
            self.logger.info("embeddings_precomputed", new=computed, total=len(files))

        if progress_callback:
            progress_callback({"type": "start", "total": len(files)})

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_asset,
                    file_path,
                    remove_background=remove_background,
                    optimize_media=optimize_media,
                ): file_path
                for file_path in files
            }

            with tqdm(
                total=len(files),
                desc="Processing assets",
                unit="file",
            ) as pbar:
                done = 0
                for future in as_completed(futures):
                    pbar.update(1)
                    done += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        self.logger.exception("asset_crashed", input=str(futures[future]))
                        result = ProcessorResult(futures[future], success=False, error=str(exc))
                        self.failed_files.append(result)
                    if progress_callback:
                        progress_callback(result_event(result, done, len(files)))

        if progress_callback:
            progress_callback({"type": "done", "summary": self.get_summary()})

    def get_summary(self) -> dict:
        """Get processing summary."""
        total_processed = len(self.processed_files)
        total_failed = len(self.failed_files)
        total = total_processed + total_failed

        total_input_size = sum(r.metadata.get("input_size", 0) for r in self.processed_files)
        total_output_size = sum(r.metadata.get("output_size", 0) for r in self.processed_files)

        return {
            "total_files": total,
            "successful": total_processed,
            "failed": total_failed,
            "success_rate": f"{(total_processed / total * 100):.1f}%" if total > 0 else "N/A",
            "total_input_size": total_input_size,
            "total_output_size": total_output_size,
            "total_compression": (
                f"{((1 - total_output_size / total_input_size) * 100):.1f}%"
                if total_input_size > 0
                else "N/A"
            ),
        }

    def print_summary(self) -> None:
        """Print processing summary."""
        summary = self.get_summary()

        self.logger.info("pipeline_summary", **summary)

        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total files:        {summary['total_files']}")
        print(f"Successful:         {summary['successful']}")
        print(f"Failed:             {summary['failed']}")
        print(f"Success rate:       {summary['success_rate']}")
        print(f"Total input size:   {summary['total_input_size'] / (1024**2):.2f} MB")
        print(f"Total output size:  {summary['total_output_size'] / (1024**2):.2f} MB")
        print(f"Compression:        {summary['total_compression']}")
        print("=" * 60 + "\n")
