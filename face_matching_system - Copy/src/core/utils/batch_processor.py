
import gc
import psutil
import numpy as np
from typing import List, Callable, Any, Optional
import logging

class OptimizedBatchProcessor:
    """
    Memory-aware batch processor for large datasets.
    Automatically adjusts batch size based on available memory.
    """

    def __init__(self, 
                 initial_batch_size: int = 50,
                 memory_threshold: float = 0.8,
                 max_batch_size: int = 200):
        self.initial_batch_size = initial_batch_size
        self.memory_threshold = memory_threshold
        self.max_batch_size = max_batch_size
        self.logger = logging.getLogger(__name__)

    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent / 100.0

    def adjust_batch_size(self, current_batch_size: int) -> int:
        """Dynamically adjust batch size based on memory usage"""
        memory_usage = self.get_memory_usage()

        if memory_usage > self.memory_threshold:
            # Reduce batch size if memory usage is high
            new_size = max(1, int(current_batch_size * 0.7))
            self.logger.info(f"High memory usage ({memory_usage:.1%}), reducing batch size to {new_size}")
            return new_size
        elif memory_usage < 0.5 and current_batch_size < self.max_batch_size:
            # Increase batch size if memory usage is low
            new_size = min(self.max_batch_size, int(current_batch_size * 1.3))
            self.logger.info(f"Low memory usage ({memory_usage:.1%}), increasing batch size to {new_size}")
            return new_size

        return current_batch_size

    def process_in_batches(self, 
                          items: List[Any], 
                          process_func: Callable,
                          progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        Process items in dynamically sized batches with memory management

        Args:
            items: List of items to process
            process_func: Function to process each batch
            progress_callback: Optional callback for progress updates

        Returns:
            List of processed results
        """
        results = []
        batch_size = self.initial_batch_size
        processed = 0

        for i in range(0, len(items), batch_size):
            # Get current batch
            batch = items[i:i + batch_size]

            try:
                # Process batch
                batch_results = process_func(batch)
                results.extend(batch_results)
                processed += len(batch)

                # Update progress
                if progress_callback:
                    progress_callback(processed, len(items))

                # Adjust batch size for next iteration
                batch_size = self.adjust_batch_size(batch_size)

                # Force garbage collection after each batch
                gc.collect()

            except MemoryError:
                self.logger.warning("Memory error encountered, reducing batch size")
                batch_size = max(1, batch_size // 2)
                # Retry with smaller batch
                continue
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                continue

        return results
