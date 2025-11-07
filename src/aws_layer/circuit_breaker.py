import time
import threading
import logging


class CircuitBreaker:
    """Implements circuit breaker pattern for API calls"""

    def __init__(self, name="default", failure_threshold=5, reset_timeout=60):
        self.name = name
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_failure_time = 0
        self.lock = threading.Lock()

    def execute(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == "OPEN":
                # Check if timeout has elapsed
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "HALF-OPEN"
                    logging.info(f"Circuit breaker '{self.name}' state changed to HALF-OPEN")
                else:
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = func(*args, **kwargs)

            with self.lock:
                if self.state == "HALF-OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logging.info(f"Circuit breaker '{self.name}' state changed to CLOSED")

            return result

        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logging.warning(
                        f"Circuit breaker '{self.name}' state changed to OPEN after {self.failure_count} failures")

            raise e