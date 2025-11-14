import pytest
from unittest.mock import Mock, patch
import time
import threading
from src.aws_layer.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    @pytest.fixture
    def circuit_breaker(self):
        return CircuitBreaker(name="test_breaker", failure_threshold=2, reset_timeout=1)

    def test_initialization(self, circuit_breaker):
        assert circuit_breaker.name == "test_breaker"
        assert circuit_breaker.failure_threshold == 2
        assert circuit_breaker.reset_timeout == 1
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0

    def test_execute_success(self, circuit_breaker):
        # Test successful execution
        mock_func = Mock(return_value="success")

        result = circuit_breaker.execute(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_execute_failure(self, circuit_breaker):
        # Test failure that doesn't trip the breaker
        mock_func = Mock(side_effect=Exception("Test error"))

        with pytest.raises(Exception) as excinfo:
            circuit_breaker.execute(mock_func)

        assert "Test error" in str(excinfo.value)
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 1

    def test_execute_trip_breaker(self, circuit_breaker):
        # Test failures that trip the breaker
        mock_func = Mock(side_effect=Exception("Test error"))

        # First failure
        with pytest.raises(Exception):
            circuit_breaker.execute(mock_func)

        # Second failure - should trip the breaker
        with pytest.raises(Exception):
            circuit_breaker.execute(mock_func)

        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.failure_count == 2

        # Third call - should fail fast with circuit breaker exception
        with pytest.raises(Exception) as excinfo:
            circuit_breaker.execute(mock_func)

        assert "Circuit breaker" in str(excinfo.value)
        assert mock_func.call_count == 2  # Function not called on third attempt

    def test_half_open_state(self, circuit_breaker):
        # Test transition to half-open state after timeout
        mock_func = Mock(side_effect=Exception("Test error"))

        # Trip the breaker
        with pytest.raises(Exception):
            circuit_breaker.execute(mock_func)
        with pytest.raises(Exception):
            circuit_breaker.execute(mock_func)

        assert circuit_breaker.state == "OPEN"

        # Wait for reset timeout
        time.sleep(1.1)

        # Next call should put breaker in HALF-OPEN state
        mock_func.side_effect = None  # Remove the exception
        mock_func.return_value = "success"

        result = circuit_breaker.execute(mock_func)

        assert result == "success"
        assert circuit_breaker.state == "CLOSED"  # Success in HALF-OPEN resets to CLOSED
        assert circuit_breaker.failure_count == 0

    def test_half_open_failure(self, circuit_breaker):
        # Test failure in half-open state
        mock_func = Mock(side_effect=Exception("Test error"))

        # Trip the breaker
        with pytest.raises(Exception):
            circuit_breaker.execute(mock_func)
        with pytest.raises(Exception):
            circuit_breaker.execute(mock_func)

        assert circuit_breaker.state == "OPEN"

        # Wait for reset timeout
        time.sleep(1.1)

        # Next call should put breaker in HALF-OPEN state, but still fail
        with pytest.raises(Exception):
            circuit_breaker.execute(mock_func)

        # Should remain in HALF-OPEN state after failure
        assert circuit_breaker.state == "HALF-OPEN"
        assert circuit_breaker.failure_count == 3  # Incremented from the failure

    def test_reset_failure_count(self, circuit_breaker):
        # Test that failure count behavior in different states
        mock_func = Mock(side_effect=[Exception("Test error"), "success"])

        # First call fails
        with pytest.raises(Exception):
            circuit_breaker.execute(mock_func)

        assert circuit_breaker.failure_count == 1

        # Second call succeeds (in CLOSED state, failure count remains)
        result = circuit_breaker.execute(mock_func)

        assert result == "success"
        assert circuit_breaker.failure_count == 1  # Failure count doesn't reset in CLOSED state

    def test_failure_count_reset_half_open_to_closed(self, circuit_breaker):
        # Test that failure count resets only when transitioning from HALF-OPEN to CLOSED
        mock_func = Mock(side_effect=Exception("Test error"))

        # Trip the breaker (2 failures)
        with pytest.raises(Exception):
            circuit_breaker.execute(mock_func)
        with pytest.raises(Exception):
            circuit_breaker.execute(mock_func)

        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.failure_count == 2

        # Wait for reset timeout to move to HALF-OPEN
        time.sleep(1.1)

        # Success in HALF-OPEN should reset failure count and move to CLOSED
        mock_func.side_effect = None
        mock_func.return_value = "success"

        result = circuit_breaker.execute(mock_func)

        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0  # Should reset to 0 when HALF-OPEN -> CLOSED

    def test_concurrent_execution(self, circuit_breaker):
        """Test circuit breaker with concurrent execution"""

        # Create a function that fails
        def failing_func():
            raise Exception("Test error")

        # Create threads that will call the circuit breaker
        threads = []
        exceptions = []

        def run_with_breaker():
            try:
                circuit_breaker.execute(failing_func)
            except Exception as e:
                exceptions.append(str(e))

        # Start multiple threads
        for _ in range(5):
            thread = threading.Thread(target=run_with_breaker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that the breaker was tripped
        assert circuit_breaker.state == "OPEN"
        assert len(exceptions) == 5  # All calls should have failed
        assert any("Circuit breaker" in e for e in exceptions)  # At least one should be a circuit breaker exception

    def test_custom_parameters(self):
        # Test with custom parameters
        cb = CircuitBreaker(name="custom", failure_threshold=5, reset_timeout=0.5)

        assert cb.name == "custom"
        assert cb.failure_threshold == 5
        assert cb.reset_timeout == 0.5

        # Should take 5 failures to trip
        mock_func = Mock(side_effect=Exception("Test error"))

        for _ in range(4):
            with pytest.raises(Exception):
                cb.execute(mock_func)
            assert cb.state == "CLOSED"  # Still closed

        # Fifth failure should trip
        with pytest.raises(Exception):
            cb.execute(mock_func)
        assert cb.state == "OPEN"

        # Wait for shorter timeout
        time.sleep(0.6)

        # Should be in HALF-OPEN now
        mock_func.side_effect = None
        mock_func.return_value = "success"
        result = cb.execute(mock_func)
        assert result == "success"
        assert cb.state == "CLOSED"
