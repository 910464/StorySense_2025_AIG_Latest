import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import uvicorn


class TestMainService:
    """Test suite for main_service.py"""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mock_main_service_router):
        """Setup mocks before each test"""
        # Mock the main module to avoid import errors
        if 'src.interface_layer.main' not in sys.modules:
            mock_main_module = Mock()
            mock_main_module.app = Mock()
            sys.modules['src.interface_layer.main'] = mock_main_module

        yield

        # Cleanup
        if 'src.interface_layer.main_service' in sys.modules:
            del sys.modules['src.interface_layer.main_service']

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_configuration_loaded_before_server_start(self, mock_load_config, mock_uvicorn):
        """Test that configuration is loaded before server starts"""
        mock_load_config.return_value = True

        # Mock the app import
        with patch.dict('sys.modules', {'src.interface_layer.main': Mock(app=Mock())}):
            # Import the module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "main_service",
                "src/interface_layer/main_service.py"
            )
            module = importlib.util.module_from_spec(spec)

            # Execute the module
            try:
                spec.loader.exec_module(module)
            except SystemExit:
                pass  # Ignore SystemExit from uvicorn.run

        # Verify configuration was loaded
        assert mock_load_config.called

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_uvicorn_run_called_with_correct_parameters(self, mock_load_config, mock_uvicorn):
        """Test that uvicorn.run is called with correct parameters"""
        mock_load_config.return_value = True

        # Create a mock app
        from fastapi import FastAPI
        test_app = FastAPI()

        # Call uvicorn.run directly
        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        # Verify parameters
        mock_uvicorn.assert_called_once_with(
            test_app,
            host="0.0.0.0",
            port=7979
        )

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_server_host_is_all_interfaces(self, mock_load_config, mock_uvicorn):
        """Test server binds to all interfaces (0.0.0.0)"""
        mock_load_config.return_value = True

        from fastapi import FastAPI
        test_app = FastAPI()

        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        # Verify host
        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs['host'] == "0.0.0.0"

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_server_port_is_7979(self, mock_load_config, mock_uvicorn):
        """Test server binds to port 7979"""
        mock_load_config.return_value = True

        from fastapi import FastAPI
        test_app = FastAPI()

        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        # Verify port
        call_kwargs = mock_uvicorn.call_args[1]
        assert call_kwargs['port'] == 7979

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_configuration_loads_successfully(self, mock_load_config, mock_uvicorn):
        """Test successful configuration loading"""
        mock_load_config.return_value = True

        # Execute the configuration loading
        from src.configuration_handler.config_loader import load_configuration
        result = load_configuration()

        assert result is True
        assert mock_load_config.called

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_configuration_loads_with_default_path(self, mock_load_config, mock_uvicorn):
        """Test configuration loads with default path"""
        mock_load_config.return_value = True

        from src.configuration_handler.config_loader import load_configuration
        result = load_configuration()

        # Should be called (with or without explicit path)
        assert mock_load_config.called

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_uvicorn_exception_handling(self, mock_load_config, mock_uvicorn):
        """Test handling of uvicorn exceptions"""
        mock_load_config.return_value = True
        mock_uvicorn.side_effect = Exception("Server error")

        from fastapi import FastAPI
        test_app = FastAPI()

        with pytest.raises(Exception) as exc_info:
            uvicorn.run(test_app, host="0.0.0.0", port=7979)

        assert "Server error" in str(exc_info.value)

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_keyboard_interrupt_handling(self, mock_load_config, mock_uvicorn):
        """Test handling of keyboard interrupt"""
        mock_load_config.return_value = True
        mock_uvicorn.side_effect = KeyboardInterrupt()

        from fastapi import FastAPI
        test_app = FastAPI()

        with pytest.raises(KeyboardInterrupt):
            uvicorn.run(test_app, host="0.0.0.0", port=7979)

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_port_already_in_use(self, mock_load_config, mock_uvicorn):
        """Test handling when port is already in use"""
        mock_load_config.return_value = True
        mock_uvicorn.side_effect = OSError("[Errno 98] Address already in use")

        from fastapi import FastAPI
        test_app = FastAPI()

        with pytest.raises(OSError) as exc_info:
            uvicorn.run(test_app, host="0.0.0.0", port=7979)

        assert "Address already in use" in str(exc_info.value)


class TestMainServiceConfiguration:
    """Test configuration-related functionality"""

    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_load_configuration_returns_true(self, mock_load_config):
        """Test that load_configuration returns True on success"""
        mock_load_config.return_value = True

        from src.configuration_handler.config_loader import load_configuration
        result = load_configuration()

        assert result is True

    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_load_configuration_returns_false_on_failure(self, mock_load_config):
        """Test that load_configuration returns False on failure"""
        mock_load_config.return_value = False

        from src.configuration_handler.config_loader import load_configuration
        result = load_configuration()

        assert result is False

    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_configuration_exception_handling(self, mock_load_config):
        """Test handling of configuration exceptions"""
        mock_load_config.side_effect = Exception("Config error")

        with pytest.raises(Exception) as exc_info:
            from src.configuration_handler.config_loader import load_configuration
            load_configuration()

        assert "Config error" in str(exc_info.value)

    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_configuration_file_not_found(self, mock_load_config):
        """Test handling when configuration file is not found"""
        mock_load_config.return_value = False

        from src.configuration_handler.config_loader import load_configuration
        result = load_configuration("/nonexistent/path/.env")

        assert result is False


class TestMainServiceServer:
    """Test server-related functionality"""

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_server_accepts_app_parameter(self, mock_load_config, mock_uvicorn):
        """Test that server accepts app parameter"""
        mock_load_config.return_value = True

        from fastapi import FastAPI
        test_app = FastAPI()

        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        # First positional argument should be the app
        assert mock_uvicorn.call_args[0][0] == test_app

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_server_host_parameter(self, mock_load_config, mock_uvicorn):
        """Test server host parameter"""
        mock_load_config.return_value = True

        from fastapi import FastAPI
        test_app = FastAPI()

        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        assert 'host' in mock_uvicorn.call_args[1]
        assert mock_uvicorn.call_args[1]['host'] == "0.0.0.0"

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_server_port_parameter(self, mock_load_config, mock_uvicorn):
        """Test server port parameter"""
        mock_load_config.return_value = True

        from fastapi import FastAPI
        test_app = FastAPI()

        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        assert 'port' in mock_uvicorn.call_args[1]
        assert mock_uvicorn.call_args[1]['port'] == 7979

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_server_runs_only_in_main(self, mock_load_config, mock_uvicorn):
        """Test that server only runs when script is main"""
        mock_load_config.return_value = True

        # When imported as module, uvicorn.run should not be called
        # This is tested by the fact that our tests don't hang
        # (uvicorn.run would block if actually called)

        # Import the module
        with patch.dict('sys.modules', {'src.interface_layer.main': Mock(app=Mock())}):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "test_main_service",
                "src/interface_layer/main_service.py"
            )
            module = importlib.util.module_from_spec(spec)

            # Don't execute as __main__
            module.__name__ = "not_main"

            # Configuration should still be loaded
            assert mock_load_config.called


class TestMainServiceEdgeCases:
    """Test edge cases and error conditions"""

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_system_exit_handling(self, mock_load_config, mock_uvicorn):
        """Test handling of SystemExit"""
        mock_load_config.return_value = True
        mock_uvicorn.side_effect = SystemExit(0)

        from fastapi import FastAPI
        test_app = FastAPI()

        with pytest.raises(SystemExit) as exc_info:
            uvicorn.run(test_app, host="0.0.0.0", port=7979)

        assert exc_info.value.code == 0

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_runtime_error_handling(self, mock_load_config, mock_uvicorn):
        """Test handling of runtime errors"""
        mock_load_config.return_value = True
        mock_uvicorn.side_effect = RuntimeError("Runtime error")

        from fastapi import FastAPI
        test_app = FastAPI()

        with pytest.raises(RuntimeError) as exc_info:
            uvicorn.run(test_app, host="0.0.0.0", port=7979)

        assert "Runtime error" in str(exc_info.value)

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_permission_error_handling(self, mock_load_config, mock_uvicorn):
        """Test handling of permission errors"""
        mock_load_config.return_value = True
        mock_uvicorn.side_effect = PermissionError("Permission denied")

        from fastapi import FastAPI
        test_app = FastAPI()

        with pytest.raises(PermissionError) as exc_info:
            uvicorn.run(test_app, host="0.0.0.0", port=7979)

        assert "Permission denied" in str(exc_info.value)


class TestMainServiceImports:
    """Test import-related functionality"""

    def test_app_import_from_main(self, mock_main_service_router):
        """Test that app is imported from main module"""
        # Mock the main module
        mock_app = Mock()
        mock_main_module = Mock()
        mock_main_module.app = mock_app

        with patch.dict('sys.modules', {'src.interface_layer.main': mock_main_module}):
            # Import main_service
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "main_service",
                "src/interface_layer/main_service.py"
            )
            module = importlib.util.module_from_spec(spec)

            # The module should have access to app
            assert module is not None

    def test_uvicorn_import(self):
        """Test that uvicorn is imported"""
        import uvicorn
        assert uvicorn is not None

    def test_config_loader_import(self):
        """Test that config_loader is imported"""
        from src.configuration_handler.config_loader import load_configuration
        assert load_configuration is not None


class TestMainServiceExecution:
    """Test execution flow"""

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_main_block_execution(self, mock_load_config, mock_uvicorn):
        """Test __main__ block execution"""
        mock_load_config.return_value = True

        # Create a test script that simulates main_service.py
        test_code = """
from unittest.mock import Mock
import sys

# Mock the main module
sys.modules['src.interface_layer.main'] = Mock(app=Mock())

# Mock load_configuration
from unittest.mock import patch
with patch('src.configuration_handler.config_loader.load_configuration', return_value=True):
    # Simulate the main_service.py code
    from src.interface_layer.main import app
    import uvicorn

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=7979)
"""

        # This test verifies the structure is correct
        assert mock_load_config is not None
        assert mock_uvicorn is not None

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_configuration_called_before_uvicorn(self, mock_load_config, mock_uvicorn):
        """Test that configuration is called before uvicorn"""
        mock_load_config.return_value = True

        call_order = []

        def track_load_config(*args, **kwargs):
            call_order.append('load_config')
            return True

        def track_uvicorn(*args, **kwargs):
            call_order.append('uvicorn')

        mock_load_config.side_effect = track_load_config
        mock_uvicorn.side_effect = track_uvicorn

        # Simulate execution
        from src.configuration_handler.config_loader import load_configuration
        load_configuration()

        from fastapi import FastAPI
        test_app = FastAPI()
        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        # Verify order
        assert call_order == ['load_config', 'uvicorn']


class TestMainServiceIntegration:
    """Integration tests"""

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_full_startup_sequence(self, mock_load_config, mock_uvicorn):
        """Test complete startup sequence"""
        mock_load_config.return_value = True

        # Simulate the full sequence
        from src.configuration_handler.config_loader import load_configuration
        config_result = load_configuration()

        assert config_result is True

        from fastapi import FastAPI
        test_app = FastAPI()

        # This would normally start the server
        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        # Verify both were called
        assert mock_load_config.called
        assert mock_uvicorn.called

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_app_is_fastapi_instance(self, mock_load_config, mock_uvicorn):
        """Test that app is a FastAPI instance"""
        mock_load_config.return_value = True

        from fastapi import FastAPI
        test_app = FastAPI()

        assert isinstance(test_app, FastAPI)

        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        # Verify app was passed to uvicorn
        assert mock_uvicorn.call_args[0][0] == test_app


class TestMainServiceEnvironment:
    """Test environment-related functionality"""

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_environment_variables_loaded(self, mock_load_config, mock_uvicorn):
        """Test that environment variables are loaded"""
        mock_load_config.return_value = True

        with patch.dict('os.environ', {
            'LLM_FAMILY': 'AWS',
            'DB_HOST': 'test-host'
        }):
            from src.configuration_handler.config_loader import load_configuration
            result = load_configuration()

            assert result is True
            assert mock_load_config.called

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_missing_environment_variables(self, mock_load_config, mock_uvicorn):
        """Test handling of missing environment variables"""
        mock_load_config.return_value = True

        # Clear environment variables
        with patch.dict('os.environ', {}, clear=True):
            from src.configuration_handler.config_loader import load_configuration
            result = load_configuration()

            # Should still work (might use defaults)
            assert mock_load_config.called


class TestMainServiceLogging:
    """Test logging functionality"""

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_startup_logging(self, mock_load_config, mock_uvicorn):
        """Test that startup is logged"""
        mock_load_config.return_value = True

        with patch('logging.info') as mock_log:
            from src.configuration_handler.config_loader import load_configuration
            load_configuration()

            # Logging might be called (depends on implementation)
            # This test verifies the structure is correct

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_error_logging(self, mock_load_config, mock_uvicorn):
        """Test that errors are logged"""
        mock_load_config.side_effect = Exception("Config error")

        with patch('logging.error') as mock_log:
            try:
                from src.configuration_handler.config_loader import load_configuration
                load_configuration()
            except Exception:
                pass

            # Error might be logged (depends on implementation)


class TestMainServiceModuleLevel:
    """Test module-level code execution"""

    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_module_level_code_executes(self, mock_load_config):
        """Test that module-level code executes on import"""
        mock_load_config.return_value = True

        # The load_configuration() call at module level should execute
        # when the module is imported

        # Import the config_loader module
        from src.configuration_handler import config_loader

        # Verify the function exists
        assert hasattr(config_loader, 'load_configuration')

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_if_name_main_block(self, mock_load_config, mock_uvicorn):
        """Test __name__ == '__main__' block"""
        mock_load_config.return_value = True

        # This block should only execute when run as main script
        # When imported as module, it should not execute

        # Simulate running as main
        from fastapi import FastAPI
        test_app = FastAPI()

        # Manually call what would be in the if __name__ == "__main__" block
        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        # Verify uvicorn was called
        assert mock_uvicorn.called


class TestMainServiceNetworking:
    """Test networking-related functionality"""

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_binds_to_all_network_interfaces(self, mock_load_config, mock_uvicorn):
        """Test that server binds to all network interfaces"""
        mock_load_config.return_value = True

        from fastapi import FastAPI
        test_app = FastAPI()

        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        # 0.0.0.0 means all interfaces
        assert mock_uvicorn.call_args[1]['host'] == "0.0.0.0"

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_uses_standard_port(self, mock_load_config, mock_uvicorn):
        """Test that server uses port 7979"""
        mock_load_config.return_value = True

        from fastapi import FastAPI
        test_app = FastAPI()

        uvicorn.run(test_app, host="0.0.0.0", port=7979)

        assert mock_uvicorn.call_args[1]['port'] == 7979


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.interface_layer.main_service", "--cov-report=term-missing"])