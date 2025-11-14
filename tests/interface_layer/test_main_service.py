import pytest
import sys
import os
import importlib.util
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from pathlib import Path

import uvicorn


class TestMainService:
    """Test suite for main_service.py"""

    def setup_method(self):
        """Setup before each test method"""
        # Clean up modules before each test
        modules_to_clean = [
            'src.interface_layer.main_service',
        ]
        
        for module in modules_to_clean:
            if module in sys.modules:
                del sys.modules[module]

    def test_main_service_import_line_1(self):
        """Test line 1: from main import app"""
        mock_app = Mock()
        mock_main_module = Mock()
        mock_main_module.app = mock_app
        
        with patch.dict('sys.modules', {'main': mock_main_module}):
            # This should not raise an error
            from src.interface_layer.main import app
            assert app == mock_app

    def test_main_service_import_line_2(self):
        """Test line 2: import uvicorn"""
        # This should not raise an error
        import uvicorn as test_uvicorn
        assert test_uvicorn is not None

    def test_main_service_import_line_3(self):
        """Test line 3: from src.configuration_handler.config_loader import load_configuration"""
        # This should not raise an error
        from src.configuration_handler.config_loader import load_configuration
        assert load_configuration is not None

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration') 
    def test_main_service_module_import_as_not_main(self, mock_load_config, mock_uvicorn):
        """Test importing main_service module when __name__ != '__main__'"""
        mock_load_config.return_value = True
        
        # Mock the main module
        mock_app = Mock()
        mock_main_module = Mock()
        mock_main_module.app = mock_app
        
        with patch.dict('sys.modules', {'main': mock_main_module}):
            # Import main_service module directly (this will execute the module code)
            import src.interface_layer.main_service
            
            # Verify configuration was loaded at module level (line 5)
            assert mock_load_config.called
            # Verify uvicorn.run was NOT called (since __name__ != "__main__")
            assert not mock_uvicorn.called

    @patch('uvicorn.run')
    @patch('src.configuration_handler.config_loader.load_configuration')
    @patch('__main__.__name__', '__main__')
    def test_main_service_as_main_script(self, mock_load_config, mock_uvicorn):
        """Test main_service behavior when executed as main script"""
        mock_load_config.return_value = True
        
        # Mock the main module
        mock_app = Mock()
        mock_main_module = Mock()
        mock_main_module.app = mock_app
        
        # Simulate running main_service.py as a script
        with patch.dict('sys.modules', {'main': mock_main_module}):
            # Create a new module spec and execute it as __main__
            spec = importlib.util.spec_from_file_location(
                "__main__", 
                "src/interface_layer/main_service.py"
            )
            module = importlib.util.module_from_spec(spec)
            module.__name__ = "__main__"
            
            # Mock sys.modules to include our module as __main__
            with patch.dict('sys.modules', {'__main__': module}):
                # Execute the module which should trigger both lines
                spec.loader.exec_module(module)
                
                # Verify both function calls were made
                assert mock_load_config.called    # Line 5
                assert mock_uvicorn.called        # Line 7
                
                # Verify uvicorn parameters
                mock_uvicorn.assert_called_with(mock_app, host="0.0.0.0", port=8998)

    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_load_configuration_called_at_module_level(self, mock_load_config):
        """Test that load_configuration is called at module level"""
        mock_load_config.return_value = True
        
        # Mock the main module
        mock_app = Mock()
        mock_main_module = Mock()
        mock_main_module.app = mock_app
        
        with patch.dict('sys.modules', {'main': mock_main_module}):
            # Import the module (this will execute line 5: load_configuration())
            import src.interface_layer.main_service
            
            # Verify load_configuration was called
            assert mock_load_config.called


class TestMainServiceErrorHandling:
    """Test error handling scenarios"""

    def setup_method(self):
        """Setup before each test method"""
        # Clean up modules before each test
        modules_to_clean = [
            'src.interface_layer.main_service',
        ]
        
        for module in modules_to_clean:
            if module in sys.modules:
                del sys.modules[module]

    @patch('src.configuration_handler.config_loader.load_configuration')
    def test_configuration_error_propagation(self, mock_load_config):
        """Test that configuration errors are properly propagated"""
        mock_load_config.side_effect = Exception("Config error")
        
        # Mock the main module
        mock_app = Mock()
        mock_main_module = Mock()
        mock_main_module.app = mock_app
        
        with patch.dict('sys.modules', {'main': mock_main_module}):
            # Try to import the module and expect error to propagate
            with pytest.raises(Exception) as exc_info:
                import src.interface_layer.main_service
            
            assert "Config error" in str(exc_info.value)
            assert mock_load_config.called








if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.interface_layer.main_service", "--cov-report=term-missing"])
