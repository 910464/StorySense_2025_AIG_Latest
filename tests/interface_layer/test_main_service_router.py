import pytest
import os
import tempfile
import shutil
import json
import zipfile
from unittest.mock import Mock, patch, MagicMock, mock_open, AsyncMock
from pathlib import Path
import pandas as pd
import uuid

from fastapi import UploadFile, HTTPException, BackgroundTasks
from fastapi.testclient import TestClient
from fastapi import FastAPI
import io

# Import the module under test
from src.interface_layer.main_service_router import (
    story_sense_router,
    process_stories,
    process_stories_with_image_support,
    process_historical_context,
    active_jobs,
    ProcessRequest,
    TEMP_DIR
)


class TestMainServiceRouter:
    """Test suite for main_service_router.py"""

    def setup_method(self):
        """Setup before each test method"""
        # Clear active jobs
        active_jobs.clear()
        
        # Create test app
        self.app = FastAPI()
        self.app.include_router(story_sense_router)
        self.client = TestClient(self.app)
        
        # Create temporary directories for testing
        self.test_temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test method"""
        # Clear active jobs
        active_jobs.clear()
        
        # Clean up test directories
        if os.path.exists(self.test_temp_dir):
            shutil.rmtree(self.test_temp_dir)

    def test_process_request_model_defaults(self):
        """Test ProcessRequest model with default values"""
        request = ProcessRequest()
        assert request.batch_size == 5
        assert request.parallel is False
        assert request.process_context is False
        assert request.force_reprocess is False

    def test_process_request_model_custom_values(self):
        """Test ProcessRequest model with custom values"""
        request = ProcessRequest(
            batch_size=10,
            parallel=True,
            process_context=True,
            force_reprocess=True
        )
        assert request.batch_size == 10
        assert request.parallel is True
        assert request.process_context is True
        assert request.force_reprocess is True

    @patch('src.interface_layer.main_service_router.StorySenseGenerator')
    @patch('src.interface_layer.main_service_router.logger')
    @patch('os.listdir')
    @pytest.mark.asyncio
    async def test_process_stories_success(self, mock_listdir, mock_logger, mock_ssg_class):
        """Test successful story processing"""
        # Setup
        job_id = "test-job-123"
        user_stories_path = "/path/to/stories.xlsx"
        context_path = "/path/to/context.xlsx"
        
        # Mock StorySenseGenerator
        mock_ssg = Mock()
        mock_ssg.output_file_path = "/output/path"
        mock_ssg_class.return_value = mock_ssg
        
        # Mock directory listing
        mock_listdir.return_value = ["result1.xlsx", "result2.html", "other.txt"]
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute
        await process_stories(
            job_id=job_id,
            user_stories_path=user_stories_path,
            context_path=context_path,
            batch_size=5,
            parallel=False,
            process_context=True,
            force_reprocess=False
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "completed"
        assert len(active_jobs[job_id]["output_files"]) == 3
        mock_ssg_class.assert_called_once_with(user_stories_path, context_path)
        mock_ssg.process_context_library.assert_called_once_with(force_reprocess=False)
        mock_ssg.process_user_stories.assert_called_once_with(batch_size=5, parallel=False)

    @patch('src.interface_layer.main_service_router.StorySenseGenerator')
    @patch('src.interface_layer.main_service_router.logger')
    @pytest.mark.asyncio
    async def test_process_stories_exception(self, mock_logger, mock_ssg_class):
        """Test story processing with exception"""
        # Setup
        job_id = "test-job-error"
        user_stories_path = "/path/to/stories.xlsx"
        
        # Mock StorySenseGenerator to raise exception
        mock_ssg_class.side_effect = Exception("Processing error")
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute
        await process_stories(
            job_id=job_id,
            user_stories_path=user_stories_path
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "failed"
        assert "Processing error" in active_jobs[job_id]["error"]
        mock_logger.error.assert_called()

    @patch('src.interface_layer.main_service_router.StorySenseGenerator')
    @patch('src.interface_layer.main_service_router.logger')
    @patch('os.listdir')
    @pytest.mark.asyncio
    async def test_process_stories_with_image_support_with_context(self, mock_listdir, mock_logger, mock_ssg_class):
        """Test story processing with image support and context"""
        # Setup
        job_id = "test-job-image"
        user_stories_path = "/path/to/stories.xlsx"
        context_path = "/path/to/context"
        
        # Mock StorySenseGenerator
        mock_ssg = Mock()
        mock_ssg.output_file_path = "/output/path"
        mock_ssg_class.return_value = mock_ssg
        
        # Mock directory listing
        mock_listdir.return_value = ["result1.xlsx", "result2.html"]
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute
        await process_stories_with_image_support(
            job_id=job_id,
            user_stories_path=user_stories_path,
            context_path=context_path,
            context_is_image=True,
            batch_size=10,
            parallel=True,
            process_context=True,
            force_reprocess=True
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "completed"
        assert len(active_jobs[job_id]["output_files"]) == 2
        mock_ssg_class.assert_called_once_with(user_stories_path)
        mock_ssg.process_context_library.assert_called_once_with(context_path, force_reprocess=True)
        mock_ssg.process_user_stories.assert_called_once_with(batch_size=10, parallel=True)

    @patch('src.interface_layer.main_service_router.StorySenseGenerator')
    @patch('src.interface_layer.main_service_router.logger')
    @patch('os.listdir')
    @pytest.mark.asyncio
    async def test_process_stories_with_image_support_without_context(self, mock_listdir, mock_logger, mock_ssg_class):
        """Test story processing with image support but no context"""
        # Setup
        job_id = "test-job-no-context"
        user_stories_path = "/path/to/stories.xlsx"
        
        # Mock StorySenseGenerator
        mock_ssg = Mock()
        mock_ssg.output_file_path = "/output/path"
        mock_ssg_class.return_value = mock_ssg
        
        # Mock directory listing
        mock_listdir.return_value = ["result1.xlsx"]
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute
        await process_stories_with_image_support(
            job_id=job_id,
            user_stories_path=user_stories_path,
            context_path=None,
            process_context=True
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "completed"
        mock_ssg_class.assert_called_once_with(user_stories_path, None)
        mock_ssg.process_context_library.assert_called_once_with(force_reprocess=False)

    @patch('src.interface_layer.main_service_router.EnhancedContextProcessor')
    @patch('src.interface_layer.main_service_router.logger')
    @patch('os.makedirs')
    @patch('os.walk')
    @patch('shutil.copy')
    @pytest.mark.asyncio
    async def test_process_historical_context_success(self, mock_copy, mock_walk, mock_makedirs, mock_logger, mock_processor_class):
        """Test successful historical context processing"""
        # Setup
        job_id = "test-historical-job"
        zip_path = "/path/to/context.zip"
        extract_dir = "/path/to/extracted"
        
        # Mock EnhancedContextProcessor
        mock_processor = Mock()
        mock_processor.process_all_context_files.return_value = {
            "processed_files": 5,
            "total_files": 5,
            "failed_files": 0
        }
        mock_processor_class.return_value = mock_processor
        
        # Mock os.walk to find image files
        mock_walk.return_value = [
            ("/path/to/extracted", [], ["image1.jpg", "image2.png", "doc.pdf"])
        ]
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute
        await process_historical_context(
            job_id=job_id,
            zip_path=zip_path,
            extract_dir=extract_dir,
            force_reprocess=True
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "completed"
        assert active_jobs[job_id]["results"]["processed_files"] == 5
        mock_processor_class.assert_called_once_with(extract_dir)
        mock_processor.process_all_context_files.assert_called_once_with(force_reprocess=True)

    @patch('src.interface_layer.main_service_router.EnhancedContextProcessor')
    @patch('src.interface_layer.main_service_router.logger')
    @pytest.mark.asyncio
    async def test_process_historical_context_exception(self, mock_logger, mock_processor_class):
        """Test historical context processing with exception"""
        # Setup
        job_id = "test-historical-error"
        zip_path = "/path/to/context.zip"
        extract_dir = "/path/to/extracted"
        
        # Mock EnhancedContextProcessor to raise exception
        mock_processor_class.side_effect = Exception("Context processing error")
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute
        await process_historical_context(
            job_id=job_id,
            zip_path=zip_path,
            extract_dir=extract_dir
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "failed"
        assert "Context processing error" in active_jobs[job_id]["error"]
        mock_logger.error.assert_called()

    def create_mock_upload_file(self, filename: str, content: bytes = b"test content"):
        """Helper method to create mock UploadFile"""
        file_obj = io.BytesIO(content)
        return UploadFile(filename=filename, file=file_obj)

    @patch('src.interface_layer.main_service_router.process_stories')
    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    @patch('uuid.uuid4')
    def test_upload_and_process_stories_only(self, mock_uuid, mock_copyfileobj, mock_makedirs, mock_process):
        """Test upload and process endpoint with stories file only"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-job-id")
        
        # Create mock file
        user_stories_file = self.create_mock_upload_file("stories.xlsx")
        
        # Mock parameters
        params = json.dumps({
            "batch_size": 10,
            "parallel": True,
            "process_context": False,
            "force_reprocess": False
        })
        
        # Execute
        response = self.client.post(
            "/process-stories",
            files={"user_stories_file": ("stories.xlsx", user_stories_file.file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"params": params}
        )
        
        # Verify
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == "test-job-id"
        assert result["status"] == "submitted"
        assert "test-job-id" in active_jobs

    @patch('src.interface_layer.main_service_router.process_stories_with_image_support')
    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    @patch('shutil.copy')
    @patch('uuid.uuid4')
    def test_upload_and_process_with_image_context(self, mock_uuid, mock_copy, mock_copyfileobj, mock_makedirs, mock_process):
        """Test upload and process endpoint with image context files"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-job-id")
        
        # Create mock files
        user_stories_file = self.create_mock_upload_file("stories.xlsx")
        context_file = self.create_mock_upload_file("wireframe.png", b"fake image data")
        
        # Mock parameters
        params = json.dumps({
            "batch_size": 5,
            "parallel": False,
            "process_context": True,
            "force_reprocess": True
        })
        
        # Execute
        response = self.client.post(
            "/process-stories",
            files=[
                ("user_stories_file", ("stories.xlsx", user_stories_file.file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
                ("context_files", ("wireframe.png", context_file.file, "image/png"))
            ],
            data={"params": params}
        )
        
        # Verify
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == "test-job-id"
        assert result["status"] == "submitted"
        assert "test-job-id" in active_jobs
        assert active_jobs["test-job-id"]["context_files"] == ["wireframe.png"]

    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    @patch('uuid.uuid4')
    def test_upload_and_process_invalid_params(self, mock_uuid, mock_copyfileobj, mock_makedirs):
        """Test upload and process endpoint with invalid parameters"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-job-id")
        
        # Create mock file
        user_stories_file = self.create_mock_upload_file("stories.xlsx")
        
        # Invalid JSON parameters
        params = "invalid json"
        
        # Execute
        response = self.client.post(
            "/process-stories",
            files={"user_stories_file": ("stories.xlsx", user_stories_file.file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"params": params}
        )
        
        # Verify
        assert response.status_code == 500

    @patch('src.interface_layer.main_service_router.process_historical_context')
    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    @patch('zipfile.ZipFile')
    @patch('uuid.uuid4')
    def test_upload_historical_context_success(self, mock_uuid, mock_zipfile, mock_copyfileobj, mock_makedirs, mock_process):
        """Test successful historical context upload"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-context-job")
        
        # Mock zipfile
        mock_zip = Mock()
        mock_zip.namelist.return_value = ["file1.pdf", "file2.docx"]
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Create mock zip file
        zip_file = self.create_mock_upload_file("context.zip", b"fake zip data")
        
        # Execute
        response = self.client.post(
            "/upload-historical-context",
            files={"zip_file": ("context.zip", zip_file.file, "application/zip")},
            data={"force_reprocess": "true"}
        )
        
        # Verify
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == "test-context-job"
        assert result["status"] == "submitted"
        assert "test-context-job" in active_jobs

    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    @patch('zipfile.ZipFile')
    @patch('uuid.uuid4')
    def test_upload_historical_context_path_traversal(self, mock_uuid, mock_zipfile, mock_copyfileobj, mock_makedirs):
        """Test historical context upload with path traversal attack"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-context-job")
        
        # Mock zipfile with malicious path
        mock_zip = Mock()
        mock_zip.namelist.return_value = ["../../../etc/passwd"]
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Create mock zip file
        zip_file = self.create_mock_upload_file("malicious.zip", b"fake zip data")
        
        # Execute
        response = self.client.post(
            "/upload-historical-context",
            files={"zip_file": ("malicious.zip", zip_file.file, "application/zip")},
            data={"force_reprocess": "false"}
        )
        
        # Verify
        assert response.status_code == 400
        assert "path traversal" in response.json()["detail"]

    def test_download_results_job_not_found(self):
        """Test download results with non-existent job"""
        response = self.client.get("/download-results/non-existent-job")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

    def test_download_results_not_completed(self):
        """Test download results with job not completed"""
        # Setup
        job_id = "test-job-processing"
        active_jobs[job_id] = {"status": "processing"}
        
        # Execute
        response = self.client.get(f"/download-results/{job_id}")
        
        # Verify
        assert response.status_code == 400
        assert "Results not ready" in response.json()["detail"]

    def test_download_results_no_output_files(self):
        """Test download results with no output files"""
        # Setup
        job_id = "test-job-no-files"
        active_jobs[job_id] = {"status": "completed", "output_files": []}
        
        # Execute
        response = self.client.get(f"/download-results/{job_id}")
        
        # Verify
        assert response.status_code == 404
        assert "No output files available" in response.json()["detail"]

    @patch('starlette.responses.FileResponse')
    @patch('os.path.basename')
    @patch('zipfile.ZipFile')
    @patch('os.path.exists')
    def test_download_results_multiple_files(self, mock_exists, mock_zipfile, mock_basename, mock_file_response):
        """Test download results with multiple files (zip creation)"""
        # Setup
        job_id = "test-job-multi-files"
        
        output_files = ["/path/to/file1.xlsx", "/path/to/file2.html"]
        active_jobs[job_id] = {
            "status": "completed",
            "output_files": output_files
        }
        
        def basename_side_effect(path):
            return path.split('\\')[-1] if '\\' in path else path.split('/')[-1]
        mock_basename.side_effect = basename_side_effect
        mock_exists.return_value = True
        
        # Mock zipfile
        mock_zip = Mock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Mock FileResponse
        mock_response = Mock()
        mock_file_response.return_value = mock_response
        
        # Execute
        response = self.client.get(f"/download-results/{job_id}")
        
        # Verify zipfile operations were called
        mock_zip.write.assert_any_call("/path/to/file1.xlsx", "file1.xlsx")
        mock_zip.write.assert_any_call("/path/to/file2.html", "file2.html")

    @patch('starlette.responses.FileResponse')
    @patch('os.path.basename')
    @patch('os.path.exists')
    def test_download_results_single_file(self, mock_exists, mock_basename, mock_file_response):
        """Test download results with single file"""
        # Setup
        job_id = "test-job-single-file"
        
        # Create temporary test file
        test_file = os.path.join(self.test_temp_dir, "single_file.xlsx")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        output_files = [test_file]
        active_jobs[job_id] = {
            "status": "completed",
            "output_files": output_files
        }
        
        mock_basename.return_value = "single_file.xlsx"
        mock_exists.return_value = True
        
        # Mock FileResponse to avoid file system interactions
        mock_response = Mock()
        mock_file_response.return_value = mock_response
        
        # Execute
        response = self.client.get(f"/download-results/{job_id}")
        
        # Verify FileResponse was called
        mock_file_response.assert_called_once_with(
            path=test_file,
            filename="single_file.xlsx",
            media_type='application/octet-stream'
        )

    @patch('src.interface_layer.main_service_router.logger')
    def test_module_level_directory_creation(self, mock_logger):
        """Test that directories are created at module level"""
        # This tests the module-level os.makedirs calls
        # We can verify this by checking if the directories exist or by mocking
        # The actual directories are created when the module is imported
        # Since the module is already imported, we'll test the TEMP_DIR creation
        assert TEMP_DIR is not None
        assert isinstance(TEMP_DIR, str)

    @patch('src.interface_layer.main_service_router.process_stories')
    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    @patch('uuid.uuid4')
    def test_upload_and_process_with_non_image_context(self, mock_uuid, mock_copyfileobj, mock_makedirs, mock_process):
        """Test upload and process endpoint with non-image context files"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-job-id")
        
        # Create mock files
        user_stories_file = self.create_mock_upload_file("stories.xlsx")
        context_file = self.create_mock_upload_file("context.pdf", b"fake pdf data")
        
        # Mock parameters
        params = json.dumps({
            "batch_size": 5,
            "parallel": False,
            "process_context": True,
            "force_reprocess": False
        })
        
        # Execute
        response = self.client.post(
            "/process-stories",
            files=[
                ("user_stories_file", ("stories.xlsx", user_stories_file.file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
                ("context_files", ("context.pdf", context_file.file, "application/pdf"))
            ],
            data={"params": params}
        )
        
        # Verify
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == "test-job-id"
        assert result["status"] == "submitted"
        assert "test-job-id" in active_jobs
        assert active_jobs["test-job-id"]["context_files"] == ["context.pdf"]

    @patch('src.interface_layer.main_service_router.process_stories_with_image_support')
    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    @patch('shutil.copy')
    @patch('uuid.uuid4')
    def test_upload_and_process_mixed_context_files(self, mock_uuid, mock_copy, mock_copyfileobj, mock_makedirs, mock_process):
        """Test upload and process endpoint with mixed context files (image and non-image)"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-job-id")
        
        # Create mock files
        user_stories_file = self.create_mock_upload_file("stories.xlsx")
        image_file = self.create_mock_upload_file("wireframe.jpg", b"fake image data")
        doc_file = self.create_mock_upload_file("requirements.pdf", b"fake pdf data")
        
        # Mock parameters
        params = json.dumps({
            "batch_size": 5,
            "parallel": False,
            "process_context": True,
            "force_reprocess": False
        })
        
        # Execute
        response = self.client.post(
            "/process-stories",
            files=[
                ("user_stories_file", ("stories.xlsx", user_stories_file.file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
                ("context_files", ("wireframe.jpg", image_file.file, "image/jpeg")),
                ("context_files", ("requirements.pdf", doc_file.file, "application/pdf"))
            ],
            data={"params": params}
        )
        
        # Verify
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == "test-job-id"
        assert result["status"] == "submitted"
        assert "test-job-id" in active_jobs
        assert "wireframe.jpg" in active_jobs["test-job-id"]["context_files"]
        assert "requirements.pdf" in active_jobs["test-job-id"]["context_files"]

    @patch('src.interface_layer.main_service_router.EnhancedContextProcessor')
    @patch('src.interface_layer.main_service_router.logger')
    @patch('os.makedirs')
    @patch('os.walk')
    @patch('os.path.exists')
    @patch('shutil.copy')
    @pytest.mark.asyncio
    async def test_process_historical_context_with_images_in_root(self, mock_copy, mock_exists, mock_walk, mock_makedirs, mock_logger, mock_processor_class):
        """Test historical context processing with images in root directory"""
        # Setup
        job_id = "test-historical-images"
        zip_path = "/path/to/context.zip"
        extract_dir = "/path/to/extracted"
        
        # Mock EnhancedContextProcessor
        mock_processor = Mock()
        mock_processor.process_all_context_files.return_value = {"processed_files": 3}
        mock_processor_class.return_value = mock_processor
        
        # Mock os.walk to find image files in root directory
        mock_walk.return_value = [
            (extract_dir, [], ["image1.jpg", "image2.png", "doc.pdf"])
        ]
        
        # Mock os.path.exists to return False for target path (so copy will be performed)
        mock_exists.return_value = False
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute
        await process_historical_context(
            job_id=job_id,
            zip_path=zip_path,
            extract_dir=extract_dir
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "completed"
        # Verify that images were copied to wireframes directory
        assert mock_copy.call_count == 2  # Two image files should be copied

    @patch('src.interface_layer.main_service_router.process_stories')
    @patch('os.makedirs')
    def test_upload_and_process_makedirs_exception(self, mock_makedirs, mock_process):
        """Test upload and process endpoint when directory creation fails"""
        # Setup
        mock_makedirs.side_effect = OSError("Permission denied")
        
        # Create mock file
        user_stories_file = self.create_mock_upload_file("stories.xlsx")
        
        # Mock parameters
        params = json.dumps({
            "batch_size": 5,
            "parallel": False,
            "process_context": False,
            "force_reprocess": False
        })
        
        # Execute
        response = self.client.post(
            "/process-stories",
            files={"user_stories_file": ("stories.xlsx", user_stories_file.file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"params": params}
        )
        
        # Verify
        assert response.status_code == 500

    @patch('src.interface_layer.main_service_router.process_historical_context')
    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    def test_upload_historical_context_copyfileobj_exception(self, mock_copyfileobj, mock_makedirs, mock_process):
        """Test historical context upload when file copy fails"""
        # Setup
        mock_copyfileobj.side_effect = IOError("Disk full")
        
        # Create mock zip file
        zip_file = self.create_mock_upload_file("context.zip", b"fake zip data")
        
        # Execute
        response = self.client.post(
            "/upload-historical-context",
            files={"zip_file": ("context.zip", zip_file.file, "application/zip")},
            data={"force_reprocess": "false"}
        )
        
        # Verify
        assert response.status_code == 500

    def test_active_jobs_global_state(self):
        """Test that active_jobs dictionary maintains state across calls"""
        # Setup
        test_job_id = "test-global-state"
        active_jobs[test_job_id] = {"status": "test"}
        
        # Verify
        assert test_job_id in active_jobs
        assert active_jobs[test_job_id]["status"] == "test"
        
        # Modify
        active_jobs[test_job_id]["status"] = "modified"
        assert active_jobs[test_job_id]["status"] == "modified"

    def test_logging_configuration(self):
        """Test that logging is properly configured"""
        # The logger should be configured at module level
        # We can test by importing and checking if it's available
        from src.interface_layer.main_service_router import logger
        assert logger is not None
        # Just verify logger exists and is callable, don't check the name since it's mocked
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')

    def test_temp_dir_creation(self):
        """Test that TEMP_DIR is properly configured"""
        from src.interface_layer.main_service_router import TEMP_DIR
        assert TEMP_DIR is not None
        assert "story_sense_temp" in TEMP_DIR

    @patch('src.interface_layer.main_service_router.StorySenseGenerator')
    @patch('src.interface_layer.main_service_router.logger')
    @patch('os.listdir')
    @pytest.mark.asyncio
    async def test_process_stories_no_context_no_process_context(self, mock_listdir, mock_logger, mock_ssg_class):
        """Test story processing without context and without process_context flag"""
        # Setup
        job_id = "test-job-no-context"
        user_stories_path = "/path/to/stories.xlsx"
        
        # Mock StorySenseGenerator
        mock_ssg = Mock()
        mock_ssg.output_file_path = "/output/path"
        mock_ssg_class.return_value = mock_ssg
        
        # Mock directory listing
        mock_listdir.return_value = ["result1.xlsx"]
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute
        await process_stories(
            job_id=job_id,
            user_stories_path=user_stories_path,
            context_path=None,
            batch_size=5,
            parallel=False,
            process_context=False,
            force_reprocess=False
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "completed"
        mock_ssg_class.assert_called_once_with(user_stories_path, None)
        mock_ssg.process_context_library.assert_not_called()  # Should not be called when process_context=False
        mock_ssg.process_user_stories.assert_called_once_with(batch_size=5, parallel=False)

    @patch('src.interface_layer.main_service_router.StorySenseGenerator')
    @patch('src.interface_layer.main_service_router.logger')
    @patch('os.listdir')
    @pytest.mark.asyncio
    async def test_process_stories_with_image_support_context_path_branch(self, mock_listdir, mock_logger, mock_ssg_class):
        """Test process_stories_with_image_support when context_path is None but process_context is True"""
        # Setup
        job_id = "test-context-path-branch"
        user_stories_path = "/path/to/stories.xlsx"
        
        # Mock StorySenseGenerator
        mock_ssg = Mock()
        mock_ssg.output_file_path = "/output/path"
        mock_ssg_class.return_value = mock_ssg
        
        # Mock directory listing
        mock_listdir.return_value = ["result1.xlsx", "result2.pdf"]
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute - context_path is None, but process_context=True (else branch)
        await process_stories_with_image_support(
            job_id=job_id,
            user_stories_path=user_stories_path,
            context_path=None,
            context_is_image=False,
            batch_size=3,
            parallel=True,
            process_context=True,
            force_reprocess=True
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "completed"
        mock_ssg_class.assert_called_once_with(user_stories_path, None)
        mock_ssg.process_context_library.assert_called_once_with(force_reprocess=True)
        mock_ssg.process_user_stories.assert_called_once_with(batch_size=3, parallel=True)

    @patch('src.interface_layer.main_service_router.process_stories')
    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    @patch('uuid.uuid4')
    def test_upload_and_process_with_context_file_extension_conditions(self, mock_uuid, mock_copyfileobj, mock_makedirs, mock_process):
        """Test upload endpoint with different file extensions to cover branch conditions"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-extension-job")
        
        # Create mock files with different extensions
        user_stories_file = self.create_mock_upload_file("stories.xlsx")
        excel_context = self.create_mock_upload_file("context.xlsx", b"excel data")
        
        # Mock parameters
        params = json.dumps({
            "batch_size": 5,
            "parallel": False,
            "process_context": True,
            "force_reprocess": False
        })
        
        # Execute - using only non-image context to avoid directory issues
        response = self.client.post(
            "/process-stories",
            files=[
                ("user_stories_file", ("stories.xlsx", user_stories_file.file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
                ("context_files", ("context.xlsx", excel_context.file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
            ],
            data={"params": params}
        )
        
        # Verify
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == "test-extension-job"
        assert "test-extension-job" in active_jobs

    @patch('src.interface_layer.main_service_router.process_historical_context')
    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    @patch('zipfile.ZipFile')
    @patch('uuid.uuid4')
    def test_upload_historical_context_with_nested_path(self, mock_uuid, mock_zipfile, mock_copyfileobj, mock_makedirs, mock_process):
        """Test historical context upload with nested paths that are safe"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-nested-job")
        
        # Mock zipfile with nested but safe paths
        mock_zip = Mock()
        mock_zip.namelist.return_value = ["folder/subfolder/file.pdf", "another/file.docx"]
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        # Create mock zip file
        zip_file = self.create_mock_upload_file("nested.zip", b"fake zip data")
        
        # Execute
        response = self.client.post(
            "/upload-historical-context",
            files={"zip_file": ("nested.zip", zip_file.file, "application/zip")},
            data={"force_reprocess": "true"}
        )
        
        # Verify
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == "test-nested-job"
        assert "test-nested-job" in active_jobs

    @patch('starlette.responses.FileResponse')
    @patch('os.listdir')
    def test_download_results_with_different_file_types(self, mock_listdir, mock_file_response):
        """Test download results with various file types covered by the filter"""
        # Setup
        job_id = "test-file-types"
        
        # Mock file paths with different extensions
        test_files = [
            "/path/to/file0.xlsx",
            "/path/to/file1.csv", 
            "/path/to/file2.pdf",
            "/path/to/file3.txt",
            "/path/to/file4.html"
        ]
        
        active_jobs[job_id] = {
            "status": "completed",
            "output_files": test_files
        }
        
        # Mock listdir to return only files with expected extensions
        mock_listdir.return_value = [f.split('/')[-1] for f in test_files]
        
        # Mock FileResponse
        mock_response = Mock()
        mock_file_response.return_value = mock_response
        
        # We need to patch the zipfile creation since we have multiple files
        with patch('zipfile.ZipFile') as mock_zipfile, \
             patch('os.path.basename') as mock_basename:
            
            mock_zip = Mock()
            mock_zipfile.return_value.__enter__.return_value = mock_zip
            def basename_side_effect(path):
                return path.split('\\')[-1] if '\\' in path else path.split('/')[-1]
            mock_basename.side_effect = basename_side_effect
            
            # Execute
            response = self.client.get(f"/download-results/{job_id}")
            
            # Verify all file types were included
            assert mock_zip.write.call_count == 5

    def test_active_jobs_status_transitions(self):
        """Test various status transitions in active_jobs"""
        # Test all possible status values
        job_id = "test-status-transitions"
        
        # Initial state
        active_jobs[job_id] = {"status": "submitted"}
        assert active_jobs[job_id]["status"] == "submitted"
        
        # Processing state
        active_jobs[job_id]["status"] = "processing"
        assert active_jobs[job_id]["status"] == "processing"
        
        # Completed state
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["output_files"] = ["file1.xlsx", "file2.html"]
        assert active_jobs[job_id]["status"] == "completed"
        assert len(active_jobs[job_id]["output_files"]) == 2
        
        # Failed state
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = "Test error message"
        assert active_jobs[job_id]["status"] == "failed"
        assert "Test error message" in active_jobs[job_id]["error"]

    @patch('src.interface_layer.main_service_router.process_stories')
    @patch('os.makedirs')
    @patch('shutil.copyfileobj')
    @patch('uuid.uuid4')
    def test_upload_and_process_edge_case_empty_params(self, mock_uuid, mock_copyfileobj, mock_makedirs, mock_process):
        """Test upload endpoint with empty parameters to use defaults"""
        # Setup
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-empty-params")
        
        # Create mock file
        user_stories_file = self.create_mock_upload_file("stories.xlsx")
        
        # Execute without params to trigger default values
        response = self.client.post(
            "/process-stories",
            files={"user_stories_file": ("stories.xlsx", user_stories_file.file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )
        
        # Verify
        assert response.status_code == 200
        result = response.json()
        assert result["job_id"] == "test-empty-params"
        assert "test-empty-params" in active_jobs
        # Verify default values were used - context_files may be None or empty list
        job_data = active_jobs["test-empty-params"]
        context_files = job_data.get("context_files", [])
        if context_files is None:
            context_files = []
        assert len(context_files) == 0

    @patch('src.interface_layer.main_service_router.EnhancedContextProcessor')
    @patch('src.interface_layer.main_service_router.logger')
    @patch('os.makedirs')
    @patch('os.walk')
    @patch('os.path.exists')
    @patch('shutil.copy')
    @pytest.mark.asyncio
    async def test_process_historical_context_no_images_found(self, mock_copy, mock_exists, mock_walk, mock_makedirs, mock_logger, mock_processor_class):
        """Test historical context processing when no image files are found"""
        # Setup
        job_id = "test-no-images"
        zip_path = "/path/to/context.zip"
        extract_dir = "/path/to/extracted"
        
        # Mock EnhancedContextProcessor
        mock_processor = Mock()
        mock_processor.process_all_context_files.return_value = {"processed_files": 2}
        mock_processor_class.return_value = mock_processor
        
        # Mock os.walk to find NO image files
        mock_walk.return_value = [
            (extract_dir, [], ["document.pdf", "text.txt", "data.csv"])
        ]
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute
        await process_historical_context(
            job_id=job_id,
            zip_path=zip_path,
            extract_dir=extract_dir
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "completed"
        # Verify that no copy operations were performed since no images were found
        mock_copy.assert_not_called()

    def test_router_path_constants(self):
        """Test that path constants are properly defined"""
        from src.interface_layer.main_service_router import TEMP_DIR
        assert TEMP_DIR is not None
        assert isinstance(TEMP_DIR, str)
        assert "story_sense_temp" in TEMP_DIR

    @patch('src.interface_layer.main_service_router.StorySenseGenerator')
    @patch('src.interface_layer.main_service_router.logger')
    @patch('os.listdir')
    @pytest.mark.asyncio
    async def test_process_stories_listdir_failure(self, mock_listdir, mock_logger, mock_ssg_class):
        """Test process_stories when os.listdir fails"""
        # Setup
        job_id = "test-listdir-fail"
        user_stories_path = "/path/to/stories.xlsx"
        
        # Mock StorySenseGenerator
        mock_ssg = Mock()
        mock_ssg.output_file_path = "/output/path"
        mock_ssg_class.return_value = mock_ssg
        
        # Mock os.listdir to raise exception
        mock_listdir.side_effect = OSError("Directory not found")
        
        # Initialize active_jobs entry
        active_jobs[job_id] = {"status": "submitted"}
        
        # Execute
        await process_stories(
            job_id=job_id,
            user_stories_path=user_stories_path,
            context_path=None,
            batch_size=5,
            parallel=False,
            process_context=False,
            force_reprocess=False
        )
        
        # Verify
        assert active_jobs[job_id]["status"] == "failed"
        assert "Directory not found" in active_jobs[job_id]["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.interface_layer.main_service_router", "--cov-report=term-missing"])
