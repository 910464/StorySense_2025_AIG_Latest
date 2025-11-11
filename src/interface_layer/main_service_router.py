from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import pandas as pd
import uuid
import shutil
from pathlib import Path
import logging
import tempfile
from src.interface_layer.StorySenseGenerator import StorySenseGenerator
# from  US_to_StorySense.StorySense.StorySenseGenerator import StorySenseGenerator
from src.context_handler.context_storage_handler.run_context_processor import EnhancedContextProcessor
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create router
story_sense_router = APIRouter(tags=["StorySense Generator"])

# Create necessary directories
os.makedirs('../Config', exist_ok=True)
os.makedirs('../Input', exist_ok=True)
os.makedirs('../Output/StorySense', exist_ok=True)
os.makedirs('../Output/RetrievalContext', exist_ok=True)
os.makedirs('../Data/SavedContexts', exist_ok=True)

# Temporary storage for uploaded files and generated results
TEMP_DIR = os.path.join(tempfile.gettempdir(), "story_sense_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Dictionary to track active jobs
active_jobs = {}


class ProcessRequest(BaseModel):
    batch_size: Optional[int] = 5
    parallel: Optional[bool] = False
    process_context: Optional[bool] = False
    force_reprocess: Optional[bool] = False


async def process_stories(job_id: str, user_stories_path: str, context_path: str = None,
                          batch_size: int = 5, parallel: bool = False,
                          process_context: bool = False, force_reprocess: bool = False):
    """Background task to process user stories"""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "processing"

        # Initialize StorySenseGenerator
        ssg = StorySenseGenerator(user_stories_path, context_path)

        # Process context if requested
        if process_context:
            ssg.process_context_library(force_reprocess=force_reprocess)

        # Process user stories
        ssg.process_user_stories(batch_size=batch_size, parallel=parallel)

        # Find the output files
        output_dir = ssg.output_file_path
        output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                        if f.endswith(('.xlsx', '.csv', '.pdf', '.txt','html'))]

        # Update job with results
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["output_files"] = output_files

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)

# New Background task to process image as additional context:
async def process_stories_with_image_support(
        job_id: str,
        user_stories_path: str,
        context_path: str = None,
        context_is_image: bool = False,
        batch_size: int = 5,
        parallel: bool = False,
        process_context: bool = False,
        force_reprocess: bool = False
):
    """Background task to process user stories with support for image context"""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "processing"

        # Step 1: If we have context files (including images), process them
        if context_path:
            # Initialize StorySenseGenerator
            ssg = StorySenseGenerator(user_stories_path)

            # Process the context library (context_path is already a directory with files)
            logger.info(f"Processing context files from directory: {context_path}")
            ssg.process_context_library(context_path, force_reprocess=force_reprocess)

            # Now process user stories (without passing context_path since we've already processed it)
            ssg.process_user_stories(batch_size=batch_size, parallel=parallel)
        else:
            # Standard processing for non-image context
            ssg = StorySenseGenerator(user_stories_path, context_path)

            # Process context if requested
            if process_context:
                ssg.process_context_library(force_reprocess=force_reprocess)

            # Process user stories
            ssg.process_user_stories(batch_size=batch_size, parallel=parallel)

        # Find the output files
        output_dir = ssg.output_file_path
        output_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)
                        if f.endswith(('.xlsx', '.csv', '.pdf', '.txt', 'html'))]

        # Update job with results
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["output_files"] = output_files

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)


# New: background task to process uploaded historical context ZIP
async def process_historical_context(job_id: str, zip_path: str, extract_dir: str, force_reprocess: bool = False):
    """Background task to unzip and process historical context using EnhancedContextProcessor"""
    try:
        active_jobs[job_id]["status"] = "processing"

        # Ensure extraction directory exists
        os.makedirs(extract_dir, exist_ok=True)

        # Safe extraction already performed at upload time; instantiate processor
        processor = EnhancedContextProcessor(extract_dir)

        # Additional logging for image files
        image_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
                    image_files.append(os.path.join(root, file))

        if image_files:
            logger.info(
                f"Found {len(image_files)} image files in context: {[os.path.basename(f) for f in image_files]}")

            # Create wireframes directory if it doesn't exist
            wireframes_dir = os.path.join(extract_dir, 'wireframes')
            os.makedirs(wireframes_dir, exist_ok=True)

            # If images aren't already in a proper subfolder, move them to wireframes
            for img_path in image_files:
                if os.path.dirname(img_path) == extract_dir:  # Image is in root dir
                    target_path = os.path.join(wireframes_dir, os.path.basename(img_path))
                    if not os.path.exists(target_path):
                        shutil.copy(img_path, target_path)
                        logger.info(f"Moved image to wireframes folder: {os.path.basename(img_path)}")

        # Process all context files (synchronous call)
        stats = processor.process_all_context_files(force_reprocess=force_reprocess)

        # Update job with results
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["results"] = stats

        logger.info(f"Historical context job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Error processing historical context job {job_id}: {str(e)}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)


@story_sense_router.post("/process-stories", response_model=dict)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    user_stories_file: UploadFile = File(...),
    # Support multiple context files (images, pdf, docx, csv, xlsx, zip, etc.)
    context_files: Optional[List[UploadFile]] = File(None),
    params: str = Form("{}")
):
    """
    Endpoint to upload user stories and optional context files,
    and process them using StorySenseGenerator
    """
    try:
        import json
        # Parse parameters
        process_params = json.loads(params)
        request_params = ProcessRequest(**process_params)

        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Create job directory
        job_dir = os.path.join(TEMP_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # Save uploaded user stories file
        user_stories_path = os.path.join(job_dir, user_stories_file.filename)
        with open(user_stories_path, "wb") as f:
            shutil.copyfileobj(user_stories_file.file, f)

        # Save context files if provided (support multiple files)
        context_path = None
        context_is_image = False
        if context_files:
            temp_context_dir = os.path.join(job_dir, "temp_context_library")
            wireframes_dir = os.path.join(temp_context_dir, "wireframes")
            os.makedirs(temp_context_dir, exist_ok=True)
            os.makedirs(wireframes_dir, exist_ok=True)

            saved_context_filenames = []
            image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']

            for cfile in context_files:
                target_path = os.path.join(temp_context_dir, cfile.filename)
                with open(target_path, "wb") as f:
                    shutil.copyfileobj(cfile.file, f)
                saved_context_filenames.append(cfile.filename)

                # If it's an image, also copy to wireframes folder for the context processor
                if any(cfile.filename.lower().endswith(ext) for ext in image_extensions):
                    context_is_image = True
                    wf_target = os.path.join(wireframes_dir, cfile.filename)
                    # copy file again (or move)
                    shutil.copy(target_path, wf_target)
                    logger.info(f"Saved image context file to wireframes: {cfile.filename}")

            context_path = temp_context_dir
            logger.info(f"Saved {len(saved_context_filenames)} context file(s) for job {job_id}: {saved_context_filenames}")

        # Create job entry
        active_jobs[job_id] = {
            "id": job_id,
            "status": "submitted",
            "user_stories_file": user_stories_file.filename,
            "context_files": [f.filename for f in context_files] if context_files else None,
            "params": request_params.dict(),
            "output_files": []
        }

        # Start background processing task
        # Start background processing task. If we detected image context files, use the
        # image-support-aware background task so images are placed correctly in the context library.
        if context_is_image:
            background_tasks.add_task(
                process_stories_with_image_support,
                job_id,
                user_stories_path,
                context_path,
                True if context_is_image else False,
                request_params.batch_size,
                request_params.parallel,
                request_params.process_context,
                request_params.force_reprocess
            )
        else:
            background_tasks.add_task(
                process_stories,
                job_id,
                user_stories_path,
                context_path,
                request_params.batch_size,
                request_params.parallel,
                request_params.process_context,
                request_params.force_reprocess
            )

        return {
            "job_id": job_id,
            "status": "submitted",
            "message": "Files uploaded and processing started"
        }

    except Exception as e:
        logger.error(f"Error initiating processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# New endpoint: Upload Historical Context (ZIP)
@story_sense_router.post("/upload-historical-context", response_model=dict)
async def upload_historical_context(
        background_tasks: BackgroundTasks,
        zip_file: UploadFile = File(...),
        force_reprocess: bool = Form(False)
):
    """Upload a ZIP of historical context files, unzip and process them in background."""
    try:
        # Create job id and directories
        job_id = str(uuid.uuid4())
        job_dir = os.path.join(TEMP_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # Save uploaded zip
        zip_path = os.path.join(job_dir, zip_file.filename)
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(zip_file.file, f)

        # Extract zip safely
        extract_dir = os.path.join(job_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        def _is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                member_path = os.path.join(extract_dir, member)
                if not _is_within_directory(extract_dir, member_path):
                    raise HTTPException(status_code=400, detail="Invalid zip file: path traversal detected")
            zf.extractall(extract_dir)

        # Create job entry
        active_jobs[job_id] = {
            "id": job_id,
            "status": "submitted",
            "zip_file": zip_file.filename,
            "extracted_path": extract_dir,
            "force_reprocess": force_reprocess,
            "results": None
        }

        # Start background processing task
        background_tasks.add_task(process_historical_context, job_id, zip_path, extract_dir, force_reprocess)

        return {"job_id": job_id, "status": "submitted", "message": "Historical context uploaded and processing started"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading historical context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# @story_sense_router.get("/job-status/{job_id}")
# async def check_job_status(job_id: str):
#     """Get the status of a submitted job"""
#     if job_id not in active_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")

#     job = active_jobs[job_id]

#     return {
#         "job_id": job_id,
#         "status": job["status"],
#         "files_available": len(job.get("output_files", [])) > 0,
#         "error": job.get("error", None)
#     }


@story_sense_router.get("/download-results/{job_id}")
async def download_results(job_id: str):
    """Download the results of a completed job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = active_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400,
                            detail=f"Results not ready. Current status: {job['status']}")

    if not job.get("output_files"):
        raise HTTPException(status_code=404, detail="No output files available")

    # If there are multiple files, create a zip archive
    if len(job["output_files"]) > 1:
        import zipfile
        zip_path = os.path.join(TEMP_DIR, f"{job_id}_results.zip")

        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for file_path in job["output_files"]:
                # Add file to zip with just the filename, not the full path
                zip_file.write(file_path, os.path.basename(file_path))

        return FileResponse(
            path=zip_path,
            filename="story_sense_results.zip",
            media_type="application/zip"
        )
    else:
        # Return single file
        file_path = job["output_files"][0]
        return FileResponse(
            path=file_path,
            filename=os.path.basename(file_path),
            media_type="application/octet-stream"
        )


# @story_sense_router.get("/list-jobs")
# async def list_all_jobs():
#     """List all jobs in the system"""
#     return {
#         "jobs": [
#             {
#                 "job_id": job_id,
#                 "status": job_info["status"],
#                 "user_stories_file": job_info["user_stories_file"],
#                 "context_file": job_info["context_file"],
#                 "submitted_at": job_info.get("submitted_at", "Unknown")
#             }
#             for job_id, job_info in active_jobs.items()
#         ]
#     }
#
#
# @story_sense_router.delete("/cleanup/{job_id}")
# async def cleanup_job(job_id: str):
#     """Clean up job files and remove from tracking"""
#     if job_id not in active_jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
#
#     # Remove job directory
#     job_dir = os.path.join(TEMP_DIR, job_id)
#     if os.path.exists(job_dir):
#         shutil.rmtree(job_dir)
#
#     # Remove potential zip file
#     zip_path = os.path.join(TEMP_DIR, f"{job_id}_results.zip")
#     if os.path.exists(zip_path):
#         os.remove(zip_path)
#
#     # Remove from tracking
#     job_info = active_jobs.pop(job_id)
#
#     return {"message": f"Job {job_id} cleaned up successfully"}