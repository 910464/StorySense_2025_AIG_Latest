import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class DocumentRegistry:
    """Handles loading, saving and querying the document registry."""

    def __init__(self, registry_dir: str):
        os.makedirs(registry_dir, exist_ok=True)
        self.registry_file = os.path.join(registry_dir, 'document_registry.json')
        self.document_registry = self._load_document_registry()

    def _load_document_registry(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"Could not load document registry: {e}. Creating new registry.")
                return {}
        return {}

    def _save_document_registry(self):
        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_registry, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Could not save document registry: {e}")

    def _calculate_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logging.error(f"Could not calculate hash for {file_path}: {e}")
            return ""

    def is_document_processed(self, file_path: str) -> Dict[str, Any]:
        """Check if a document has already been processed."""
        file_hash = self._calculate_file_hash(file_path)
        file_name = Path(file_path).name
        try:
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)
        except Exception:
            file_size = None
            file_mtime = None

        # Check by hash first (most reliable)
        for doc_id, doc_info in self.document_registry.items():
            if doc_info.get('file_hash') == file_hash and file_hash:
                return {
                    'exists': True,
                    'reason': 'identical_content',
                    'existing_doc_id': doc_id,
                    'existing_info': doc_info
                }

        # Check by name, size, and modification time
        for doc_id, doc_info in self.document_registry.items():
            if (doc_info.get('file_name') == file_name and
                    doc_info.get('file_size') == file_size and
                    doc_info.get('file_mtime') == file_mtime):
                return {
                    'exists': True,
                    'reason': 'same_file_attributes',
                    'existing_doc_id': doc_id,
                    'existing_info': doc_info
                }

        return {'exists': False}

    def register_document(self, file_path: str, processing_result: Dict[str, Any]) -> str:
        """Register a processed document in the registry and persist it."""
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.document_registry)}"

        try:
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)
        except Exception:
            file_size = None
            file_mtime = None

        self.document_registry[doc_id] = {
            'file_name': Path(file_path).name,
            'file_path': file_path,
            'file_hash': self._calculate_file_hash(file_path),
            'file_size': file_size,
            'file_mtime': file_mtime,
            'file_type': Path(file_path).suffix.lower(),
            'processed_date': datetime.now().isoformat(),
            'document_count': processing_result.get('document_count', 0),
            'collections': processing_result.get('collections', []),
            'context_type': processing_result.get('context_type', 'unknown'),
            'processing_time': processing_result.get('processing_time', 0),
            'success': processing_result.get('success', False)
        }

        self._save_document_registry()
        return doc_id

    def get_status_summary(self) -> Dict[str, Any]:
        status = {
            'registry_exists': os.path.exists(self.registry_file),
            'total_registered_documents': len(self.document_registry),
            'last_update': None
        }

        if self.document_registry:
            dates = [doc.get('processed_date') for doc in self.document_registry.values() if doc.get('processed_date')]
            if dates:
                status['last_update'] = max(dates)

        return status

    def get_all_documents(self) -> Dict[str, Any]:
        return self.document_registry
