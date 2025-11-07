import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.context_handler.context_storage_handler.pgvector_connector import PGVectorConnector
from src.metrics.metrics_manager import MetricsManager
from colorama import Fore, Style

# New modular components
from src.context_handler.context_file_handler.document_registry import DocumentRegistry
from src.context_handler.context_file_handler.document_processor import DocumentProcessor
from src.context_handler.context_file_handler.context_type_detector import ContextTypeDetector


class ContextManager:
    """Enhanced context management system for StorySense with smart document handling

    This class orchestrates scanning a context library, delegating parsing to DocumentProcessor,
    registry handling to DocumentRegistry, and context-type heuristics to ContextTypeDetector.
    """

    def __init__(self, metrics_manager=None):
        self.metrics_manager = metrics_manager or MetricsManager()
        self.context_storage_path = '../Data/ContextLibrary'
        self.processed_contexts_path = '../Data/ProcessedContexts'
        self.document_registry_path = '../Data/DocumentRegistry'

        # Create directories
        os.makedirs(self.context_storage_path, exist_ok=True)
        os.makedirs(self.processed_contexts_path, exist_ok=True)
        os.makedirs(self.document_registry_path, exist_ok=True)

        # Initialize document registry, processor, and detector
        self.document_registry = DocumentRegistry(self.document_registry_path)
        self.document_processor = DocumentProcessor()
        self.context_detector = ContextTypeDetector()

        # Initialize PGVector connectors for different context types
        self.context_collections = {
            'business_rules': PGVectorConnector(collection_name="business_rules", metrics_manager=self.metrics_manager),
            'requirements': PGVectorConnector(collection_name="requirements", metrics_manager=self.metrics_manager),
            'documentation': PGVectorConnector(collection_name="documentation", metrics_manager=self.metrics_manager),
            'policies': PGVectorConnector(collection_name="policies", metrics_manager=self.metrics_manager),
            'examples': PGVectorConnector(collection_name="examples", metrics_manager=self.metrics_manager),
            'glossary': PGVectorConnector(collection_name="glossary", metrics_manager=self.metrics_manager)
        }

    def check_and_process_context_library(self, context_directory: str = None) -> Dict[str, Any]:
        """
        Smart context library processing:
        1. Check if context directory exists
        2. If no files, continue without context
        3. If files exist, check for duplicates and process new ones
        4. Return status for the main application
        """
        if context_directory is None:
            context_directory = './Input/ContextLibrary'

        print(Fore.CYAN + f"\n🔍 Checking Context Library: {context_directory}" + Style.RESET_ALL)

        # Check if context directory exists
        if not os.path.exists(context_directory):
            print(Fore.YELLOW + f"📁 Context directory not found: {context_directory}" + Style.RESET_ALL)
            print(Fore.CYAN + "💡 Creating directory structure for future use..." + Style.RESET_ALL)
            self._create_context_directory_structure(context_directory)
            return {
                'status': 'no_context_directory',
                'message': 'Context directory created. Add files and run again to use enhanced context.',
                'has_context': False,
                'processed_files': 0,
                'skipped_files': 0,
                'new_files': 0
            }

        # Check if directory has any supported files
        supported_files = self._find_supported_files(context_directory)

        if not supported_files:
            print(Fore.YELLOW + f"📂 No supported files found in context directory" + Style.RESET_ALL)
            print(Fore.CYAN + f"💡 Supported formats: {', '.join(self.document_processor.file_processors.keys())}" + Style.RESET_ALL)
            print(Fore.CYAN + "📝 Add your context files and run again to use enhanced context." + Style.RESET_ALL)
            return {
                'status': 'no_context_files',
                'message': 'No supported context files found. Running without enhanced context.',
                'has_context': False,
                'processed_files': 0,
                'skipped_files': 0,
                'new_files': 0
            }

        # Process files (checking for duplicates)
        print(Fore.GREEN + f"📄 Found {len(supported_files)} supported files" + Style.RESET_ALL)
        results = self._process_files_with_duplicate_check(supported_files)

        # Display results
        if results['new_files'] > 0:
            print(Fore.GREEN + f"✅ Processed {results['new_files']} new files" + Style.RESET_ALL)

        if results['skipped_files'] > 0:
            print(Fore.YELLOW + f"⏭️  Skipped {results['skipped_files']} existing files" + Style.RESET_ALL)

        if results['failed_files'] > 0:
            print(Fore.RED + f"❌ Failed to process {results['failed_files']} files" + Style.RESET_ALL)

        print(Fore.CYAN + f"📊 Total documents in context library: {results['total_documents']}" + Style.RESET_ALL)

        return {
            'status': 'context_processed',
            'message': f"Context library processed. {results['new_files']} new files, {results['skipped_files']} existing files.",
            'has_context': results['total_documents'] > 0,
            'processed_files': results['new_files'],
            'skipped_files': results['skipped_files'],
            'new_files': results['new_files'],
            'total_documents': results['total_documents'],
            'collections_updated': results['collections_updated']
        }

    def _find_supported_files(self, directory: str) -> List[str]:
        """Find all supported files in the directory"""
        supported_files = []
        supported_exts = set(self.document_processor.file_processors.keys())

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file_path).suffix.lower()

                if file_ext in supported_exts:
                    supported_files.append(file_path)

        return supported_files

    def _process_files_with_duplicate_check(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process files while checking for duplicates"""
        results = {
            'new_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_documents': 0,
            'collections_updated': set(),
            'processing_details': [],
            'skipped_details': []
        }

        for file_path in file_paths:
            try:
                # Check if file already processed
                duplicate_check = self.document_registry.is_document_processed(file_path)

                if duplicate_check['exists']:
                    results['skipped_files'] += 1
                    results['skipped_details'].append({
                        'file': Path(file_path).name,
                        'reason': duplicate_check['reason'],
                        'existing_doc_id': duplicate_check.get('existing_doc_id')
                    })

                    print(
                        Fore.YELLOW + f"⏭️  Skipping {Path(file_path).name} - {duplicate_check['reason']}" + Style.RESET_ALL)
                    continue

                # Process new file
                print(Fore.CYAN + f"🔄 Processing {Path(file_path).name}..." + Style.RESET_ALL)
                file_result = self._process_single_file(file_path)

                if file_result['success']:
                    results['new_files'] += 1
                    results['total_documents'] += file_result['document_count']
                    results['collections_updated'].update(file_result['collections'])

                    # Register the document
                    doc_id = self.document_registry.register_document(file_path, file_result)
                    file_result['doc_id'] = doc_id

                    print(
                        Fore.GREEN + f"✅ Successfully processed {Path(file_path).name} ({file_result['document_count']} documents)" + Style.RESET_ALL)
                else:
                    results['failed_files'] += 1
                    print(
                        Fore.RED + f"❌ Failed to process {Path(file_path).name}: {file_result.get('error')}" + Style.RESET_ALL)

                results['processing_details'].append(file_result)

            except Exception as e:
                results['failed_files'] += 1
                error_msg = f"Unexpected error processing {file_path}: {str(e)}"
                logging.error(error_msg)
                print(Fore.RED + f"❌ {error_msg}" + Style.RESET_ALL)

        # Convert set to list for JSON serialization
        results['collections_updated'] = list(results['collections_updated'])

        return results

    def _process_single_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file and extract context"""
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name

        result = {
            'file_name': file_name,
            'file_path': file_path,
            'file_type': file_ext,
            'success': False,
            'document_count': 0,
            'collections': [],
            'context_type': None,
            'error': None,
            'processing_time': 0
        }

        start_time = datetime.now()

        try:
            # Use DocumentProcessor to parse the file
            documents = self.document_processor.process(file_path)

            if not documents:
                result['error'] = "No content extracted from file"
                return result

            # Determine context type and store documents
            context_type = self.context_detector.determine(file_path, documents)
            result['context_type'] = context_type

            # Store in appropriate collection
            if context_type in self.context_collections:
                connector = self.context_collections[context_type]
                success = connector.vector_store_documents(documents)

                if success:
                    result['success'] = True
                    result['document_count'] = len(documents)
                    result['collections'] = [context_type]
                else:
                    result['error'] = "Failed to store documents in vector database"
            else:
                result['error'] = f"Unknown context type: {context_type}"

        except Exception as e:
            result['error'] = str(e)
            logging.error(f"Error processing {file_path}: {e}")

        finally:
            result['processing_time'] = (datetime.now() - start_time).total_seconds()

        return result

    def _create_context_directory_structure(self, base_dir: str):
        """Create context directory structure with examples and README files"""
        structure = {
            'business_rules': {
                'description': 'Business logic, rules, constraints, and validation requirements',
                'examples': [
                    'payment_processing_rules.pdf',
                    'user_validation_rules.docx',
                    'business_constraints.txt'
                ]
            },
            'requirements': {
                'description': 'Functional and non-functional requirements, specifications',
                'examples': [
                    'functional_requirements.docx',
                    'system_requirements.pdf',
                    'feature_specifications.xlsx'
                ]
            },
            'documentation': {
                'description': 'Technical documentation, user guides, API docs',
                'examples': [
                    'api_documentation.md',
                    'user_guide.pdf',
                    'technical_architecture.pptx'
                ]
            },
            'policies': {
                'description': 'Company policies, procedures, guidelines, standards',
                'examples': [
                    'data_privacy_policy.pdf',
                    'security_guidelines.docx',
                    'compliance_procedures.txt'
                ]
            },
            'examples': {
                'description': 'Example user stories, templates, samples, best practices',
                'examples': [
                    'good_user_stories.xlsx',
                    'story_templates.md',
                    'acceptance_criteria_examples.csv'
                ]
            },
            'glossary': {
                'description': 'Terms, definitions, acronyms, domain vocabulary',
                'examples': [
                    'business_terms.csv',
                    'technical_glossary.json',
                    'acronyms_definitions.txt'
                ]
            }
        }

        # Create main README
        main_readme = os.path.join(base_dir, 'README.md')
        os.makedirs(base_dir, exist_ok=True)

        with open(main_readme, 'w', encoding='utf-8') as f:
            f.write("""# StorySense Context Library

This directory contains context files that will be used to enhance user story analysis.

## How It Works

1. **Add Files**: Place your context files in the appropriate subdirectories
2. **Automatic Processing**: The system will automatically detect and process new files
3. **Duplicate Detection**: Files are checked for duplicates to avoid reprocessing
4. **Enhanced Analysis**: Context is automatically used during user story analysis

## Supported File Types

- **PDF** (.pdf) - Documents, reports, specifications
- **Word** (.docx, .doc) - Requirements, policies, documentation
- **PowerPoint** (.pptx) - Presentations, architecture diagrams
- **Excel** (.xlsx, .xls) - Data, examples, templates
- **Text** (.txt, .md) - Plain text, markdown documentation
- **JSON** (.json) - Structured data, configurations
- **CSV** (.csv) - Tabular data, glossaries, examples

## Directory Structure

Each subdirectory serves a specific purpose in providing context for user story analysis:
""")

        for context_type, info in structure.items():
            type_dir = os.path.join(base_dir, context_type)
            os.makedirs(type_dir, exist_ok=True)

            # Add to main README
            f.write(f"\n### {context_type.replace('_', ' ').title()}\n")
            f.write(f"**Purpose**: {info['description']}\n\n")
            f.write("**Example files you might add**:\n")
            for example in info['examples']:
                f.write(f"- {example}\n")
            f.write("\n")

            # Create subdirectory README
            readme_path = os.path.join(type_dir, 'README.md')
            with open(readme_path, 'w', encoding='utf-8') as sub_f:
                sub_f.write(f"""# {context_type.replace('_', ' ').title()} Context

**Purpose**: {info['description']}

## What to Put Here

Add files that contain {context_type.replace('_', ' ')} information that would be helpful when analyzing user stories.

## Example Files

{chr(10).join(f'- **{file}** - {self._get_example_description(file)}' for file in info['examples'])}

## Supported Formats

- PDF (.pdf)
- Word Documents (.docx, .doc)
- PowerPoint (.pptx)
- Excel (.xlsx, .xls)
- Text Files (.txt, .md)
- JSON (.json)
- CSV (.csv)

## Processing

When you run StorySense:
1. New files will be automatically detected and processed
2. Existing files will be skipped (no duplicate processing)
3. Content will be made available as context for user story analysis
4. You'll see a summary of what was processed

## Tips

- Use descriptive file names
- Keep files focused on specific topics
- Update files as needed - the system will detect changes
- Remove outdated files to keep context relevant
""")

        print(Fore.GREEN + f"📁 Created context directory structure at: {base_dir}" + Style.RESET_ALL)
        print(
            Fore.CYAN + "📖 Check the README.md files for detailed instructions on what to put in each directory." + Style.RESET_ALL)

    def _get_example_description(self, filename: str) -> str:
        """Get description for example files"""
        descriptions = {
            'payment_processing_rules.pdf': 'Rules for handling payments, refunds, and transactions',
            'user_validation_rules.docx': 'User input validation and authentication rules',
            'business_constraints.txt': 'Business logic constraints and limitations',
            'functional_requirements.docx': 'Detailed functional requirements document',
            'system_requirements.pdf': 'System and technical requirements',
            'feature_specifications.xlsx': 'Detailed feature specifications and acceptance criteria',
            'api_documentation.md': 'API endpoints, parameters, and usage examples',
            'user_guide.pdf': 'End-user documentation and guides',
            'technical_architecture.pptx': 'System architecture and design documents',
            'data_privacy_policy.pdf': 'Data handling and privacy policies',
            'security_guidelines.docx': 'Security requirements and guidelines',
            'compliance_procedures.txt': 'Regulatory compliance procedures',
            'good_user_stories.xlsx': 'Examples of well-written user stories',
            'story_templates.md': 'Templates and formats for user stories',
            'acceptance_criteria_examples.csv': 'Examples of good acceptance criteria',
            'business_terms.csv': 'Business terminology and definitions',
            'technical_glossary.json': 'Technical terms and their meanings',
            'acronyms_definitions.txt': 'List of acronyms and abbreviations'
        }
        return descriptions.get(filename, 'Context file for user story analysis')

    def get_context_status(self) -> Dict[str, Any]:
        """Get current status of the context library"""
        status = self.document_registry.get_status_summary()

        # Get collection statistics
        all_docs = self.document_registry.get_all_documents()
        status['collections'] = {}

        for context_type in self.context_collections.keys():
            docs_in_collection = sum(1 for doc in all_docs.values()
                                     if context_type in doc.get('collections', []))
            status['collections'][context_type] = docs_in_collection

        return status

    def search_context(self, query, context_types=None, k=5):
        """Search for relevant context across specified collections"""
        if context_types is None:
            context_types = list(self.context_collections.keys())

        results = {}
        file_type_stats = {}  # Track file types

        for context_type in context_types:
            if context_type in self.context_collections:
                connector = self.context_collections[context_type]
                try:
                    context, docs_with_score, docs_metadata, threshold = connector.retrieval_context(query, k)

                    # Extract file types from metadata
                    file_types = {}
                    for similarity, metadata in docs_metadata.items():
                        file_type = metadata.get('file_type', 'unknown')
                        source_file = metadata.get('source_file', 'unknown')
                        file_types[similarity] = {
                            'type': file_type,
                            'source': source_file
                        }

                        # Update overall stats
                        if file_type not in file_type_stats:
                            file_type_stats[file_type] = 0
                        file_type_stats[file_type] += 1

                    results[context_type] = {
                        'context': context,
                        'documents': docs_with_score,
                        'metadata': docs_metadata,
                        'file_types': file_types,
                        'threshold': threshold,
                        'document_count': len(docs_with_score)
                    }
                except Exception as e:
                    logging.error(f"Error searching {context_type}: {e}")
                    results[context_type] = {
                        'context': '',
                        'documents': {},
                        'metadata': {},
                        'file_types': {},
                        'threshold': 0.5,
                        'document_count': 0,
                        'error': str(e)
                    }

        # Add overall file type statistics
        results['file_type_stats'] = file_type_stats

        return results

    def get_file_type_distribution(self, search_results):
        """
        Analyze the distribution of file types in search results

        Args:
            search_results: Results from search_context method

        Returns:
            Dictionary with file type statistics
        """
        if 'file_type_stats' in search_results:
            return search_results['file_type_stats']

        # If not already calculated, compute it
        file_type_stats = {}

        for context_type, result in search_results.items():
            if isinstance(result, dict) and 'file_types' in result:
                for similarity, file_info in result['file_types'].items():
                    file_type = file_info.get('type', 'unknown')
                    if file_type not in file_type_stats:
                        file_type_stats[file_type] = 0
                    file_type_stats[file_type] += 1

        return file_type_stats