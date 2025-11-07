import configparser
import os
import sys
import warnings
import pandas as pd
from botocore import args
from colorama import Fore, Style
import threading
import argparse
from pathlib import Path
from src.aws_layer.aws_titan_embedding import AWSTitanEmbeddings
from src.metrics.metrics_manager import MetricsManager
import threading
from src.html_report.storysense_processor import StorySenseProcessor


# Create necessary directories
os.makedirs('../Config', exist_ok=True)
os.makedirs('../Input', exist_ok=True)
os.makedirs('../Output/StorySense', exist_ok=True)
os.makedirs('../Output/RetrievalContext', exist_ok=True)
os.makedirs('../Data/SavedContexts', exist_ok=True)


class StorySenseGenerator:
    def __init__(self, user_stories_path=None, additional_context_path=None):
        self.config_path = '../Config'
        self.config_parser_io = configparser.ConfigParser()

        # Set AWS as default LLM family
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(os.path.join(self.config_path, 'Config.properties'))
        if not self.config_parser.has_section('LLM'):
            self.config_parser.add_section('LLM')
        if not self.config_parser.has_option('LLM', 'LLM_FAMILY'):
            self.config_parser.set('LLM', 'LLM_FAMILY', 'AWS')
            with open(os.path.join(self.config_path, 'Config.properties'), 'w') as f:
                self.config_parser.write(f)

        # Create default Config.properties if it doesn't exist
        if not os.path.exists(os.path.join(self.config_path, 'Config.properties')):
            self.create_default_config()

        # Create default ConfigIO.properties if it doesn't exist
        config_io_path = os.path.join(self.config_path, 'ConfigIO.properties')
        if not os.path.exists(config_io_path):
            self.create_default_config_io()

        self.config_parser_io.read(config_io_path)

        # Reading values from configuration file
        default_input_path = self.config_parser_io.get('Input', 'input_file_path')
        default_context_path = self.config_parser_io.get('Input', 'additional_context_path')

        # Override with command line arguments if provided
        self.input_file_path = user_stories_path if user_stories_path else default_input_path
        self.additional_context_path = additional_context_path if additional_context_path else default_context_path

        # Convert to absolute paths if they're relative
        if self.input_file_path and not os.path.isabs(self.input_file_path):
            self.input_file_path = os.path.abspath(self.input_file_path)
        if self.additional_context_path and not os.path.isabs(self.additional_context_path):
            self.additional_context_path = os.path.abspath(self.additional_context_path)

        self.input_filename = os.path.basename(self.input_file_path) if self.input_file_path else "No file"
        self.additional_context_name = os.path.basename(
            self.additional_context_path) if self.additional_context_path else "No file"
        self.output_file_path = self.config_parser_io.get('Output', 'output_file_path')

        # Create output directory if it doesn't exist
        os.makedirs(self.output_file_path, exist_ok=True)

    def create_default_config(self):
        """Create a default Config.properties file"""
        config = configparser.ConfigParser()
        config['AdvancedConfigurations'] = {
            'embedding_model_name': 'amazon.titan-embed-text-v1',
            'embedding_model_path': '../Data/ExternalEmbeddingModel',
            'external_model_threshold': '0.7',  # Increased threshold for better matches
            'default_model_threshold': '0.50',
            'local_embeddings_path': '../Data/LocalEmbeddings'
        }
        config['LLM'] = {
            'LLM_FAMILY': 'AWS',
            'TEMPERATURE': '0.05'
        }
        config['Guardrails'] = {
            'guardrail_id': '3xr1mcliy9u6',
            'region': 'us-east-1',
            'description': 'Default guardrail for general use'
        }

        os.makedirs(self.config_path, exist_ok=True)
        with open(os.path.join(self.config_path, 'Config.properties'), 'w') as f:
            config.write(f)
        print(Fore.GREEN + "Created default Config.properties file" + Style.RESET_ALL)

    def create_default_config_io(self):
        """Create a default ConfigIO.properties file"""
        config = configparser.ConfigParser()
        config['Input'] = {
            'input_file_path': '../Input/UserStories.xlsx',
            'additional_context_path': '../Input/AdditionalContext.xlsx'
        }
        config['Output'] = {
            'output_file_path': '../Output/StorySense',
            'retrieval_context': '../Output/RetrievalContext',
            'num_context_retrieve': '8',
            'manual_test_type': 'Functional'
        }
        config['Processing'] = {
            'batch_size': '5',
            'parallel_processing': 'false'
        }
        config['Context'] = {
            'context_library_path': '../Input/ContextLibrary',
            'context_types': 'business_rules,requirements,documentation,policies,examples,glossary'
        }

        os.makedirs(self.config_path, exist_ok=True)
        with open(os.path.join(self.config_path, 'ConfigIO.properties'), 'w') as f:
            config.write(f)
        print(Fore.GREEN + "Created default ConfigIO.properties file" + Style.RESET_ALL)

    def process_context_library(self, context_folder=None, force_reprocess=False):
        """Process context library before analyzing user stories"""
        from src.context_handler.context_file_handler.enhanced_context_processor import EnhancedContextProcessor

        # Use default context folder if not specified
        if context_folder is None:
            context_folder = self.config_parser_io.get('Context', 'context_library_path',
                                                       fallback='../Input/ContextLibrary')

        print(Fore.CYAN + f"\nProcessing context library: {context_folder}" + Style.RESET_ALL)

        # Create metrics manager
        metrics_manager = MetricsManager()

        # Initialize context processor
        processor = EnhancedContextProcessor(context_folder, metrics_manager=metrics_manager)

        # Process context files
        stats = processor.process_all_context_files(force_reprocess=force_reprocess)

        # Display results
        if stats['processed_files'] > 0:
            print(Fore.GREEN + f"\nProcessed {stats['processed_files']} context files successfully" + Style.RESET_ALL)
        else:
            print(Fore.YELLOW + "\nNo new context files processed" + Style.RESET_ALL)

        return stats

    def process_user_stories(self, batch_size=5, parallel=False):
        """Process user stories with optional batch processing and parallelization"""
        # Create a single metrics manager instance
        metrics_manager = MetricsManager()
        print(f"DEBUG: Created metrics_manager in StorySenseGenerator: {id(metrics_manager)}")

        # Pass it to StorySenseProcessor
        story_sense_processor = StorySenseProcessor(metrics_manager=metrics_manager)

        header = "STORY SENSE ANALYZER"
        line_length = 60
        padding_length = (line_length - len(header)) // 2
        padding = " " * padding_length
        print("*" * line_length)
        print("*" + padding + header + padding + "*")
        print("*" * line_length + "\n")
        print(f"Processing options: batch_size={batch_size}, parallel={parallel}\n")

        if self.input_file_path:
            try:
                # Check if file exists
                if not os.path.exists(self.input_file_path):
                    print(
                        Fore.RED + f"Input user story file not found: {self.input_file_path}. Check the path and try again."
                        + Style.RESET_ALL)
                    sys.exit(1)

                # Try to read the file based on extension
                file_ext = os.path.splitext(self.input_file_path)[1].lower()
                if file_ext == '.xlsx' or file_ext == '.xls':
                    input_us = pd.read_excel(self.input_file_path)
                elif file_ext == '.csv':
                    input_us = pd.read_csv(self.input_file_path)
                else:
                    print(
                        Fore.RED + f"Unsupported file format: {file_ext}. Please provide an Excel (.xlsx, .xls) or CSV file."
                        + Style.RESET_ALL)
                    sys.exit(1)

                print("Processing input User Stories from", self.input_filename, "\n")

                # Verify required columns exist
                required_columns = ['ID', 'Description', 'AcceptanceCriteria']
                missing_columns = [col for col in required_columns if col not in input_us.columns]
                if missing_columns:
                    print(
                        Fore.RED + f"Missing required columns in user stories file: {', '.join(missing_columns)}"
                        + Style.RESET_ALL)
                    sys.exit(1)

            except Exception as e:
                print(
                    Fore.RED + f"Error reading user story file: {str(e)}"
                    + Style.RESET_ALL)
                sys.exit(1)
        else:
            print("Please provide input user story file path in configuration file or as a command line argument")
            sys.exit(1)

        # Flag to track if we've already processed the stories
        stories_processed = False

        if self.additional_context_path:
            try:
                # Check if file exists
                if not os.path.exists(self.additional_context_path):
                    print(
                        Fore.YELLOW + f'Input Additional Context File not found: {self.additional_context_path}. Analyzing without context.' + Style.RESET_ALL)
                    story_sense_processor.analyze_stories_in_batches(input_us, self.output_file_path, batch_size,
                                                                     parallel)
                    stories_processed = True
                    return

                # Try to read the file based on extension
                file_ext = os.path.splitext(self.additional_context_path)[1].lower()
                # Check if it's an image file
                if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                    print(
                        Fore.GREEN + f"\nProcessing with image context: {self.additional_context_name}\n" + Style.RESET_ALL)
                    # The image is already processed in the context library, just analyze with context
                    story_sense_processor.analyze_stories_with_context_in_batches(input_us, None, self.output_file_path,
                                                                                  batch_size, parallel)
                    stories_processed = True
                    return

                if file_ext == '.xlsx' or file_ext == '.xls':
                    input_context = pd.read_excel(self.additional_context_path)
                elif file_ext == '.csv':
                    input_context = pd.read_csv(self.additional_context_path)
                else:
                    print(
                        Fore.YELLOW + f"Unsupported context file format: {file_ext}. Analyzing without context." + Style.RESET_ALL)
                    story_sense_processor.analyze_stories_in_batches(input_us, self.output_file_path, batch_size,
                                                                     parallel)
                    stories_processed = True
                    return

                print("Processing with input Additional Context from", self.additional_context_name, "\n")
                try:
                    # Process with context automatically
                    story_sense_processor.analyze_stories_with_context_in_batches(input_us, input_context,
                                                                                  self.output_file_path, batch_size,
                                                                                  parallel)
                    stories_processed = True
                except KeyError as e:
                    if str(e) == "'vector_db'":
                        # Fix the specific KeyError without rerunning the analysis
                        print(f"Warning: Missing metrics key: {str(e)}. Continuing without combined report.")
                        # Mark as processed since we don't want to reprocess
                        stories_processed = True
                    else:
                        print(f"Error in context analysis: {str(e)}. Analyzing without context.")
                        # Only process without context if we haven't already processed the stories
                        if not stories_processed:
                            story_sense_processor.analyze_stories_in_batches(input_us, self.output_file_path,
                                                                             batch_size, parallel)
                            stories_processed = True
                except Exception as e:
                    print(f"Error processing with context: {str(e)}. Analyzing without context.")
                    # Only process without context if we haven't already processed the stories
                    if not stories_processed:
                        story_sense_processor.analyze_stories_in_batches(input_us, self.output_file_path,
                                                                         batch_size, parallel)
                        stories_processed = True
            except Exception as e:
                print(
                    Fore.YELLOW + f'Error reading Additional Context File: {str(e)}. Analyzing without context.' + Style.RESET_ALL)
                # Only process without context if we haven't already processed the stories
                if not stories_processed:
                    story_sense_processor.analyze_stories_in_batches(input_us, self.output_file_path, batch_size,
                                                                     parallel)
                    stories_processed = True
        else:
            print("\nNo Additional Context file provided. Analyzing User Stories without context\n")
            story_sense_processor.analyze_stories_in_batches(input_us, self.output_file_path, batch_size, parallel)
            stories_processed = True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='StorySense - User Story Analysis Tool')

    parser.add_argument('--user-stories', '-u',
                        help='Path to the user stories Excel file')

    parser.add_argument('--manage-context', '-m', action='store_true',
                        help='Manage context library (process documents)')

    parser.add_argument('--process-context', '-pc', action='store_true',
                        help='Process context library before analysis')

    parser.add_argument('--context', '-c',
                        help='Path to the additional context Excel file')

    parser.add_argument('--context-folder', '-cf',
                        help='Path to the context library folder')

    parser.add_argument('--batch-size', '-b', type=int, default=5,
                        help='Number of user stories to process in each batch')

    parser.add_argument('--parallel', '-p', action='store_true',
                        help='Enable parallel processing within batches')

    parser.add_argument('--force-reprocess', '-fr', action='store_true',
                        help='Force reprocessing of all context files')

    # Allow positional argument for user stories file (for simpler usage)
    parser.add_argument('user_stories_path', nargs='?', default=None,
                        help='Path to the user stories Excel file (positional argument)')

    args = parser.parse_args()

    # Prioritize named arguments over positional
    user_stories_path = args.user_stories or args.user_stories_path
    additional_context_path = args.context
    context_folder = args.context_folder
    batch_size = args.batch_size
    parallel = args.parallel
    force_reprocess = args.force_reprocess

    return user_stories_path, additional_context_path, batch_size, parallel, args, context_folder, force_reprocess


# Entry point
if __name__ == "__main__":
    # Parse command line arguments
    user_stories_path, additional_context_path, batch_size, parallel, args, context_folder, force_reprocess = parse_arguments()

    # Initialize StorySenseGenerator with command line arguments
    ssg = StorySenseGenerator(user_stories_path, additional_context_path)

    # Process context library if requested
    if args.process_context:
        ssg.process_context_library(context_folder, force_reprocess)

    # Manage context if requested
    if args.manage_context:
        from src.context_handler.context_file_handler.context_manager import ContextManager

        metrics_manager = MetricsManager()
        context_manager = ContextManager(metrics_manager=metrics_manager)
        status = context_manager.check_and_process_context_library(context_folder)
        print(f"\nContext library status: {status['message']}")

    # Process user stories
    ssg.process_user_stories(batch_size=batch_size, parallel=parallel)

    print("\nProcessing complete. Exiting application...")

    # Print active threads for debugging
    print(f"Active threads: {len(threading.enumerate())}")
    for thread in threading.enumerate():
        print(f"- {thread.name} ({'daemon' if thread.daemon else 'non-daemon'})")

    # Force exit
    sys.exit(0)