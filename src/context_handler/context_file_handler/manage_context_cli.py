
#!/usr/bin/env python3
import argparse
import os
import sys
from colorama import init, Fore, Style
from src.context_handler.context_file_handler.context_manager import ContextManager
from src.metrics.metrics_manager import MetricsManager

init()  # Initialize colorama


def main():
    parser = argparse.ArgumentParser(description="StorySense Context Library Manager")

    parser.add_argument('--process', '-p', action='store_true',
                        help='Process all documents in the context library')

    parser.add_argument('--status', '-s', action='store_true',
                        help='Show status of the context library')

    parser.add_argument('--directory', '-d', type=str,
                        help='Specify context library directory (default: ../Input/ContextLibrary)')

    parser.add_argument('--create-structure', '-c', action='store_true',
                        help='Create the context library directory structure')

    args = parser.parse_args()

    # Initialize context manager
    metrics_manager = MetricsManager()
    context_manager = ContextManager(metrics_manager=metrics_manager)

    if args.create_structure:
        directory = args.directory or '../Input/ContextLibrary'
        context_manager._create_context_directory_structure(directory)
        print(Fore.GREEN + f"Created context library structure at {directory}" + Style.RESET_ALL)
        return

    if args.status:
        status = context_manager.get_context_status()
        print(Fore.CYAN + "\n📊 CONTEXT LIBRARY STATUS" + Style.RESET_ALL)
        print(f"Total registered documents: {status['total_registered_documents']}")
        print(f"Last update: {status['last_update'] or 'Never'}")
        print("\nDocuments by collection:")
        for collection, count in status['collections'].items():
            print(f"  - {collection.replace('_', ' ').title()}: {count}")
        return

    if args.process:
        directory = args.directory or '../Input/ContextLibrary'
        status = context_manager.check_and_process_context_library(directory)
        print(Fore.GREEN + f"\nProcessing complete: {status['message']}" + Style.RESET_ALL)
        return

    # If no arguments, show help
    parser.print_help()


if __name__ == "__main__":
    main()