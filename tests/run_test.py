#!/usr/bin/env python3

import os
import subprocess
import webbrowser
import time
import sys


def run_tests():
    """Run all tests and generate coverage reports"""
    print("Running tests with coverage...")

    # Create directories for reports if they don't exist
    os.makedirs("reports", exist_ok=True)

    # Run pytest with coverage
    result = subprocess.run([
        "pytest",
        "--cov=src",
        "--cov-report=xml:reports/coverage.xml",
        "--cov-report=html:reports/htmlcov",
        "--cov-report=term",
        "--cov-branch",
        "--junitxml=reports/junit.xml",
        "-v"
    ], capture_output=True, text=True)

    # Print test output
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

    # Open HTML report in browser
    html_report = os.path.join("reports", "htmlcov", "index.html")
    if os.path.exists(html_report):
        print(f"Opening HTML coverage report: {html_report}")
        webbrowser.open(f"file://{os.path.abspath(html_report)}")
    else:
        print(f"HTML report not found at {html_report}")

    return result.returncode


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
