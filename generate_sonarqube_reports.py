#!/usr/bin/env python3
"""
SonarQube Test Coverage Report Generator
Generates coverage reports in formats required by SonarQube
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def generate_coverage_reports():
    """Generate comprehensive coverage reports for SonarQube"""
    
    print("🚀 Generating SonarQube Coverage Reports for StorySense")
    print("=" * 60)
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Step 1: Clean previous reports
    print("\n🧹 Cleaning previous reports...")
    cleanup_commands = [
        "Remove-Item -Force -Recurse htmlcov -ErrorAction SilentlyContinue",
        "Remove-Item -Force coverage.xml -ErrorAction SilentlyContinue", 
        "Remove-Item -Force test-results.xml -ErrorAction SilentlyContinue",
        "Remove-Item -Force .coverage -ErrorAction SilentlyContinue"
    ]
    
    for cmd in cleanup_commands:
        subprocess.run(cmd, shell=True, capture_output=True)
    
    # Step 2: Run tests with coverage for different modules
    test_modules = [
        ("tests/llm_layer/", "LLM Layer"),
        ("tests/metrics/", "Metrics Layer"), 
        ("tests/prompt_layer/", "Prompt Layer"),
        ("tests/embedding_handler/", "Embedding Handler"),
        ("tests/configuration_handler/", "Configuration Handler")
    ]
    
    all_passed = True
    
    for test_path, description in test_modules:
        if os.path.exists(test_path):
            cmd = f"python -m pytest {test_path} --cov=src --cov-append --cov-report=term-missing -v"
            result = run_command(cmd, f"Testing {description}")
            if result is None:
                all_passed = False
        else:
            print(f"⚠️  {test_path} not found, skipping {description}")
    
    # Step 3: Generate final coverage reports for SonarQube
    print("\n📊 Generating SonarQube-compatible reports...")
    
    # XML coverage report for SonarQube
    xml_cmd = "python -m pytest tests/ --cov=src --cov-report=xml:coverage.xml --junitxml=test-results.xml --tb=short"
    run_command(xml_cmd, "Generating XML coverage report")
    
    # HTML report for human review
    html_cmd = "python -m pytest tests/ --cov=src --cov-report=html:htmlcov --tb=short"
    run_command(html_cmd, "Generating HTML coverage report")
    
    # Step 4: Generate additional quality reports
    print("\n🔍 Generating additional quality reports...")
    
    # Pylint report (if pylint is installed)
    pylint_cmd = "pylint src/ --output-format=text --reports=yes > pylint-report.txt"
    run_command(pylint_cmd, "Generating Pylint report")
    
    # Bandit security report (if bandit is installed)
    bandit_cmd = "bandit -r src/ -f json -o bandit-report.json"
    run_command(bandit_cmd, "Generating Bandit security report")
    
    # Step 5: Display summary
    print("\n" + "=" * 60)
    print("📈 SONARQUBE REPORT GENERATION SUMMARY")
    print("=" * 60)
    
    reports_generated = []
    if os.path.exists("coverage.xml"):
        reports_generated.append("✅ coverage.xml (SonarQube coverage)")
    if os.path.exists("test-results.xml"):
        reports_generated.append("✅ test-results.xml (JUnit test results)")
    if os.path.exists("htmlcov/index.html"):
        reports_generated.append("✅ htmlcov/ (HTML coverage report)")
    if os.path.exists("pylint-report.txt"):
        reports_generated.append("✅ pylint-report.txt (Code quality)")
    if os.path.exists("bandit-report.json"):
        reports_generated.append("✅ bandit-report.json (Security analysis)")
    
    if reports_generated:
        print("\n🎉 Successfully generated reports:")
        for report in reports_generated:
            print(f"   {report}")
    
    # Display coverage summary if available
    if os.path.exists("coverage.xml"):
        try:
            with open("coverage.xml", 'r') as f:
                content = f.read()
                if 'line-rate=' in content:
                    import re
                    match = re.search(r'line-rate="([0-9.]+)"', content)
                    if match:
                        coverage_rate = float(match.group(1)) * 100
                        print(f"\n📊 Overall Coverage: {coverage_rate:.1f}%")
                        if coverage_rate >= 80:
                            print("✅ Coverage meets SonarQube quality gate (80%+)")
                        else:
                            print(f"⚠️  Coverage below SonarQube threshold. Need {80-coverage_rate:.1f}% more")
        except Exception as e:
            print(f"Could not parse coverage: {e}")
    
    print("\n🔧 Next Steps for SonarQube:")
    print("1. Upload these reports to your SonarQube server")
    print("2. Review the HTML coverage report: htmlcov/index.html")
    print("3. Fix any critical/blocker issues identified")
    print("4. Ensure coverage is above 80% threshold")
    print("5. Configure CI/CD pipeline to run this script automatically")
    
    return all_passed

if __name__ == "__main__":
    success = generate_coverage_reports()
    sys.exit(0 if success else 1)