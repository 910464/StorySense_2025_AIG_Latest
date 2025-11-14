#!/usr/bin/env python3
"""
Optimized test runner for StorySense project
Addresses issues with running large test suites
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests_by_category():
    """Run tests by category to avoid resource conflicts"""
    print("🧪 Running tests by category to avoid resource conflicts...")
    
    test_categories = [
        ("Unit Tests - AWS Layer", "tests/aws_layer/"),
        ("Unit Tests - Configuration", "tests/config/ tests/configuration_handler/"),
        ("Unit Tests - Context Handlers", "tests/context_handler/"),
        ("Unit Tests - Embedding", "tests/embedding_handler/"),
        ("Unit Tests - Interface Layer", "tests/interface_layer/"),
        ("Unit Tests - LLM Layer", "tests/llm_layer/"),
        ("Unit Tests - Metrics", "tests/metrics/"),
        ("Unit Tests - Prompt Layer", "tests/prompt_layer/"),
        ("Unit Tests - HTML Report", "tests/html_report/")
    ]
    
    overall_success = True
    results = []
    
    for category_name, test_paths in test_categories:
        print(f"\n🔍 Running {category_name}...")
        
        # Build pytest command for this category
        cmd = [
            sys.executable, "-m", "pytest",
            "--tb=short",
            "--maxfail=5",
            "--timeout=30",
            "-v"
        ] + test_paths.split()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {category_name}: PASSED")
                results.append((category_name, "PASSED", None))
            else:
                print(f"❌ {category_name}: FAILED")
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                print("STDERR:", result.stderr[-500:])  # Last 500 chars
                results.append((category_name, "FAILED", result.stderr))
                overall_success = False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {category_name}: TIMEOUT")
            results.append((category_name, "TIMEOUT", "Test execution exceeded 5 minutes"))
            overall_success = False
        except Exception as e:
            print(f"💥 {category_name}: ERROR - {e}")
            results.append((category_name, "ERROR", str(e)))
            overall_success = False
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    for category, status, error in results:
        status_emoji = "✅" if status == "PASSED" else "❌"
        print(f"{status_emoji} {category}: {status}")
        if error and len(error) > 100:
            print(f"   Error: {error[:100]}...")
    
    return overall_success


def run_fast_smoke_test():
    """Run a quick smoke test on a subset of important tests"""
    print("🚀 Running fast smoke tests...")
    
    smoke_tests = [
        "tests/aws_layer/test_circuit_breaker.py::TestCircuitBreaker::test_initialization",
        "tests/configuration_handler/test_config_loader.py::TestConfigLoader::test_load_valid_configuration",
        "tests/embedding_handler/test_embedding_cache.py::TestEmbeddingCache::test_initialization_default",
        "tests/metrics/test_metrics_manager.py::TestMetricsManager::test_init_default_values"
    ]
    
    cmd = [
        sys.executable, "-m", "pytest",
        "--tb=short",
        "-v",
        "--timeout=10"
    ] + smoke_tests
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Smoke tests: PASSED")
            return True
        else:
            print("❌ Smoke tests: FAILED")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"💥 Smoke tests: ERROR - {e}")
        return False


def run_coverage_report():
    """Generate coverage report for the entire codebase"""
    print("📊 Generating coverage report...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=src",
        "--cov-report=xml:coverage.xml",
        "--cov-report=html:htmlcov",
        "--cov-report=term",
        "--cov-branch",
        "--cov-fail-under=70",  # Fail if coverage below 70%
        "--maxfail=50",  # Allow more failures for coverage
        "--timeout=60",  # Longer timeout for coverage
        "tests/"
    ]
    
    try:
        result = subprocess.run(cmd, timeout=1200)  # 20 minute timeout
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Coverage generation timed out")
        return False
    except Exception as e:
        print(f"💥 Coverage generation failed: {e}")
        return False


def run_parallel_tests():
    """Run tests in parallel using pytest-xdist if available"""
    print("⚡ Attempting to run tests in parallel...")
    
    # Check if pytest-xdist is available
    try:
        subprocess.run([sys.executable, "-c", "import xdist"], check=True, capture_output=True)
        has_xdist = True
    except subprocess.CalledProcessError:
        has_xdist = False
        print("📦 Installing pytest-xdist for parallel execution...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest-xdist"], check=True)
            has_xdist = True
        except subprocess.CalledProcessError:
            print("❌ Could not install pytest-xdist")
            return False
    
    if has_xdist:
        cmd = [
            sys.executable, "-m", "pytest",
            "-n", "auto",  # Use all available CPUs
            "--tb=short",
            "--maxfail=20",
            "--timeout=45",
            "--dist=loadfile",  # Distribute by file to reduce conflicts
            "tests/"
        ]
        
        try:
            result = subprocess.run(cmd, timeout=1800)  # 30 minute timeout
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("⏰ Parallel test execution timed out")
            return False
        except Exception as e:
            print(f"💥 Parallel test execution failed: {e}")
            return False
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Optimized StorySense Test Runner")
    parser.add_argument("--mode", 
                       choices=["smoke", "category", "parallel", "coverage"], 
                       default="category",
                       help="Test execution mode")
    parser.add_argument("--timeout", type=int, default=30, help="Test timeout in seconds")
    
    args = parser.parse_args()
    
    # Set working directory
    os.chdir(Path(__file__).parent)
    
    print("🧪 StorySense Optimized Test Runner")
    print("="*50)
    
    success = False
    
    if args.mode == "smoke":
        success = run_fast_smoke_test()
    elif args.mode == "category":
        success = run_tests_by_category()
    elif args.mode == "parallel":
        success = run_parallel_tests()
    elif args.mode == "coverage":
        success = run_coverage_report()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()