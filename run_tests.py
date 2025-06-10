#!/usr/bin/env python3
"""
QueenTrack Test Runner
Runs comprehensive tests for the QueenTrack backend application.
"""

import sys
import subprocess
import os
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=Path(__file__).parent
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode == 0, result
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Run QueenTrack tests")
    parser.add_argument('--fast', action='store_true', help='Run only fast tests (skip performance tests)')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--pattern', help='Run tests matching pattern')
    args = parser.parse_args()
    
    print("🚀 QueenTrack Backend Test Suite")
    print("================================")
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Test categories to run
    test_files = [
        ("tests/test_crud.py", "CRUD Operations & Basic API Tests"),
        ("tests/test_database_service.py", "Database Service Layer Tests"),
        ("tests/test_api_routes.py", "API Routes & Endpoints Tests"),
        ("tests/test_video_processing.py", "Video Processing & Computer Vision Tests"),
    ]
    
    if not args.fast:
        test_files.append(("tests/test_performance.py", "Performance & Load Tests"))
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.coverage:
        pytest_cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])
    
    if args.pattern:
        pytest_cmd.extend(["-k", args.pattern])
    
    # Track results
    all_passed = True
    results = []
    
    # Run individual test files
    for test_file, description in test_files:
        if os.path.exists(test_file):
            cmd = pytest_cmd + [test_file]
            success, result = run_command(cmd, description)
            results.append((description, success))
            
            if not success:
                all_passed = False
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((description, False))
            all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {description}")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your QueenTrack backend is ready for deployment! 🚀")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the test output and fix issues before deployment.")
        
    print(f"{'='*60}")
    
    # Additional information
    print("\n📋 Next Steps:")
    if all_passed:
        print("1. ✅ All tests passed - your code is ready!")
        print("2. 🚀 You can now deploy to production")
        print("3. 📊 Review test coverage if --coverage was used")
    else:
        print("1. 🔍 Review failed test output above")
        print("2. 🛠️  Fix any issues in your code")
        print("3. 🔄 Re-run tests with: python run_tests.py")
    
    print("\n📖 Test Categories Explained:")
    print("• CRUD Tests: Database operations (create, read, update, delete)")
    print("• Service Tests: Business logic and data processing")
    print("• API Tests: HTTP endpoints and request/response handling")
    print("• Video Tests: Computer vision, frame processing, bee detection")
    print("• Performance Tests: Load testing, speed, memory usage")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 