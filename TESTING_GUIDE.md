# 🧪 Local Testing Guide for QueenTrack Backend

This guide explains how to run the comprehensive test suite locally while the application is running in Docker containers.

## 🎯 Overview

The QueenTrack backend has a robust testing system with **75+ test cases** covering:
- Database operations and services
- API endpoints and routes
- Video processing and AI functionality
- Performance and load testing
- WebSocket connections
- Error handling and edge cases

## 📋 Prerequisites

### 1. Install Python Dependencies Locally
```bash
# Make sure you have Python 3.9+ installed
python --version

# Install all dependencies locally (needed for running tests)
pip install -r requirements.txt

# Verify pytest installation
pytest --version
```

### 2. Start Docker Services
```bash
# Start the application stack
docker-compose up -d

# Verify services are running
docker-compose ps

# Check service health
curl http://localhost:8000/health
```

## 🚀 Testing Methods

### Method 1: Using the Custom Test Runner (Recommended)

The project includes a custom test runner with enhanced features:

```bash
# Run all tests with detailed output and timing
python run_tests.py

# Run specific test categories
python run_tests.py --category database    # Database & service tests
python run_tests.py --category api         # API route tests  
python run_tests.py --category video       # Video processing tests
python run_tests.py --category performance # Performance tests

# Run tests with verbose output
python run_tests.py --verbose

# Run tests and generate HTML coverage report
python run_tests.py --coverage
```

**Test Runner Features:**
- ✅ Colored output with emojis
- ⏱️ Detailed timing information
- 📊 Test categorization
- 🎯 Coverage reporting
- 🐛 Enhanced error reporting

### Method 2: Direct pytest Commands

For more control, use pytest directly:

```bash
# Run all tests
pytest -v

# Run tests with short traceback
pytest -v --tb=short

# Run specific test files
pytest tests/test_api_routes.py -v
pytest tests/test_video_processing.py -v
pytest tests/test_database_service.py -v
pytest tests/test_performance.py -v

# Run tests matching a pattern
pytest -k "test_create" -v              # Tests with "create" in name
pytest -k "api" -v                      # Tests with "api" in name
pytest -k "not performance" -v          # Exclude performance tests

# Run tests and stop on first failure
pytest -x

# Run tests with detailed output
pytest -v -s

# Run tests with coverage
pytest --cov=app --cov-report=html --cov-report=term
```

### Method 3: Testing Against Docker Services

Test the application while it's running in Docker containers:

```bash
# Start services and wait for them to be ready
docker-compose up -d
sleep 10

# Run integration tests against Docker containers
pytest tests/test_crud.py -v              # Database integration tests
pytest tests/test_api_routes.py -v        # API integration tests

# Test specific endpoints
curl -X GET http://localhost:8000/health
curl -X GET http://localhost:8000/events/
curl -X GET http://localhost:8000/videos/list
```

## 📁 Test File Structure

```
tests/
├── conftest.py                 # Test configuration and fixtures
├── test_database_service.py    # Database & service tests (15 tests)
├── test_api_routes.py         # API endpoint tests (20 tests)
├── test_video_processing.py   # Video & AI tests (25 tests)
├── test_performance.py        # Performance tests (15 tests)
├── test_crud.py              # Integration tests
└── __pycache__/              # Python cache files
```

## 🧪 Test Categories Explained

### 🗃️ Database & Service Tests (15 tests)
**File:** `tests/test_database_service.py`

Tests database operations and business logic:
```bash
# Run only database tests
pytest tests/test_database_service.py -v
```

**Coverage:**
- ✅ Event creation and validation
- ✅ CRUD operations (Create, Read, Update, Delete)
- ✅ Database connection handling
- ✅ Data validation and error cases
- ✅ ObjectId validation
- ✅ Schema validation with Pydantic

### 🌐 API Route Tests (20 tests)
**File:** `tests/test_api_routes.py`

Tests all REST API endpoints:
```bash
# Run only API tests
pytest tests/test_api_routes.py -v
```

**Coverage:**
- ✅ Events endpoints (GET, POST, PUT, DELETE)
- ✅ Video endpoints (upload, processing)
- ✅ WebSocket connections
- ✅ Error responses (404, 400, 500)
- ✅ Request validation
- ✅ Response format validation

### 🎥 Video Processing Tests (25 tests)
**File:** `tests/test_video_processing.py`

Tests video processing and AI functionality:
```bash
# Run only video tests
pytest tests/test_video_processing.py -v
```

**Coverage:**
- ✅ YOLO model integration (mocked)
- ✅ Video upload and processing
- ✅ Frame processing and bee detection
- ✅ External camera control
- ✅ Video format validation
- ✅ Error handling for corrupted files

### ⚡ Performance Tests (15 tests)
**File:** `tests/test_performance.py`

Tests system performance and load handling:
```bash
# Run only performance tests
pytest tests/test_performance.py -v
```

**Coverage:**
- ✅ Response time benchmarks
- ✅ Memory usage monitoring
- ✅ Concurrent request handling
- ✅ Database query performance
- ✅ Video processing performance
- ✅ Load testing scenarios

## 🔧 Docker + Local Testing Workflow

### Option A: Services in Docker, Tests Local
```bash
# 1. Start Docker services
docker-compose up -d

# 2. Wait for services to be ready
sleep 5

# 3. Run tests locally
python run_tests.py

# 4. View service logs if needed
docker-compose logs -f backend
```

### Option B: Everything in Docker
```bash
# Run tests inside Docker container
docker-compose exec backend python -m pytest tests/ -v

# Or run specific test files
docker-compose exec backend python -m pytest tests/test_api_routes.py -v
```

### Option C: Mixed Environment Testing
```bash
# Start only database in Docker
docker-compose up -d mongo

# Run backend locally for debugging
export MONGO_URI=mongodb://localhost:27017
export MONGO_DB_NAME=queentrack_test
python -m uvicorn app.main:app --reload

# Run tests against local backend
pytest tests/ -v
```

## 📊 Coverage Reports

Generate detailed test coverage reports:

```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html

# View coverage in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows

# Generate terminal coverage report
pytest --cov=app --cov-report=term-missing

# Generate XML coverage report (for CI/CD)
pytest --cov=app --cov-report=xml
```

## 🔄 Continuous Testing During Development

### Watch Mode (Re-run tests on file changes)
```bash
# Install pytest-watch
pip install pytest-watch

# Watch for changes and re-run tests
ptw --runner "pytest -v"

# Watch specific directory
ptw app/ tests/ --runner "pytest -v"
```

### Pre-commit Testing
```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running tests before commit..."
python run_tests.py
if [ $? -ne 0 ]; then
  echo "Tests failed! Commit aborted."
  exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
```

## 🐛 Debugging Failed Tests

### Verbose Test Output
```bash
# Run with maximum verbosity
pytest -vvv --tb=long

# Show print statements in tests
pytest -v -s

# Drop into debugger on failures
pytest --pdb
```

### Debug Specific Test
```bash
# Run single test with debugging
pytest tests/test_api_routes.py::test_create_event -v -s

# Add print statements in test code
def test_my_function():
    result = my_function()
    print(f"Debug: result is {result}")  # Will show with -s flag
    assert result == expected
```

### Check Docker Service Logs
```bash
# View all service logs
docker-compose logs

# View backend logs only
docker-compose logs backend

# Follow logs in real-time
docker-compose logs -f backend

# View recent logs
docker-compose logs --tail=50 backend
```

## 🔍 Test Environment Variables

Tests use special environment variables:

```bash
# Set test-specific variables
export TESTING=true
export MONGO_DB_NAME=queentrack_test
export LOG_LEVEL=DEBUG

# Run tests with test variables
TESTING=true pytest tests/ -v
```

## 📝 Writing New Tests

### Test Template
```python
import pytest
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_my_new_feature():
    """Test description."""
    # Arrange
    test_data = {"key": "value"}
    
    # Act
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/endpoint", json=test_data)
    
    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "success"
```

### Best Practices
- ✅ Use descriptive test names
- ✅ Follow AAA pattern (Arrange, Act, Assert)
- ✅ Mock external dependencies
- ✅ Test both success and failure cases
- ✅ Use fixtures for common setup
- ✅ Keep tests independent and isolated

## 🚨 Troubleshooting

### Common Issues

#### Tests Can't Connect to Database
```bash
# Check if MongoDB is running
docker-compose ps mongo

# Check MongoDB logs
docker-compose logs mongo

# Verify connection string
echo $MONGO_URI
```

#### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python path
echo $PYTHONPATH

# Run from project root
cd /path/to/QueenTrack-backend
pytest tests/
```

#### Port Conflicts
```bash
# Check if port 8000 is in use
netstat -tulpn | grep :8000
lsof -i :8000

# Use different port for testing
export PORT=8001
docker-compose -p queentrack_test up -d
```

#### Permission Issues
```bash
# Fix video directory permissions
sudo chmod -R 755 data/videos/

# Fix test file permissions
chmod +x run_tests.py
```

## 📈 Performance Testing

### Load Testing with Locust
```bash
# Install locust
pip install locust

# Create locustfile.py for load testing
# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

### Memory Profiling
```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler tests/test_performance.py
```

## ✅ Test Checklist

Before pushing code, ensure:

- [ ] All tests pass: `python run_tests.py`
- [ ] Coverage is adequate: `pytest --cov=app`
- [ ] No lint errors: `flake8 app/ tests/`
- [ ] Docker services start: `docker-compose up -d`
- [ ] API endpoints respond: `curl http://localhost:8000/health`
- [ ] No memory leaks in long-running tests
- [ ] Performance tests meet benchmarks

## 🎉 Success!

You're now ready to run comprehensive tests locally while your QueenTrack backend runs in Docker! 

For any issues, check:
1. **README.md** - General setup and configuration
2. **Docker logs** - `docker-compose logs backend`
3. **Test output** - Run with `-v` flag for details
4. **GitHub Issues** - Report bugs and get help

Happy testing! 🐝✨ 