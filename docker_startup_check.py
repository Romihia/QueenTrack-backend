#!/usr/bin/env python3
"""
Queen Track System Docker Startup Check
Quick verification that all components are working properly inside Docker containers
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

def print_header():
    print("=" * 70)
    print("🐳 Queen Track Docker System v2.0 - Startup Check")
    print("=" * 70)

def check_docker_environment():
    print("\n🐳 Checking Docker environment...")
    
    # Check if we're inside a container
    if os.path.exists('/.dockerenv'):
        print("✅ Running inside Docker container")
    else:
        print("⚠️ Not running inside Docker container")
    
    # Check container name
    try:
        hostname = subprocess.check_output(['hostname'], text=True).strip()
        print(f"📦 Container hostname: {hostname}")
    except:
        print("❌ Could not determine container hostname")

def check_directories():
    print("\n📁 Checking directories...")
    
    directories = [
        "/data/logs",
        "/data/videos",
        "/data/videos/outside_videos", 
        "/data/videos/temp_videos",
        "/data/videos/uploaded_videos",
        "/data/videos/processed_videos",
        "/app",
        "/app/app"
    ]
    
    for dir_path in directories:
        if os.path.exists(dir_path):
            # Get directory size
            try:
                size = subprocess.check_output(['du', '-sh', dir_path], text=True).split()[0]
                print(f"✅ {dir_path} (Size: {size})")
            except:
                print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
            # Try to create missing directories
            if dir_path.startswith('/data'):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"   ↳ Created successfully")
                except Exception as e:
                    print(f"   ↳ Failed to create: {e}")

def check_environment_variables():
    print("\n🔧 Checking environment variables...")
    
    required_vars = [
        "EMAIL_USER",
        "EMAIL_PASS", 
        "SEND_EMAIL",
        "MONGO_URI",
        "MONGO_DB_NAME"
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive information
            if "PASS" in var or "URI" in var:
                if len(value) > 10:
                    masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:]
                else:
                    masked_value = "****"
                print(f"✅ {var} = {masked_value}")
            else:
                print(f"✅ {var} = {value}")
        else:
            print(f"❌ {var} - NOT SET")

def check_python_packages():
    print("\n📦 Checking Python packages...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'cv2',  # This checks for OpenCV (works with both opencv-python and opencv-python-headless)
        'ultralytics',
        'numpy',
        'pymongo',
        'websockets'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")

def check_yolo_models():
    print("\n🤖 Checking YOLO models...")
    
    models = [
        "/app/yolov8n.pt",
        "/app/best.pt"
    ]
    
    for model_path in models:
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"✅ {model_path} ({size:.1f} MB)")
        else:
            print(f"❌ {model_path} - MISSING")

def check_container_resources():
    print("\n💻 Checking container resources...")
    
    try:
        # Check memory
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if 'MemTotal' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / 1024 / 1024
                    print(f"💾 Total Memory: {mem_gb:.1f} GB")
                    break
        
        # Check CPU
        cpu_count = os.cpu_count()
        print(f"⚡ CPU Cores: {cpu_count}")
        
        # Check disk space
        disk_usage = subprocess.check_output(['df', '-h', '/'], text=True)
        lines = disk_usage.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            print(f"💽 Disk Space: {parts[1]} total, {parts[2]} used, {parts[3]} available")
            
    except Exception as e:
        print(f"⚠️ Could not check resources: {e}")

def wait_for_server(url="http://localhost:8000", timeout=60):
    print(f"\n⏳ Waiting for server at {url}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Use curl since requests might not be available in minimal containers
            result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', url], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip() == '200':
                print(f"✅ Server is responding")
                return True
        except:
            pass
        
        time.sleep(2)
        print(".", end="", flush=True)
    
    print(f"\n❌ Server did not respond within {timeout} seconds")
    return False

def test_endpoints():
    print("\n🧪 Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    
    endpoints = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/system/health", "System health"),
        ("/video/external-camera-status", "Camera status"),
        ("/video/videos-list", "Video listing")
    ]
    
    for endpoint, description in endpoints:
        try:
            result = subprocess.run(['curl', '-s', '-w', '%{http_code}', f"{base_url}{endpoint}"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Extract status code from the end of the output
                output = result.stdout
                if len(output) >= 3 and output[-3:].isdigit():
                    status_code = output[-3:]
                    if status_code == '200':
                        print(f"✅ {endpoint} - {description}")
                    else:
                        print(f"⚠️ {endpoint} - {description} (Status: {status_code})")
                else:
                    print(f"⚠️ {endpoint} - {description} (Unknown response)")
            else:
                print(f"❌ {endpoint} - {description} (Connection failed)")
        except Exception as e:
            print(f"❌ {endpoint} - {description} (Error: {e})")

def test_email_service():
    print("\n📧 Testing email service...")
    
    try:
        result = subprocess.run(['curl', '-s', '-X', 'POST', 'http://localhost:8000/video/test-email'], 
                              capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                if data.get("send_test"):
                    print("✅ Email test successful - Check your inbox!")
                else:
                    print("⚠️ Email connection OK but send failed")
                    print(f"   Message: {data.get('message', 'Unknown error')}")
            except json.JSONDecodeError:
                print("⚠️ Email test response not JSON - check manually")
        else:
            print(f"❌ Email test failed (curl error)")
    except Exception as e:
        print(f"❌ Email test error: {e}")

def check_container_logs():
    print("\n📋 Checking container logs...")
    
    log_files = [
        "/data/logs/queen_track.log"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                # Get last 5 lines of log
                result = subprocess.run(['tail', '-5', log_file], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {log_file} (Last 5 lines):")
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            print(f"   {line}")
                else:
                    print(f"⚠️ {log_file} - Could not read")
            except Exception as e:
                print(f"⚠️ {log_file} - Error reading: {e}")
        else:
            print(f"❌ {log_file} - Does not exist")

def show_docker_next_steps():
    print("\n" + "=" * 70)
    print("🎯 Docker Next Steps:")
    print("=" * 70)
    print("1. 📁 Add sample video to frontend container:")
    print("   docker cp your-video.mp4 container_name:/app/public/sample-videos/sample-hive-video.mp4")
    print()
    print("2. 🌐 Access services:")
    print("   - Backend API: http://localhost:8000")
    print("   - Frontend: http://localhost:3000 (if running)")
    print("   - API Docs: http://localhost:8000/docs")
    print()
    print("3. 🔧 Monitor containers:")
    print("   docker-compose logs -f          # All logs")
    print("   docker-compose logs -f backend  # Backend only")
    print("   docker stats                    # Resource usage")
    print()
    print("4. 🧪 Test inside container:")
    print("   docker-compose exec backend python3 docker_startup_check.py")
    print()
    print("5. 📧 Email notifications will be sent when bees are detected")
    print()
    print("📚 Full documentation: SYSTEM_UPGRADE_README.md")
    print("🐳 Container logs: docker-compose logs -f")
    print()
    print("🐝 Queen Track Docker v2.0 is ready! Happy bee monitoring!")

def show_troubleshooting():
    print("\n" + "=" * 70)
    print("🔧 Docker Troubleshooting:")
    print("=" * 70)
    print("If issues occur:")
    print()
    print("🔄 Restart containers:")
    print("   docker-compose down")
    print("   docker-compose up -d")
    print()
    print("🧹 Clean restart:")
    print("   docker-compose down -v")
    print("   docker-compose build --no-cache")
    print("   docker-compose up -d")
    print()
    print("📋 Check logs:")
    print("   docker-compose logs backend")
    print("   docker exec -it bee_backend tail -f /data/logs/queen_track.log")
    print()
    print("💻 Access container shell:")
    print("   docker-compose exec backend bash")
    print("   docker-compose exec backend python3 -c 'import cv2; print(cv2.__version__)'")

def main():
    print_header()
    
    # Docker-specific checks
    check_docker_environment()
    check_directories()
    check_environment_variables()
    check_python_packages()
    check_yolo_models()
    check_container_resources()
    
    # Wait for server to start
    if not wait_for_server(timeout=60):
        print("\n❌ Server startup check failed!")
        show_troubleshooting()
        sys.exit(1)
    
    # Test endpoints
    test_endpoints()
    
    # Test email (optional)
    email_user = os.getenv("EMAIL_USER")
    if email_user:
        test_email_service()
    else:
        print("\n📧 Skipping email test (EMAIL_USER not configured)")
    
    # Check logs
    check_container_logs()
    
    # Show next steps
    show_docker_next_steps()

if __name__ == "__main__":
    main() 