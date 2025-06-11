#!/usr/bin/env python3
"""
Queen Track System Startup Check
Quick verification that all components are working properly
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

def print_header():
    print("=" * 60)
    print("🐝 Queen Track System v2.0 - Startup Check")
    print("=" * 60)

def check_directories():
    print("\n📁 Checking directories...")
    
    directories = [
        "/data/logs",
        "/data/videos",
        "/data/videos/outside_videos", 
        "/data/videos/temp_videos",
        "/data/videos/uploaded_videos",
        "/data/videos/processed_videos"
    ]
    
    for dir_path in directories:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
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
        "SEND_EMAIL"
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive information
            if "PASS" in var:
                masked_value = "*" * (len(value) - 4) + value[-4:] if len(value) > 4 else "****"
                print(f"✅ {var} = {masked_value}")
            else:
                print(f"✅ {var} = {value}")
        else:
            print(f"❌ {var} - NOT SET")

def wait_for_server(url="http://localhost:8000", timeout=30):
    print(f"\n⏳ Waiting for server at {url}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ Server is responding")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
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
        ("/videos/list", "Video listing")
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} - {description}")
            else:
                print(f"⚠️ {endpoint} - {description} (Status: {response.status_code})")
        except Exception as e:
            print(f"❌ {endpoint} - {description} (Error: {e})")

def test_email_service():
    print("\n📧 Testing email service...")
    
    try:
        response = requests.post("http://localhost:8000/video/test-email", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("send_test"):
                print("✅ Email test successful - Check your inbox!")
            else:
                print("⚠️ Email connection OK but send failed")
                print(f"   Message: {data.get('message', 'Unknown error')}")
        else:
            print(f"❌ Email test failed (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Email test error: {e}")

def show_next_steps():
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("=" * 60)
    print("1. 📁 Add sample video to: public/sample-videos/sample-hive-video.mp4")
    print("2. 🌐 Open frontend: http://localhost:3000")
    print("3. 🔧 Configure cameras in the web interface")
    print("4. ▶️ Start video streaming to test bee detection")
    print("5. 📧 Watch for email notifications when bees are detected")
    print()
    print("📚 Full documentation: QueenTrack-backend/SYSTEM_UPGRADE_README.md")
    print("🐛 Logs location: /data/logs/queen_track.log")
    print()
    print("🐝 Queen Track v2.0 is ready! Happy bee monitoring!")

def main():
    print_header()
    
    # Basic checks
    check_directories()
    check_environment_variables()
    
    # Wait for server to start
    if not wait_for_server():
        print("\n❌ Server startup check failed!")
        sys.exit(1)
    
    # Test endpoints
    test_endpoints()
    
    # Test email (optional)
    email_user = os.getenv("EMAIL_USER")
    if email_user:
        test_email_service()
    else:
        print("\n📧 Skipping email test (EMAIL_USER not configured)")
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main() 