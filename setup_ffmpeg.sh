#!/bin/bash

# Setup script for FFmpeg installation
# This script installs FFmpeg for video format conversion

echo "🎬 Setting up FFmpeg for video conversion..."

# Check if we're in a container or native environment
if [ -f /.dockerenv ]; then
    echo "📦 Detected Docker environment"
    
    # Update package list
    apt-get update
    
    # Install FFmpeg
    apt-get install -y ffmpeg
    
    # Verify installation
    if command -v ffmpeg &> /dev/null; then
        echo "✅ FFmpeg installed successfully"
        ffmpeg -version | head -1
    else
        echo "❌ FFmpeg installation failed"
        exit 1
    fi
    
else
    echo "🖥️ Detected native environment"
    
    # Check the operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "📋 Installing FFmpeg on Linux..."
        
        if command -v apt-get &> /dev/null; then
            # Debian/Ubuntu
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            sudo yum install -y epel-release
            sudo yum install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            # Fedora
            sudo dnf install -y ffmpeg
        else
            echo "❌ Unsupported Linux distribution"
            exit 1
        fi
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "📋 Installing FFmpeg on macOS..."
        
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "❌ Homebrew not found. Please install Homebrew first."
            exit 1
        fi
        
    elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win32"* ]]; then
        # Windows
        echo "📋 FFmpeg installation on Windows requires manual setup"
        echo "Please download FFmpeg from https://ffmpeg.org/download.html"
        echo "And add it to your system PATH"
        exit 1
        
    else
        echo "❌ Unsupported operating system: $OSTYPE"
        exit 1
    fi
fi

# Verify installation
if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg setup completed successfully"
    echo "📊 FFmpeg version:"
    ffmpeg -version | head -1
    echo ""
    echo "🎯 Supported codecs for H.264:"
    ffmpeg -codecs | grep -i h264 | head -3
else
    echo "❌ FFmpeg setup failed"
    exit 1
fi

echo "🚀 FFmpeg is ready for video conversion!" 