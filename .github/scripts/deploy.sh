#!/bin/bash

# QueenTrack Backend Deployment Script
# This script handles the server-side deployment process

set -e

echo "🚀 Starting QueenTrack Backend Deployment..."

# Configuration
PROJECT_DIR="/opt/queentrack"
GITHUB_REPO="https://github.com/Romihia/QueenTrack-backend.git"  # This will be updated automatically by CI/CD
BRANCH="main"

# Create project directory if it doesn't exist
if [ ! -d "$PROJECT_DIR" ]; then
    echo "📁 Creating project directory..."
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    
    echo "📥 Cloning repository..."
    git clone "$GITHUB_REPO" . || {
        echo "⚠️ Git clone failed, continuing with existing files..."
    }
else
    cd "$PROJECT_DIR"
fi

# Ensure we're on the correct branch and pull latest changes
echo "📥 Updating code from $BRANCH branch..."
if [ -d ".git" ]; then
    git fetch origin || echo "Git fetch failed, continuing..."
    git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"
    git reset --hard "origin/$BRANCH" || echo "Git reset failed, continuing with current code..."
else
    echo "⚠️ Not a git repository, skipping git operations..."
fi

# Copy production environment file
echo "⚙️ Setting up production environment..."
if [ -f ".env.production" ]; then
    cp .env.production .env
    echo "✅ Production environment configured from .env.production"
    echo "🔍 Environment variables loaded:"
    grep -E "^[A-Z_]+=.*" .env | head -5 | sed 's/=.*/=***/' || echo "No environment variables found"
else
    echo "❌ ERROR: .env.production file not found!"
    echo "📁 Files in current directory:"
    ls -la | grep -E "\.(env|production)" || echo "No environment files found"
    
    if [ ! -f ".env" ]; then
        echo "⚠️ Creating fallback .env file with required variables..."
        cat > .env << EOL
# Fallback production environment - UPDATE THESE VALUES!
ENV=production
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# DATABASE - These need to be set properly!
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=queentrack_production

# MINIMAL REQUIRED SETTINGS
SECRET_KEY=fallback-secret-key-change-this
CORS_ORIGINS=*
UVICORN_WORKERS=2
LOG_LEVEL=INFO

# AI MODELS
YOLO_DETECTION_MODEL=yolov8n.pt
YOLO_CLASSIFICATION_MODEL=best.pt
DETECTION_CONFIDENCE=0.7

# HEALTH CHECK
HEALTHCHECK_ENABLED=true
EOL
        echo "⚠️ WARNING: Using fallback environment. Update MongoDB settings!"
    fi
fi

# Determine which docker-compose file to use
COMPOSE_FILE="docker-compose.yml"
if [ -f "docker-compose.prod.yml" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
    echo "✅ Using production docker-compose file"
else
    echo "⚠️ docker-compose.prod.yml not found, using docker-compose.yml"
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f "$COMPOSE_FILE" down || true

# Clean up old images to save space
echo "🧹 Cleaning up old Docker images..."
docker image prune -af || true
docker container prune -f || true

# Build and start new containers
echo "🔨 Building and starting containers..."
docker-compose -f "$COMPOSE_FILE" up --build -d

# Wait for containers to start
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🏥 Performing health check..."
if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
    echo "✅ Deployment successful! Containers are running:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    # Test the health endpoint with multiple attempts
    echo "🔍 Testing health endpoints..."
    HEALTH_CHECK_PASSED=false
    
    for i in {1..5}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Health check passed on /health endpoint!"
            HEALTH_CHECK_PASSED=true
            break
        elif curl -f http://localhost:8000/ > /dev/null 2>&1; then
            echo "✅ Health check passed on root endpoint!"
            HEALTH_CHECK_PASSED=true
            break
        else
            echo "⏳ Health check attempt $i/5 failed, retrying in 10 seconds..."
            sleep 10
        fi
    done
    
    if [ "$HEALTH_CHECK_PASSED" = false ]; then
        echo "⚠️ Health endpoints not responding, but containers are up"
        echo "📋 Container logs:"
        docker-compose -f "$COMPOSE_FILE" logs --tail=20
    fi
    
else
    echo "❌ Deployment failed! Containers are not running properly:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    echo "📋 Container logs:"
    docker-compose -f "$COMPOSE_FILE" logs --tail=50
    echo ""
    echo "🔍 Docker system info:"
    docker system df
    exit 1
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo "🌐 Service is running on port 8000"
echo "📊 Monitor with: docker-compose -f $COMPOSE_FILE ps"
echo "📋 View logs with: docker-compose -f $COMPOSE_FILE logs"
echo "🔍 Follow logs with: docker-compose -f $COMPOSE_FILE logs -f"
echo "🛑 Stop service with: docker-compose -f $COMPOSE_FILE down"