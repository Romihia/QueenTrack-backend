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
    git clone "$GITHUB_REPO" .
else
    cd "$PROJECT_DIR"
fi

# Ensure we're on the correct branch and pull latest changes
echo "📥 Pulling latest code from $BRANCH branch..."
git fetch origin
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"

# Copy production environment file
echo "⚙️ Setting up production environment..."
if [ -f ".env.production" ]; then
    cp .env.production .env
    echo "✅ Production environment configured"
else
    echo "⚠️ Warning: .env.production not found, using existing .env"
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down || true

# Clean up old images to save space
echo "🧹 Cleaning up old Docker images..."
docker image prune -af || true

# Build and start new containers
echo "🔨 Building and starting containers..."
docker-compose -f docker-compose.prod.yml up --build -d

# Wait for containers to start
echo "⏳ Waiting for services to start..."
sleep 20

# Health check
echo "🏥 Performing health check..."
if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
    echo "✅ Deployment successful! Containers are running:"
    docker-compose -f docker-compose.prod.yml ps
    
    # Test the health endpoint
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Health check passed!"
    else
        echo "⚠️ Health endpoint not responding, but containers are up"
    fi
else
    echo "❌ Deployment failed! Containers are not running properly:"
    docker-compose -f docker-compose.prod.yml ps
    echo "📋 Container logs:"
    docker-compose -f docker-compose.prod.yml logs
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo "🌐 Service is running on port 8000"
echo "📊 Monitor with: docker-compose -f docker-compose.prod.yml ps"
echo "📋 View logs with: docker-compose -f docker-compose.prod.yml logs" 