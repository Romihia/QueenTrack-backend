#!/bin/bash

# QueenTrack Production Deployment Script
# This script handles the deployment process on the production server

set -e  # Exit on any error

echo "🚀 QueenTrack Production Deployment Started"
echo "=================================================="

# Configuration
PROJECT_DIR="/opt/queentrack"
BACKUP_DIR="/opt/queentrack-backup"
LOG_FILE="/var/log/queentrack-deployment.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to create backup
create_backup() {
    log_message "📦 Creating backup of current deployment..."
    if [ -d "$PROJECT_DIR" ]; then
        sudo rm -rf "$BACKUP_DIR" || true
        sudo cp -r "$PROJECT_DIR" "$BACKUP_DIR"
        log_message "✅ Backup created successfully"
    fi
}

# Function to rollback in case of failure
rollback() {
    log_message "🔄 Rolling back to previous version..."
    if [ -d "$BACKUP_DIR" ]; then
        sudo rm -rf "$PROJECT_DIR"
        sudo mv "$BACKUP_DIR" "$PROJECT_DIR"
        cd "$PROJECT_DIR"
        docker-compose up -d
        log_message "✅ Rollback completed"
    else
        log_message "❌ No backup found for rollback"
    fi
}

# Main deployment function
main_deployment() {
    log_message "🏁 Starting main deployment process..."
    
    # Ensure project directory exists
    sudo mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    
    # Initialize git repository if it doesn't exist
    if [ ! -d ".git" ]; then
        log_message "🔧 Initializing Git repository..."
        git init
        git remote add origin https://github.com/YOUR_USERNAME/QueenTrack-backend.git
    fi
    
    # Pull latest code
    log_message "📥 Pulling latest code from main branch..."
    git fetch origin
    git reset --hard origin/main
    
    # Stop existing containers
    log_message "🛑 Stopping existing containers..."
    docker-compose down || true
    
    # Clean up old Docker images and containers
    log_message "🧹 Cleaning up Docker resources..."
    docker container prune -f || true
    docker image prune -af || true
    
    # Build new Docker image
    log_message "🔨 Building new Docker image..."
    docker-compose build --no-cache
    
    # Start new containers
    log_message "🚀 Starting new containers..."
    docker-compose up -d
    
    # Wait for service to start
    log_message "⏳ Waiting for service to start..."
    sleep 20
    
    # Health check
    log_message "🏥 Performing health check..."
    if docker-compose ps | grep -q "Up"; then
        log_message "✅ Deployment successful! Service is running."
        docker-compose ps
        
        # Additional health check with curl if possible
        if command -v curl &> /dev/null; then
            if curl -f http://localhost:8000/health &> /dev/null || curl -f http://localhost:8000/ &> /dev/null; then
                log_message "✅ HTTP health check passed"
            else
                log_message "⚠️ HTTP health check failed, but containers are running"
            fi
        fi
        
        return 0
    else
        log_message "❌ Deployment failed! Service is not running."
        docker-compose logs
        return 1
    fi
}

# Main script execution
main() {
    log_message "🎬 Deployment script started"
    
    # Create backup before deployment
    create_backup
    
    # Attempt deployment
    if main_deployment; then
        log_message "🎉 Deployment completed successfully!"
        log_message "🌐 Service should be available at: http://$(hostname -I | awk '{print $1}'):8000"
    else
        log_message "❌ Deployment failed, attempting rollback..."
        rollback
        exit 1
    fi
    
    log_message "🏁 Deployment script completed"
}

# Execute main function
main "$@" 