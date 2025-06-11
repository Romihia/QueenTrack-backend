#!/bin/bash

# 🐳 Queen Track Docker Manager v2.0
# Easy Docker container management and testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "${PURPLE}"
    echo "=================================================="
    echo "🐝 Queen Track Docker Manager v2.0"
    echo "=================================================="
    echo -e "${NC}"
}

# Function to check if Docker is running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    print_status "Docker and Docker Compose are available"
}

# Function to start containers
start_containers() {
    print_info "Starting Queen Track containers..."
    
    # Create data directories on host
    mkdir -p ./data/logs ./data/videos/outside_videos ./data/videos/temp_videos ./data/videos/uploaded_videos ./data/videos/processed_videos
    
    # Build and start containers
    docker-compose up -d --build
    
    print_status "Containers started"
    
    # Wait a moment for containers to initialize
    print_info "Waiting for containers to initialize..."
    sleep 10
    
    # Show container status
    docker-compose ps
}

# Function to stop containers
stop_containers() {
    print_info "Stopping Queen Track containers..."
    docker-compose down
    print_status "Containers stopped"
}

# Function to restart containers
restart_containers() {
    print_info "Restarting Queen Track containers..."
    docker-compose down
    docker-compose up -d --build
    print_status "Containers restarted"
    
    # Wait for restart
    sleep 10
    docker-compose ps
}

# Function to clean restart (remove volumes)
clean_restart() {
    print_warning "This will remove ALL container data including logs and videos!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Performing clean restart..."
        docker-compose down -v
        docker system prune -f
        docker-compose build --no-cache
        docker-compose up -d
        print_status "Clean restart completed"
        sleep 10
        docker-compose ps
    else
        print_info "Clean restart cancelled"
    fi
}

# Function to show logs
show_logs() {
    local service=${1:-""}
    
    if [ -n "$service" ]; then
        print_info "Showing logs for $service..."
        docker-compose logs -f "$service"
    else
        print_info "Showing all container logs..."
        docker-compose logs -f
    fi
}

# Function to run startup check inside container
run_startup_check() {
    print_info "Running startup check inside container..."
    
    if ! docker-compose ps | grep -q "bee_backend.*Up"; then
        print_error "Backend container is not running. Please start containers first."
        return 1
    fi
    
    docker-compose exec backend python3 docker_startup_check.py
}

# Function to run health check
run_health_check() {
    print_info "Running health check..."
    
    # Test if backend is responding
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000 | grep -q "200"; then
        print_status "Backend is responding"
        
        # Test specific endpoints
        echo "Testing endpoints:"
        curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "Health endpoint response received"
        curl -s http://localhost:8000/system/health | jq . 2>/dev/null || echo "System health endpoint response received"
    else
        print_error "Backend is not responding"
        return 1
    fi
}

# Function to enter container shell
enter_container() {
    local service=${1:-"backend"}
    
    print_info "Entering $service container shell..."
    
    if docker-compose ps | grep -q "${service}.*Up"; then
        docker-compose exec "$service" /bin/bash
    else
        print_error "$service container is not running"
        return 1
    fi
}

# Function to show container stats
show_stats() {
    print_info "Container resource usage:"
    docker stats --no-stream
}

# Function to copy sample video
copy_sample_video() {
    local video_file="$1"
    
    if [ -z "$video_file" ]; then
        print_error "Please provide video file path"
        echo "Usage: $0 copy-video /path/to/your/video.mp4"
        return 1
    fi
    
    if [ ! -f "$video_file" ]; then
        print_error "Video file not found: $video_file"
        return 1
    fi
    
    print_info "Copying video to frontend container..."
    
    # Copy to local public directory first (if frontend is local)
    if [ -d "../queen-track-frontend/public/sample-videos/" ]; then
        cp "$video_file" "../queen-track-frontend/public/sample-videos/sample-hive-video.mp4"
        print_status "Video copied to frontend public directory"
    fi
    
    # If frontend is also in Docker, copy there too
    if docker ps | grep -q "frontend"; then
        docker cp "$video_file" frontend_container:/app/public/sample-videos/sample-hive-video.mp4
        print_status "Video copied to frontend container"
    fi
}

# Function to show help
show_help() {
    print_header
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start               Start all containers"
    echo "  stop                Stop all containers"
    echo "  restart             Restart all containers"
    echo "  clean               Clean restart (removes all data)"
    echo "  logs [service]      Show logs (optionally for specific service)"
    echo "  check               Run startup check inside container"
    echo "  health              Run health check"
    echo "  shell [service]     Enter container shell (default: backend)"
    echo "  stats               Show container resource usage"
    echo "  copy-video <file>   Copy sample video to frontend"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start all containers"
    echo "  $0 logs backend             # Show backend logs"
    echo "  $0 shell                    # Enter backend shell"
    echo "  $0 copy-video bee_video.mp4 # Copy video for testing"
    echo ""
    echo "🐝 Queen Track Docker Manager v2.0"
}

# Main execution
main() {
    check_docker
    
    case "${1:-help}" in
        "start")
            start_containers
            ;;
        "stop")
            stop_containers
            ;;
        "restart")
            restart_containers
            ;;
        "clean")
            clean_restart
            ;;
        "logs")
            show_logs "$2"
            ;;
        "check")
            run_startup_check
            ;;
        "health")
            run_health_check
            ;;
        "shell")
            enter_container "$2"
            ;;
        "stats")
            show_stats
            ;;
        "copy-video")
            copy_sample_video "$2"
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 