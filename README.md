# QueenTrack Backend 🐝

A sophisticated AI-powered bee tracking system that monitors Queen bee activities using computer vision and YOLO object detection.

## 🚀 Features

- **Real-time Video Processing**: Live stream processing with WebSocket support
- **AI-Powered Detection**: YOLO-based bee detection and classification
- **Event Tracking**: Automatic tracking of bee entry/exit events
- **External Camera Control**: Automated recording when bees exit the hive
- **Video Management**: Upload, process, and serve video files
- **RESTful API**: Complete CRUD operations for events
- **Docker Support**: Containerized deployment with Docker Compose
- **HTTPS Support**: Production-ready SSL/TLS configuration
- **Comprehensive Testing**: Full test suite with 75+ test cases

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Video Access](#video-access)
- [HTTPS Setup (Production)](#https-setup-production)
- [Running Tests Locally](#running-tests-locally)
- [API Documentation](#api-documentation)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

- **Docker & Docker Compose**: Latest version
- **Python 3.9+**: For local development and testing
- **Git**: For version control
- **OpenSSL**: For HTTPS certificate generation
- **MongoDB**: Database (can be local or cloud-based)

## 🔒 Environment Configuration

Create a `.env` file in the project root with the following variables:

### Required Variables

```bash
# Database Configuration
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=queentrack_db

# Application Environment
ENV=development  # Options: development, production

# Server Configuration (for production)
SERVER_HOST=162.55.53.52  # Your production server IP or domain
SERVER_PORT=8000

# SSL/HTTPS Configuration (for production)
SERVER_DOMAIN=yourdomain.com  # Your domain name
SSL_EMAIL=your-email@example.com  # Email for Let's Encrypt

# Camera Configuration (optional)
DEFAULT_INTERNAL_CAMERA=0  # Default internal camera ID
DEFAULT_EXTERNAL_CAMERA=1  # Default external camera ID

# Video Storage Configuration
VIDEOS_RETENTION_DAYS=30  # Days to keep videos (optional)
MAX_VIDEO_SIZE_MB=100  # Maximum video file size (optional)

# AI Model Configuration
YOLO_DETECTION_MODEL=yolov8n.pt  # YOLO detection model file
YOLO_CLASSIFICATION_MODEL=best.pt  # YOLO classification model file
DETECTION_CONFIDENCE=0.7  # Detection confidence threshold
```

### Optional Variables (Advanced Configuration)

```bash
# MongoDB Authentication (if using authenticated MongoDB)
MONGO_USERNAME=your_mongo_user
MONGO_PASSWORD=your_mongo_password

# Production Security
SECRET_KEY=your-secret-key-here  # For JWT or session management
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Monitoring & Logging
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR
SENTRY_DSN=https://your-sentry-dsn  # Error tracking (optional)

# Performance Tuning
UVICORN_WORKERS=2  # Number of worker processes
MAX_CONNECTIONS=1000  # Maximum concurrent connections

# External Services
TELEGRAM_BOT_TOKEN=your-telegram-token  # For notifications (optional)
TELEGRAM_CHAT_ID=your-chat-id  # Telegram chat ID for alerts

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=24
BACKUP_RETENTION_DAYS=7
```

### Environment-Specific Examples

#### Development (.env.development)
```bash
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=queentrack_dev
ENV=development
SERVER_HOST=localhost
SERVER_PORT=8000
LOG_LEVEL=DEBUG
```

#### Production (.env.production)
```bash
MONGO_URI=mongodb://your-production-mongo-host:27017
MONGO_DB_NAME=queentrack_production
ENV=production
SERVER_HOST=162.55.53.52
SERVER_PORT=8000
SERVER_DOMAIN=queentrack.yourdomain.com
SSL_EMAIL=admin@yourdomain.com
UVICORN_WORKERS=4
LOG_LEVEL=INFO
```

## 🏗️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/QueenTrack-backend.git
cd QueenTrack-backend
```

### 2. Create Environment File
```bash
# Copy template and edit
cp .env.example .env
# Edit .env with your configuration
nano .env
```

### 3. Build and Start Services
```bash
# For development
docker-compose up -d

# For production (HTTP)
docker-compose -f docker-compose.prod.yml up -d

# For production (HTTPS)
docker-compose -f docker-compose.https.yml up -d
```

## 🎬 Video Access

Videos are automatically served and accessible through multiple methods:

### Direct Video Access
```
http://[SERVER_IP]:8000/videos/[folder]/[filename.mp4]
```

**Examples:**
- `http://162.55.53.52:8000/videos/outside_videos/video_1234567890.mp4`
- `http://162.55.53.52:8000/videos/processed_videos/processed_upload.mp4`
- `http://162.55.53.52:8000/videos/uploaded_videos/my_video.mp4`

### Video Listing API
```bash
# List all videos
curl http://162.55.53.52:8000/videos/list

# List videos in specific folder
curl http://162.55.53.52:8000/videos/list?folder=outside_videos

# List available folders
curl http://162.55.53.52:8000/videos/folders
```

### Video Folder Structure
```
/data/videos/
├── outside_videos/     # External camera recordings
├── processed_videos/   # AI-processed videos
├── uploaded_videos/    # User-uploaded videos
└── temp_videos/        # Temporary processing files
```

### Browser Video Player
Videos support HTML5 video player features:
- **Streaming**: Range requests for efficient playback
- **Seeking**: Jump to any position in the video
- **Multiple Formats**: MP4, AVI, MOV, MKV, WebM
- **Mobile Friendly**: Responsive video player

## 🔐 HTTPS Setup (Production)

For production deployment with HTTPS support:

### 1. Configure SSL Variables
Add to your `.env` file:
```bash
SERVER_DOMAIN=yourdomain.com
SSL_EMAIL=admin@yourdomain.com
```

### 2. Run SSL Setup Script
```bash
chmod +x scripts/setup-ssl.sh
./scripts/setup-ssl.sh
```

### 3. Choose Certificate Option
- **Let's Encrypt** (Recommended): Free, trusted certificates
- **Self-signed**: For development/testing
- **Existing certificates**: If you already have certificates

### 4. Start HTTPS Services
```bash
docker-compose -f docker-compose.https.yml up -d
```

### 5. Access Your Site
- HTTP: `http://yourdomain.com` (redirects to HTTPS)
- HTTPS: `https://yourdomain.com`

### SSL Certificate Renewal
Certificates auto-renew via cron job:
```bash
# Add to crontab
0 12 * * * /path/to/QueenTrack-backend/scripts/renew-ssl.sh
```

## 🧪 Running Tests Locally

You can run the comprehensive test suite locally while the application runs in Docker:

### Prerequisites for Local Testing
```bash
# Install Python dependencies locally
pip install -r requirements.txt
```

### Method 1: Using the Test Runner
```bash
# Run all tests with detailed output
python run_tests.py

# Run specific test categories
python run_tests.py --category api
python run_tests.py --category video
python run_tests.py --category performance
```

### Method 2: Direct pytest Commands
```bash
# Run all tests
pytest -v

# Run specific test files
pytest tests/test_api_routes.py -v
pytest tests/test_video_processing.py -v
pytest tests/test_database_service.py -v

# Run tests with coverage
pytest --cov=app --cov-report=html

# Run performance tests
pytest tests/test_performance.py -v
```

### Method 3: Testing Against Docker Services
```bash
# Start the Docker services
docker-compose up -d

# Wait for services to be ready
sleep 10

# Run tests against running Docker containers
pytest tests/ -v --tb=short

# Run integration tests
pytest tests/test_crud.py -v
```

### Test Categories

#### 🗃️ Database & Service Tests (15 tests)
- Database connection and operations
- CRUD operations for events
- Data validation and error handling

#### 🌐 API Route Tests (20 tests)
- All REST endpoints
- WebSocket connections
- Error responses and edge cases

#### 🎥 Video Processing Tests (25 tests)
- YOLO model integration
- Video upload and processing
- Camera control simulation
- Stream processing

#### ⚡ Performance Tests (15 tests)
- Load testing
- Memory usage monitoring
- Response time benchmarks
- Concurrent request handling

### Test Configuration
Tests use mocked dependencies for:
- MongoDB connections
- External camera hardware
- YOLO model predictions
- File system operations

## 📚 API Documentation

### Core Endpoints

#### Health Check
```bash
GET /health
# Returns: {"status": "healthy", "service": "QueenTrack Backend"}
```

#### Events Management
```bash
# Create event
POST /events/
Content-Type: application/json
{
  "time_out": "2024-01-01T10:00:00",
  "time_in": "2024-01-01T10:30:00",
  "video_url": "/videos/outside_videos/video_123.mp4"
}

# Get all events
GET /events/

# Get specific event
GET /events/{event_id}

# Update event
PUT /events/{event_id}

# Delete event
DELETE /events/{event_id}
```

#### Video Processing
```bash
# Upload video for processing
POST /video/upload
Content-Type: multipart/form-data
file: [video file]

# Live stream WebSocket
WS /video/live-stream

# Camera configuration
POST /video/camera-config
{
  "internal_camera_id": "0",
  "external_camera_id": "1"
}

# External camera status
GET /video/external-camera-status
```

#### Video Access
```bash
# List videos
GET /videos/list?folder=outside_videos

# List folders
GET /videos/folders

# Direct video access
GET /videos/{folder}/{filename}
```

### Interactive Documentation
When running the server, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🚀 CI/CD Pipeline

The project includes automated CI/CD with GitHub Actions:

### Pipeline Features
- **Automated Testing**: Runs on `stage` branch pushes
- **Docker Build**: Validates Docker images
- **Production Deployment**: Auto-deploys `main` branch to production
- **Health Checks**: Verifies deployment success
- **Rollback Support**: Automatic rollback on deployment failure

### Required GitHub Secrets
```bash
PRODUCTION_HOST=162.55.53.52
PRODUCTION_USER=your_server_user
PRODUCTION_SSH_KEY=your_private_ssh_key
```

### Deployment Workflow
1. Push to `stage` → Run tests
2. Push to `main` → Deploy to production
3. Automatic health checks
4. Notification on success/failure

## 🔧 Running the Application

### Development Mode
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Mode
```bash
# HTTP production
docker-compose -f docker-compose.prod.yml up -d

# HTTPS production
docker-compose -f docker-compose.https.yml up -d
```

### Manual Deployment Commands
```bash
# SSH into production server
ssh user@162.55.53.52

# Update and deploy
cd /opt/queentrack
git pull origin main
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d --build
```

## 🐛 Troubleshooting

### Common Issues

#### MongoDB Connection Failed
```bash
# Check MongoDB service
docker-compose ps
docker-compose logs mongo

# Verify connection string
echo $MONGO_URI
```

#### Video Files Not Accessible
```bash
# Check video directory permissions
ls -la data/videos/
chmod 755 data/videos/

# Verify Docker volume mounting
docker-compose exec backend ls -la /data/videos/
```

#### SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in nginx/ssl/live/queentrack/fullchain.pem -noout -dates

# Renew Let's Encrypt certificates
./scripts/renew-ssl.sh
```

#### High CPU Usage
```bash
# Monitor container resources
docker stats

# Reduce YOLO model complexity
# Edit .env: YOLO_DETECTION_MODEL=yolov8n.pt (nano model)
```

### Performance Optimization

#### For High Video Loads
```bash
# Increase worker processes
ENV: UVICORN_WORKERS=4

# Enable video caching
# Add to nginx.conf: proxy_cache_path
```

#### For Large Video Files
```bash
# Increase client body size
ENV: MAX_VIDEO_SIZE_MB=500

# Enable streaming uploads
# Configure nginx with appropriate timeouts
```

### Monitoring & Logs

#### Application Logs
```bash
# View real-time logs
docker-compose logs -f backend

# View specific service logs
docker-compose logs nginx
docker-compose logs backend
```

#### System Monitoring
```bash
# Resource usage
docker stats --no-stream

# Disk usage
df -h data/

# Network connections
netstat -tulpn | grep :8000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `python run_tests.py`
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📧 Email: support@queentrack.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/QueenTrack-backend/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/QueenTrack-backend/wiki)

---

**Made with ❤️ for Bee Conservation** 🐝🌍
