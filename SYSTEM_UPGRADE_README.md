# Queen Track System v2.0 - Professional Upgrade

## 🚀 **Major System Improvements**

This upgrade transforms the Queen Track system into a professional-grade bee monitoring solution with advanced features.

---

## ✨ **New Features**

### 1. 📧 **Professional Email Notification System**

- **HTML-formatted emails** with embedded bee images
- **Automatic notifications** when marked bee exits/enters hive
- **Detailed event information** including timestamps and coordinates
- **Error handling and logging** for reliable email delivery

### 2. 🎥 **Advanced Video Streaming & Serving**

- **Range request support** for proper video playback in browsers
- **Professional video service** with security checks
- **Multiple video formats** support (MP4, AVI, MOV, etc.)
- **Video metadata extraction** (duration, FPS, resolution)
- **Proper MIME types** and HTTP headers

### 3. 📹 **Real External Camera Integration**

- **Automatic camera detection** and connection
- **Fallback to mock camera** if real camera unavailable
- **Camera specs detection** (resolution, FPS)
- **Improved error handling** and logging

### 4. 🔍 **System Health Monitoring**

- **Comprehensive health checks** for all components
- **Camera availability testing**
- **Email service validation**
- **YOLO model verification**
- **Full system workflow testing**

### 5. 📊 **Professional Logging**

- **Structured logging** with timestamps and levels
- **File and console output**
- **Component-specific loggers**
- **Error tracking and debugging**

---

## 🛠 **Installation & Setup**

### Prerequisites

```bash
# Install additional Python packages
pip install python-multipart
```

### Environment Variables

Make sure your `.env` file includes:

```env
EMAIL_USER=your_gmail@gmail.com
EMAIL_PASS=your_app_password
SEND_EMAIL=recipient@email.com
```

---

## 🔧 **API Endpoints**

### System Health & Testing

```
GET  /system/health              - Comprehensive system health check
POST /system/test-full-system    - Test complete system workflow
```

### Email Testing

```
POST /video/test-email           - Test email functionality
```

### Video Streaming (NEW)

```
GET  /video/videos/{file_path}   - Stream videos with range support
GET  /video/videos-list          - List all available videos
GET  /video/video-info/{path}    - Get video file information
```

### Existing Endpoints (Enhanced)

```
POST /video/camera-config        - Save camera configuration
GET  /video/external-camera-status - Get camera status
WS   /video/live-stream         - Live video streaming with email alerts
POST /video/upload              - Upload video with email alerts
```

---

## 📧 **Email Notification Features**

### Email Content Includes:

- **Event type** (Exit/Entrance) with colored indicators
- **Timestamp** and date information
- **Bee image** (if detected) with professional formatting
- **System status** information
- **External camera status**
- **Hive entrance coordinates**

### Email Template:

- **Professional HTML design** with responsive layout
- **Branded headers** with Queen Track logo styling
- **Color-coded status badges**
- **Embedded images** for immediate viewing
- **Detailed event information** tables

---

## 🎥 **Video System Improvements**

### Video Serving Features:

- **HTTP Range requests** support for video seeking
- **Proper Content-Type** headers
- **Browser-compatible streaming**
- **Security path validation**
- **Automatic MIME type detection**

### Video Storage Structure:

```
/data/videos/
├── outside_videos/    # External camera recordings
├── temp_videos/       # Live stream recordings
├── uploaded_videos/   # User uploaded videos
└── processed_videos/  # Processed output videos
```

### Video Access URLs:

```
http://your-server:8000/video/videos/outside_videos/video_123456.mp4
http://your-server:8000/video/videos/processed_videos/processed_stream_123456.mp4
```

---

## 📹 **External Camera System**

### Real Camera Support:

- **Automatic detection** of available cameras
- **Camera specification** extraction (resolution, FPS)
- **Real-time frame capture** and recording
- **Professional overlay** with timestamp and status

### Mock Camera Fallback:

- **Simulation mode** when no real camera available
- **Timestamp generation** for testing
- **Status information** display
- **Consistent recording format**

### Camera Configuration:

```json
{
  "internal_camera_id": "0",
  "external_camera_id": "1"
}
```

---

## 🔍 **System Monitoring**

### Health Check Response:

```json
{
  "timestamp": "2024-12-16T10:30:00",
  "overall_status": "healthy",
  "components": {
    "email_service": {
      "status": "healthy",
      "connection_test": true,
      "configured": true
    },
    "video_service": {
      "status": "healthy",
      "videos_count": 15,
      "directories": ["outside_videos", "temp_videos"]
    },
    "cameras": {
      "status": "healthy",
      "available_cameras": [
        { "id": 0, "width": 1920, "height": 1080, "fps": 30 },
        { "id": 1, "width": 640, "height": 480, "fps": 20 }
      ],
      "count": 2
    },
    "yolo_models": {
      "status": "healthy",
      "models": {
        "yolov8n.pt": { "status": "available", "size_mb": 6.2 },
        "best.pt": { "status": "available", "size_mb": 12.5 }
      }
    }
  }
}
```

---

## 📝 **Logging System**

### Log Levels:

- **INFO**: Normal operations, successful events
- **WARNING**: Non-critical issues, fallbacks
- **ERROR**: Critical errors, failed operations

### Log Locations:

- **File**: `/data/logs/queen_track.log`
- **Console**: Real-time output during development

### Log Format:

```
2024-12-16 10:30:15 - email_service - INFO - ✅ Email notification sent successfully for exit event
2024-12-16 10:30:16 - video_service - INFO - Streaming video: outside_videos/video_123456.mp4 (Range: 0-8191/2048576)
```

---

## 🔧 **Troubleshooting**

### Common Issues:

#### 1. **Email Not Sending**

```bash
# Test email connection
curl -X POST http://localhost:8000/video/test-email

# Check logs
tail -f /data/logs/queen_track.log | grep email
```

#### 2. **Video Not Playing**

```bash
# Check video service
curl http://localhost:8000/video/videos-list

# Test video access
curl -I http://localhost:8000/video/videos/outside_videos/video_123456.mp4
```

#### 3. **External Camera Issues**

```bash
# Check camera availability
curl http://localhost:8000/system/health

# Test camera configuration
curl -X POST http://localhost:8000/video/camera-config \
  -H "Content-Type: application/json" \
  -d '{"internal_camera_id": "0", "external_camera_id": "1"}'
```

#### 4. **System Health Check**

```bash
# Full system test
curl -X POST http://localhost:8000/system/test-full-system
```

---

## 🔄 **Migration from v1.0**

### Changes Made:

1. **Enhanced email system** with professional templates
2. **Improved video serving** with range support
3. **Real camera integration** with fallback mechanism
4. **Professional logging** throughout the system
5. **System monitoring** and health checks

### Breaking Changes:

- **Video URLs** now use `/video/videos/` prefix
- **Email configuration** requires all environment variables
- **Logging output** format has changed

### Migration Steps:

1. **Update environment variables** with email configuration
2. **Test email functionality** using `/video/test-email`
3. **Verify video access** using new video endpoints
4. **Check system health** using `/system/health`

---

## 📞 **Support & Testing**

### Quick Test Commands:

```bash
# 1. Test system health
curl http://localhost:8000/system/health

# 2. Test email functionality
curl -X POST http://localhost:8000/video/test-email

# 3. List available videos
curl http://localhost:8000/video/videos-list

# 4. Test video streaming
curl -I http://localhost:8000/video/videos/temp_videos/processed_stream_123456.mp4

# 5. Full system test
curl -X POST http://localhost:8000/system/test-full-system
```

### Success Indicators:

- ✅ **Email notifications** working
- ✅ **Video playback** in browser
- ✅ **External camera** recording
- ✅ **System health** all green
- ✅ **Logs** showing proper operations

---

## 🎯 **Key Benefits**

1. **Professional email alerts** with bee images
2. **Reliable video playback** in web browsers
3. **Real camera support** with automatic fallback
4. **Comprehensive monitoring** and error tracking
5. **Production-ready logging** and debugging
6. **Scalable architecture** for future enhancements

---

**Queen Track v2.0** - _Professional Bee Monitoring Solution_ 🐝
