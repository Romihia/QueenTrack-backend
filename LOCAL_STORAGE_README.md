# Local Video Storage Implementation

This document describes the transition from AWS S3 to local video storage for the Queen Track application.

## Overview

The application now stores all videos locally in the `/data/videos/` directory instead of uploading to AWS S3. This provides:

- Faster video access and serving
- No dependency on external cloud services
- Reduced costs
- Simplified deployment

## Directory Structure

```
/data/videos/
├── outside_videos/     # Videos recorded by external camera when bee exits
├── temp_videos/        # Temporary videos from live streaming
├── uploaded_videos/    # Original uploaded video files
└── processed_videos/   # Processed versions of uploaded videos
```

## Key Changes Made

### Backend Changes

1. **FastAPI Static Files**: Added static file serving for videos
   ```python
   app.mount("/videos", StaticFiles(directory="/data/videos"), name="videos")
   ```

2. **Removed AWS Dependencies**: 
   - Removed `boto3` from requirements.txt
   - Removed AWS configuration from config.py
   - Replaced `upload_to_s3()` with `save_video_locally()`

3. **Updated Video Paths**: All video processing functions now use `/data/videos/` as the base directory

4. **Local URL Generation**: Videos are now accessible via `/videos/{filename}` endpoints

### Frontend Changes

1. **Updated TrackPage**: Added helper function to construct full video URLs from relative paths
2. **Fixed Upload Form**: Changed form field name from 'videoFile' to 'file' to match backend

### Docker Configuration

1. **Volume Mount**: Added volume mount in docker-compose.yml:
   ```yaml
   volumes:
     - ./data/videos:/data/videos
   ```

## Video URL Format

Videos are now accessible via:
- **Relative URL**: `/videos/outside_videos/video_outside_1234567890.mp4`
- **Full URL**: `http://localhost:8000/videos/outside_videos/video_outside_1234567890.mp4`

## Database Schema

The MongoDB events collection now stores local video URLs:
```json
{
  "id": "event_id",
  "time_out": "2024-01-01T10:00:00Z",
  "time_in": "2024-01-01T10:30:00Z",
  "video_url": "/videos/outside_videos/video_outside_1234567890.mp4"
}
```

## Testing the Implementation

1. **Start the backend**:
   ```bash
   cd QueenTrack-backend
   docker-compose up
   ```

2. **Test video serving**: Visit `http://localhost:8000/videos/` to see if static files are served

3. **Test live streaming**: Use the frontend to start camera streaming and verify videos are saved locally

4. **Test video upload**: Upload a video via the frontend and verify it's processed and saved locally

5. **Test event tracking**: Verify that bee tracking events create videos in the correct directories

## Monitoring and Maintenance

- **Disk Usage**: Monitor the `/data/videos/` directory for disk usage
- **Video Cleanup**: Consider implementing a cleanup policy for old videos
- **Backup**: Ensure the `/data/videos/` directory is included in backup procedures

## Troubleshooting

### Videos Not Accessible
- Check that the `/data/videos/` directory exists and has proper permissions
- Verify the Docker volume mount is working correctly
- Check FastAPI logs for static file serving errors

### Upload Failures
- Verify the upload directories exist and are writable
- Check available disk space
- Review backend logs for processing errors

### Database Issues
- Ensure video URLs in the database start with `/videos/`
- Check that the frontend is constructing full URLs correctly 