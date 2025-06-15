# Video Format Conversion System Documentation

## Overview

The Queen Track system now includes automatic video format conversion to ensure browser compatibility. Videos are recorded in mp4v format and automatically converted to avc1 (H.264) format for optimal web playback.

## Why Video Conversion is Needed

- **Browser Compatibility**: mp4v codec is not universally supported by web browsers
- **HTML5 Video**: avc1 (H.264) format ensures compatibility with HTML5 `<video>` elements
- **Streaming Performance**: avc1 format provides better streaming performance
- **File Size**: Optimized encoding reduces file sizes while maintaining quality

## System Architecture

### Components

1. **VideoFormatConverter Service** (`app/services/video_format_converter.py`)

   - Handles video conversion using FFmpeg
   - Converts mp4v to avc1 format
   - Manages conversion status and error handling

2. **VideoRecordingService** (`app/services/video_recording_service.py`)

   - Records videos during bee tracking events
   - Triggers conversion after recording ends
   - Manages video buffers and file paths

3. **Database Schema Updates** (`app/schemas/schema.py`)
   - Added fields for converted video URLs
   - Conversion status tracking
   - Error logging for failed conversions

### Workflow

```
1. Bee Event Detected
   ↓
2. Start Video Recording (mp4v format)
   ↓
3. Event Ends → Stop Recording
   ↓
4. Trigger Conversion Process (Background)
   ↓
5. Convert to avc1 Format
   ↓
6. Update Database with Converted URLs
   ↓
7. Videos Available for Browser Playback
```

## Configuration

### Prerequisites

1. **FFmpeg Installation**

   ```bash
   # Run the setup script
   ./setup_ffmpeg.sh

   # Or install manually:
   # Ubuntu/Debian: sudo apt-get install ffmpeg
   # macOS: brew install ffmpeg
   # Windows: Download from https://ffmpeg.org/
   ```

2. **Directory Structure**
   ```
   /data/videos/
   ├── events/           # Event recordings
   ├── converted/        # Converted videos (avc1)
   ├── temp_videos/      # Temporary files
   └── ...
   ```

### Environment Variables

No additional environment variables are required. The system uses the existing video directory configuration.

## API Endpoints

### Conversion Management

#### Get Conversion Status

```
GET /video/conversion-status/{event_id}
```

Returns the conversion status for a specific event.

#### Manual Conversion

```
POST /video/manual-convert/{event_id}
```

Manually trigger conversion for a specific event.

#### Batch Conversion

```
POST /video/batch-convert-events
```

Convert all events that need conversion.

#### Events Needing Conversion

```
GET /video/events-needing-conversion
```

List all events that require conversion.

### System Health

#### Conversion Service Status

```
GET /video/debug/conversion-status
```

Get detailed status of the conversion service.

#### Full System Health

```
GET /system/health
```

Includes video conversion status in the health check.

## Database Schema

### Event Fields

```python
class EventBase(BaseModel):
    # Original video URLs (mp4v format)
    internal_video_url: Optional[str] = None
    external_video_url: Optional[str] = None

    # Converted video URLs (avc1 format) - NEW
    internal_video_url_converted: Optional[str] = None
    external_video_url_converted: Optional[str] = None

    # Conversion status tracking - NEW
    conversion_status: Optional[str] = None  # "pending", "processing", "completed", "failed"
    conversion_error: Optional[str] = None   # Error message if conversion fails
```

### Conversion Status Values

- `pending`: Conversion not yet started
- `processing`: Conversion in progress
- `completed`: Conversion successful
- `failed`: Conversion failed (check `conversion_error`)

## Usage Examples

### Frontend Integration

```javascript
// Check if converted video is available
const event = await fetch(`/events/${eventId}`).then(r => r.json());

const videoUrl = event.internal_video_url_converted || event.internal_video_url;

// Use in HTML5 video element
<video controls>
    <source src={videoUrl} type="video/mp4">
    Your browser does not support the video tag.
</video>
```

### Manual Conversion

```bash
# Convert a specific event
curl -X POST "http://localhost:8000/video/manual-convert/EVENT_ID"

# Check conversion status
curl "http://localhost:8000/video/conversion-status/EVENT_ID"

# Batch convert all pending events
curl -X POST "http://localhost:8000/video/batch-convert-events"
```

## Troubleshooting

### Common Issues

#### 1. FFmpeg Not Found

```
Error: FFmpeg not available - video conversion will not work
```

**Solution**: Install FFmpeg using the setup script or manually.

#### 2. Conversion Timeout

```
Error: Video conversion timed out
```

**Solution**: Large videos may take longer. Check system resources and consider adjusting timeout in the converter.

#### 3. File Not Found

```
Error: Input video file not found
```

**Solution**: Verify the original video was recorded successfully and the file path is correct.

#### 4. Permission Issues

```
Error: Permission denied when creating converted video
```

**Solution**: Check write permissions for `/data/videos/converted/` directory.

### Debugging

#### Check Conversion Service

```bash
curl "http://localhost:8000/video/debug/conversion-status"
```

#### View System Health

```bash
curl "http://localhost:8000/system/health"
```

#### Check Logs

```bash
# Look for conversion-related log messages
grep -i "conversion\|ffmpeg" /data/logs/queen_track.log
```

## Technical Details

### FFmpeg Conversion Command

The system uses the following FFmpeg command for conversion:

```bash
ffmpeg -i input.mp4 \
  -c:v libx264 \           # H.264 codec (avc1)
  -profile:v baseline \     # Baseline profile for compatibility
  -level 3.0 \             # Level 3.0 for wide support
  -c:a aac \               # AAC audio codec
  -b:a 128k \              # Audio bitrate
  -movflags +faststart \   # Enable fast start for web
  -y output.mp4
```

### Performance Considerations

- **CPU Usage**: Video conversion is CPU-intensive
- **Background Processing**: Conversions run in separate threads
- **Timeout**: 5-minute timeout per conversion
- **Queue Management**: Small delays between batch conversions

### File Management

- **Original Files**: Kept by default for backup
- **Converted Files**: Stored in `/data/videos/converted/`
- **Naming Convention**: `{original_name}_avc1_{timestamp}.mp4`
- **Cleanup**: Optional cleanup of original files available

## Monitoring and Maintenance

### Regular Checks

1. **Disk Space**: Monitor `/data/videos/` directory size
2. **Failed Conversions**: Check for events with `conversion_status: "failed"`
3. **FFmpeg Updates**: Keep FFmpeg updated for best compatibility

### Automated Maintenance

```bash
# Find and retry failed conversions
curl "http://localhost:8000/video/events-needing-conversion"

# Batch convert pending events
curl -X POST "http://localhost:8000/video/batch-convert-events"
```

## Future Enhancements

- Queue-based conversion system for high volume
- Multiple quality/resolution outputs
- Progress tracking for long conversions
- Automatic cleanup of old original files
- Integration with cloud storage for converted videos

## Support

For issues or questions regarding the video conversion system:

1. Check the logs: `/data/logs/queen_track.log`
2. Verify FFmpeg installation: `ffmpeg -version`
3. Test conversion service: `GET /video/debug/conversion-status`
4. Review system health: `GET /system/health`
