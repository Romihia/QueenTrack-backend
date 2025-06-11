# 🚀 Queen Track v2.0 - Quick Start Guide

## ⚡ **Immediate Fix Applied**

The logging directory issue has been resolved! The system now:

- ✅ Creates `/data/logs` directory before logging initialization
- ✅ Has fallback to console-only logging if file logging fails
- ✅ Shows clear startup messages

---

## 🏃‍♂️ **Quick Start Steps**

### 🐳 **Docker Method (Recommended)**

```bash
# 1. Start containers
./docker_manager.sh start

# 2. Run Docker startup check inside container
./docker_manager.sh check

# 3. Monitor logs
./docker_manager.sh logs backend

# 4. Access container shell (if needed)
./docker_manager.sh shell
```

### 💻 **Local Development Method**

### 1. **Restart the Backend** (if needed)

```bash
# If the backend crashed, restart it:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. **Run Startup Check** (Recommended)

```bash
# Run the automated system check:
python3 startup_check.py
```

### 3. **Verify Everything Works**

```bash
# Test the main endpoints:
curl http://localhost:8000/                    # Root endpoint
curl http://localhost:8000/system/health       # System health
curl -X POST http://localhost:8000/video/test-email  # Email test
```

---

## 🐛 **If You Still See Errors**

### **Permission Issues:**

```bash
# Fix permissions for data directory:
sudo chmod -R 755 /data/
sudo chown -R $USER:$USER /data/

# Or create in home directory instead:
mkdir -p ~/queen-track-data/{logs,videos}
# Then update paths in code to use ~/queen-track-data/
```

### **Docker Environment:**

```bash
# For Docker issues, use the Docker manager:
./docker_manager.sh restart        # Restart containers
./docker_manager.sh clean          # Clean restart (removes data)
./docker_manager.sh logs backend   # Check logs
./docker_manager.sh health         # Quick health check

# Manual Docker commands:
docker-compose down
docker-compose up -d --build
docker-compose logs -f backend
```

### **Alternative Logging Setup:**

If file logging still fails, you can disable it temporarily:

```python
# In app/main.py, replace the logging setup with:
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Console only
)
```

---

## ✅ **Expected Startup Messages**

### 🐳 **Docker Messages:**

```
Creating network "queentrack-backend_default" with the default driver
Creating bee_backend ... done
✅ Containers started
ℹ️  Waiting for containers to initialize...
```

### 💻 **Application Messages:**

You should see:

```
INFO: Uvicorn running on http://0.0.0.0:8000
INFO:app.services.email_service:Email service initialized. Sending from: your@email.com
INFO:app.services.video_service:Video service initialized. Base directory: /data/videos
INFO:app.main:🚀 Queen Track Backend v2.0 initializing...
INFO:app.main:✅ All routers loaded successfully
INFO:app.main:📧 Email service ready
INFO:app.main:🎥 Video service ready
INFO:app.main:🔍 System monitoring ready
INFO:app.main:🐝 Queen Track Backend v2.0 ready to serve!
```

---

## 🧪 **Quick Test Commands**

```bash
# 1. System health
curl http://localhost:8000/system/health

# 2. Email test (will send actual email if configured)
curl -X POST http://localhost:8000/video/test-email

# 3. Video list
curl http://localhost:8000/video/videos-list

# 4. Camera status
curl http://localhost:8000/video/external-camera-status

# 5. Complete system test
curl -X POST http://localhost:8000/system/test-full-system
```

---

## 📁 **Frontend Setup**

1. **Add Sample Video:**

   ```bash
   # Place your test video in the frontend:
   cp your-bee-video.mp4 queen-track-frontend/public/sample-videos/sample-hive-video.mp4
   ```

2. **Start Frontend:**

   ```bash
   cd queen-track-frontend
   npm start
   # Opens on http://localhost:3000
   ```

3. **Test the System:**
   - Select "שידור קובץ וידאו לדוגמה" (Sample Video Broadcasting)
   - Click "התחל שידור וידאו" (Start Video Broadcasting)
   - Watch for bee detection and email notifications

---

## 🔧 **Environment Variables Check**

Make sure your `.env` file has:

```env
EMAIL_USER=your_gmail@gmail.com
EMAIL_PASS=your_app_password_not_regular_password
SEND_EMAIL=recipient@email.com

# Other existing variables...
MONGO_URI=mongodb+srv://...
AWS_ACCESS_KEY_ID=...
# etc.
```

---

## 📧 **Email Configuration**

For Gmail (most common):

1. **Enable 2-factor authentication**
2. **Generate app password:** Google Account → Security → App passwords
3. **Use app password** (not your regular password) in `EMAIL_PASS`

---

## 🆘 **Still Having Issues?**

### **Check Logs:**

```bash
# Backend logs
tail -f /data/logs/queen_track.log

# Or if file logging failed, check console output
```

### **Check Process:**

```bash
# See if backend is running
ps aux | grep uvicorn

# Check port usage
netstat -tulpn | grep :8000
```

### **Reset Everything:**

```bash
# Clean restart
pkill -f uvicorn
rm -rf /data/logs/* /data/videos/*
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🎯 **Success Indicators**

✅ **Backend starts without errors**  
✅ **All API endpoints respond**  
✅ **Email test works**  
✅ **Video streaming starts**  
✅ **System health shows all green**

When all these work → **You're ready to monitor bees!** 🐝

---

## 🐳 **Docker Quick Reference**

```bash
# Essential Docker commands:
./docker_manager.sh start          # Start everything
./docker_manager.sh check          # Run full system check
./docker_manager.sh logs backend   # Monitor backend logs
./docker_manager.sh shell          # Enter container shell
./docker_manager.sh stop           # Stop everything
./docker_manager.sh help           # See all commands

# Direct Docker commands:
docker-compose ps                   # See container status
docker-compose exec backend python3 docker_startup_check.py
docker-compose logs -f backend      # Follow logs
docker stats                        # See resource usage
```

---

**Need help?** Check the full documentation in `SYSTEM_UPGRADE_README.md`
