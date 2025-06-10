# 🚀 CI/CD Quick Reference

## 📋 What's Been Set Up

### ✅ Files Created/Modified:
- `.github/workflows/ci.yml` - Updated CI/CD pipeline
- `.github/scripts/deploy.sh` - Deployment script  
- `docker-compose.prod.yml` - Production Docker config
- `scripts/setup-production-server.sh` - Server setup script
- `DEPLOYMENT_GUIDE.md` - Comprehensive setup guide
- `CI_CD_SUMMARY.md` - This summary file

### ✅ What the Pipeline Does:
1. **Push to `stage`** → Runs tests only
2. **Push to `main`** → Runs tests + automatic deployment

## 🔧 Quick Setup Checklist

### 🎯 GitHub Secrets (Required)
Go to: **Repository → Settings → Secrets and variables → Actions**

Add these secrets:
```
PRODUCTION_HOST=162.55.53.52
PRODUCTION_USER=root  
PRODUCTION_SSH_KEY=[Your private SSH key content]
```

### 🖥️ Server Setup Commands
```bash
# 1. Connect to server
ssh root@162.55.53.52

# 2. Download and run setup script
curl -o setup.sh https://raw.githubusercontent.com/YOUR_USERNAME/QueenTrack-backend/main/scripts/setup-production-server.sh
chmod +x setup.sh
sudo ./setup.sh

# 3. Add your SSH public key
echo "YOUR_SSH_PUBLIC_KEY" >> ~/.ssh/authorized_keys

# 4. Clone repository
cd /opt/queentrack
git clone https://github.com/YOUR_USERNAME/QueenTrack-backend.git .

# 5. Configure environment
nano .env  # Add your environment variables
```

### 🔑 SSH Key Generation
```bash
# Generate key pair
ssh-keygen -t rsa -b 4096 -C "github-actions-queentrack"

# Copy private key to GitHub Secret
cat ~/.ssh/id_rsa

# Copy public key to server
cat ~/.ssh/id_rsa.pub
```

## 🚀 How to Deploy

### Automatic Deployment
```bash
git add .
git commit -m "Your changes"
git push origin main  # This triggers automatic deployment!
```

### Manual Deployment (if needed)
```bash
ssh root@162.55.53.52
cd /opt/queentrack
git pull origin main
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📊 Monitoring

### Check Deployment Status
- **GitHub**: Repository → Actions tab
- **Server Logs**: `tail -f /var/log/queentrack-deployment.log`
- **Container Status**: `docker-compose ps`
- **Service URL**: http://162.55.53.52:8000

### Common Commands
```bash
# Check containers
docker-compose ps

# View logs  
docker-compose logs -f backend

# Restart service
docker-compose restart backend

# Full rebuild
docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

## 🔍 Troubleshooting

### Pipeline Fails
1. Check GitHub Actions logs
2. Verify GitHub Secrets are correct
3. Test SSH connection manually

### Deployment Fails
1. SSH to server: `ssh root@162.55.53.52`
2. Check logs: `tail -f /var/log/queentrack-deployment.log`
3. Check Docker: `docker-compose ps`
4. Manual deploy to test

### SSH Issues
```bash
# Test SSH connection
ssh -i ~/.ssh/id_rsa root@162.55.53.52

# Check authorized_keys on server
cat ~/.ssh/authorized_keys
```

## 🎯 Success Indicators

✅ **Working CI/CD:**
- Green checkmark in GitHub Actions
- Container shows "Up" status
- Service responds at http://162.55.53.52:8000
- No errors in deployment logs

## 📱 Quick Commands Reference

```bash
# Server connection
ssh root@162.55.53.52

# Project directory
cd /opt/queentrack

# View deployment logs
tail -f /var/log/queentrack-deployment.log

# Container management
docker-compose ps              # Check status
docker-compose logs -f backend # View logs
docker-compose restart backend # Restart
docker-compose down            # Stop all
docker-compose up -d           # Start all

# Git operations
git status
git pull origin main
git log --oneline -5
```

---

🎉 **That's it!** Push to `main` and watch your code automatically deploy to production! 