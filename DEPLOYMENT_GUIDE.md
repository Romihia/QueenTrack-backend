# 🚀 QueenTrack CI/CD Deployment Guide

## 📋 Overview

This guide will help you set up fully automated CI/CD deployment for QueenTrack from GitHub to your production server.

## 🎯 What You'll Achieve

- **Push to `main`** → **Automatic tests** → **Automatic deployment** → **Production ready!**
- No more manual SSH, docker-compose build, or manual deployment steps
- Automatic rollback on deployment failures
- Health checks and deployment verification

## 🔧 Setup Steps

### Step 1: Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following repository secrets:

```
PRODUCTION_HOST=162.55.53.52
PRODUCTION_USER=root
PRODUCTION_SSH_KEY=[Your private SSH key - see below]
```

#### 🔑 Generate SSH Key Pair

On your local machine, run:

```bash
# Generate SSH key pair
ssh-keygen -t rsa -b 4096 -c "github-actions-queentrack"

# This will create:
# ~/.ssh/id_rsa (private key) - Copy this to PRODUCTION_SSH_KEY secret
# ~/.ssh/id_rsa.pub (public key) - Copy this to server's authorized_keys
```

**Add private key to GitHub Secrets:**
```bash
cat ~/.ssh/id_rsa
# Copy the entire output (including -----BEGIN/END----- lines) to PRODUCTION_SSH_KEY secret
```

### Step 2: Setup Production Server

#### 2.1 Connect to your server:
```bash
ssh root@162.55.53.52
```

#### 2.2 Install required dependencies:
```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Git
apt install git -y

# Install curl (for health checks)
apt install curl -y
```

#### 2.3 Setup SSH access for GitHub Actions:
```bash
# Add the public key to authorized_keys
mkdir -p ~/.ssh
echo "YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

#### 2.4 Setup project directory:
```bash
# Create project directory
mkdir -p /opt/queentrack
cd /opt/queentrack

# Clone your repository (replace with your actual repo URL)
git clone https://github.com/YOUR_USERNAME/QueenTrack-backend.git .

# Create necessary directories
mkdir -p data/videos
mkdir -p logs

# Create environment file
nano .env
```

#### 2.5 Configure environment variables in `.env`:
```env
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=queentrack_prod
ENV=production

# Add any other environment variables your app needs
```

#### 2.6 Test manual deployment:
```bash
# Build and run to test everything works
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d

# Check if it's running
docker-compose ps
curl http://localhost:8000/health || curl http://localhost:8000/
```

### Step 3: Update Deployment Script

In the `.github/scripts/deploy.sh` file, update the GitHub repository URL:

```bash
# Find this line in deploy.sh:
git remote add origin https://github.com/YOUR_USERNAME/QueenTrack-backend.git

# Replace YOUR_USERNAME with your actual GitHub username
```

### Step 4: Test the CI/CD Pipeline

1. **Push to `stage` branch** (optional - for testing):
   ```bash
   git checkout -b stage
   git push origin stage
   # This will run tests but not deploy
   ```

2. **Push to `main` branch**:
   ```bash
   git checkout main
   git add .
   git commit -m "Setup automated CI/CD deployment"
   git push origin main
   # This will run tests AND deploy to production automatically!
   ```

## 📊 Monitoring Deployment

### GitHub Actions Logs
- Go to your repository → Actions tab
- Click on the latest workflow run
- Monitor the deployment progress in real-time

### Production Server Logs
```bash
# SSH to your server
ssh root@162.55.53.52

# Check deployment logs
tail -f /var/log/queentrack-deployment.log

# Check Docker containers
docker-compose ps

# Check application logs
docker-compose logs -f backend
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. SSH Connection Failed
```bash
# Verify SSH key is properly added to server
ssh -i ~/.ssh/id_rsa root@162.55.53.52

# Check if key is in authorized_keys
cat ~/.ssh/authorized_keys
```

#### 2. Docker Build Failed
```bash
# SSH to server and check manually
cd /opt/queentrack
docker-compose build --no-cache
docker-compose logs
```

#### 3. Port 8000 Already in Use
```bash
# Check what's using port 8000
sudo lsof -i :8000

# Stop existing containers
docker-compose down
```

#### 4. Permission Issues
```bash
# Fix directory permissions
chown -R root:root /opt/queentrack
chmod -R 755 /opt/queentrack
```

## 🎉 Success Indicators

✅ **Successful deployment will show:**
- Green checkmark in GitHub Actions
- Container status shows "Up" 
- Service responds to HTTP requests
- No errors in deployment logs

✅ **Your service will be available at:**
- http://162.55.53.52:8000

## 🛡️ Security Notes

- SSH keys are securely stored in GitHub Secrets
- Production server only accepts key-based SSH authentication
- Docker containers run with appropriate user privileges
- Environment variables are kept in separate `.env` file

## 🔄 Deployment Process Flow

```
1. Push to main branch
2. GitHub Actions triggers
3. Code checkout and testing
4. Docker image build and test
5. SSH connection to production server
6. Code pull from GitHub
7. Docker containers stop
8. New Docker image build
9. New containers start
10. Health checks
11. Deployment success notification
```

## 📞 Support

If you encounter any issues:
1. Check GitHub Actions logs first
2. Check production server logs: `/var/log/queentrack-deployment.log`
3. Verify all secrets are properly configured
4. Test SSH connection manually

---

🎉 **Congratulations!** You now have fully automated CI/CD deployment for QueenTrack! 