# QueenTrack Backend - Deployment Setup Guide

This guide will help you set up automatic deployment from GitHub to your production server.

## 🔐 GitHub Secrets Setup

You need to configure the following secrets in your GitHub repository:

### How to Add Secrets:

1. Go to your GitHub repository
2. Click on **Settings** tab
3. Click on **Secrets and variables** → **Actions**
4. Click **New repository secret**

### Required Secrets:

| Secret Name          | Description                       | Example Value           |
| -------------------- | --------------------------------- | ----------------------- |
| `PRODUCTION_HOST`    | Your server's IP address          | `162.55.53.52`          |
| `PRODUCTION_USER`    | SSH username for your server      | `root`                  |
| `PRODUCTION_SSH_KEY` | Private SSH key for server access | See SSH Key Setup below |

## 🗝️ SSH Key Setup

### 1. Generate SSH Key (if you don't have one):

```bash
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
```

### 2. Copy Public Key to Server:

```bash
ssh-copy-id your-username@your-server-ip
# Or manually add to ~/.ssh/authorized_keys on the server
```

### 3. Get Private Key for GitHub Secret:

```bash
cat ~/.ssh/id_rsa
```

Copy the entire output (including `-----BEGIN OPENSSH PRIVATE KEY-----` and `-----END OPENSSH PRIVATE KEY-----`) and paste it as `PRODUCTION_SSH_KEY` secret.

## 🐳 Server Preparation

### 1. Install Docker and Docker Compose on your server:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
```

### 2. Create project directory:

```bash
sudo mkdir -p /opt/queentrack
sudo chown $USER:$USER /opt/queentrack
```

### 3. Install Git:

```bash
sudo apt install git -y
```

## 🚀 Deployment Process

The deployment happens automatically when you push to the `main` branch:

1. **Stage Branch**: Push to `stage` branch runs tests only
2. **Main Branch**: Push to `main` branch runs tests AND deploys to production

### Manual Deployment:

If you need to deploy manually, SSH to your server and run:

```bash
cd /opt/queentrack
./deploy.sh
```

## 📝 Environment Configuration

The production environment is configured in `.env.production` file, which includes:

- Database connection (MongoDB Atlas)
- Email configuration for notifications
- AI model settings
- Performance optimizations
- Security settings

**Note**: AWS services are excluded from production environment as requested.

## 🔍 Monitoring

### Check Service Status:

```bash
cd /opt/queentrack
docker-compose -f docker-compose.prod.yml ps
```

### View Logs:

```bash
docker-compose -f docker-compose.prod.yml logs
```

### Health Check:

```bash
curl http://your-server-ip:8000/health
```

## 🏗️ Architecture

```
GitHub Repository (main branch)
         ↓ (push trigger)
    GitHub Actions
         ↓ (SSH deployment)
    Production Server
         ↓
    Docker Compose → QueenTrack Backend Container
         ↓
    Port 8000 (accessible via internet)
```

## 🛠️ Troubleshooting

### Common Issues:

1. **SSH Connection Failed**:

   - Check if `PRODUCTION_SSH_KEY` secret is correct
   - Verify server IP and username
   - Ensure SSH access is enabled on server

2. **Docker Build Failed**:

   - Check server disk space: `df -h`
   - Clean up old images: `docker system prune -a`

3. **Port 8000 Not Accessible**:

   - Check firewall: `sudo ufw allow 8000`
   - Verify container is running: `docker ps`

4. **Database Connection Issues**:
   - Verify MongoDB connection string in `.env.production`
   - Check if MongoDB Atlas allows connections from server IP

### Logs Location:

- GitHub Actions logs: Available in your repository's Actions tab
- Container logs: `docker-compose -f docker-compose.prod.yml logs`

## 📋 Deployment Checklist

- [ ] GitHub secrets configured
- [ ] SSH key setup complete
- [ ] Docker installed on server
- [ ] Project directory created
- [ ] Firewall configured (port 8000 open)
- [ ] MongoDB connection tested
- [ ] First deployment successful

## 🎯 Production URL

After successful deployment, your QueenTrack backend will be available at:
`http://YOUR_SERVER_IP:8000`

Health check endpoint: `http://YOUR_SERVER_IP:8000/health`
