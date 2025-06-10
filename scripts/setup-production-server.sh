#!/bin/bash

# QueenTrack Production Server Setup Script
# Run this script on your production server (162.55.53.52) to set it up for CI/CD

set -e  # Exit on any error

echo "🚀 QueenTrack Production Server Setup"
echo "====================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run this script as root (use sudo)"
    exit 1
fi

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# Install Docker Compose
echo "🐳 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed"
else
    echo "✅ Docker Compose already installed"
fi

# Install Git
echo "📂 Installing Git..."
if ! command -v git &> /dev/null; then
    apt install git -y
    echo "✅ Git installed"
else
    echo "✅ Git already installed"
fi

# Install curl
echo "🌐 Installing curl..."
if ! command -v curl &> /dev/null; then
    apt install curl -y
    echo "✅ curl installed"
else
    echo "✅ curl already installed"
fi

# Create project directory
echo "📁 Setting up project directory..."
mkdir -p /opt/queentrack
cd /opt/queentrack

# Create necessary subdirectories
mkdir -p data/videos
mkdir -p logs
mkdir -p scripts

echo "✅ Project directories created"

# Set up SSH directory (if not exists)
echo "🔑 Setting up SSH directory..."
mkdir -p /root/.ssh
chmod 700 /root/.ssh

if [ ! -f /root/.ssh/authorized_keys ]; then
    touch /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

echo "✅ SSH directory configured"

# Create environment file template
echo "⚙️ Creating environment file template..."
cat > /opt/queentrack/.env << 'EOF'
# QueenTrack Production Environment Variables
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=queentrack_prod
ENV=production

# Add your specific environment variables below:
# EXAMPLE_VAR=example_value

EOF

echo "✅ Environment file template created at /opt/queentrack/.env"

# Create log file
echo "📝 Setting up logging..."
touch /var/log/queentrack-deployment.log
chmod 644 /var/log/queentrack-deployment.log
echo "✅ Log file created"

# Enable Docker service
echo "🔧 Enabling Docker service..."
systemctl enable docker
systemctl start docker
echo "✅ Docker service enabled and started"

# Create deployment user (optional - if you want to use a specific user instead of root)
# echo "👤 Creating deployment user..."
# if ! id "queentrack" &>/dev/null; then
#     useradd -m -s /bin/bash queentrack
#     usermod -aG docker queentrack
#     echo "✅ Deployment user created"
# else
#     echo "✅ Deployment user already exists"
# fi

# Display system information
echo ""
echo "📊 System Information:"
echo "======================"
echo "🐳 Docker version: $(docker --version)"
echo "🐳 Docker Compose version: $(docker-compose --version)"
echo "📂 Git version: $(git --version)"
echo "🌐 curl version: $(curl --version | head -1)"
echo "💾 Available disk space:"
df -h /opt/queentrack
echo ""

# Final instructions
echo "🎉 Production server setup completed!"
echo ""
echo "📝 Next steps:"
echo "1. Add your SSH public key to /root/.ssh/authorized_keys"
echo "2. Configure environment variables in /opt/queentrack/.env"
echo "3. Clone your repository manually once for initial setup:"
echo "   cd /opt/queentrack"
echo "   git clone https://github.com/YOUR_USERNAME/QueenTrack-backend.git ."
echo "4. Set up GitHub Secrets with your SSH private key"
echo "5. Push to main branch to trigger automated deployment!"
echo ""
echo "🔗 For detailed instructions, see DEPLOYMENT_GUIDE.md"

# Test Docker installation
echo "🧪 Testing Docker installation..."
if docker run --rm hello-world > /dev/null 2>&1; then
    echo "✅ Docker test passed"
else
    echo "⚠️ Docker test failed - you may need to restart the server"
fi

echo ""
echo "🏁 Setup script completed successfully!"
echo "Server is ready for CI/CD deployment! 🚀" 