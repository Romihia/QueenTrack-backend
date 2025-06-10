#!/bin/bash

# QueenTrack SSL Setup Script
# This script sets up SSL certificates for HTTPS support

set -e

echo "🔒 QueenTrack SSL Setup"
echo "======================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root for security reasons"
   exit 1
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "❌ .env file not found. Please create it first."
    exit 1
fi

# Check required variables
REQUIRED_VARS=("SERVER_DOMAIN" "SSL_EMAIL")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set in .env"
        exit 1
    fi
done

echo "📧 Email: $SSL_EMAIL"
echo "🌐 Domain: $SERVER_DOMAIN"

# Create SSL directory structure
echo "📁 Creating SSL directory structure..."
mkdir -p nginx/ssl/live/queentrack
mkdir -p nginx/ssl-challenge

# Option 1: Let's Encrypt (Recommended for production)
echo ""
echo "Choose SSL certificate option:"
echo "1) Let's Encrypt (Free, recommended for production)"
echo "2) Self-signed certificate (For development/testing)"
echo "3) Use existing certificates"
echo ""
read -p "Enter choice (1-3): " ssl_choice

case $ssl_choice in
    1)
        echo "🔐 Setting up Let's Encrypt certificates..."
        
        # Start nginx for HTTP challenge
        echo "🚀 Starting HTTP server for domain verification..."
        docker-compose -f docker-compose.https.yml up -d nginx
        sleep 5
        
        # Get Let's Encrypt certificate
        echo "📜 Requesting SSL certificate from Let's Encrypt..."
        docker-compose -f docker-compose.https.yml run --rm certbot \
            certonly --webroot \
            --webroot-path=/var/www/certbot \
            --email $SSL_EMAIL \
            --agree-tos \
            --no-eff-email \
            -d $SERVER_DOMAIN
        
        # Copy certificates to expected location
        if [ -d "nginx/ssl/live/$SERVER_DOMAIN" ]; then
            cp nginx/ssl/live/$SERVER_DOMAIN/fullchain.pem nginx/ssl/live/queentrack/
            cp nginx/ssl/live/$SERVER_DOMAIN/privkey.pem nginx/ssl/live/queentrack/
            echo "✅ Let's Encrypt certificates installed successfully!"
        else
            echo "❌ Certificate generation failed"
            exit 1
        fi
        ;;
    
    2)
        echo "🔐 Creating self-signed certificates..."
        
        # Generate self-signed certificate
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/live/queentrack/privkey.pem \
            -out nginx/ssl/live/queentrack/fullchain.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=$SERVER_DOMAIN"
        
        echo "✅ Self-signed certificates created successfully!"
        echo "⚠️  Note: Self-signed certificates will show security warnings in browsers"
        ;;
    
    3)
        echo "📋 Using existing certificates..."
        echo "Please ensure your certificates are located at:"
        echo "  - nginx/ssl/live/queentrack/fullchain.pem"
        echo "  - nginx/ssl/live/queentrack/privkey.pem"
        
        if [ ! -f "nginx/ssl/live/queentrack/fullchain.pem" ] || [ ! -f "nginx/ssl/live/queentrack/privkey.pem" ]; then
            echo "❌ Certificate files not found!"
            exit 1
        fi
        
        echo "✅ Using existing certificates"
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Set proper permissions
echo "🔧 Setting certificate permissions..."
chmod 644 nginx/ssl/live/queentrack/fullchain.pem
chmod 600 nginx/ssl/live/queentrack/privkey.pem

# Create certificate renewal script
echo "🔄 Creating certificate renewal script..."
cat > scripts/renew-ssl.sh << 'EOF'
#!/bin/bash
# Renew Let's Encrypt certificates

echo "🔄 Renewing SSL certificates..."
docker-compose -f docker-compose.https.yml run --rm certbot renew

# Copy renewed certificates
if [ -d "nginx/ssl/live/$SERVER_DOMAIN" ]; then
    cp nginx/ssl/live/$SERVER_DOMAIN/fullchain.pem nginx/ssl/live/queentrack/
    cp nginx/ssl/live/$SERVER_DOMAIN/privkey.pem nginx/ssl/live/queentrack/
    
    # Reload nginx
    docker-compose -f docker-compose.https.yml exec nginx nginx -s reload
    echo "✅ Certificates renewed and nginx reloaded"
else
    echo "❌ Certificate renewal failed"
    exit 1
fi
EOF

chmod +x scripts/renew-ssl.sh

# Add to crontab suggestion
echo ""
echo "✅ SSL setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Start the HTTPS-enabled services:"
echo "   docker-compose -f docker-compose.https.yml up -d"
echo ""
echo "2. Your site will be available at:"
echo "   https://$SERVER_DOMAIN"
echo ""
echo "3. For Let's Encrypt certificates, add this to your crontab for auto-renewal:"
echo "   0 12 * * * /path/to/your/project/scripts/renew-ssl.sh"
echo ""
echo "4. Test your SSL configuration at:"
echo "   https://www.ssllabs.com/ssltest/analyze.html?d=$SERVER_DOMAIN"
echo ""

# Verify certificates
echo "🔍 Certificate information:"
openssl x509 -in nginx/ssl/live/queentrack/fullchain.pem -noout -subject -dates

echo ""
echo "🎉 SSL setup completed successfully!" 