import smtplib
import os
import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
from typing import Optional
import cv2
import numpy as np
from app.services.timezone_service import get_timezone_service

# Ensure logs directory exists
os.makedirs('/data/logs', exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)

class EmailService:
    """
    Professional email service for Queen Track notifications with dynamic settings
    """
    
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_user = os.getenv("EMAIL_USER")
        self.email_pass = os.getenv("EMAIL_PASS")
        self.default_email = os.getenv("SEND_EMAIL")  # Fallback email
        
        # Dynamic settings
        self.notifications_enabled = False
        self.notification_email = ""
        self.email_on_exit = True
        self.email_on_entrance = True
        
        # Timezone service for local time
        try:
            self.timezone_service = get_timezone_service()
        except Exception as e:
            logger.warning(f"⚠️ Timezone service not available, using datetime.now(): {e}")
            self.timezone_service = None
        
        if not all([self.email_user, self.email_pass]):
            logger.error("Email authentication missing. Please check EMAIL_USER and EMAIL_PASS environment variables.")
            raise ValueError("Email authentication incomplete")
        
        logger.info(f"Email service initialized. Sending from: {self.email_user}")
        logger.info(f"📧 Email service initial settings: enabled={self.notifications_enabled}, recipient={self.get_recipient_email()}")
    
    async def load_settings_from_database(self):
        """Load email settings from database on startup"""
        try:
            from app.services.settings_service import get_settings_service
            settings_service = await get_settings_service()
            settings = await settings_service.get_current_settings()
            
            if "processing" in settings:
                logger.info(f"🔧 Debug - Raw processing settings from DB: {settings['processing']}")
                
                email_config = {
                    "notifications_enabled": settings["processing"].get("email_notifications_enabled", False),
                    "notification_email": settings["processing"].get("notification_email", ""),
                    "email_on_exit": settings["processing"].get("email_on_exit", True),
                    "email_on_entrance": settings["processing"].get("email_on_entrance", True)
                }
                
                logger.info(f"🔧 Debug - Email config extracted: {email_config}")
                self.update_settings(email_config)
                logger.info("✅ Email service loaded settings from database")
            
        except Exception as e:
            logger.error(f"❌ Failed to load email settings from database: {e}")
    
    def update_settings(self, email_config: dict):
        """Update email settings dynamically"""
        try:
            self.notifications_enabled = email_config.get("notifications_enabled", False)
            self.notification_email = email_config.get("notification_email", "")
            self.email_on_exit = email_config.get("email_on_exit", True)
            self.email_on_entrance = email_config.get("email_on_entrance", True)
            
            # Use user-defined email or fallback to default
            effective_email = self.notification_email or self.default_email
            
            logger.info(f"📧 Email settings updated: enabled={self.notifications_enabled}, email={effective_email}")
            
        except Exception as e:
            logger.error(f"❌ Error updating email settings: {e}")
    
    def get_recipient_email(self) -> str:
        """Get the current recipient email address"""
        return self.notification_email or self.default_email or ""
    
    def is_notifications_enabled(self) -> bool:
        """Check if email notifications are enabled"""
        return self.notifications_enabled and bool(self.get_recipient_email())

    def create_bee_detection_email(self, 
                                 event_type: str, 
                                 timestamp: datetime, 
                                 bee_image: Optional[np.ndarray] = None,
                                 additional_info: dict = None) -> MIMEMultipart:
        """
        Create formatted email for bee detection events
        
        Args:
            event_type: 'exit' or 'entrance'
            timestamp: When the event occurred
            bee_image: OpenCV image array of the detected bee
            additional_info: Additional event information
        
        Returns:
            MIMEMultipart: Formatted email message
        """
        
        msg = MIMEMultipart('related')
        
        # Email headers
        recipient_email = self.get_recipient_email()
        msg['From'] = self.email_user
        msg['To'] = recipient_email
        
        # Convert timestamp to local time for display
        if self.timezone_service:
            local_timestamp = self.timezone_service.to_local_time(timestamp)
        else:
            # Fallback: add 3 hours manually
            local_timestamp = timestamp + timedelta(hours=3)
        
        if event_type == 'exit':
            msg['Subject'] = f"🐝 Queen Track Alert: Marked Bee Has Left the Hive - {local_timestamp.strftime('%H:%M:%S')}"
            event_title = "Bee Exit Detected"
            event_description = "The marked bee has left the hive and is now outside."
            event_color = "#FF6B35"  # Orange
        else:
            msg['Subject'] = f"🏠 Queen Track Alert: Marked Bee Has Returned - {local_timestamp.strftime('%H:%M:%S')}"
            event_title = "Bee Return Detected"
            event_description = "The marked bee has returned to the hive."
            event_color = "#4CAF50"  # Green
        
        # HTML email body
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, {event_color}, #FFD700); color: white; padding: 20px; border-radius: 10px 10px 0 0; text-align: center; }}
                .content {{ padding: 20px; }}
                .event-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }}
                .timestamp {{ font-size: 18px; font-weight: bold; color: {event_color}; }}
                .bee-image {{ text-align: center; margin: 20px 0; }}
                .footer {{ background-color: #333; color: white; padding: 15px; text-align: center; border-radius: 0 0 10px 10px; }}
                .status-badge {{ background-color: {event_color}; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🐝 Queen Track System</h1>
                    <h2>{event_title}</h2>
                </div>
                
                <div class="content">
                    <div class="event-info">
                        <h3>📊 Event Details</h3>
                        <p><strong>Event Type:</strong> <span class="status-badge">{event_type.upper()}</span></p>
                        <p><strong>Date:</strong> {local_timestamp.strftime('%Y-%m-%d')}</p>
                        <p><strong>Time:</strong> <span class="timestamp">{local_timestamp.strftime('%H:%M:%S')}</span></p>
                        <p><strong>Description:</strong> {event_description}</p>
                    </div>
                    
                    {'<div class="bee-image"><h3>📸 Detected Bee Image</h3><img src="cid:bee_image" style="max-width: 400px; border: 2px solid ' + event_color + '; border-radius: 8px;"></div>' if bee_image is not None else ''}
                    
                    <div class="event-info">
                        <h3>ℹ️ Additional Information</h3>
                        <p><strong>System Status:</strong> Active Monitoring</p>
                        <p><strong>ROI Position:</strong> Hive Entrance</p>
                        {'<p><strong>External Camera:</strong> ' + ('Recording' if additional_info and additional_info.get('external_camera_active') else 'Standby') + '</p>' if additional_info else ''}
                    </div>
                </div>
                
                <div class="footer">
                    <p>This is an automated message from Queen Track System</p>
                    <p>Generated at: {(self.timezone_service.get_local_now() if self.timezone_service else datetime.now() + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Attach HTML body
        msg.attach(MIMEText(html_body, 'html'))
        
        # Attach bee image if provided
        if bee_image is not None:
            try:
                # Convert OpenCV image to JPEG bytes
                _, img_encoded = cv2.imencode('.jpg', bee_image)
                img_bytes = img_encoded.tobytes()
                
                # Create image attachment
                img_attachment = MIMEImage(img_bytes)
                img_attachment.add_header('Content-ID', '<bee_image>')
                img_attachment.add_header('Content-Disposition', 'inline', filename='detected_bee.jpg')
                msg.attach(img_attachment)
                
                logger.info("Bee image attached to email")
            except Exception as e:
                logger.error(f"Failed to attach bee image: {e}")
        
        return msg

    def send_bee_detection_notification(self, 
                                      event_type: str, 
                                      timestamp: datetime, 
                                      bee_image: Optional[np.ndarray] = None,
                                      additional_info: dict = None) -> bool:
        """
        Send bee detection notification email
        
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        try:
            # Check if notifications are enabled and configured
            if not self.is_notifications_enabled():
                logger.info(f"📧 Email notifications disabled or no recipient configured - skipping {event_type} notification")
                return False
            
            # Check event-specific settings
            if event_type == 'exit' and not self.email_on_exit:
                logger.info(f"📧 Exit email notifications disabled - skipping {event_type} notification")
                return False
            elif event_type == 'entrance' and not self.email_on_entrance:
                logger.info(f"📧 Entrance email notifications disabled - skipping {event_type} notification")
                return False
            # Create email message
            msg = self.create_bee_detection_email(event_type, timestamp, bee_image, additional_info)
            
            # Connect to SMTP server and send email
            logger.info(f"Connecting to SMTP server: {self.smtp_server}:{self.smtp_port}")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_pass)
                
                text = msg.as_string()
                recipient_email = self.get_recipient_email()
                server.sendmail(self.email_user, recipient_email, text)
                
            logger.info(f"✅ Email notification sent successfully for {event_type} event at {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email notification: {e}")
            return False

    def test_email_connection(self) -> bool:
        """
        Test email connection and configuration
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Testing email connection...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_pass)
            
            logger.info("✅ Email connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Email connection test failed: {e}")
            return False

# Create singleton instance
email_service = EmailService()

async def update_email_settings(email_config: dict):
    """Update the global email service settings"""
    try:
        email_service.update_settings(email_config)
        logger.info("🔄 Global email service settings updated")
    except Exception as e:
        logger.error(f"❌ Error updating global email service settings: {e}")

def get_email_service() -> EmailService:
    """Get the global email service instance"""
    return email_service 