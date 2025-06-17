import smtplib
import os
import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from typing import Optional
import cv2
import numpy as np

# Ensure logs directory exists
os.makedirs('/data/logs', exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)

class EmailService:
    """
    Professional email service for Queen Track notifications
    """
    
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_user = os.getenv("EMAIL_USER")
        self.email_pass = os.getenv("EMAIL_PASS")
        self.send_to_email = os.getenv("SEND_EMAIL")
        
        if not all([self.email_user, self.email_pass, self.send_to_email]):
            logger.error("Email configuration missing. Please check environment variables.")
            raise ValueError("Email configuration incomplete")
        
        logger.info(f"Email service initialized. Sending from: {self.email_user} to: {self.send_to_email}")

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
        msg['From'] = self.email_user
        msg['To'] = self.send_to_email
        
        if event_type == 'exit':
            msg['Subject'] = f"🐝 Queen Track Alert: Marked Bee Has Left the Hive - {timestamp.strftime('%H:%M:%S')}"
            event_title = "Bee Exit Detected"
            event_description = "The marked bee has left the hive and is now outside."
            event_color = "#FF6B35"  # Orange
        else:
            msg['Subject'] = f"🏠 Queen Track Alert: Marked Bee Has Returned - {timestamp.strftime('%H:%M:%S')}"
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
                        <p><strong>Date:</strong> {timestamp.strftime('%Y-%m-%d')}</p>
                        <p><strong>Time:</strong> <span class="timestamp">{timestamp.strftime('%H:%M:%S')}</span></p>
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
                    <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
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
            # Create email message
            msg = self.create_bee_detection_email(event_type, timestamp, bee_image, additional_info)
            
            # Connect to SMTP server and send email
            logger.info(f"Connecting to SMTP server: {self.smtp_server}:{self.smtp_port}")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_pass)
                
                text = msg.as_string()
                server.sendmail(self.email_user, self.send_to_email, text)
                
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