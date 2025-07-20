#!/usr/bin/env python3
"""
Simple WebSocket client to test the live-stream endpoint
"""
import asyncio
import websockets
import json
import sys

async def test_websocket():
    uri = "ws://localhost:8000/video/live-stream"
    
    print(f"Attempting to connect to {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("WebSocket connection established!")
            
            # Wait for the initial ping message
            print("Waiting for server messages...")
            
            try:
                # Listen for messages for 5 seconds
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"Received message: {message}")
                
                # Try to parse as JSON
                try:
                    data = json.loads(message)
                    print(f"Parsed JSON: {json.dumps(data, indent=2)}")
                except json.JSONDecodeError:
                    print(f"Raw message: {message}")
                
            except asyncio.TimeoutError:
                print("No messages received within 5 seconds")
            
            print("Connection test completed successfully!")
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection was closed: code={e.code}, reason={e.reason}")
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"Invalid status code: {e.status_code}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    print("WebSocket Connection Test")
    print("=" * 40)
    asyncio.run(test_websocket())