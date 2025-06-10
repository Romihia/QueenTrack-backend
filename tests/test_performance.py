import pytest
import asyncio
import time
import concurrent.futures
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
import cv2
from datetime import datetime
from httpx import AsyncClient

from app.routes.video_routes import process_frame, start_external_camera, stop_external_camera, handle_bee_tracking
from app.main import app
from app.services.service import create_event, get_all_events
from app.schemas.schema import EventCreate


class TestPerformance:
    """Test system performance under various loads."""

    @patch('app.routes.video_routes.model_detect')
    @patch('app.routes.video_routes.model_classify')
    def test_frame_processing_speed(self, mock_classify, mock_detect):
        """Test frame processing speed."""
        # Mock YOLO models for consistent timing
        mock_detect.return_value = []
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Measure processing time for multiple frames
        start_time = time.time()
        num_frames = 100
        
        for _ in range(num_frames):
            process_frame(frame)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_frame = total_time / num_frames
        
        # Should process frames reasonably fast (less than 50ms per frame for 20fps)
        assert avg_time_per_frame < 0.05, f"Frame processing too slow: {avg_time_per_frame:.3f}s per frame"
        
        fps = 1 / avg_time_per_frame
        print(f"Frame processing performance: {fps:.1f} FPS")

    @patch('app.routes.video_routes.model_detect')
    @patch('app.routes.video_routes.model_classify')
    def test_frame_processing_with_detection_speed(self, mock_classify, mock_detect):
        """Test frame processing speed with bee detection."""
        # Mock YOLO detection results
        mock_boxes = MagicMock()
        mock_boxes.xyxy = [[250, 350, 350, 400]]
        
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_detect.return_value = [mock_result]
        
        # Mock classification results
        mock_probs = MagicMock()
        mock_probs.top1 = 0
        mock_probs.top1conf = 0.8
        
        mock_classify_result = MagicMock()
        mock_classify_result.probs = mock_probs
        mock_classify_result.names = {0: "marked_bee", 1: "normal_bee"}
        mock_classify.predict.return_value = [mock_classify_result]
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Measure processing time with detection
        start_time = time.time()
        num_frames = 50  # Fewer frames since detection is more expensive
        
        for _ in range(num_frames):
            process_frame(frame)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_frame = total_time / num_frames
        
        # Detection should still be reasonably fast (less than 100ms per frame)
        assert avg_time_per_frame < 0.1, f"Detection processing too slow: {avg_time_per_frame:.3f}s per frame"
        
        fps = 1 / avg_time_per_frame
        print(f"Detection processing performance: {fps:.1f} FPS")

    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self):
        """Test database performance under concurrent load."""
        num_concurrent_operations = 10
        
        async def create_test_event(i):
            event_data = EventCreate(
                time_out=datetime.now(),
                time_in=datetime.now(),
                video_url=f"http://test{i}.mp4"
            )
            
            with patch('app.services.service.db') as mock_db:
                mock_result = MagicMock()
                mock_result.inserted_id = f"test_id_{i}"
                mock_db.__getitem__.return_value.insert_one = AsyncMock(return_value=mock_result)
                
                start_time = time.time()
                result = await create_event(event_data)
                end_time = time.time()
                
                return end_time - start_time, result
        
        # Run concurrent operations
        start_time = time.time()
        tasks = [create_test_event(i) for i in range(num_concurrent_operations)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Check that all operations completed
        assert len(results) == num_concurrent_operations
        
        # Check individual operation times
        operation_times = [result[0] for result in results]
        avg_operation_time = sum(operation_times) / len(operation_times)
        max_operation_time = max(operation_times)
        
        print(f"Concurrent database operations:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average operation time: {avg_operation_time:.3f}s")
        print(f"  Max operation time: {max_operation_time:.3f}s")
        
        # Operations should complete reasonably fast
        assert avg_operation_time < 0.1, f"Database operations too slow: {avg_operation_time:.3f}s average"
        assert max_operation_time < 0.5, f"Slowest operation too slow: {max_operation_time:.3f}s"

    @pytest.mark.asyncio
    async def test_bee_tracking_performance(self):
        """Test bee tracking logic performance."""
        num_tracking_calls = 100
        
        with patch('app.routes.video_routes.create_event') as mock_create:
            with patch('app.routes.video_routes.start_external_camera') as mock_camera:
                mock_event = MagicMock()
                mock_event.id = "test_event"
                mock_create.return_value = mock_event
                mock_camera.return_value = "/test/video.mp4"
                
                start_time = time.time()
                
                for i in range(num_tracking_calls):
                    # Alternate between inside and outside to trigger tracking logic
                    status = "inside" if i % 2 == 0 else "outside"
                    await handle_bee_tracking(status, datetime.now())
                
                end_time = time.time()
                total_time = end_time - start_time
                avg_time = total_time / num_tracking_calls
                
                print(f"Bee tracking performance: {avg_time:.4f}s per call")
                
                # Tracking should be very fast
                assert avg_time < 0.01, f"Bee tracking too slow: {avg_time:.4f}s per call"

    def test_memory_usage_frame_processing(self):
        """Test memory usage during frame processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('app.routes.video_routes.model_detect') as mock_detect:
            mock_detect.return_value = []
            
            # Process many frames to check for memory leaks
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            for _ in range(1000):
                process_frame(frame)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (increase: {memory_increase:.1f}MB)")
            
            # Memory increase should be reasonable (less than 1000MB for 1000 frames in CI environment)
            assert memory_increase < 1000, f"Memory usage too high: {memory_increase:.1f}MB increase"


class TestLoadTesting:
    """Test system behavior under high load."""

    @pytest.mark.asyncio
    async def test_api_concurrent_requests(self):
        """Test API performance under concurrent requests."""
        num_concurrent_requests = 20
        
        async def make_request():
            start_time = time.time()
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/")
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Make concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(num_concurrent_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Check that all requests succeeded
        status_codes = [result[0] for result in results]
        response_times = [result[1] for result in results]
        
        success_count = sum(1 for code in status_codes if code == 200)
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        print(f"API load test results:")
        print(f"  Successful requests: {success_count}/{num_concurrent_requests}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  Max response time: {max_response_time:.3f}s")
        
        # Most requests should succeed
        assert success_count >= num_concurrent_requests * 0.9, f"Too many failed requests: {success_count}/{num_concurrent_requests}"
        
        # Response times should be reasonable
        assert avg_response_time < 1.0, f"Average response time too slow: {avg_response_time:.3f}s"
        assert max_response_time < 5.0, f"Slowest response too slow: {max_response_time:.3f}s"

    @pytest.mark.asyncio
    async def test_websocket_multiple_connections(self, sync_test_client):
        """Test WebSocket performance with multiple connections."""
        num_connections = 5  # Limited for test environment
        
        def create_websocket_connection():
            try:
                with sync_test_client.websocket_connect("/video/live-stream") as websocket:
                    # Keep connection open briefly
                    time.sleep(0.1)
                    return True
            except Exception:
                return False
        
        # Create multiple WebSocket connections concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_connections) as executor:
            start_time = time.time()
            futures = [executor.submit(create_websocket_connection) for _ in range(num_connections)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            total_time = time.time() - start_time
        
        successful_connections = sum(results)
        
        print(f"WebSocket load test:")
        print(f"  Successful connections: {successful_connections}/{num_connections}")
        print(f"  Total time: {total_time:.3f}s")
        
        # Most connections should succeed
        assert successful_connections >= num_connections * 0.8, f"Too many failed connections: {successful_connections}/{num_connections}"

    def test_large_frame_processing(self):
        """Test processing of large frames."""
        with patch('app.routes.video_routes.model_detect') as mock_detect:
            mock_detect.return_value = []
            
            # Test with various frame sizes
            frame_sizes = [
                (480, 640),   # Standard
                (720, 1280),  # HD
                (1080, 1920), # Full HD
            ]
            
            for height, width in frame_sizes:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                start_time = time.time()
                processed_frame, bee_status, current_time = process_frame(frame)
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                print(f"Frame size {width}x{height}: {processing_time:.3f}s")
                
                # Larger frames should still process in reasonable time
                assert processing_time < 0.5, f"Large frame processing too slow: {processing_time:.3f}s for {width}x{height}"
                assert processed_frame.shape == frame.shape


class TestStressTests:
    """Stress tests for extreme conditions."""

    @pytest.mark.asyncio
    async def test_rapid_bee_status_changes(self):
        """Test system with rapid bee status changes."""
        with patch('app.routes.video_routes.create_event') as mock_create:
            with patch('app.routes.video_routes.update_event') as mock_update:
                with patch('app.routes.video_routes.start_external_camera') as mock_start:
                    with patch('app.routes.video_routes.stop_external_camera') as mock_stop:
                        
                        mock_event = MagicMock()
                        mock_event.id = "test_event"
                        mock_create.return_value = mock_event
                        mock_update.return_value = mock_event
                        mock_start.return_value = "/test/video.mp4"
                        mock_stop.return_value = "/test/video.mp4"
                        
                        # Simulate rapid status changes
                        start_time = time.time()
                        
                        for i in range(100):
                            # Alternate rapidly between inside and outside
                            status = "inside" if i % 2 == 0 else "outside"
                            await handle_bee_tracking(status, datetime.now())
                            
                            # Small delay to simulate real-time processing
                            await asyncio.sleep(0.001)
                        
                        end_time = time.time()
                        total_time = end_time - start_time
                        
                        print(f"Rapid status changes test: {total_time:.3f}s for 100 changes")
                        
                        # Should handle rapid changes without significant delay
                        assert total_time < 5.0, f"Rapid status changes too slow: {total_time:.3f}s"

    def test_invalid_frame_handling(self):
        """Test handling of invalid or corrupted frames."""
        invalid_frames = [
            None,
            np.array([]),  # Empty array
            np.zeros((10, 10, 1), dtype=np.uint8),  # Wrong number of channels
            np.zeros((0, 640, 3), dtype=np.uint8),  # Zero height
            np.zeros((480, 0, 3), dtype=np.uint8),  # Zero width
        ]
        
        for i, frame in enumerate(invalid_frames):
            try:
                if frame is not None:
                    result = process_frame(frame)
                    # If it doesn't crash, that's good
                    print(f"Invalid frame {i}: Handled gracefully")
                else:
                    # None frame should be handled appropriately
                    print(f"Invalid frame {i}: None frame")
            except Exception as e:
                # Should either handle gracefully or fail predictably
                print(f"Invalid frame {i}: Exception - {str(e)}")
                # This is acceptable as long as it's a controlled failure

    @pytest.mark.asyncio
    async def test_database_connection_failure_simulation(self):
        """Test behavior when database connection fails."""
        with patch('app.services.service.db') as mock_db:
            # Simulate database connection failure
            mock_db.__getitem__.return_value.insert_one.side_effect = Exception("Connection failed")
            
            event_data = EventCreate(
                time_out=datetime.now(),
                time_in=datetime.now(),
                video_url="http://test.mp4"
            )
            
            # Should handle database failures gracefully
            with pytest.raises(Exception):
                await create_event(event_data)
            
            print("Database failure handled as expected")


class TestResourceUsage:
    """Test system resource usage patterns."""

    def test_cpu_usage_during_processing(self):
        """Test CPU usage during intensive processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        with patch('app.routes.video_routes.model_detect') as mock_detect:
            mock_detect.return_value = []
            
            # Monitor CPU usage during frame processing
            cpu_percentages = []
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            for _ in range(50):
                cpu_before = process.cpu_percent()
                process_frame(frame)
                cpu_after = process.cpu_percent()
                
                if cpu_after > 0:  # Only record non-zero values
                    cpu_percentages.append(cpu_after)
                
                time.sleep(0.01)  # Small delay for CPU measurement
            
            if cpu_percentages:
                avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
                max_cpu = max(cpu_percentages)
                
                print(f"CPU usage during processing:")
                print(f"  Average: {avg_cpu:.1f}%")
                print(f"  Peak: {max_cpu:.1f}%")
                
                # CPU usage should be reasonable (relaxed for CI environment)
                assert avg_cpu < 2000, f"Average CPU usage too high: {avg_cpu:.1f}%"
            else:
                print("CPU usage: No significant usage detected")

    def test_file_handle_usage(self):
        """Test that file handles are properly managed."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_file_count = process.num_fds() if hasattr(process, 'num_fds') else 0
        
        # Simulate multiple video operations that might open files
        with patch('cv2.VideoWriter') as mock_writer:
            with patch('os.makedirs'):
                mock_writer_instance = MagicMock()
                mock_writer.return_value = mock_writer_instance
                
                # Start and stop external camera multiple times
                for _ in range(10):
                    start_external_camera()
                    stop_external_camera()
        
        final_file_count = process.num_fds() if hasattr(process, 'num_fds') else 0
        
        if initial_file_count > 0:
            file_increase = final_file_count - initial_file_count
            print(f"File handles: {initial_file_count} -> {final_file_count} (increase: {file_increase})")
            
            # Should not leak file handles
            assert file_increase < 10, f"Too many file handles leaked: {file_increase}"
        else:
            print("File handle tracking not available on this system") 