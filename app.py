import os
import json
import csv
import time
import zipfile
import tempfile
import threading
import random
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import concurrent.futures

import pandas as pd
import streamlit as st
import google.generativeai as genai
from tqdm import tqdm
import cv2  # For video processing and frame extraction
import numpy as np
from PIL import Image  # For better image handling

# --- Configuration ---
MODEL_NAME = "gemini-2.0-flash"
MIN_KEY_COOLDOWN = 6
MAX_RETRIES = 10
RETRY_BASE_DELAY = 70

generation_config = {
    "temperature": 0.5,
    "top_p": 0.5,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Load prompt template
def load_prompt():
    try:
        with open("st_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        # Default prompt if file not found
        return """
Please analyze the following image and detect Indian vehicle number plates. Return your analysis in the following JSON format:

{
  "number_plate_detected": "Yes/No",
  "extracted_text": "Full number plate text as read",
  "plate_type": "Private/Commercial/Electric/Diplomatic/Unknown",
  "vehicle_type": "Car/Motorcycle/Truck/Bus/Auto-rickshaw/Unknown",
  "confidence_score": "High/Medium/Low",
  "state_code": "State abbreviation (e.g., MH, DL, KA)",
  "district_code": "District number if visible",
  "additional_details": "Any other relevant information about the plate or vehicle"
}
"""

# Load video prompt template
def load_video_prompt():
    try:
        with open("st_video_prompt.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        # Default video prompt if file not found
        return """
Please analyze the following video and detect Indian vehicle number plates throughout the video. 
Return your analysis as a JSON with array of detected vehicles including precise timestamps.
"""

# --- API Key Management ---
# Thread-safe key management
class KeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        initial_time = datetime.now() - timedelta(seconds=MIN_KEY_COOLDOWN * 2)
        self.last_used = {key: initial_time for key in api_keys}
        self.key_lock = threading.Lock()
    
    def get_next_available_key(self):
        """Thread-safe method to get the next available API key"""
        while True:
            best_key = None
            min_wait_time = float('inf')
            now = datetime.now()

            shuffled_keys = list(self.api_keys)
            random.shuffle(shuffled_keys)

            with self.key_lock:
                sorted_keys = sorted(self.last_used.items(), key=lambda item: item[1])

                for key in shuffled_keys:
                    last_use_time = self.last_used[key]
                    time_since_last_use = (now - last_use_time).total_seconds()
                    if time_since_last_use >= MIN_KEY_COOLDOWN:
                        self.last_used[key] = now
                        best_key = key
                        break

                if best_key:
                    return best_key

                soonest_key, soonest_time = sorted_keys[0]
                wait_time = MIN_KEY_COOLDOWN - (now - soonest_time).total_seconds()
                min_wait_time = max(0.1, wait_time)

            time.sleep(min_wait_time)

# --- Video Processing Functions ---
def convert_timestamp_to_seconds(timestamp, video_duration=None):
    """
    Convert timestamp to decimal seconds, handling both formats:
    - Decimal seconds: 75.5 -> 75.5
    - MM.SS format: 1.15 -> 75.0 (1 minute 15 seconds)
    
    Args:
        timestamp: The timestamp value to convert
        video_duration: Optional video duration in seconds to help with decision
    """
    try:
        timestamp_float = float(timestamp)
        
        # Check if this might be MM.SS format
        whole_part = int(timestamp_float)
        decimal_part = int(round((timestamp_float - whole_part) * 100))
        
        # More precise check for MM.SS format
        timestamp_str = f"{timestamp_float:.2f}"
        has_two_decimals = len(timestamp_str.split('.')[-1]) <= 2
        
        # Only convert MM.SS for very specific cases that are clearly time format
        # Be very conservative - only convert obvious MM.SS cases like 1.15, 2.30, etc.
        # Don't convert larger numbers like 45.5 which are likely already in seconds
        should_convert_mmss = (
            whole_part <= 10 and 
            decimal_part <= 59 and 
            has_two_decimals and 
            decimal_part >= 10 and  # At least 10 seconds to be clearly MM.SS
            whole_part > 0  # Must have minutes
        )
        
        # Additional check: if we have video duration context, be even more conservative
        if video_duration is not None and should_convert_mmss:
            converted_seconds = whole_part * 60 + decimal_part
            # Don't convert if the result would exceed video duration
            if converted_seconds > video_duration:
                print(f"Not converting {timestamp} to MM.SS format - would exceed video duration ({video_duration}s)")
                should_convert_mmss = False
        
        if should_convert_mmss:
            # Likely MM.SS format - convert to seconds
            minutes = whole_part
            seconds = decimal_part
            total_seconds = minutes * 60 + seconds
            print(f"Converted timestamp {timestamp} (MM.SS format) to {total_seconds} seconds")
            return total_seconds
        else:
            # Already in decimal seconds format
            print(f"Using as decimal seconds: {timestamp_float}")
            return timestamp_float
            
    except (ValueError, TypeError):
        print(f"Invalid timestamp format: {timestamp}, defaulting to 0")
        return 0.0

def extract_frame_from_video(video_path, timestamp_seconds, output_path):
    """Extract a frame from video at specified timestamp and save as image."""
    try:
        # Validate input parameters
        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
            return False
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video file: {video_path}")
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps if fps > 0 else 0
        
        if fps <= 0:
            print(f"Invalid FPS: {fps}")
            cap.release()
            return False
            
        if timestamp_seconds > duration:
            print(f"Timestamp {timestamp_seconds}s exceeds video duration {duration}s")
            cap.release()
            return False
            
        frame_number = int(timestamp_seconds * fps)
        
        # Set position and read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret and frame is not None and frame.size > 0:
            # Ensure the output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create PIL Image and save
            pil_image = Image.fromarray(frame_rgb)
            
            # Save with PIL for better compatibility
            pil_image.save(output_path, 'JPEG', quality=95, optimize=True)
            cap.release()
            
            # Verify the file was created and is readable
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                # Test if the image can be read back by PIL
                try:
                    test_img = Image.open(output_path)
                    test_img.verify()  # Verify it's a valid image
                    return True
                except Exception as verify_error:
                    print(f"Saved image verification failed: {output_path}, error: {verify_error}")
                    return False
            else:
                print(f"Failed to save frame to: {output_path}")
                return False
        else:
            print(f"Failed to read frame at timestamp {timestamp_seconds}s")
            cap.release()
            return False
            
    except Exception as e:
        print(f"Error extracting frame from {video_path} at {timestamp_seconds}s: {e}")
        return False

def get_video_duration(video_path):
    """Get video duration in seconds."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        return duration
    except:
        return 0

# --- LLM Call Functions ---
def call_llm_with_key(api_key, media_path, prompt, is_video=False):
    """
    Calls the Gemini LLM with a specific API key for image or video.
    Returns parsed JSON on success, or an error dictionary on failure.
    """
    try:
        # Configure genai for this call
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=generation_config,
        )

        try:
            with open(media_path, "rb") as media_file:
                media_data = media_file.read()
            
            if is_video:
                # For video files
                media_part = {"mime_type": "video/mp4", "data": media_data}
            else:
                # For image files
                media_part = {"mime_type": "image/jpeg", "data": media_data}
                
        except FileNotFoundError:
            return {"error": "File Not Found During Read", "is_retriable": False}
        except Exception as e:
            return {"error": f"Media Read Error: {e}", "is_retriable": False}

        if is_video:
            prompt_parts = [prompt, "INPUT VIDEO:", media_part]
        else:
            prompt_parts = [prompt, "INPUT IMAGE:", media_part]
        response = model.generate_content(prompt_parts)

        # Process Response
        if response.candidates and response.candidates[0].content.parts:
            if response.candidates[0].finish_reason == 'SAFETY':
                return {"error": "Blocked by safety settings", "is_retriable": False}

            response_text = "".join(part.text for part in response.candidates[0].content.parts).strip()
            
            # Clean JSON
            if response_text.startswith("```json"): response_text = response_text[7:]
            if response_text.endswith("```"): response_text = response_text[:-3]
            response_text = response_text.strip()

            try:
                # SUCCESS CASE
                json_result = json.loads(response_text)
                return json_result
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON response: {e}", "raw_text": response_text, "is_retriable": True}

        else:
            # Handle blocked/empty responses
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            is_retriable = block_reason != 'SAFETY'
            return {"error": f"Empty or blocked response: {block_reason}", "is_retriable": is_retriable}

    except Exception as e:
        error_str = str(e).lower()
        error_type_name = type(e).__name__
        
        # Rate Limiting
        if "rate limit exceeded" in error_str or "quota exceeded" in error_str or "429" in error_str:
            return {"error": f"Rate limit exceeded ({error_type_name})", "is_retriable": True}

        # Invalid Key
        elif "api key not valid" in error_str:
            return {"error": "Invalid API Key", "is_retriable": False}
        
        # Network errors
        elif "deadlineexceeded" in error_str or "connection error" in error_str:
            return {"error": f"Network/Timeout Error ({error_type_name}): {e}", "is_retriable": True}

        # Catch-all for other unexpected errors
        else:
            return {"error": f"Unexpected LLM Error ({error_type_name}): {e}", "is_retriable": True}

def create_result_dict(media_path, status, data=None, error_msg=None, raw_text=None, is_video=False, timestamp=None, vehicle_id=None):
    """Creates a standardized result dictionary."""
    media_name = os.path.basename(media_path)
    base = {
        'image_name': media_name,  # Keep same key for compatibility
        'image_path': media_path,  # Keep same key for compatibility
        'status': status,
        'error': error_msg,
        'raw_llm_response': raw_text,
        'is_video_frame': is_video,
        'timestamp_seconds': timestamp,
        'vehicle_id': vehicle_id
    }
    if status == 'Success' and data:
        base.update({
            'number_plate_detected': data.get('number_plate_detected', "Parse Error"),
            'extracted_text': data.get('extracted_text', "Parse Error"),
            'plate_type': data.get('plate_type', "Parse Error"),
            'vehicle_type': data.get('vehicle_type', "Parse Error"),
            'confidence_score': data.get('confidence_score', "Parse Error"),
            'state_code': data.get('state_code', "Parse Error"),
            'district_code': data.get('district_code', "Parse Error"),
            'additional_details': data.get('additional_details', "")
        })
    else:
        base.update({
            'number_plate_detected': 'N/A',
            'extracted_text': 'N/A',
            'plate_type': 'N/A',
            'vehicle_type': 'N/A',
            'confidence_score': 'N/A',
            'state_code': 'N/A',
            'district_code': 'N/A',
            'additional_details': 'N/A'
        })
    return base

def process_video_with_retry(video_path, key_manager, prompt):
    """Processes a single video, retrying on failure up to MAX_RETRIES."""
    last_error = "No attempts made"
    raw_llm_text = None

    # Initial File Check
    if not os.path.exists(video_path):
        return [create_result_dict(video_path, 'Failed', error_msg="Video File Not Found")]

    for attempt in range(MAX_RETRIES + 1):
        api_key = None
        try:
            api_key = key_manager.get_next_available_key()
            llm_result = call_llm_with_key(api_key, video_path, prompt, is_video=True)

            # Check Result
            if isinstance(llm_result, dict) and 'error' in llm_result:
                # Error occurred during the call
                last_error = llm_result['error']
                raw_llm_text = llm_result.get('raw_text')
                is_retriable = llm_result.get('is_retriable', False)

                if not is_retriable:
                    return [create_result_dict(video_path, 'Failed', error_msg=last_error, raw_text=raw_llm_text)]
                # Continue to backoff logic if retriable

            elif isinstance(llm_result, dict) and 'vehicles' in llm_result:
                # SUCCESS! Process each detected vehicle
                results = []
                temp_dir = tempfile.mkdtemp()
                
                # Get video duration for smart timestamp conversion
                video_duration = get_video_duration(video_path)
                
                for i, vehicle in enumerate(llm_result.get('vehicles', [])):
                    timestamp_raw = vehicle.get('best_timestamp_seconds', vehicle.get('timestamp_seconds', 0))
                    timestamp = convert_timestamp_to_seconds(timestamp_raw, video_duration)
                    vehicle_id = vehicle.get('vehicle_id', f'Vehicle_{i+1}')
                    
                    # Clean vehicle_id for filename (remove special characters)
                    clean_vehicle_id = "".join(c for c in vehicle_id if c.isalnum() or c in ('_', '-'))
                    if not clean_vehicle_id:
                        clean_vehicle_id = f"Vehicle_{i+1}"
                    
                    # Extract frame at timestamp
                    frame_filename = f"{clean_vehicle_id}_frame_{timestamp:.1f}s.jpg"
                    frame_path = os.path.join(temp_dir, frame_filename)
                    
                    print(f"Attempting to extract frame for {vehicle_id} at {timestamp}s to {frame_path}")
                    
                    if extract_frame_from_video(video_path, timestamp, frame_path):
                        print(f"Successfully extracted frame to: {frame_path}")
                        # Verify the extracted frame one more time
                        if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                            # Create result with extracted frame
                            result = create_result_dict(
                                frame_path, 'Success', 
                                data=vehicle, 
                                is_video=True, 
                                timestamp=timestamp,
                                vehicle_id=vehicle_id
                            )
                            results.append(result)
                            print(f"Added successful result for {vehicle_id}")
                        else:
                            print(f"Frame file validation failed: {frame_path}")
                            result = create_result_dict(
                                video_path, 'Failed', 
                                error_msg=f"Frame file validation failed at {timestamp}s",
                                is_video=True, 
                                timestamp=timestamp,
                                vehicle_id=vehicle_id
                            )
                            results.append(result)
                    else:
                        print(f"Frame extraction failed for {vehicle_id} at {timestamp}s")
                        # Frame extraction failed
                        result = create_result_dict(
                            video_path, 'Failed', 
                            error_msg=f"Failed to extract frame at {timestamp}s",
                            is_video=True, 
                            timestamp=timestamp,
                            vehicle_id=vehicle_id
                        )
                        results.append(result)
                
                return results if results else [create_result_dict(video_path, 'Failed', error_msg="No vehicles detected")]

            else:
                # Unexpected result type
                last_error = "Unexpected non-dict/non-vehicle result from call_llm"
                is_retriable = True

        except Exception as e:
            last_error = f"Worker loop exception: {type(e).__name__} - {e}"
            is_retriable = True

        # Backoff Logic if attempt failed and is retriable
        if attempt < MAX_RETRIES:
            # Exponential backoff with jitter
            wait_time = RETRY_BASE_DELAY 
            wait_time = random.uniform(wait_time * 1.0, wait_time * 1.2)
            time.sleep(wait_time)
        else:
            # Max retries reached
            return [create_result_dict(video_path, 'Failed_Retries', error_msg=f"Max retries reached. Last: {last_error}", raw_text=raw_llm_text)]

    # Fallback if loop finishes unexpectedly
    return [create_result_dict(video_path, 'Failed', error_msg=f"Unexpected loop exit. Last: {last_error}", raw_text=raw_llm_text)]

def process_image_with_retry(image_path, key_manager, prompt):
    """Processes a single image, retrying on failure up to MAX_RETRIES."""
    last_error = "No attempts made"
    raw_llm_text = None

    # Initial File Check
    if not os.path.exists(image_path):
        return create_result_dict(image_path, 'Failed', error_msg="File Not Found")

    for attempt in range(MAX_RETRIES + 1):
        api_key = None
        try:
            api_key = key_manager.get_next_available_key()
            llm_result = call_llm_with_key(api_key, image_path, prompt, is_video=False)

            # Check Result
            if isinstance(llm_result, dict) and 'error' in llm_result:
                # Error occurred during the call
                last_error = llm_result['error']
                raw_llm_text = llm_result.get('raw_text')
                is_retriable = llm_result.get('is_retriable', False)

                if not is_retriable:
                    return create_result_dict(image_path, 'Failed', error_msg=last_error, raw_text=raw_llm_text)
                # Continue to backoff logic if retriable

            elif isinstance(llm_result, dict):
                # SUCCESS!
                return create_result_dict(image_path, 'Success', data=llm_result)

            else:
                # Unexpected result type
                last_error = "Unexpected non-dict/non-error result from call_llm"
                is_retriable = True

        except Exception as e:
            last_error = f"Worker loop exception: {type(e).__name__} - {e}"
            is_retriable = True

        # Backoff Logic if attempt failed and is retriable
        if attempt < MAX_RETRIES:
            # Exponential backoff with jitter
            wait_time = RETRY_BASE_DELAY 
            wait_time = random.uniform(wait_time * 1.0, wait_time * 1.2)
            time.sleep(wait_time)
        else:
            # Max retries reached
            return create_result_dict(image_path, 'Failed_Retries', error_msg=f"Max retries reached. Last: {last_error}", raw_text=raw_llm_text)

    # Fallback if loop finishes unexpectedly (shouldn't happen with current logic)
    return create_result_dict(image_path, 'Failed', error_msg=f"Unexpected loop exit. Last: {last_error}", raw_text=raw_llm_text)

def process_images_concurrent(image_paths, api_keys, progress_callback=None):
    """Process images concurrently with thread pool."""
    prompt = load_prompt()
    key_manager = KeyManager(api_keys)
    max_workers = min(len(api_keys), 10)  # Limit workers based on API keys
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image_with_retry, img_path, key_manager, prompt) 
                  for img_path in image_paths]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(i + 1, len(image_paths))
                    
            except Exception as exc:
                # Handle unexpected errors in the future
                image_path = image_paths[image_paths.index(list(futures.keys())[list(futures.values()).index(future)])]
                results.append(create_result_dict(image_path, 'Failed', error_msg=f"Executor Error: {exc}"))
                
    return pd.DataFrame(results)

def extract_zip_to_temp(zip_file):
    """Extract uploaded zip to temporary directory and return media file paths."""
    temp_dir = tempfile.mkdtemp()
    media_paths = []
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    # Get all media files (images and videos)
    image_extensions = ['.jpg', '.jpeg', '.png']
    video_extensions = ['.mp4', '.avi', '.mov']
    all_extensions = image_extensions + video_extensions
    
    for root, _, files in os.walk(temp_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in all_extensions:
                media_paths.append(os.path.join(root, file))
                
    return temp_dir, media_paths

def get_image_paths_from_uploaded_files(uploaded_files):
    """Save uploaded individual files to temp directory and return paths."""
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(file_path)
    
    return temp_dir, image_paths

def get_video_paths_from_uploaded_files(uploaded_files):
    """Save uploaded video files to temp directory and return paths."""
    temp_dir = tempfile.mkdtemp()
    video_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_paths.append(file_path)
    
    return temp_dir, video_paths

def process_videos_concurrent(video_paths, api_keys, progress_callback=None):
    """Process videos concurrently with thread pool."""
    prompt = load_video_prompt()
    key_manager = KeyManager(api_keys)
    max_workers = min(len(api_keys), 5)  # Limit workers for video processing
    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_video_with_retry, video_path, key_manager, prompt) 
                  for video_path in video_paths]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                video_results = future.result()  # This returns a list of results
                all_results.extend(video_results)
                
                # Update progress if callback provided
                if progress_callback:
                    progress_callback(i + 1, len(video_paths))
                    
            except Exception as exc:
                # Handle unexpected errors in the future
                # Create a generic error result since we can't reliably map back to specific video
                error_result = create_result_dict("unknown_video", 'Failed', error_msg=f"Executor Error: {exc}")
                all_results.append(error_result)
                
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

def display_batch_results(results_df, batch_size=5):
    """Display results in batches with navigation."""
    if results_df.empty:
        st.warning("No results to display")
        return
    
    # Initialize session state for batch navigation
    if 'current_batch' not in st.session_state:
        st.session_state.current_batch = 0
    
    total_results = len(results_df)
    total_batches = (total_results + batch_size - 1) // batch_size
    
    # Navigation controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", disabled=(st.session_state.current_batch == 0)):
            st.session_state.current_batch = 0
            st.rerun()
    
    with col2:
        if st.button("⏪ Previous", disabled=(st.session_state.current_batch == 0)):
            st.session_state.current_batch -= 1
            st.rerun()
    
    with col3:
        st.write(f"Batch {st.session_state.current_batch + 1} of {total_batches}")
    
    with col4:
        if st.button("Next ⏩", disabled=(st.session_state.current_batch >= total_batches - 1)):
            st.session_state.current_batch += 1
            st.rerun()
    
    with col5:
        if st.button("Last ⏭️", disabled=(st.session_state.current_batch >= total_batches - 1)):
            st.session_state.current_batch = total_batches - 1
            st.rerun()
    
    # Display current batch
    start_idx = st.session_state.current_batch * batch_size
    end_idx = min(start_idx + batch_size, total_results)
    batch_df = results_df.iloc[start_idx:end_idx]
    
    st.markdown("---")
    
    # Display each result in the batch
    for idx, row in batch_df.iterrows():
        # Create a container for each result to better control layout
        with st.container():
            col1, col2 = st.columns([1, 1.5])  # Adjusted ratio for better balance
            
            with col1:
                # Create a fixed-size container for the image
                with st.container():
                    # Display image if path exists and file exists
                    try:
                        image_path = row['image_path']
                        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                            # Test if image can be read by PIL first
                            try:
                                # Open and verify image with PIL
                                with Image.open(image_path) as test_img:
                                    test_img.verify()
                                
                                # Image is valid, display with constrained size
                                st.image(
                                    image_path, 
                                    caption=row['image_name'], 
                                    width=250,  # Reduced width further
                                    use_container_width=False  # Don't auto-resize
                                )
                            except Exception as img_test_error:
                                st.error(f"Invalid image file: {row['image_name']}")
                                st.markdown("📷 **Corrupted image file**")
                                print(f"Image validation failed for {image_path}: {img_test_error}")
                        else:
                            st.error(f"Image file not found or empty: {row['image_name']}")
                            # Show placeholder for missing image
                            st.markdown("📷 **Image extraction failed**")
                    except Exception as e:
                        st.error(f"Error displaying image: {row['image_name']}")
                        st.markdown("📷 **Image display error**")
                        print(f"Display error for {row.get('image_path', 'unknown')}: {e}")
        
        with col2:
            if row.get('is_video_frame', False):
                st.subheader(f"🎬 {row.get('vehicle_id', 'Vehicle')} (Frame @ {row.get('timestamp_seconds', 0):.1f}s)")
            else:
                st.subheader(f"🚗 {row['image_name']}")
            
            if row['status'] == 'Success':
                # Create a styled container for the results
                with st.container():
                    if row['number_plate_detected'] == 'Yes':
                        st.success("✅ Number Plate Detected")
                        
                        # Video-specific info
                        if row.get('is_video_frame', False):
                            st.info(f"⏱️ Detected at: {row.get('timestamp_seconds', 0):.1f} seconds")
                        
                        # Main details in columns
                        det_col1, det_col2 = st.columns(2)
                        with det_col1:
                            st.metric("📝 Extracted Text", row['extracted_text'])
                            st.metric("🚙 Vehicle Type", row['vehicle_type'])
                        with det_col2:
                            st.metric("🎯 Confidence", row['confidence_score'])
                            st.metric("🏷️ Plate Type", row['plate_type'])
                        
                        # Additional details
                        if row['state_code'] != 'N/A' and row['state_code'] != 'Parse Error':
                            st.info(f"📍 State: {row['state_code']}")
                        
                        if row['district_code'] != 'N/A' and row['district_code'] != 'Parse Error':
                            st.info(f"🏘️ District: {row['district_code']}")
                        
                        if row['additional_details'] and row['additional_details'] not in ['N/A', 'Parse Error', '']:
                            st.text_area("📋 Additional Details", row['additional_details'], height=80, key=f"details_{row['image_name']}_{idx}")
                    else:
                        st.warning("❌ No Number Plate Detected")
                        st.info("The AI could not detect any number plate in this image.")
            else:
                st.error(f"❌ Processing Failed: {row['error']}")
        
        st.markdown("---")

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Indian Vehicle Number Plate Detection", layout="wide", page_icon="🚗")
    
    # Custom CSS to prevent image overflow
    st.markdown("""
    <style>
    .stImage > div > div > img {
        max-width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
    }
    .stImage {
        max-width: 100% !important;
        overflow: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with custom styling
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #ff6b35, #f7931e); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🚗 Indian Vehicle Number Plate Detection</h1>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1em;">AI-Powered License Plate Recognition System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Upload images or videos of Indian vehicles to automatically detect and extract number plate information.
    The AI will analyze media files and provide detailed information about detected license plates.
    
    **Supported formats:** 
    - Images: JPG, JPEG, PNG  
    - Videos: MP4, AVI, MOV
    **Input methods:** Individual files, ZIP folder, or video files  
    **Features:** Text extraction, vehicle type identification, plate type classification, timestamp detection for videos
    """)
    
    api_keys = [
        "AIzaSyApOn5xquEmwaxPOalb_oCqV2olQ9P5UKg",
        "AIzaSyDwai1l5RkIx6YQiSY2Nfapfi9loZN_qYE", 
        "AIzaSyAu0UbliYyn8jib7oWpaX-giBwZ2IfrITM", 
        "AIzaSyD0EXPhEoI1n9Gy-wgohdz-mSGZHc66uko",
        "AIzaSyAPuU29o6Fj5CMsDXP_td-D2EhxYR-QBZU",
        "AIzaSyC95HTi0KbCxHUB_82omoMCXJ30y9IeSGw",
    ]
    
    # Input method selection
    st.subheader("📤 Upload Media Files")
    input_method = st.radio(
        "Choose your input method:",
        ["Individual Images", "Video Files", "ZIP Folder"],
        horizontal=True
    )
    
    uploaded_files = None
    uploaded_zip = None
    uploaded_videos = None
    
    # Generate a unique key for file uploaders to enable clearing
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    
    uploader_key = st.session_state.uploader_key
    
    if input_method == "Individual Images":
        uploaded_files = st.file_uploader(
            "Upload image files", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            help="Select one or more image files containing Indian vehicles",
            key=f"image_uploader_{uploader_key}"
        )
    elif input_method == "Video Files":
        uploaded_videos = st.file_uploader(
            "Upload video files", 
            type=["mp4", "avi", "mov"], 
            accept_multiple_files=True,
            help="Select one or more video files containing Indian vehicles",
            key=f"video_uploader_{uploader_key}"
        )
    else:
        uploaded_zip = st.file_uploader(
            "Upload ZIP folder containing images/videos", 
            type="zip",
            help="Upload a ZIP file containing multiple vehicle images or videos",
            key=f"zip_uploader_{uploader_key}"
        )
    
    # Check if we have results to display (either new or from session state)
    if 'results_df' in st.session_state and not st.session_state.results_df.empty:
        # Display existing results with metrics
        results_df = st.session_state.results_df
        
        # Calculate success metrics
        success_count = results_df[results_df['status'] == 'Success'].shape[0]
        failed_count = results_df[results_df['status'] != 'Success'].shape[0]
        detected_count = results_df[results_df['number_plate_detected'] == 'Yes'].shape[0]
        
        # Display metrics
        st.subheader("📊 Processing Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        # Determine if we have video frames or images
        has_video_frames = results_df['is_video_frame'].any() if 'is_video_frame' in results_df.columns else False
        label = "Total Detections" if has_video_frames else "Total Images"
        
        col1.metric(label, len(results_df))
        col2.metric("Successfully Processed", success_count)
        col3.metric("Number Plates Detected", detected_count)
        col4.metric("Failed", failed_count)
        
        # Display results in batches
        st.subheader("🔍 Detection Results")
        display_batch_results(results_df)
        
        # Provide download link for CSV
        csv_columns = ['image_name', 'number_plate_detected', 'extracted_text', 
                      'plate_type', 'vehicle_type', 'confidence_score', 
                      'state_code', 'district_code', 'additional_details']
        
        # Add video-specific columns if present
        if 'is_video_frame' in results_df.columns:
            csv_columns.extend(['is_video_frame', 'timestamp_seconds', 'vehicle_id'])
        
        available_columns = [col for col in csv_columns if col in results_df.columns]
        csv = results_df[available_columns].to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Results",
            data=csv,
            file_name=f"numberplate_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        # Add option to start new detection
        st.markdown("---")
        if st.button("🔄 Start New Detection"):
            # Clean up any temporary directories from previous detection
            if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
                try:
                    shutil.rmtree(st.session_state.temp_dir)
                except:
                    pass  # Ignore cleanup errors
            
            # Clear ALL session state to start completely fresh
            keys_to_clear = [
                'results_df', 
                'current_batch', 
                'temp_dir'
            ]
            
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Increment uploader key to force file uploaders to reset
            st.session_state.uploader_key += 1
            
            # Clear any other session state that might persist
            for key in list(st.session_state.keys()):
                if key.startswith(('file_uploader', 'upload')) and 'uploader_key' not in key:
                    del st.session_state[key]
            
            st.success("🔄 Interface reset! Upload new files to start fresh detection.")
            st.rerun()
    
    # Process files when uploaded and no results exist
    elif (uploaded_files and len(uploaded_files) > 0) or uploaded_zip or (uploaded_videos and len(uploaded_videos) > 0):
        if api_keys:
            # Show file details
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} images uploaded")
                for file in uploaded_files[:5]:  # Show first 5 filenames
                    st.write(f"📁 {file.name} ({file.size/1024:.1f} KB)")
                if len(uploaded_files) > 5:
                    st.write(f"... and {len(uploaded_files) - 5} more files")
            elif uploaded_videos:
                st.success(f"✅ {len(uploaded_videos)} videos uploaded")
                for file in uploaded_videos[:5]:  # Show first 5 filenames
                    st.write(f"🎬 {file.name} ({file.size/1024/1024:.1f} MB)")
                if len(uploaded_videos) > 5:
                    st.write(f"... and {len(uploaded_videos) - 5} more files")
            else:
                st.success(f"✅ ZIP file uploaded: {uploaded_zip.name} ({uploaded_zip.size/1024:.1f} KB)")
            
            if st.button("🚀 Start Number Plate Detection", type="primary"):
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Extract/prepare files
                with st.spinner("📂 Preparing files..."):
                    if uploaded_files:
                        temp_dir, media_paths = get_image_paths_from_uploaded_files(uploaded_files)
                        media_type = "images"
                    elif uploaded_videos:
                        temp_dir, media_paths = get_video_paths_from_uploaded_files(uploaded_videos)
                        media_type = "videos"
                    else:
                        temp_dir, media_paths = extract_zip_to_temp(uploaded_zip)
                        # Check if zip contains videos or images
                        video_extensions = ['.mp4', '.avi', '.mov']
                        has_videos = any(any(path.lower().endswith(ext) for ext in video_extensions) for path in media_paths)
                        media_type = "videos" if has_videos else "images"
                        
                    # Validate media files
                    valid_media_paths = []
                    for path in media_paths:
                        if os.path.exists(path) and os.path.getsize(path) > 0:
                            valid_media_paths.append(path)
                        else:
                            st.warning(f"Skipping invalid file: {os.path.basename(path)}")
                    
                    media_paths = valid_media_paths
                    
                    st.write(f"🖼️ Found {len(media_paths)} {media_type} to process")
                    
                if not media_paths:
                    st.error(f"❌ No valid {media_type} found")
                    return
                    
                # Process files with progress updates
                def update_progress(current, total):
                    progress_bar.progress(current / total)
                    status_text.text(f"🔍 Processing: {current}/{total} {media_type}")
                
                with st.spinner(f"🤖 AI is analyzing {media_type}..."):
                    start_time = time.time()
                    if media_type == "videos":
                        results_df = process_videos_concurrent(
                            media_paths, 
                            api_keys,
                            progress_callback=update_progress
                        )
                    else:
                        results_df = process_images_concurrent(
                            media_paths, 
                            api_keys,
                            progress_callback=update_progress
                        )
                    end_time = time.time()
                    
                # Show results
                progress_bar.progress(1.0)
                status_text.text(f"✅ Completed! Processed {len(media_paths)} {media_type} in {end_time-start_time:.1f} seconds")
                
                # Store results in session state for batch navigation
                st.session_state.results_df = results_df
                st.session_state.temp_dir = temp_dir
                # Reset batch navigation
                st.session_state.current_batch = 0
                
                # Rerun to display results
                st.rerun()
        else:
            st.error("❌ No API keys configured. Please check the configuration.")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 🇮🇳 Indian Number Plate Formats")
        st.markdown("""
        **Standard Format:**
        - MH12 AB 1234 (Maharashtra)
        - DL01 CA 1234 (Delhi)
        - KA03 HB 1234 (Karnataka)
        
        **Plate Types:**
        - ⚪ White: Private vehicles
        - 🟡 Yellow: Commercial/Taxi
        - 🟢 Green: Electric vehicles
        - 🔵 Blue: Diplomatic vehicles
        """)
        
        st.markdown("### ℹ️ Tips for Best Results")
        st.markdown("""
        **For Images:**
        - Use clear, well-lit images
        - Ensure number plate is visible
        - Avoid heavily blurred images
        - Multiple angles are okay
        
        **For Videos:**
        - Good resolution and lighting
        - Steady recording preferred
        - AI will find best timestamps
        - Multiple vehicles per video OK
        - Works with dashcam footage
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🤖 Powered by Google Gemini AI | 🇮🇳 Specialized for Indian Vehicles</p>
        <p><small>Supports all Indian states and union territories</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()