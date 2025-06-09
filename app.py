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

# --- LLM Call Functions ---
def call_llm_with_key(api_key, image_path, prompt):
    """
    Calls the Gemini LLM with a specific API key.
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
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()
            image_part = {"mime_type": "image/jpeg", "data": image_data}
        except FileNotFoundError:
            return {"error": "File Not Found During Read", "is_retriable": False}
        except Exception as e:
            return {"error": f"Image Read Error: {e}", "is_retriable": False}

        prompt_parts = [prompt, "INPUT IMAGE:", image_part]
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

def create_result_dict(image_path, status, data=None, error_msg=None, raw_text=None):
    """Creates a standardized result dictionary."""
    image_name = os.path.basename(image_path)
    base = {
        'image_name': image_name,
        'image_path': image_path,
        'status': status,
        'error': error_msg,
        'raw_llm_response': raw_text
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
            llm_result = call_llm_with_key(api_key, image_path, prompt)

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
    """Extract uploaded zip to temporary directory and return image paths."""
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    # Get all image files (.jpg, .jpeg, .png)
    image_extensions = ['.jpg', '.jpeg', '.png']
    for root, _, files in os.walk(temp_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(root, file))
                
    return temp_dir, image_paths

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
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display image if path exists and file exists
            if os.path.exists(row['image_path']):
                st.image(row['image_path'], caption=row['image_name'], width=300)
            else:
                st.error(f"Image not found: {row['image_name']}")
        
        with col2:
            st.subheader(f"🚗 {row['image_name']}")
            
            if row['status'] == 'Success':
                # Create a styled container for the results
                with st.container():
                    if row['number_plate_detected'] == 'Yes':
                        st.success("✅ Number Plate Detected")
                        
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
    
    # Header with custom styling
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #ff6b35, #f7931e); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🚗 Indian Vehicle Number Plate Detection</h1>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1em;">AI-Powered License Plate Recognition System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Upload images of Indian vehicles to automatically detect and extract number plate information.
    The AI will analyze each image and provide detailed information about the detected license plates.
    
    **Supported formats:** JPG, JPEG, PNG  
    **Input methods:** Individual images or ZIP folder  
    **Features:** Text extraction, vehicle type identification, plate type classification
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
    st.subheader("📤 Upload Images")
    input_method = st.radio(
        "Choose your input method:",
        ["Individual Images", "ZIP Folder"],
        horizontal=True
    )
    
    uploaded_files = None
    uploaded_zip = None
    
    if input_method == "Individual Images":
        uploaded_files = st.file_uploader(
            "Upload image files", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            help="Select one or more image files containing Indian vehicles"
        )
    else:
        uploaded_zip = st.file_uploader(
            "Upload ZIP folder containing images", 
            type="zip",
            help="Upload a ZIP file containing multiple vehicle images"
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
        col1.metric("Total Images", len(results_df))
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
        csv = results_df[csv_columns].to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Results",
            data=csv,
            file_name=f"numberplate_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        
        # Add option to start new detection
        st.markdown("---")
        if st.button("🔄 Start New Detection"):
            # Clear session state to start fresh
            if 'results_df' in st.session_state:
                del st.session_state.results_df
            if 'current_batch' in st.session_state:
                del st.session_state.current_batch
            st.rerun()
    
    # Process images when files are uploaded and no results exist
    elif (uploaded_files and len(uploaded_files) > 0) or uploaded_zip:
        if api_keys:
            # Show file details
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} images uploaded")
                for file in uploaded_files[:5]:  # Show first 5 filenames
                    st.write(f"📁 {file.name} ({file.size/1024:.1f} KB)")
                if len(uploaded_files) > 5:
                    st.write(f"... and {len(uploaded_files) - 5} more files")
            else:
                st.success(f"✅ ZIP file uploaded: {uploaded_zip.name} ({uploaded_zip.size/1024:.1f} KB)")
            
            if st.button("🚀 Start Number Plate Detection", type="primary"):
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Extract/prepare image files
                with st.spinner("📂 Preparing images..."):
                    if uploaded_files:
                        temp_dir, image_paths = get_image_paths_from_uploaded_files(uploaded_files)
                    else:
                        temp_dir, image_paths = extract_zip_to_temp(uploaded_zip)
                    
                    st.write(f"🖼️ Found {len(image_paths)} images to process")
                    
                if not image_paths:
                    st.error("❌ No valid images found")
                    return
                    
                # Process images with progress updates
                def update_progress(current, total):
                    progress_bar.progress(current / total)
                    status_text.text(f"🔍 Processing: {current}/{total} images")
                
                with st.spinner("🤖 AI is analyzing images..."):
                    start_time = time.time()
                    results_df = process_images_concurrent(
                        image_paths, 
                        api_keys,
                        progress_callback=update_progress
                    )
                    end_time = time.time()
                    
                # Show results
                progress_bar.progress(1.0)
                status_text.text(f"✅ Completed! Processed {len(image_paths)} images in {end_time-start_time:.1f} seconds")
                
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
        - Use clear, well-lit images
        - Ensure number plate is visible
        - Avoid heavily blurred images
        - Multiple angles are okay
        - Works with partial plates too
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