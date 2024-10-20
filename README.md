
# Shorts Extractor Web App from Long Videos

## Overview
This web application allows users to upload long video files and extract shorter clips that are suitable for YouTube Shorts. The application leverages Google Generative AI and AssemblyAI to transcribe audio, generate summaries, and extract video segments. The identified clips are then trimmed and sped up for faster playback, making them perfect for short-form video platforms.

## Features
- **File Upload:** Upload videos in `mp4`, `avi`, or `mov` format.
- **Video Processing:** Extracts audio from video, transcribes it, generates a summary, and identifies matching video segments.
- **Video Trimming & Speed-Up:** The identified segments are trimmed and sped up.
- **Download:** Allows users to download the processed video clips.

## Installation

### Prerequisites
- Python 3.x
- `pip` for package management

### Required Libraries

Install the required Python libraries listed in the `requirements.txt` file:

    
    pip install -r requirements.txt

Download the required spaCy model (if not already installed):
    
    python -m spacy download en_core_web_lg
    

### API Keys Setup

The project requires API keys for Google Generative AI and AssemblyAI. You can directly set them within the code:

Google Generative AI API key: Replace GOOGLE_API_KEY with your Google API key in the final function.
AssemblyAI API key: Replace ASSEMBLY_API_KEY with your AssemblyAI API key in the final function.

   
    GOOGLE_API_KEY = "your_google_api_key_here"
    ASSEMBLY_API_KEY = "your_assemblyai_api_key_here"
### Usage

Run the app using:

     streamlit run app.py

### Output 

After processing, the app will display the generated video clips, allowing you to view and download each clip individually.
     
     

     
