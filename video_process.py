import requests
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
import google.generativeai as genai
import os
from pydub import AudioSegment
import spacy
import assemblyai as aai
import streamlit as st
import spacy.cli
import tempfile

model_name = "en_core_web_lg"

# Function to load the spaCy model
def load_spacy_model():
    model_name = "en_core_web_lg"
    try:
        nlp = spacy.load(model_name)  # Try loading the model
    except OSError:
        # If the model is not found, download it
        spacy.cli.download(model_name)  # No need for 'target' argument
        nlp = spacy.load(model_name)  # Load the model after installation
    return nlp

# Load the spaCy model
nlp = load_spacy_model()


# Access API keys from Streamlit secrets
GOOGLE_API_KEY = st.secrets["google_api"]["api_key"]
ASSEMBLY_API_KEY = st.secrets["assembly_ai"]["api_key"]
aai.settings.api_key = ASSEMBLY_API_KEY

def final(video_path):
    # Configure Google Generative AI
    genai.configure(api_key=GOOGLE_API_KEY)

    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    try:
        # Process the video to extract audio
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile("output_audio.wav")
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return

    FILE_URL = "output_audio.wav"

    try:
        # Transcribe audio using AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(FILE_URL)
        transcript_text = transcript.text

        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"Transcription error: {transcript.error}")
            return
        else:
            st.write(f"Transcript: {transcript.text}")
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return

    try:
        # Save transcript to a file
        transcript_file_path = "transcript.txt"
        with open(transcript_file_path, 'w') as file:
            file.write(transcript_text)

        # Generate summary using Generative AI model
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": transcript.text
                        }
                    ]
                }
            ]
        )

        ad = "\nExtract important sentences from the given video transcript, providing long detailed points with a length of more than 100 words. The extracted points should be suitable for creating concise video shorts."
        prompt = transcript.text + ad
        response = chat_session.send_message(
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        )
        summary = transcript.text[:100] + response.text.replace("*", "").replace("-", "")

        # Save the summary to a text file
        with open('summary.txt', 'w', encoding='utf-8') as file:
            file.write(summary)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return

    def get_audio_duration(file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000  # Converting milliseconds to seconds
        except Exception as e:
            st.error(f"Error reading audio file: {e}")
            return 0

    def find_segments(video_transcript_path, summarized_text_path, total_video_duration):
        try:
            # Read the video transcript
            with open(video_transcript_path, 'r') as file:
                video_transcript = file.read()

            # Read the summarized text
            with open(summarized_text_path, 'r') as file:
                summarized_text = file.read()
        except Exception as e:
            st.error(f"Error reading files: {e}")
            return []

        # Process the texts with spaCy
        doc_transcript = nlp(video_transcript)
        doc_summary = nlp(summarized_text)

        # Initialize an empty list to store matching segments
        matching_segments = []

        # Iterate over sentences in the video transcript
        for sentence in doc_transcript.sents:
            similarity_score = sentence.similarity(doc_summary)
            similarity_threshold = 0.5

            if similarity_score > similarity_threshold:
                start_char = sentence.start_char
                end_char = sentence.end_char
                start_time = round(start_char / len(doc_transcript.text) * total_video_duration, 0)
                end_time = round(end_char / len(doc_transcript.text) * total_video_duration, 0)

                if 10 < (end_time - start_time) < 90:
                    matching_segments.append((start_time, end_time))

        return matching_segments

    total_video_duration = get_audio_duration("output_audio.wav")
    video_transcript = "transcript.txt"
    summarized_text = "summary.txt"
    matching_segments = find_segments(video_transcript, summarized_text, total_video_duration)
    st.write("Matching Segments:", matching_segments)

    for start_time, end_time in matching_segments:
        st.write(f"Cut video from {start_time} seconds to {end_time} seconds.")

    def trim_and_speedup_video(video_path, output_folder, start_time, end_time, speed_factor=1.5):
        try:
            start_seconds = start_time
            end_seconds = end_time
            output_filename = f"clip_{start_seconds}_{end_seconds}_speedup.mp4"  # Include both times in the filename
            output_path = os.path.join(output_folder, output_filename)

            command = f"ffmpeg -i {video_path} -ss {start_seconds} -to {end_seconds} -vf 'setpts={1/speed_factor}*PTS' -af 'atempo={speed_factor}' {output_path}"
            os.system(command)
            st.write(f"Trimmed and sped up video: {output_path}")
        except Exception as e:
            st.error(f"Error trimming video: {e}")

    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)

    for start_time, end_time in matching_segments:
        start_time = max(0, start_time - 20)
        end_time = end_time + 20

        if end_time - start_time < 60:
            end_time = start_time + 60

        trim_and_speedup_video(video_path, output_folder, start_time, end_time)

    st.write("All video clips trimmed and sped up successfully!")

# Assuming this function is called in your Streamlit app with the uploaded video path
