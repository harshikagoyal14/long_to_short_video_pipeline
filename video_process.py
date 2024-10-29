import os
import tempfile
import shutil
import requests
import google.generativeai as genai
import assemblyai as aai
import spacy
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment

model_name = "en_core_web_lg"

# Function to load the spaCy model
def load_spacy_model():
    try:
        nlp = spacy.load(model_name)
    except OSError:
        # Download the model if not available
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp

# Load the spaCy model
nlp = load_spacy_model()

def final(video_path, google_api_key, assembly_api_key):
    # Configure Google Generative AI
    genai.configure(api_key=google_api_key)
    aai.settings.api_key = assembly_api_key

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

    # Process the video to extract audio
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_output_path = os.path.join(tempfile.gettempdir(), "output_audio.wav")
        audio_clip.write_audiofile(audio_output_path)
    except Exception as e:
        return f"Error processing video: {e}"

    # Transcribe audio using AssemblyAI
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_output_path)
        if transcript.status == aai.TranscriptStatus.error:
            return f"Transcription error: {transcript.error}"
        transcript_text = transcript.text
    except Exception as e:
        return f"Error transcribing audio: {e}"

    # Save transcript to a temporary file
    transcript_file_path = os.path.join(tempfile.gettempdir(), "transcript.txt")
    with open(transcript_file_path, 'w') as file:
        file.write(transcript_text)

    # Generate summary using Generative AI model
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    {"text": transcript_text}
                ]
            }
        ]
    )

    prompt = f"{transcript_text}\nExtract important sentences from the given video transcript, providing long detailed points with a length of more than 100 words. The extracted points should be suitable for creating concise video shorts."
    response = chat_session.send_message(
        {
            "role": "user",
            "parts": [
                {"text": prompt}
            ]
        }
    )
    summary = response.text.replace("*", "").replace("-", "")

    # Save the summary to a temporary file
    summary_file_path = os.path.join(tempfile.gettempdir(), 'summary.txt')
    with open(summary_file_path, 'w', encoding='utf-8') as file:
        file.write(summary)

    # Function to get audio duration
    def get_audio_duration(file_path):
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000  # Convert milliseconds to seconds

    # Function to find matching segments
    def find_segments(video_transcript_path, summarized_text_path, total_video_duration):
        with open(video_transcript_path, 'r') as file:
            video_transcript = file.read()
        with open(summarized_text_path, 'r') as file:
            summarized_text = file.read()

        doc_transcript = nlp(video_transcript)
        doc_summary = nlp(summarized_text)

        matching_segments = []

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

    total_video_duration = get_audio_duration(audio_output_path)
    matching_segments = find_segments(transcript_file_path, summary_file_path, total_video_duration)

    # Function to trim and speed up video
    def trim_and_speedup_video(video_path, start_time, end_time, speed_factor=1.5):
        output_filename = f"clip_{start_time}_{end_time}_speedup.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        command = f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -vf 'setpts={1/speed_factor}*PTS' -af 'atempo={speed_factor}' {output_path}"
        os.system(command)
        return output_path

    output_paths = []
    for start_time, end_time in matching_segments:
        start_time = max(0, start_time - 20)
        end_time = end_time + 20

        if end_time - start_time < 60:
            end_time = start_time + 60

        clip_path = trim_and_speedup_video(video_path, start_time, end_time)
        if clip_path:
            output_paths.append(clip_path)

    return output_paths
