import requests
import time
from moviepy.video.io.VideoFileClip import VideoFileClip
import google.generativeai as genai
import os
from pydub import AudioSegment
import spacy
import assemblyai as aai

nlp = spacy.load("en_core_web_lg")

GOOGLE_API_KEY = 'AIzaSyAmyK2-L52gXmIdHbaY5ZwQxaouJaLalBM'
aai.settings.api_key = "df88c6042da143fd9a94044557aea851"

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
        print(f"Error processing video: {e}")
        return

    FILE_URL = "output_audio.wav"

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(FILE_URL)
    transcript_text = transcript.text
    transcript_file_path = "transcript.txt"

    with open(transcript_file_path, 'w') as file:
        file.write(transcript_text)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription error: {transcript.error}")
        return
    else:
        print(f"Transcript: {transcript.text}")

    try:
        # Generate summary using generative AI model
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
        print(f"Error generating summary: {e}")

    def get_audio_duration(file_path):
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000  # Converting milliseconds to seconds

    def find_segments(video_transcript_path, summarized_text_path, total_video_duration):
        # Load spaCy model
        nlp = spacy.load("en_core_web_lg")

        try:
            # Read the video transcript
            with open(video_transcript_path, 'r') as file:
                video_transcript = file.read()

            # Read the summarized text
            with open(summarized_text_path, 'r') as file:
                summarized_text = file.read()
        except Exception as e:
            print(f"Error reading files: {e}")
            return []

        # Process the texts with spaCy
        doc_transcript = nlp(video_transcript)
        doc_summary = nlp(summarized_text)

        # Initialize an empty list to store matching segments
        matching_segments = []

        # Iterate over sentences in the video transcript
        for sentence in doc_transcript.sents:
            # Calculate the similarity between the summarized text and the current sentence in the video transcript
            similarity_score = sentence.similarity(doc_summary)

            # Define a similarity threshold
            similarity_threshold = 0.5

            # If the similarity score is above the threshold, consider it a match
            if similarity_score > similarity_threshold:
                # Extract the start and end character offsets of the matching sentence
                start_char = sentence.start_char
                end_char = sentence.end_char

                # Convert character offsets to time (in seconds) using the start and end time of the video
                start_time = round(start_char / len(doc_transcript.text) * total_video_duration, 0)
                end_time = round(end_char / len(doc_transcript.text) * total_video_duration, 0)

                # Ensure the segment length is within the specified range
                if 10 < (end_time - start_time) < 90:
                    # Add the matching segment to the list
                    matching_segments.append((start_time, end_time))

        return matching_segments

    audio_path = "output_audio.wav"
    total_video_duration = get_audio_duration(audio_path)
    video_transcript = "transcript.txt"
    summarized_text = "summary.txt"
    matching_segments = find_segments(video_transcript, summarized_text, total_video_duration)
    print("Matching Segments:", matching_segments)

    for start_time, end_time in matching_segments:
        print(f"Cut video from {start_time} seconds to {end_time} seconds.")

    def trim_and_speedup_video(video_path, output_folder, start_time, end_time, speed_factor=1.5):
        start_seconds = start_time
        end_seconds = end_time
        output_filename = f"clip_{end_time}_speedup.mp4"
        output_path = os.path.join(output_folder, output_filename)

        # Use FFmpeg to trim and speed up the video
        command = f"ffmpeg -i {video_path} -ss {start_seconds} -to {end_seconds} -vf 'setpts={1/speed_factor}*PTS' -af 'atempo={speed_factor}' {output_path}"
        print(f"Trimming and speeding up video: {output_path}")
        os.system(command)

    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)

    timestamps = matching_segments
    for i in timestamps:
        start_time = i[0] - 20 if i[0] > 20 else i[0]
        end_time = i[1] + 20

        if end_time - start_time < 60:
            end_time = start_time + 60

        trim_and_speedup_video(video_path, output_folder, start_time, end_time)

    print("All video clips trimmed and sped up successfully!")
