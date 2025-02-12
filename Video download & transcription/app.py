import os
import time
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import yt_dlp
import re
from moviepy import VideoFileClip
from pydub import AudioSegment
from whisper import load_model, load_audio


default_model = "base.en" 
model = load_model(default_model)


def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def get_video_id_from_url(video_url):
    if "watch?v=" in video_url:
        return video_url.split("watch?v=")[1].split("&")[0]
    return None


def get_video_transcription(video_id, output_name):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        print("YouTube transcript fetched successfully.")

        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)

        file_path = f"{output_name}_youtube_transcription.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        print(f"YouTube transcription saved as: {file_path}")
        return True
    except Exception as e:
        print(f"Error fetching YouTube transcription: {e}")
        return False


def generate_whisper_transcription(audio_path, output_path):
    try:
        audio = load_audio(audio_path)
        print("Audio loaded successfully for Whisper.")

        start_time = time.time()
        transcription = model.transcribe(audio)
        end_time = time.time()

        with open(output_path, "w", encoding="utf-8") as text_file:
            text_file.write(transcription["text"])
        print(f"Whisper transcription saved to: {output_path}")

        elapsed_time = end_time - start_time
        print(f"Time Taken by Whisper: {elapsed_time:.4f} seconds")
    except Exception as e:
        print(f"Error generating Whisper transcription: {e}")



def check_transcription_validity(transcription):
    patterns = [
        r"\[.*?\]", 
        r"\(.*?\)",  
    ]
    
    combined_pattern = f"^({'|'.join(patterns)})+$"
    
    if re.fullmatch(combined_pattern, transcription.strip()):
        return "not valid"
    else:
        return "valid"


def extract_audio_from_video(video_path, audio_trimmed_path):
    try:
        clip = VideoFileClip(video_path)
        audio_path = "output_audio.wav"
        clip.audio.write_audiofile(audio_path)

        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(audio_trimmed_path, format="wav")

        print(f"Audio extracted and saved to: {audio_trimmed_path}")
        return audio_trimmed_path
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return None



if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=WRONQKUzXTo&ab_channel=CasRaven3D'

 
    opts = {
        'format': 'best',
        'outtmpl': '%(title)s.%(ext)s', 
        'noplaylist': True,
        'postprocessors': [
        {'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}
    ]
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            video_title = sanitize_filename(info_dict.get('title', 'Unknown_Title'))
            print(f"Video downloaded as: {video_title}.mp4")
    except yt_dlp.DownloadError as e:
        print(f"Error downloading video: {e}")
        video_title = None

    if video_title:
        video_id = get_video_id_from_url(video_url)
        if video_id:
            print(f"Attempting to fetch transcription for video ID: {video_id}...")
            transcription_successful = get_video_transcription(video_id, video_title)

            if transcription_successful:
                with open(f"{video_title}_youtube_transcription.txt", "r", encoding="utf-8") as f:
                    transcription_text = f.read()

                transcription_validity = check_transcription_validity(transcription_text)
                if transcription_validity == "not valid":
                    print("Transcription is not valid. Falling back to Whisper transcription...")
                    video_path = f"{video_title}.mp4"
                    audio_trimmed_path = "output_audio_trimmed.wav"
                    audio_path = extract_audio_from_video(video_path, audio_trimmed_path)
                    if audio_path:
                        whisper_output_path = os.path.join(os.path.dirname(audio_trimmed_path), "whisper_transcription.txt")
                        generate_whisper_transcription(audio_trimmed_path, whisper_output_path)
                else:
                    print("Valid transcription found, no need for Whisper transcription.")
            else:
                print("No transcription available. Falling back to Whisper transcription...")
                video_path = f"{video_title}.mp4"
                audio_trimmed_path = "output_audio_trimmed.wav"
                audio_path = extract_audio_from_video(video_path, audio_trimmed_path)
                if audio_path:
                    whisper_output_path = os.path.join(os.path.dirname(audio_trimmed_path), "whisper_transcription.txt")
                    generate_whisper_transcription(audio_trimmed_path, whisper_output_path)
        else:
            print("Error: Could not extract video ID from the provided URL.")
