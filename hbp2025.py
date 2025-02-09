import os
import requests
import threading
import pandas as pd
import spotipy
import wave
import pyaudio
from flask import Flask, request, jsonify, session, redirect, render_template, copy_current_request_context
from spotipy.oauth2 import SpotifyOAuth
from urllib.parse import urlencode
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
import asyncio
import time
import random  # For random song selection

from pairing_model import get_pairings


# Your API keys and credentials
HUME_API_KEY = "xXQA3btAKN3pAcTkq0etLlqEcTns4jcZWNCJPMFdQ2AXS1oQ"
SPOTIFY_CLIENT_ID = "bcf78faf174e4bcabcce73773e1f650f"
SPOTIFY_CLIENT_SECRET = "93c3a69720d24a55ad13d22c6058ae5e"

SPOTIFY_REDIRECT_URI = "http://localhost:8080/callback"
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"

app = Flask(__name__)
app.secret_key = "supersecretkey"

df = pd.read_csv("spotify_dataset.csv")

recording = False
recording_thread = None
last_recording_file = None  # Global variable to store the most recent recording file path

def get_spotify_auth_url():
    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": "user-modify-playback-state user-read-playback-state",
        "show_dialog": "true"
    }
    return f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"

@app.route("/")
def home():
    logged_in = "token_info" in session
    return render_template("index.html", logged_in=logged_in)

@app.route("/login")
def login():
    return redirect(get_spotify_auth_url())

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/callback")
def callback():
    if "error" in request.args:
        return jsonify({"error": request.args["error"]})
    if "code" in request.args:
        code = request.args["code"]
        req_body = {
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": SPOTIFY_REDIRECT_URI,
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET
        }
        response = requests.post(SPOTIFY_TOKEN_URL, data=req_body)
        token_info = response.json()
        if "access_token" not in token_info:
            return jsonify({"error": "Spotify authentication failed"}), 401
        session["token_info"] = token_info
        access_token = token_info["access_token"]
        return redirect(f"http://localhost:3000/voices?accessToken={access_token}")

def record_audio():
    global recording, last_recording_file
    # Generate a unique filename using the current timestamp.
    timestamp = int(time.time())
    file_path = f"uploads/recorded_audio_{timestamp}.wav"
    os.makedirs("uploads", exist_ok=True)
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=1024
    )
    frames = []
    print(f"[{time.strftime('%X')}] Recording started... Speak now!")
    while recording:
        try:
            data = stream.read(1024)
        except Exception as e:
            print("Error reading stream:", e)
            break
        frames.append(data)
    print(f"[{time.strftime('%X')}] Recording stopped.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
    
    print(f"Recording saved: {file_path}")
    last_recording_file = file_path
    print("Global last_recording_file set to:", last_recording_file)
    return file_path

@app.route('/start_recording', methods=['POST'])
def start_recording_route():
    global recording, recording_thread
    if recording:
        return jsonify({"error": "Already recording"}), 400
    recording = True

    @copy_current_request_context
    def record():
        record_audio()
    recording_thread = threading.Thread(target=record)
    recording_thread.start()
    return jsonify({"message": "Recording started"}), 200

@app.route('/stop_recording', methods=['POST'])
def stop_recording_route():
    global recording, recording_thread
    if not recording:
        return jsonify({"error": "Not recording"}), 400
    recording = False
    recording_thread.join()  # Wait for the recording thread to finish.
    return jsonify({"message": "Recording stopped"}), 200

async def analyze_emotion(file_path):
    # Convert the audio file for Hume processing.
    os.system(f"ffmpeg -y -i {file_path} -acodec pcm_s16le -ar 44100 {file_path}_fixed.wav")
    fixed_file = f"{file_path}_fixed.wav"
    client = AsyncHumeClient(api_key=HUME_API_KEY)
    model_config = Config(prosody={})
    stream_options = StreamConnectOptions(config=model_config)
    async with client.expression_measurement.stream.connect(options=stream_options) as socket:
        result = await socket.send_file(fixed_file)
        if result.prosody and result.prosody.predictions:
            emotions = result.prosody.predictions[0].emotions
            strongest_emotion = max(emotions, key=lambda e: e.score)
            print("Analyzed emotion:", strongest_emotion.name)
            return strongest_emotion.name
    return None

def mood_to_music_features(mood):
    # (Use your extensive mood mapping as before)
    mood_map = {
        "Admiration": {"valence": 0.8, "energy": 0.6, "danceability": 0.7, "acousticness": 0.4, "tempo": 110},
        # ... (all other moods) ...
        "Triumph": {"valence": 0.9, "energy": 0.9, "danceability": 0.8, "acousticness": 0.2, "tempo": 140},
    }
    return mood_map.get(mood, {"valence": 0.5, "energy": 0.5, "danceability": 0.5, "acousticness": 0.8, "tempo": 100})

def find_closest_song(valence, energy, danceability, acousticness, tempo):
    filtered_df = df[~df["decade"].isin(["60s", "70s"])].copy()
    filtered_df["score"] = (
        abs(filtered_df["valence"] - valence) +
        abs(filtered_df["energy"] - energy) +
        abs(filtered_df["acousticness"] - acousticness) +
        abs(filtered_df["danceability"] - danceability) +
        abs(filtered_df["tempo"] - tempo) / 200
    )
    top_candidates = filtered_df.nsmallest(20, "score")
    if not top_candidates.empty:
        selected_song = top_candidates.sample(n=1).iloc[0]
        track_uri = str(selected_song["uri"]).split(":")[-1]
        return {
            "track_id": track_uri,
            "track_name": str(selected_song["track"]),
            "artist": str(selected_song["artist"]),
        }
    return None

def queue_song(track_id, provided_token=None):
    user_token = provided_token or session.get("token_info", {}).get("access_token")
    if not user_token:
        print("Spotify authentication required.")
        return
    sp = spotipy.Spotify(auth=user_token)
    sp.add_to_queue(f"spotify:track:{track_id}")
    
@app.route('/process_audio', methods=['POST'])
def process_audio_route():
    # Check if an audio file was sent with the request.
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # Save the uploaded file to disk with a unique name.
    file = request.files["audio"]
    timestamp = int(time.time())
    file_path = f"uploads/recorded_audio_{timestamp}.wav"
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    print(f"Recording saved from request: {file_path}")

    # Process the file (analyze emotion, etc.).
    strongest_emotion = asyncio.run(analyze_emotion(file_path))
    print("Strongest emotion returned:", strongest_emotion)
    
    # Extract token from the header if provided.
    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    provided_token = None
    if auth_header and auth_header.startswith("Bearer "):
        provided_token = auth_header[len("Bearer "):].strip()
    
    if strongest_emotion:
        music_features = mood_to_music_features(strongest_emotion)
        closest_song = find_closest_song(**music_features)
        if closest_song:
            queue_song(closest_song["track_id"], provided_token=provided_token)
            try:
                # Give a short delay to ensure the file is no longer in use.
                time.sleep(3)
                os.remove(file_path)
                print(f"Deleted recording: {file_path}")
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")
            return jsonify({"emotion": strongest_emotion, "matched_song": closest_song}), 200
    return jsonify({"error": "No emotion detected"}), 500

#---------------------Friends Matching api -------------------------------------
@app.route('/', methods=['GET'])
def default():
    return "Hello World!"

@app.route('/matchPairs', methods=['GET'])
def get_match_pairs():
    return pair_results

@app.route('/matchPairs', methods=['POST'])
def add_match_pair():
    global pair_results  # Declare pair_results as global
    pairings = get_pairings()
    pairs_jsonify = jsonify(pairings)
    pair_results = pairs_jsonify
    return pair_results


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, port=8080)
