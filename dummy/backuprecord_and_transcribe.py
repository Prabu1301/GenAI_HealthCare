import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import wave
import requests
import os
import subprocess

st.title("🎤 Record Audio and Transcribe")

# 🎙️ Audio processor class with clean mono shaping
class AudioProcessor:
    def __init__(self):
        self.buffer = []

    def recv(self, frame: av.AudioFrame):
        audio_array = frame.to_ndarray().reshape(-1)  # Clean 1D waveform
        self.buffer.append(audio_array.astype(np.int16))
        print("Frame received")  # Debug
        return frame

# 🎛️ Start recording stream
streamer_context = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False}
)

# 🧠 Cache processor after recording starts
if streamer_context and streamer_context.audio_processor:
    st.session_state["processor"] = streamer_context.audio_processor

# 🚫 Prevent transcription during recording
if streamer_context and streamer_context.state.playing:
    st.warning("Please stop recording before transcribing.")
else:
    if st.button("📝 Transcribe"):
        processor = st.session_state.get("processor", None)

        if processor and hasattr(processor, "buffer"):
            st.write(f"Captured frames: {len(processor.buffer)}")

            if processor.buffer:
                # Combine and normalize audio frames
                audio_data = np.concatenate(processor.buffer).astype(np.int16)
                normalized = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

                # Duration check
                duration_sec = len(normalized) / 48000
                st.write(f"Audio duration: {duration_sec:.2f} seconds")
                if duration_sec < 2:
                    st.warning("Recording too short — try speaking for at least 3 seconds.")

                # Save as WAV file
                wav_path = "recorded_audio.wav"
                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(48000)
                    wf.writeframes(normalized.tobytes())

                st.audio(wav_path, format="audio/wav")
                st.write(f"WAV file size: {os.path.getsize(wav_path)} bytes")

                # Convert WAV to MP3 using ffmpeg
                mp3_path = "recorded_audio.mp3"
                subprocess.run([
                    "ffmpeg", "-y", "-i", wav_path, mp3_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                st.audio(mp3_path, format="audio/mp3")
                st.write(f"MP3 file size: {os.path.getsize(mp3_path)} bytes")

                # Send MP3 to FastAPI backend
                with open(mp3_path, "rb") as f:
                    files = {"file": (mp3_path, f, "audio/mp3")}
                    response = requests.post("http://127.0.0.1:8000/transcribe/", files=files)

                # Display transcript or error
                if response.status_code == 200:
                    st.subheader("📝 Transcript")
                    st.write(response.json().get("transcript", "No transcript found."))
                else:
                    st.error(f"Transcription failed: {response.text}")
            else:
                st.warning("No audio recorded yet.")
        else:
            st.warning("Audio processor not initialized.")