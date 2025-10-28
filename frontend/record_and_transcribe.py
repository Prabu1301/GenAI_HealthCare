import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import wave
import requests
import os
import subprocess
from io import BytesIO

st.title("🎤 Record Audio and Transcribe")

# 🎙️ Robust Audio Processor
class AudioProcessor:
    def __init__(self):
        self.frames = []  # Store raw PyAV frames
        self.sample_rate = None

    def recv(self, frame: av.AudioFrame):
        # Store the entire frame object
        self.frames.append(frame)
        
        # Capture sample rate from first frame
        if self.sample_rate is None:
            self.sample_rate = frame.sample_rate
            st.session_state["detected_rate"] = frame.sample_rate
            print(f"🎵 Detected: {frame.sample_rate}Hz, format={frame.format.name}")
        
        return frame

# 🎛️ WebRTC Streamer
streamer_context = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
        },
        "video": False
    }
)

# Cache processor
if streamer_context and streamer_context.audio_processor:
    st.session_state["processor"] = streamer_context.audio_processor

# Display detected sample rate
if "detected_rate" in st.session_state:
    st.info(f"🎵 Detected Sample Rate: {st.session_state['detected_rate']}Hz")

# Prevent transcription during recording
if streamer_context and streamer_context.state.playing:
    st.warning("🔴 Recording in progress... Stop before transcribing.")
else:
    if st.button("📝 Transcribe Recording"):
        processor = st.session_state.get("processor", None)

        if not processor or not hasattr(processor, "frames") or not processor.frames:
            st.warning("⚠️ No audio recorded. Click START to begin recording.")
        else:
            with st.spinner("Processing audio..."):
                try:
                    st.write(f"✅ Processing {len(processor.frames)} frames")
                    
                    # Use PyAV resampler to properly handle audio format conversion
                    
                    # Create resampler for 16kHz mono output
                    resampler = av.audio.resampler.AudioResampler(
                        format='s16',
                        layout='mono',
                        rate=16000
                    )
                    
                    # Resample all frames
                    resampled_frames = []
                    for frame in processor.frames:
                        resampled = resampler.resample(frame)
                        resampled_frames.extend(resampled)
                    
                    # Flush resampler
                    resampled_frames.extend(resampler.resample(None))
                    
                    # Combine all resampled audio data
                    audio_data = []
                    for frame in resampled_frames:
                        array = frame.to_ndarray()
                        # Handle both planar and packed formats
                        if array.ndim > 1:
                            array = array.flatten('F')  # Fortran order for planar
                        audio_data.append(array)
                    
                    # Concatenate all audio
                    if not audio_data:
                        st.error("❌ No audio data after resampling")
                        st.stop()
                    
                    audio_array = np.concatenate(audio_data).astype(np.int16)
                    
                    # Ensure it's 1D
                    if audio_array.ndim > 1:
                        audio_array = audio_array.flatten()
                    
                    audio_bytes = audio_array.tobytes()
                    
                    # Save as WAV with proper headers
                    wav_path = "recorded_audio.wav"
                    with wave.open(wav_path, "wb") as wf:
                        wf.setnchannels(1)  # Mono
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(16000)  # 16kHz
                        wf.writeframes(audio_bytes)
                    
                    # Verify file
                    file_size = os.path.getsize(wav_path)
                    st.write(f"💾 WAV Size: {file_size} bytes")
                    
                    if file_size < 1000:
                        st.error("❌ Audio file too small. Recording may have failed.")
                    else:
                        # Play audio for verification
                        st.audio(wav_path, format="audio/wav")
                        
                        # Calculate duration
                        with wave.open(wav_path, 'rb') as wf:
                            frames = wf.getnframes()
                            rate = wf.getframerate()
                            duration = frames / float(rate)
                            st.write(f"⏱️ Duration: {duration:.2f}s")
                        
                        if duration < 0.5:
                            st.warning("⚠️ Recording too short. Please record at least 1 second.")
                        else:
                            # Convert to MP3 for backend (optional, or send WAV directly)
                            mp3_path = "recorded_audio.mp3"
                            result = subprocess.run([
                                "ffmpeg", "-y", "-i", wav_path,
                                "-ar", "16000",
                                "-ac", "1",
                                "-b:a", "64k",
                                "-map_metadata", "-1",  # Remove metadata
                                mp3_path
                            ], capture_output=True, text=True)
                            
                            if result.returncode != 0:
                                st.error(f"❌ FFmpeg Error: {result.stderr}")
                            else:
                                st.write(f"💾 MP3 Size: {os.path.getsize(mp3_path)} bytes")
                                st.audio(mp3_path, format="audio/mp3")
                                
                                # Send to backend
                                with st.spinner("Transcribing..."):
                                    with open(mp3_path, "rb") as f:
                                        files = {"file": ("recording.mp3", f, "audio/mp3")}
                                        try:
                                            response = requests.post(
                                                "http://127.0.0.1:8000/transcribe/",
                                                files=files,
                                                timeout=60
                                            )
                                            
                                            if response.status_code == 200:
                                                transcript = response.json().get("transcript", "")
                                                if transcript:
                                                    st.subheader("📝 Transcript")
                                                    st.success(transcript)
                                                else:
                                                    st.warning("⚠️ No transcript returned from API")
                                            else:
                                                st.error(f"❌ API Error ({response.status_code}): {response.text}")
                                        
                                        except requests.exceptions.Timeout:
                                            st.error("❌ Request timeout. Backend may be slow.")
                                        except requests.exceptions.ConnectionError:
                                            st.error("❌ Cannot connect to backend at http://127.0.0.1:8000")
                                        except Exception as e:
                                            st.error(f"❌ Unexpected error: {str(e)}")
                
                except Exception as e:
                    st.error(f"❌ Processing Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Add clear button
if st.button("🗑️ Clear Recording"):
    if "processor" in st.session_state:
        processor = st.session_state["processor"]
        if hasattr(processor, "frames"):
            processor.frames = []
            st.success("✅ Recording cleared")
    st.rerun()