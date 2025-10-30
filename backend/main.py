from fastapi import FastAPI, UploadFile, File
import io

app = FastAPI()

# Lazy-load whisper model and anthropic client inside route
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    # Import inside handler to avoid loading on app startup
    import whisper
    from anthropic import Anthropic

    # Load smallest whisper model inside route for memory saving
    model = whisper.load_model("tiny")

    # Read file contents 
    audio_bytes = await file.read()

    # Use Whisper for transcription
    audio_buffer = io.BytesIO(audio_bytes)
    transcription = model.transcribe(audio_buffer)

    # Lazy-load Anthropic client and use it for correction
    anthropic_client = Anthropic(api_key="YOUR_API_KEY")
    correction_response = anthropic_client.completions.create(
        model="claude-v1",
        prompt=f"Correct this transcription: {transcription['text']}",
        max_tokens_to_sample=100,
    )

    return {
        "transcription": transcription["text"],
        "corrected": correction_response.completion,
    }
