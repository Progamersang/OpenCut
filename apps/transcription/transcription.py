import modal
from pydantic import BaseModel

app = modal.App("opencut-transcription")

# List of supported Whisper models
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v3"]

class TranscribeRequest(BaseModel):
    filename: str
    language: str = "auto"  # Default is auto-detect
    model_size: str = "base"  # Default is 'base'
    decryptionKey: str = None
    iv: str = None
    # Add offset option for flexibility
    time_offset: float = 0.5 # Default remains 0.5s as before, can be changed via request

@app.function(
    image=modal.Image.debian_slim()
        .apt_install(["ffmpeg"]) # Required for Whisper audio processing
        .pip_install(["openai-whisper", "boto3", "fastapi[standard]", "pydantic", "cryptography"]),
    gpu="A10G", # Can be adjusted based on model_size if needed
    timeout=600, # Increased to 10 minutes to support larger models or longer files
    secrets=[modal.Secret.from_name("opencut-r2-secrets")]
)
@modal.fastapi_endpoint(method="POST")
def transcribe_audio(request: TranscribeRequest):
    import whisper
    import boto3
    import tempfile
    import os
    import json
    import logging

    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        filename = request.filename
        language = request.language
        model_size = request.model_size.lower()
        decryption_key = request.decryptionKey
        iv = request.iv
        time_offset = request.time_offset # Get offset value from request

        # Check if model_size is valid
        if model_size not in WHISPER_MODELS:
            return {
                "error": f"Invalid model size. Supported models: {', '.join(WHISPER_MODELS)}"
            }

        if not filename:
            return {
                "error": "Missing filename parameter"
            }

        logger.info(f"Starting transcription for {filename} using model {model_size} and language {language}")

        # Initialize R2 client
        s3_client = boto3.client(
            's3',
            endpoint_url=f'https://{os.environ["CLOUDFLARE_ACCOUNT_ID"]}.r2.cloudflarestorage.com',
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name='auto'
        )

        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name

            try:
                # Download audio from R2
                logger.info(f"Downloading {filename} from R2...")
                s3_client.download_file(
                    os.environ["R2_BUCKET_NAME"],
                    filename,
                    temp_path
                )
                logger.info(f"Downloaded {filename} successfully.")

                # If decryption key is provided, perform decryption
                if decryption_key and iv:
                    logger.info("Decrypting audio file...")
                    import base64
                    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                    from cryptography.hazmat.backends import default_backend

                    with open(temp_path, 'rb') as f:
                        encrypted_data = f.read()

                    key_bytes = base64.b64decode(decryption_key)
                    iv_bytes = base64.b64decode(iv)

                    # Assume data was encrypted using AES-GCM, need to extract tag
                    tag = encrypted_data[-16:]
                    ciphertext = encrypted_data[:-16]

                    cipher = Cipher(
                        algorithms.AES(key_bytes),
                        modes.GCM(iv_bytes, tag),
                        backend=default_backend()
                    )
                    decryptor = cipher.decryptor()
                    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

                    with open(temp_path, 'wb') as f:
                        f.write(decrypted_data)
                    logger.info("Decryption completed.")

                # Load Whisper model
                logger.info(f"Loading Whisper model: {model_size}...")
                # Whisper will automatically use GPU if available
                model = whisper.load_model(model_size)
                logger.info(f"Model {model_size} loaded successfully.")

                # Transcribe audio
                logger.info(f"Starting transcription...")
                options = {"task": "transcribe"}
                if language != "auto":
                    options["language"] = language.lower()

                # Additional options like beam_size, best_of, temperature can be added here if needed
                # options["beam_size"] = 5
                # options["best_of"] = 5
                # options["temperature"] = 0.0

                result = model.transcribe(temp_path, **options)
                logger.info(f"Transcription completed.")

                # Delete audio file from R2 (privacy/security cleanup)
                logger.info(f"Deleting {filename} from R2 after processing...")
                s3_client.delete_object(
                    Bucket=os.environ["R2_BUCKET_NAME"],
                    Key=filename
                )
                logger.info(f"{filename} deleted from R2.")

                # Adjust segment timing
                logger.info("Adjusting segment timings...")
                adjusted_segments = []
                for segment in result["segments"]:
                    adjusted_segment = segment.copy()
                    # Shift start/end times by the specified offset, ensure start >= 0
                    adjusted_segment["start"] = max(0, segment["start"] - time_offset)
                    # Ensure end time is at least start time + min duration (e.g., 0.5s) or original adjusted end
                    adjusted_segment["end"] = max(adjusted_segment["start"] + 0.5, segment["end"] - time_offset)
                    adjusted_segments.append(adjusted_segment)
                logger.info("Timing adjustment completed.")

                # Return adjusted results
                return {
                    "text": result["text"],
                    "segments": adjusted_segments,
                    "language": result["language"],
                    "model_used": model_size # Return the model used for tracking
                }

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.info(f"Temporary file {temp_path} deleted.")

    except Exception as e:
        import traceback
        logger.error(f"Transcription error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return error response matching expected format
        return {
            "error": str(e),
            "text": "",
            "segments": [],
            "language": "unknown",
            "model_used": model_size if 'model_size' in locals() else "unknown" # Return model if possible
        }

@app.local_entrypoint()
def main():
    # Test function - can be called with `modal run transcription.py`
    print("Transcription service is ready to deploy!")
    print("Deploy with: modal deploy transcription.py")
    print("Supported models:", WHISPER_MODELS)
