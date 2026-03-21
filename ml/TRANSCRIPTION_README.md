# Local Audio/Video Transcription (Stage 1)

This project supports **local Whisper** transcription for audio/video files and flows the text into the existing Stage 2 (chunking) and Stage 4 (structured extraction) pipeline.

## Requirements

- Python 3.10+ (tested with 3.12)
- `ffmpeg` on PATH
- Local Whisper package: `openai-whisper`
- For this repo, `.env` is loaded via `python-dotenv`

## Local-Only (No API)

This setup uses **local Whisper** only. The OpenAI API is **not** used.

### Install

```bash
python3 -m pip install -U openai-whisper
```

### Environment (.env)

Example keys (use `.env.example` as a template):

```
XDG_CACHE_HOME=/Users/gegekang/Desktop/note-agent-ML-chunk-embed/derived/whisper_cache
SSL_CERT_FILE=/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/certifi/cacert.pem

# Optional tuning
WHISPER_MODEL=base
```

### Common Issues

- **SSL cert errors when downloading models**: ensure `SSL_CERT_FILE` is set in `.env` (see above).
- **Permission errors for cache**: ensure `XDG_CACHE_HOME` points to a writable path.
- **MP4/MP3/M4A need ffmpeg**: confirm `ffmpeg -version` works.

## Run (Bash)

Use the provided script to transcribe a file to `ml/outputs/full_transcript.txt`:

```bash
./ml/transcribe_media.sh /path/to/file.mp4
```

You can also provide a note id and/or output path:

```bash
./ml/transcribe_media.sh /path/to/file.mp4 123 /absolute/path/output.txt
```

## Output

The script writes the full transcript to:

```
/Users/gegekang/Desktop/note-agent-ML-chunk-embed/ml/outputs/full_transcript.txt
```

The pipeline also stores the cleaned text in `derived/<workspace>/<note_id>/extracted.txt`.
