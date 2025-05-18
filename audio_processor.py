import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
from jamaibase import JamAI, protocol as p

load_dotenv()

PROJECT_ID = os.getenv("JAMAI_PROJECT_ID")
PAT = os.getenv("JAMAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = JamAI(project_id=PROJECT_ID, token=PAT)
openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=120)


def transcribe_audio_whisper(audio_path):
    with open(audio_path, "rb") as f:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
    return transcript.text


def upload_transcription_to_knowledge(transcription_text: str, title: str, blueprint: str = None, table_id: str = None, file_id: str = None):

    row = {
        "Title": title,
        "Text": transcription_text,
        "Source": title,
        "File ID": file_id or str(uuid.uuid4())  # 🔑 required for history view
    }
    if blueprint:
        row["Linked Blueprint"] = blueprint  # ✅ must go here

    client.table.add_table_rows(
        "knowledge",
        p.RowAddRequest(
            table_id=table_id or "knowledge-digital-products",
            data=[row],
            stream=False
        )
    )
