
# -*- coding: utf-8 -*-

import streamlit as st
from jamaibase import JamAI, protocol as p
from audio_processor import transcribe_audio_whisper, upload_transcription_to_knowledge
import os
import time
import json
import re
from dotenv import load_dotenv
from notion_client import Client as NotionClient
import whisper
import cv2
import pytesseract
from collections import Counter
import yt_dlp
from urllib.parse import urlparse
from auth_app import authenticate_user

st.set_page_config(page_title="Digital Product Creator", layout="wide")

# Load environment variables
load_dotenv()

# Initialize JamAI client
jamai = JamAI(
    project_id=os.getenv("JAMAI_PROJECT_ID"),
    token=os.getenv("JAMAI_API_KEY")
)

# Authenticate user first
name, email, paid_status, authenticated = authenticate_user()
is_paid_user = paid_status

st.title("Digital Product Creator - Audio to Guide")

if "audio_upload_count" not in st.session_state:
    st.session_state.audio_upload_count = 0

if "pdf_upload_count" not in st.session_state:
    st.session_state.pdf_upload_count = 0

if "remix_upload_count" not in st.session_state:
    st.session_state.remix_upload_count = 0

CONNECTION_FILE = ".notion_connection.json"
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Product Blueprint", "Upload + Transcribe", "Create Outlines / Guides", "Connect to Notion", "Remix Existing Video"])

def download_video_from_url(url, output_path="downloaded_video.mp4"):
    """
    Downloads a video and extracts metadata like caption using yt-dlp.
    """
    ydl_opts = {
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'format': 'mp4',
        'noplaylist': True,
        'skip_download': False,
        'writesubtitles': False,
        'writeinfojson': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            caption = info.get("description", "")  # this is the TikTok caption
        return output_path, caption
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, ""

def shorten_url(url):
    try:
        parsed = urlparse(url)
        if parsed.netloc:
            return parsed.netloc.replace("www.", "")  # e.g. tiktok.com
        return url
    except:
        return url
    
def clean_remix_text(text):
    # Remove leading markdown headers like ### or ##
    return re.sub(r"^#+\s*", "", text.strip())

# Whisper Transcription + Filter
def transcribe_video_audio(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    text = result["text"].strip()

    # Reject very short or repetitive transcripts
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 5:
        return ""
    most_common = Counter(words).most_common(1)
    if most_common[0][1] / len(words) > 0.5:
        return ""
    return text

def extract_text_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_texts = []
    frame_rate = 30  # 1 frame per second (for 30fps)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            if text.strip():
                frame_texts.append(text.strip())
        frame_count += 1
    cap.release()
    return "\n".join(frame_texts)

def clean_temp_files():
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")

def show_progress(total_steps=100):
    progress_bar = st.progress(0)
    for i in range(total_steps):
        time.sleep(0.03)
        progress_bar.progress(i + 1)
    progress_bar.empty()

def save_connection_to_file(api_key, parent_page_id):
    with open(CONNECTION_FILE, "w") as f:
        json.dump({"api_key": api_key, "parent_page_id": parent_page_id}, f)

def load_connection_from_file():
    if os.path.exists(CONNECTION_FILE):
        with open(CONNECTION_FILE, "r") as f:
            data = json.load(f)
            return data.get("api_key", ""), data.get("parent_page_id", "")
    return "", ""

def trim_incomplete_sentence(text: str) -> str:
    """
    Trims the final sentence if it ends mid-way (i.e. no punctuation).
    """
    if not text.strip():
        return text

    endings = [".", "!", "?"]
    last_good = max(text.rfind(e) for e in endings)
    
    # If no complete sentence found, return as-is
    if last_good == -1:
        return text

    return text[:last_good + 1]

def create_notion_page(api_key, parent_page_id, title, content):
    notion = NotionClient(auth=api_key)
    try:
        content = trim_incomplete_sentence(content)
        cleaned_content = clean_content(content)
        blocks = split_text_into_blocks(cleaned_content)
        first_100_blocks = blocks[:100]
        page = notion.pages.create(
            parent={"page_id": parent_page_id},
            properties={"title": {"title": [{"type": "text", "text": {"content": title}}]}},
            children=first_100_blocks
        )
        page_id = page["id"]
        for i in range(100, len(blocks), 100):
            batch = blocks[i:i+100]
            notion.blocks.children.append(block_id=page_id, children=batch)
        return True
    except Exception as e:
        print(e)
        return False

def clean_content(content: str) -> str:
    content = re.sub(r"#+ ", "", content)
    content = content.replace("**", "")
    content = re.sub(r"\n-{2,}\n", "\n", content)
    return content

def split_text_into_blocks(content: str) -> list:
    lines = content.split("\n")
    blocks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("- "):
            blocks.append({"object": "block", "type": "bulleted_list_item",
                           "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line[2:].strip()}}]}})
        else:
            blocks.append({"object": "block", "type": "paragraph",
                           "paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]}})
    return blocks

def extract_lesson_titles(text):
    matches = []
    for line in text.splitlines():
        clean_line = re.sub(r"[*_#>`~-]", "", line).strip()
        if re.match(r"^(Module|Lesson|Week)\s+(?:\d+|[IVXLCDM]+):", clean_line, re.IGNORECASE):
            matches.append(clean_line)
    return matches

def generate_video_scripts(guide_text):
    lessons = extract_lesson_titles(guide_text)
    if not lessons:
        return [], "❌ No lesson titles found. Try using headings like 'Lesson 1:' or 'Module 2:' in your guide."

    scripts = []
    previous_script_summary = ""

    for i, lesson in enumerate(lessons):
        # Add summary to context for next lesson
        context = f"Previous Summary: {previous_script_summary}\n\n" if previous_script_summary else ""

        prompt = f"""
You are a YouTube educator and expert course scriptwriter. Write a detailed, high-quality video script for the lesson titled:
"{lesson}"

{context}This lesson is part of a larger video course. Do NOT repeat intros. Avoid clichés like “Alright, listen up” or “Let’s dive in.”
Each script should feel like a continuation — not a restart.

Include:
- 3–5 key teaching points
- Transitions that feel natural between ideas
- Examples, analogies, or frameworks
- A motivating takeaway at the end

Use second-person voice (“you”) to speak directly to the viewer. No fluff.
        """

        completion = jamai.table.add_table_rows(
            "action",
            p.RowAddRequest(
                table_id="action-video-script-generator",
                data=[{"lesson_topic": prompt}],
                stream=True,
            ),
        )

        script_text = "".join(chunk.text for chunk in completion if hasattr(chunk, "text"))
        script_text = trim_incomplete_sentence(script_text)

        scripts.append((lesson, script_text))

        # Optional: summarize or clip part of this to inject later
        previous_script_summary = script_text.strip().replace("\n", " ")[:350]

    return scripts, None

def parse_remix_ideas(text):
    """
    Parses the remix output into 3 separate remix sections.
    It assumes each idea starts with '### Remix Idea' header.
    """
    parts = re.split(r"###\s*Remix Idea \d+[:：]?", text)
    ideas = [p.strip() for p in parts if p.strip()]
    return ideas

def log_remix_to_jamai(video_url, remix_ideas, caption=""):
    try:
        jamai.table.add_table_rows(
            "action",
            p.RowAddRequest(
                table_id="action-remix-history",
                data=[{
                    "video_link": video_url,
                    "video_caption": caption,
                    "remix_1": remix_ideas[0] if len(remix_ideas) > 0 else "",
                    "remix_2": remix_ideas[1] if len(remix_ideas) > 1 else "",
                    "remix_3": remix_ideas[2] if len(remix_ideas) > 2 else "",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }],
                stream=False
            )
        )
    except Exception as e:
        st.warning(f"⚠️ Failed to log remix to history: {e}")

def log_outline_or_guide(user_input, content, is_guide=False):
    try:
        jamai.table.add_table_rows(
            "action",
            p.RowAddRequest(
                table_id="action-outline-guide-history",
                data=[{
                    "user_input": user_input,
                    "type": "guide" if is_guide else "outline",
                    "content": content,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }],
                stream=False
            )
        )
    except Exception as e:
        st.warning(f"⚠️ Failed to save to history: {e}")

# 📜 New History Viewer Function
def display_remix_history():
    st.subheader("📜 Remix History")

    try:
        rows = jamai.table.list_table_rows("action", "action-remix-history")
        if not rows.items:
            st.info("No remix history yet.")
            return

        for row in rows.items:
            video_url = row.get("video_link", {}).get("value", "")
            video_caption = row.get("video_caption", {}).get("value", "")
            remix_1 = clean_remix_text(row.get("remix_1", {}).get("value", ""))
            remix_2 = clean_remix_text(row.get("remix_2", {}).get("value", ""))
            remix_3 = clean_remix_text(row.get("remix_3", {}).get("value", ""))
            row_id = row.get("ID")

            label = (
                "📁 Local Upload" if not video_url or video_url == "Uploaded file"
                else shorten_url(video_url)
            )

            with st.expander(f"🎬 {label}"):
                if video_url and video_url != "Uploaded file":
                    st.markdown(f"[🔗 View Original Video]({video_url})", unsafe_allow_html=True)

                if video_caption:
                    st.markdown(f"📝 **Caption:** {video_caption}")

                if remix_1:
                        st.markdown(f"**Remix 1:**\n\n{remix_1}")
                        st.divider()

                if remix_2:
                    st.markdown(f"**Remix 2:**\n\n{remix_2}")
                    st.divider()

                if remix_3:
                    st.markdown(f"**Remix 3:**\n\n{remix_3}")

                if st.button("❌ Delete this entry", key=f"delete_{row_id}"):
                    jamai.table.delete_table_rows(
                        "action",
                        p.RowDeleteRequest(
                            table_id="action-remix-history",
                            row_ids=[row_id]
                        )
                    )
                    st.rerun()

    except Exception as e:
        st.warning(f"⚠️ Failed to load history: {e}")

def display_outline_and_guide_history():
    st.subheader("📚 Outline and Guide History")

    # --- Outlines ---
    st.markdown("### 📝 Outlines")
    try:
        outline_rows = jamai.table.list_table_rows("action", "action-outline-generator")
        if not outline_rows.items:
            st.info("No outlines generated yet.")
        else:
            for row in outline_rows.items:
                topic = row.get("user_instruction", {}).get("value", "")
                outline = row.get("outline", {}).get("value", "")
                row_id = row.get("ID")

                with st.expander(f"📝 {topic}"):
                    st.markdown(outline or "_No outline found._")
                    if st.button(f"❌ Delete this outline", key=f"delete_outline_{row_id}"):
                        jamai.table.delete_table_rows(
                            "action",
                            p.RowDeleteRequest(
                                table_id="action-outline-generator",
                                row_ids=[row_id]
                            )
                        )
                        st.success(f"✅ Deleted outline for '{topic}'")
                        st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Failed to load outlines: {e}")

    st.divider()

    # --- Full Guides ---
    st.markdown("### 📖 Full Guides")
    try:
        guide_rows = jamai.table.list_table_rows("action", "action-full-guide-generator")
        if not guide_rows.items:
            st.info("No full guides generated yet.")
        else:
            for row in guide_rows.items:
                topic = row.get("user_instruction", {}).get("value", "")
                guide = row.get("guide", {}).get("value", "")
                row_id = row.get("ID")

                with st.expander(f"📖 {topic}"):
                    st.markdown(guide or "_No guide found._")
                    if st.button(f"❌ Delete this guide", key=f"delete_guide_{row_id}"):
                        jamai.table.delete_table_rows(
                            "action",
                            p.RowDeleteRequest(
                                table_id="action-full-guide-generator",
                                row_ids=[row_id]
                            )
                        )
                        st.success(f"✅ Deleted guide for '{topic}'")
                        st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Failed to load guides: {e}")

def display_product_blueprint_history():
    st.subheader("📐 Product Blueprint History")

    try:
        rows = jamai.table.list_table_rows("action", "action-product-blueprint")
        if not rows.items:
            st.info("No product blueprints generated yet.")
            return

        for row in rows.items:
            product_title = row.get("title", {}).get("value", "Untitled")
            timestamp = row.get("timestamp", {}).get("value", "")
            blueprint = row.get("product_blueprint", {}).get("value", "")
            row_id = row.get("ID")

            with st.expander(f"🧠 {product_title} — {timestamp}"):
                st.markdown(blueprint or "_No blueprint found._")
                if st.button(f"❌ Delete this blueprint", key=f"delete_blueprint_{row_id}"):
                    jamai.table.delete_table_rows(
                        "action",
                        p.RowDeleteRequest(
                            table_id="action-product-blueprint",
                            row_ids=[row_id]
                        )
                    )
                    st.success("✅ Deleted blueprint.")
                    st.rerun()

    except Exception as e:
        st.warning(f"⚠️ Failed to load blueprint history: {e}")

if "notion_api_key" not in st.session_state or "notion_parent_page_id" not in st.session_state:
    api_key, parent_page_id = load_connection_from_file()
    st.session_state.notion_api_key = api_key
    st.session_state.notion_parent_page_id = parent_page_id

with tab1:
    st.header("🎯 Product Blueprint")

    st.markdown("Define the **core strategy** behind your digital product. This helps us generate outlines, guides, and creative assets that are aligned and monetizable.")

    with st.form("blueprint_form"):
        col1, col2 = st.columns(2)
        with col1:
            target_audience = st.text_input("Who is this product for? (Target Audience)")
        with col2:
            transformation = st.text_input("What outcome does it deliver? (Big Promise)")

        col3, col4 = st.columns(2)
        with col3:
            delivery_method = st.selectbox("How will it be delivered?", ["Video Course", "eBook", "Live Cohort", "Templates", "Toolkit", "Workshop", "PDF Guide"])
        with col4:
            product_title = st.text_input("What is the working title?")

        product_pitch = st.text_area("One-Line Product Pitch (optional)", placeholder="e.g. A 7-day toolkit to help side hustlers launch their first online product with zero audience.")

        blueprint_submitted = st.form_submit_button("🧠 Generate Product Blueprint")

    if blueprint_submitted and target_audience and transformation and product_title:
        blueprint_prompt = f"""
You're an elite digital product strategist. Based on the following inputs, create a concise product blueprint that defines the offer and pitch for a profitable digital product.

Target Audience: {target_audience}
Big Promise / Outcome: {transformation}
Delivery Method: {delivery_method}
Product Title: {product_title}
Optional Pitch: {product_pitch}

Include:
- Product Subtitle or Tagline
- Core Transformation (One Sentence)
- 3–5 Sales Bullet Points
- 1–2 Positioning Notes (what makes it unique or timely)
"""

        with st.spinner("🎨 Generating your product blueprint..."):
            blueprint_response = jamai.table.add_table_rows(
                "action",
                p.RowAddRequest(
                    table_id="action-product-blueprint",
                    data=[{
                        "title": product_title,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "blueprint_prompt": blueprint_prompt
                    }],
                    stream=True,
                ),
            )

            blueprint_output = "".join(chunk.text for chunk in blueprint_response if hasattr(chunk, "text"))

        st.subheader("🧱 Product Blueprint")
        st.markdown(blueprint_output)
        st.download_button("📥 Download Blueprint", blueprint_output, file_name="product_blueprint.txt", mime="text/plain")

    elif blueprint_submitted:
        st.warning("❗ Please fill in all required fields to generate a blueprint.")

    st.divider()
    st.subheader("📐 Product Blueprint History")

    try:
        rows = jamai.table.list_table_rows("action", "action-product-blueprint")
        if not rows.items:
            st.info("No product blueprints generated yet.")
        else:
            sorted_rows = sorted(rows.items, key=lambda r: r.get("timestamp", {}).get("value", ""), reverse=True)
            for row in sorted_rows:
                product_title = row.get("title", {}).get("value", "Untitled")
                timestamp = row.get("timestamp", {}).get("value", "")
                blueprint = row.get("product_blueprint", {}).get("value", "")
                row_id = row.get("ID")

                with st.expander(f"🧠 {product_title} — {timestamp}"):
                    st.markdown(blueprint or "_No blueprint found._")
                    if st.button(f"❌ Delete this blueprint", key=f"delete_blueprint_{row_id}"):
                        jamai.table.delete_table_rows(
                            "action",
                            p.RowDeleteRequest(
                                table_id="action-product-blueprint",
                                row_ids=[row_id]
                            )
                        )
                        st.success("✅ Deleted blueprint.")
                        st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Failed to load blueprint history: {e}")

with tab2:
    st.header("Upload Your Files")

    # Limits
    AUDIO_LIMIT = 3
    PDF_LIMIT = 5

    # Track counters (initialize if missing)
    if "audio_upload_count" not in st.session_state:
        st.session_state.audio_upload_count = 0
    if "pdf_upload_count" not in st.session_state:
        st.session_state.pdf_upload_count = 0

    # Upload components
    audio_files = st.file_uploader("Upload up to 3 audio files", type=["mp3", "wav"], accept_multiple_files=True, key="audio_upload")
    pdf_files = st.file_uploader("Upload up to 5 PDF documents", type=["pdf"], accept_multiple_files=True, key="pdf_upload")

    # --- Free user limit handling
    if not is_paid_user:
        st.info(f"🎧 Audio uploads used: {st.session_state.audio_upload_count}/{AUDIO_LIMIT}")
        st.info(f"📄 PDF uploads used: {st.session_state.pdf_upload_count}/{PDF_LIMIT}")

        if st.session_state.audio_upload_count >= AUDIO_LIMIT:
            st.warning("❌ Free tier audio upload limit reached.")
            audio_files = []  # Disable new uploads

        if st.session_state.pdf_upload_count >= PDF_LIMIT:
            st.warning("❌ Free tier PDF upload limit reached.")
            pdf_files = []  # Disable new uploads

    # --- Button conditions
    process_disabled = (not audio_files and not pdf_files)

    if (audio_files or pdf_files) and st.button("▶️ Start Processing Uploaded Files", disabled=process_disabled):
        st.info("🚀 Starting upload and embedding...")
        total_files = len(audio_files) + len(pdf_files)
        current_step = 0
        progress_bar = st.progress(0)

        for audio in audio_files:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.read())
            transcription = transcribe_audio_whisper("temp_audio.wav")
            upload_transcription_to_knowledge(transcription, title=audio.name)
            clean_temp_files()
            current_step += 1
            st.session_state.audio_upload_count += 1  # ✅ Increment
            progress_bar.progress(current_step / total_files)

        for pdf in pdf_files:
            with open("temp.pdf", "wb") as f:
                f.write(pdf.read())
            jamai.table.embed_file("temp.pdf", "knowledge-digital-products")
            clean_temp_files()
            current_step += 1
            st.session_state.pdf_upload_count += 1  # ✅ Increment
            progress_bar.progress(current_step / total_files)

        progress_bar.empty()
        st.success("✅ All files processed and embedded!")

    # --- Display Uploaded Files (Knowledge Table)
    st.divider()
    st.subheader("📚 Uploaded Files (Knowledge)")

    try:
        rows = jamai.table.list_table_rows("knowledge", "knowledge-digital-products")
        if not rows.items:
            st.info("No uploaded files yet.")
        else:
            titles = {}
            for row in rows.items:
                title = row.get("Title", {}).get("value", "")
                if title:
                    titles[title] = titles.get(title, 0) + 1

            for title_name, count in titles.items():
                icon = "🎵" if title_name.lower().endswith((".mp3", ".wav", ".m4a")) else "📄"

                with st.expander(f"{icon} {title_name} ({count} pages)"):
                    st.markdown(f"**Uploaded File:** {title_name}")
                    if st.button(f"❌ Delete '{title_name}'", key=f"delete_{title_name}"):
                        to_delete = [
                            row["ID"]
                            for row in rows.items
                            if row.get("Title", {}).get("value", "") == title_name
                        ]
                        jamai.table.delete_table_rows(
                            "knowledge",
                            p.RowDeleteRequest(
                                table_id="knowledge-digital-products",
                                row_ids=to_delete
                            )
                        )
                        st.success(f"✅ Deleted {len(to_delete)} pages from '{title_name}'")
                        st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Failed to load knowledge uploads: {e}")

with tab3:
    st.header("Generate Outlines or Full Guides")
    st.subheader("Account Access")
    user_tier = st.radio("Select your plan:", ("Free / Starter", "Pro / Paid"), index=0, key="user_tier")
    st.divider()
    st.subheader("Step 1: Enter Your Guide Topic")
    with st.form("generation_form"):
        user_topic = st.text_input("Enter your topic or idea (e.g., 'Starting a fitness coaching business'):")
        col1, col2 = st.columns(2)
        with col1:
            outline_submitted = st.form_submit_button("📝 Generate Outline Only")
        with col2:
            guide_submitted = st.form_submit_button("📚 Generate Full Guide (Pro Only)")
    if outline_submitted and user_topic:
        st.session_state.user_topic = user_topic
        with st.spinner("🧠 Generating Outline..."):
            completion = jamai.table.add_table_rows("action", p.RowAddRequest(table_id="action-outline-generator", data=[{"user_instruction": user_topic}], stream=True))
            st.session_state.outline = "".join(chunk.text for chunk in completion if hasattr(chunk, "text"))
            log_outline_or_guide(user_topic, st.session_state.outline, is_guide=False)

    if guide_submitted and user_topic:
        if user_tier == "Pro / Paid":
            st.session_state.user_topic = user_topic
            with st.spinner("📚 Writing Full Guide..."):
                completion = jamai.table.add_table_rows("action", p.RowAddRequest(table_id="action-full-guide-generator", data=[{"user_instruction": user_topic}], stream=True))
                st.session_state.guide = "".join(chunk.text for chunk in completion if hasattr(chunk, "text"))
                log_outline_or_guide(user_topic, st.session_state.guide, is_guide=True)

        else:
            st.warning("🚀 Upgrade to Pro to generate full guides!")
    if "outline" in st.session_state:
        st.subheader("📝 Generated Outline")

        formatted_outline = st.session_state.outline.strip()

        st.markdown(formatted_outline)

        st.download_button(
            label="📥 Download as Text",
            data=formatted_outline,
            file_name="outline.txt",
            mime="text/plain"
        )

        if st.session_state.notion_api_key and st.session_state.notion_parent_page_id:
            if st.button("📤 Upload Outline to Notion"):
                success = create_notion_page(st.session_state.notion_api_key, st.session_state.notion_parent_page_id, f"Outline - {st.session_state.user_topic}", st.session_state.outline)
                if success:
                    st.success("✅ Outline uploaded to Notion!")
                else:
                    st.error("❌ Failed to upload to Notion.")
    if "guide" in st.session_state:
        st.subheader("📚 Full Digital Product Guide")

        formatted_guide = st.session_state.guide.strip()

        st.markdown(formatted_guide)

        st.download_button(
            label="📥 Download Full Guide as Text",
            data=formatted_guide,
            file_name="full_guide.txt",
            mime="text/plain"
        )

        if st.session_state.notion_api_key and st.session_state.notion_parent_page_id:
            if st.button("📤 Upload Full Guide to Notion"):
                success = create_notion_page(st.session_state.notion_api_key, st.session_state.notion_parent_page_id, f"Full Guide - {st.session_state.user_topic}", st.session_state.guide)
                if success:
                    st.success("✅ Full guide uploaded to Notion!")
                else:
                    st.error("❌ Failed to upload to Notion!")
    st.divider()
    st.subheader("🎥 Generate Video Scripts from Guide")
    if st.button("🎬 Generate Video Scripts for Each Lesson"):
        if "guide" not in st.session_state or not st.session_state.guide.strip():
            st.error("❌ No guide found. Please generate a full guide first.")
        else:
            with st.spinner("🧠 Generating scripts..."):
                scripts, error = generate_video_scripts(st.session_state.guide)
                if error:
                    st.error(error)
                else:
                    st.session_state.generated_scripts = scripts
                    st.success("✅ Scripts generated!")

    if "generated_scripts" in st.session_state:
        st.subheader("🎬 Your Generated Video Scripts")
        for lesson, script in st.session_state.generated_scripts:
            with st.expander(f"🎥 {lesson}"):
                st.markdown(script)
        all_scripts = "\n\n---\n\n".join(f"{lesson}\n\n{script}" for lesson, script in st.session_state.generated_scripts)
        st.download_button("📥 Download All Scripts", data=all_scripts, file_name="video_scripts.txt")

    st.divider()
    display_outline_and_guide_history()
with tab4:
    st.header("Connect Your Notion")
    st.markdown("""
    ### How to Connect
    1. Go to [Notion Developers](https://www.notion.com/my-integrations) and create a new Integration.
    2. Copy your Internal Integration Token (API Key).
    3. Create a blank page (e.g., "My Guides") in your Notion workspace.
    4. Open that page, click `...` → `Connections` → add your Integration.
    5. Copy the Page URL and paste it below.
    """)
    with st.form("notion_connection_form"):
        api_key = st.text_input("Paste your Notion Integration API Key:", type="password", value=st.session_state.notion_api_key)
        parent_page_link = st.text_input("Paste your Parent Page Link:", value=st.session_state.notion_parent_page_id)
        submitted = st.form_submit_button("Save Connection")
    if submitted:
        st.session_state.notion_api_key = api_key
        if "/" in parent_page_link:
            raw_page_id = parent_page_link.split("-")[-1]
            cleaned_page_id = raw_page_id.split("?")[0]
            st.session_state.notion_parent_page_id = cleaned_page_id
        else:
            st.session_state.notion_parent_page_id = parent_page_link
        save_connection_to_file(api_key, st.session_state.notion_parent_page_id)
        st.success("✅ Notion connection saved!")

with tab5:
    st.header("🎬 Remix an Existing Video")

    REMIX_LIMIT = 10

    if "remix_upload_count" not in st.session_state:
        st.session_state.remix_upload_count = 0

    if not is_paid_user:
        st.info(f"🎥 You have used {st.session_state.remix_upload_count}/{REMIX_LIMIT} remix uploads.")
        if st.session_state.remix_upload_count >= REMIX_LIMIT:
            st.warning("❌ Free remix upload limit reached.")
            st.stop()  # 🔒 Prevent further code from running

    st.markdown("Upload a video **or** paste a TikTok/Instagram link:")
    video_url = st.text_input("Paste a video URL (TikTok, IG Reels, etc.)")
    uploaded_video = st.file_uploader("Or upload a .mp4 file", type=["mp4"], key="video_upload")

    video_path = None

    # Download from URL
    if video_url and st.button("📥 Download from URL"):
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")

        with st.spinner("📥 Downloading video..."):
            if download_video_from_url(video_url, "temp_video.mp4"):
                video_path, caption = download_video_from_url(video_url, "temp_video.mp4")
                st.success("✅ Video downloaded!")
            else:
                st.error("❌ Failed to download video.")

    # Handle uploaded file
    if uploaded_video:
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        video_path = "temp_video.mp4"
        st.success("✅ Video uploaded!")


    # Process the video once available
    if video_path:
        with st.spinner("🔊 Transcribing audio..."):
            transcript = transcribe_video_audio(video_path)

        with st.spinner("📝 Extracting text from visuals..."):
            screen_text = extract_text_from_video(video_path)

        remix_prompt = f"""
You are a social media content strategist. Analyze the following video and generate 3 remix ideas.
Original Caption:
\"\"\"{caption}\"\"\"

Audio Transcript:
\"\"\"{transcript}\"\"\"

On-Screen Captions:
\"\"\"{screen_text}\"\"\"

Each remix idea must include:
- A unique hook or angle
- Optional new caption or visual suggestion
- Format suggestion (voiceover, skit, reaction, storytelling, etc.)

Constraints:
- Only ONE remix idea may use structured "Scene 1, Scene 2..." storytelling, if it makes sense.
- The other remix ideas must be written naturally in paragraph format without forced scene breakdowns.
- Do NOT automatically split every idea into scenes unless it's critical for clarity.
- Prioritize fluid, conversational description over rigid numbered scenes.
- Only ONE idea may be a light, fun "challenge" or skit.
- One idea must be informative, educational, or provide value (e.g., tips, facts, lessons).
- One idea must focus on storytelling or emotional engagement (e.g., day-in-the-life, personal journey).
- Avoid repeating the same format across all three ideas.
- Avoid rigid numbering like Scene 1, Scene 2 unless it fits naturally.
- Prioritize clear paragraphs and natural flow over forced steps.

Tone:
- Creative, high-performance content.
- Balance between entertainment, education, and emotional connection.
- Suitable for short-form platforms (TikTok, Reels, Shorts).

Be creative, but follow the balance strictly.
"""

        with st.spinner("🎨 Generating remix ideas..."):
            remix_response = jamai.table.add_table_rows(
                "action",
                p.RowAddRequest(
                    table_id="action-content-remix-generator",  # reusing the table
                    data=[{"video_analysis_prompt": remix_prompt}],
                    stream=True,
                ),
            )

            remix_output = "".join(chunk.text for chunk in remix_response if hasattr(chunk, "text"))

        st.session_state.remix_upload_count += 1

        st.subheader("🌀 Remix Ideas")
        st.markdown(remix_output)

        source_url = video_url if video_url else "Uploaded file"
        remix_ideas = parse_remix_ideas(remix_output)
        log_remix_to_jamai(source_url, remix_ideas, caption)

    st.divider()
    display_remix_history()
