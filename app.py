
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
from auth_app import authenticator, credentials, users
from collections import defaultdict
from slugify import slugify  # Add this import at the top if not already

file_map = defaultdict(list)

# Load environment variables
load_dotenv()

# ✅ Must come before calling any JamAI functions
jamai = JamAI(
    project_id=os.getenv("JAMAI_PROJECT_ID"),
    token=os.getenv("JAMAI_API_KEY")
)

st.set_page_config(page_title="Digital Product Creator", layout="wide")

# Initialize session state for authentication if not present
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None
if "name" not in st.session_state:
    st.session_state["name"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None

# Try to use the login widget if not authenticated
if not st.session_state["authentication_status"]:
    # Call the authenticator login method with the correct parameters
    # The key parameter is what you were using as the first parameter
    login_result = authenticator.login(location="main", key="Login")
    
    # According to docs, this returns None when location is not 'unrendered'
    # The authentication status is stored in session_state instead
    if st.session_state["authentication_status"]:
        # Successful login
        name = st.session_state["name"]
        username = st.session_state["username"]
    elif st.session_state["authentication_status"] is False:
        st.error("❌ Incorrect username or password.")
        st.stop()
    else:
        st.warning("🔐 Please log in to continue.")
        st.stop()

# If code reaches here, user is authenticated
username = st.session_state["username"]
name = st.session_state["name"]

# Extract email & paid status
email = credentials["usernames"][username]["email"]
is_paid_user = users[email]["paid"]

# Optional logout button
if authenticator.logout("Logout", "sidebar"):
    st.session_state["authentication_status"] = None
    st.session_state["name"] = None
    st.session_state["username"] = None
    st.experimental_rerun()

# Main app content
st.title("Digital Product Creator - Audio to Guide")
st.success(f"Welcome {name}! Your paid status: {'Premium' if is_paid_user else 'Free'}")

# Rest of your application...

# Rest of your application...

if "audio_upload_count" not in st.session_state:
    st.session_state.audio_upload_count = 0

if "pdf_upload_count" not in st.session_state:
    st.session_state.pdf_upload_count = 0

if "remix_upload_count" not in st.session_state:
    st.session_state.remix_upload_count = 0

CONNECTION_FILE = ".notion_connection.json"
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Product Blueprint", "Upload + Transcribe", "Create Guides", "Connect to Notion", "Remix Existing Video", "Course Launch Kit"])

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
    
def clean_generated_markdown(text):
    """
    Cleans LLM-generated markdown to render nicely in Streamlit without formatting artifacts.
    """
    if not text:
        return ""
    text = re.sub(r'(\*{1,3})(.*?)\1', r'\2', text)             # *bold* and **bold**
    text = re.sub(r'_([^_]+)_', r'\1', text)                    # _italic_
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # markdown headers
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)       # blockquotes
    text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)    # unordered lists
    text = re.sub(r'`([^`]*)`', r'\1', text)                    # inline code
    text = text.replace("*", "").replace("_", "")
    return text.strip()

def fix_token_spacing(text):
    # Inserts spaces between letters and numbers that were jammed together
    return re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)

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
        # Trim off incomplete sentence endings for better formatting
        content = trim_incomplete_sentence(content)

        # Clean up markdown (optional: keep if your content has extra characters)
        cleaned_content = clean_content(content)

        # ✅ Convert to properly formatted Notion blocks
        blocks = split_text_into_blocks(cleaned_content)

        if not blocks:
            print("❌ No blocks to upload.")
            return False

        # Create first batch of blocks
        first_100_blocks = blocks[:100]

        page = notion.pages.create(
            parent={"page_id": parent_page_id},
            properties={
                "title": {
                    "title": [
                        {"type": "text", "text": {"content": title[:200]}}  # ⛑ Limit title to 200 chars max
                    ]
                }
            },
            children=first_100_blocks
        )

        page_id = page["id"]

        # If there are more than 100 blocks, append in batches
        for i in range(100, len(blocks), 100):
            batch = blocks[i:i+100]
            notion.blocks.children.append(block_id=page_id, children=batch)

        return True

    except Exception as e:
        print(f"❌ Notion upload failed: {e}")
        return False

def clean_content(content: str) -> str:
    content = re.sub(r"#+ ", "", content)
    content = content.replace("**", "")
    content = re.sub(r"\n-{2,}\n", "\n", content)
    return content

def format_rich_text(text):
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', text)
    rich_text = []

    for token in tokens:
        if token.startswith("**") and token.endswith("**"):
            rich_text.append({
                "type": "text",
                "text": {"content": token[2:-2]},
                "annotations": {"bold": True, "italic": False, "code": False}
            })
        elif token.startswith("*") and token.endswith("*"):
            rich_text.append({
                "type": "text",
                "text": {"content": token[1:-1]},
                "annotations": {"bold": False, "italic": True, "code": False}
            })
        elif token.startswith("`") and token.endswith("`"):
            rich_text.append({
                "type": "text",
                "text": {"content": token[1:-1]},
                "annotations": {"bold": False, "italic": False, "code": True}
            })
        else:
            rich_text.append({
                "type": "text",
                "text": {"content": token},
                "annotations": {"bold": False, "italic": False, "code": False}
            })
    return rich_text

def split_text_into_blocks(content: str) -> list:
    lines = content.splitlines()
    blocks = []
    inside_code_block = False
    code_buffer = []
    table_buffer = []

    def flush_table():
        if len(table_buffer) < 3:
            return
        headers = [h.strip() for h in table_buffer[0].strip("|").split("|")]
        rows = table_buffer[2:]
        for row in rows:
            cells = [c.strip() for c in row.strip("|").split("|")]
            pairs = zip(headers, cells)
            line = " · ".join(f"{k}: {v}" for k, v in pairs if k and v)
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": format_rich_text(line)}
            })
        table_buffer.clear()

    for line in lines:
        line = line.rstrip()

        if inside_code_block:
            if line.strip().startswith("```"):
                blocks.append({
                    "object": "block",
                    "type": "code",
                    "code": {
                        "rich_text": [{"type": "text", "text": {"content": "\n".join(code_buffer)}}],
                        "language": "plain text"
                    }
                })
                code_buffer = []
                inside_code_block = False
            else:
                code_buffer.append(line)
            continue

        if line.strip().startswith("```"):
            inside_code_block = True
            continue

        # Table detection
        if line.strip().startswith("|") and "|" in line:
            table_buffer.append(line)
            continue
        elif table_buffer:
            flush_table()

        stripped = line.strip()
        if not stripped:
            continue

        # Headings
        if stripped.startswith("### "):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": format_rich_text(stripped[4:].strip())}
            })
        elif stripped.startswith("## "):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": format_rich_text(stripped[3:].strip())}
            })
        elif stripped.startswith("# "):
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {"rich_text": format_rich_text(stripped[2:].strip())}
            })
        elif re.match(r"- \[( |x|X)\] ", stripped):
            checked = "[x]" in stripped.lower()
            content = re.sub(r"- \[( |x|X)\] ", "", stripped)
            blocks.append({
                "object": "block",
                "type": "to_do",
                "to_do": {"rich_text": format_rich_text(content), "checked": checked}
            })
        elif re.match(r"\d+\.\s", stripped):
            content = re.sub(r"^\d+\.\s*", "", stripped)
            blocks.append({
                "object": "block",
                "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": format_rich_text(content)}
            })
        elif stripped.startswith("- "):
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": format_rich_text(stripped[2:].strip())}
            })
        else:
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": format_rich_text(stripped)}
            })

    if table_buffer:
        flush_table()

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
        next_lesson = lessons[i + 1] if i + 1 < len(lessons) else None

        # Add summary to context for next lesson
        context = f"Previous Summary: {previous_script_summary}\n\n" if previous_script_summary else ""

        prompt = f"""
You are a course content strategist and expert video scriptwriter. Write a detailed, high-quality video script for the lesson titled:
"{lesson}"

{context}This lesson comes from a digital product guide designed to teach or inspire transformation. Do NOT repeat intros. Avoid clichés like “Alright, listen up” or “Let’s dive in.”

Each script should feel like a continuation — not a restart.

Include:
- 3–5 key teaching points
- Transitions that feel natural between ideas
- Examples, analogies, or frameworks
- A motivating takeaway at the end

Conclude with a smooth preview of the next lesson:
"{next_lesson}"{'' if next_lesson else ' (This is the final module.)'}

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

def log_guide_history(user_input, content, is_guide=False, is_premium=False):
    try:
        guide_type = "premium" if is_premium else "essential"

        jamai.table.add_table_rows(
            "action",
            p.RowAddRequest(
                table_id="action-guide-history",
                data=[{
                    "user_input": user_input,
                    "type": guide_type,
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
            remix_1 = clean_generated_markdown(row.get("remix_1", {}).get("value", ""))
            remix_2 = clean_generated_markdown(row.get("remix_2", {}).get("value", ""))
            remix_3 = clean_generated_markdown(row.get("remix_3", {}).get("value", ""))
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

def display_guide_history():
    st.subheader("📚 Generated Guide History")

    # --- Essential Guides ---
    st.markdown("### 📝 Essential Guides")
    try:
        guide_rows = jamai.table.list_table_rows("action", "action-guide-history")
        essentials = [r for r in guide_rows.items if r.get("type", {}).get("value") == "essential"]
        if not essentials:
            st.info("No essential guides generated yet.")
        else:
            for row in essentials:
                topic = row.get("user_input", {}).get("value", "")
                content = row.get("content", {}).get("value", "")
                row_id = row.get("ID")

                with st.expander(f"📝 {topic}"):
                    st.markdown(content or "_No guide found._")
                    if st.button(f"❌ Delete this guide", key=f"delete_essential_{row_id}"):
                        jamai.table.delete_table_rows(
                            "action",
                            p.RowDeleteRequest(
                                table_id="action-guide-history",
                                row_ids=[row_id]
                            )
                        )
                        st.success(f"✅ Deleted guide for '{topic}'")
                        st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Failed to load essential guides: {e}")

    st.divider()

    # --- Premium Guides ---
    st.markdown("### 💎 Premium Guides")
    try:
        guide_rows = jamai.table.list_table_rows("action", "action-guide-history")
        premiums = [r for r in guide_rows.items if r.get("type", {}).get("value") == "premium"]
        if not premiums:
            st.info("No premium guides generated yet.")
        else:
            for row in premiums:
                topic = row.get("user_input", {}).get("value", "")
                content = row.get("content", {}).get("value", "")
                row_id = row.get("ID")

                with st.expander(f"💎 {topic}"):
                    st.markdown(content or "_No guide found._")
                    if st.button(f"❌ Delete this guide", key=f"delete_premium_{row_id}"):
                        jamai.table.delete_table_rows(
                            "action",
                            p.RowDeleteRequest(
                                table_id="action-guide-history",
                                row_ids=[row_id]
                            )
                        )
                        st.success(f"✅ Deleted guide for '{topic}'")
                        st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Failed to load premium guides: {e}")

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


def display_course_assets_history():
    st.subheader("📜 Past Course Launch Kits")
    try:
        rows = jamai.table.list_table_rows("action", "action-course-assets-history")
        if not rows.items:
            st.info("No previous asset bundles yet.")
            return
        for row in rows.items:
            title = row.get("title", {}).get("value", "Untitled")
            timestamp = row.get("timestamp", {}).get("value", "")
            with st.expander(f"📘 {title} — {timestamp}"):
                for key in ["slides", "workbook", "emails", "checklist", "discord"]:
                    content = row.get(key, {}).get("value", "")
                    if content:
                        st.markdown(f"### {key.capitalize()}")
                        st.markdown(content)
                if st.button("❌ Delete", key=f"delete_{row['ID']}"):
                    jamai.table.delete_table_rows(
                        "action",
                        p.RowDeleteRequest(
                            table_id="action-course-assets-history",
                            row_ids=[row["ID"]]
                        )
                    )
                    st.success("✅ Deleted asset bundle.")
                    st.rerun()
    except Exception as e:
        st.warning(f"⚠️ Failed to load asset history: {e}")

def fix_token_spacing(text):
    return re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)

def clean_generated_markdown(text):
    if not text:
        return ""
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    text = re.sub(r'(\*{3})(.*?)\1', r'<b><i>\2</i></b>', text)
    text = re.sub(r'(\*{2})(.*?)\1', r'<b>\2</b>', text)
    text = re.sub(r'(?<!\w)\*(\w.*?)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'^#{1,6}\s+(.*)', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)
    return text.strip()
        
if "notion_api_key" not in st.session_state or "notion_parent_page_id" not in st.session_state:
    api_key, parent_page_id = load_connection_from_file()
    st.session_state.notion_api_key = api_key
    st.session_state.notion_parent_page_id = parent_page_id

with tab1:
    st.header("🎯 Product Blueprint")

    st.markdown("Define the **core strategy** behind your digital product. This helps us generate guides and creative assets that are aligned and monetizable.")

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
        with st.spinner("🎨 Generating your product blueprint..."):
            # 🔹 Step 1: Create a unique knowledge table for this product
            knowledge_table_id = f"knowledge-{slugify(product_title)}"

            # Check if it already exists to avoid errors
            existing_ktables = jamai.table.list_tables("knowledge").items
            existing_ids = [t.id for t in existing_ktables]

            if knowledge_table_id not in existing_ids:
                jamai.table.create_knowledge_table(
                    p.KnowledgeTableSchemaCreate(
                        id=knowledge_table_id,
                        cols=[],  # ✅ Do not define reserved columns
                        embedding_model="ellm/BAAI/bge-m3",
                    )
                )

            # 🔹 Step 2: Store blueprint metadata, including the linked knowledge table
            blueprint_response = jamai.table.add_table_rows(
                "action",
                p.RowAddRequest(
                    table_id="action-product-blueprint",
                    data=[{
                        "title": product_title,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "user_instruction": product_title,
                        "delivery_method": delivery_method,
                        "audience": target_audience,
                        "promise": transformation,
                        "pitch": product_pitch,
                        "knowledge_table_id": knowledge_table_id  # ✅ Save for later use
                    }],
                    stream=True,
                ),
            )

            # 🔹 Step 3: Clean and display the blueprint
            raw_output = "".join(chunk.text for chunk in blueprint_response if hasattr(chunk, "text"))
            fixed = fix_token_spacing(raw_output)
            cleaned = clean_generated_markdown(fixed)

        st.subheader("🧱 Product Blueprint")
        st.markdown(cleaned, unsafe_allow_html=True)

        safe_title = re.sub(r'\W+', '_', product_title.lower())[:50]
        st.download_button("📥 Download Blueprint", cleaned, file_name=f"blueprint_{safe_title}.txt", mime="text/plain")

    elif blueprint_submitted:
        st.warning("❗ Please fill in all required fields to generate a blueprint.")

    st.divider()
    st.subheader("📐 Product Blueprint History")

    try:
        rows = jamai.table.list_table_rows("action", "action-product-blueprint")
        rows = [r for r in rows.items if r.get("title", {}).get("value")]
        sorted_rows = sorted(rows, key=lambda r: r.get("timestamp", {}).get("value") or "", reverse=True)

        if not sorted_rows:
            st.info("No product blueprints generated yet.")
        else:
            for row in sorted_rows:
                product_title = row.get("title", {}).get("value", "Untitled")
                timestamp = row.get("timestamp", {}).get("value", "")
                blueprint = row.get("product_blueprint", {}).get("value", "")
                row_id = row.get("ID")

                with st.expander(f"🧠 {product_title} — {timestamp}"):
                    st.markdown(blueprint or "_No blueprint found._", unsafe_allow_html=True)
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
    st.header("📂 Upload for PDF → Course or Podcast → Course")

    st.markdown("""
    Upload content you'd like included in your course:

    - **Podcast**: Upload a podcast episode or voice note
    - **PDF**: Upload a PDF study, framework, or research file

    _(Optional: You can skip this step to use our expert-built knowledge base instead.)_
    """)

    AUDIO_LIMIT = 3
    PDF_LIMIT = 5

    if "audio_upload_count" not in st.session_state:
        st.session_state.audio_upload_count = 0
    if "pdf_upload_count" not in st.session_state:
        st.session_state.pdf_upload_count = 0

    audio_files = st.file_uploader("🎧 Upload a podcast or voice note", type=["mp3", "wav"], accept_multiple_files=True, key="audio_upload")
    pdf_files = st.file_uploader("📄 Upload a PDF document", type=["pdf"], accept_multiple_files=True, key="pdf_upload")

    if not is_paid_user:
        st.info(f"🎧 Audio uploads used: {st.session_state.audio_upload_count}/{AUDIO_LIMIT}")
        st.info(f"📄 PDF uploads used: {st.session_state.pdf_upload_count}/{PDF_LIMIT}")

        if st.session_state.audio_upload_count >= AUDIO_LIMIT:
            st.warning("❌ Free tier audio upload limit reached.")
            audio_files = []

        if st.session_state.pdf_upload_count >= PDF_LIMIT:
            st.warning("❌ Free tier PDF upload limit reached.")
            pdf_files = []

    process_disabled = (not audio_files and not pdf_files)

    knowledge_table_id = None  # Ensure it's always defined
    
    try:
        blueprint_rows = jamai.table.list_table_rows("action", "action-product-blueprint").items
        if not blueprint_rows:
            st.warning("❗ No product blueprints found. Please create one in Tab 1 first.")
            st.stop()

        blueprint_options = {row["title"]["value"]: row.get("title", {}).get("value") for row in blueprint_rows if "title" in row and row.get("title", {}).get("value")}
        selected_blueprint = st.selectbox("📌 Link uploads to a product blueprint:", list(blueprint_options.keys()))
        selected_row = next(row for row in blueprint_rows if row["title"]["value"] == selected_blueprint)
        kt_field = selected_row.get("knowledge_table_id")
        if isinstance(kt_field, dict) and "value" in kt_field:
            knowledge_table_id = str(kt_field["value"])
        else:
            knowledge_table_id = None

    except Exception as e:
        st.error(f"❌ Failed to load product blueprints: {e}")
        selected_blueprint = None

    if st.button("⏭️ Skip and Use Built-In Knowledge Only"):
        st.session_state.skipped_upload = True
        st.success("✅ You can now proceed without uploading. The guide will use our expert knowledge base.")

    if (audio_files or pdf_files) and st.button("▶️ Start Processing Uploaded Files", disabled=process_disabled):
        st.info("🚀 Uploading and embedding files...")
        total_files = len(audio_files) + len(pdf_files)
        current_step = 0
        progress_bar = st.progress(0)

        for audio in audio_files:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.read())
            transcription = transcribe_audio_whisper("temp_audio.wav")
            upload_transcription_to_knowledge(transcription, title=audio.name, blueprint=selected_blueprint)
            clean_temp_files()
            st.session_state.audio_upload_count += 1
            current_step += 1
            progress_bar.progress(current_step / total_files)

        
        for pdf in pdf_files:
            original_name = pdf.name
            
            # Check if the filename contains spaces or special characters that need special handling
            needs_special_handling = " " in original_name or any(c in original_name for c in "&+,;=?@#%<>{}[]|^~`")
            
            if needs_special_handling:
                # Use the two-step process for files with spaces or special characters
                with open("temp.pdf", "wb") as f:
                    f.write(pdf.read())
                
                # Embed with generic name
                jamai.table.embed_file("temp.pdf", knowledge_table_id)
                
                # Add a separate row with correct title
                jamai.table.add_table_rows(
                    "knowledge",
                    p.RowAddRequest(
                        table_id=knowledge_table_id,
                        data=[{
                            "Title": original_name,
                            "Text": "",
                            "Source": original_name,
                            "Linked Blueprint": selected_blueprint
                        }],
                        stream=False
                    )
                )
                
                # Clean up
                os.remove("temp.pdf")
            else:
                # For filenames without spaces or special chars, use direct embedding
                # This avoids creating duplicate entries
                with open(original_name, "wb") as f:
                    f.write(pdf.read())
                
                # Embed directly with original name
                jamai.table.embed_file(original_name, knowledge_table_id)
                
                # Clean up
                os.remove(original_name)
            
            # Update progress
            st.session_state.pdf_upload_count += 1
            current_step += 1
            progress_bar.progress(current_step / total_files)

        progress_bar.empty()
        st.success("✅ All files processed and embedded!")

    # --- Display Uploaded Files ---
    st.divider()
    st.subheader("📚 Uploaded Files (Knowledge)")

    try:
        rows = jamai.table.list_table_rows("knowledge", knowledge_table_id)
        if not rows.items:
            st.info("No uploaded files yet.")
        else:
            # Group all rows by title
            for row in rows.items:
                title = row.get("Title", {}).get("value", "")
                if title:
                    file_map[title].append(row)

            # Show only one expander per file
            for title, grouped_rows in file_map.items():
                is_audio = title.lower().endswith((".mp3", ".wav", ".m4a"))
                is_pdf = title.lower().endswith(".pdf")
                icon = "🎵" if is_audio else "📄"

                label = f"{icon} {title}"
                if is_pdf:
                    text = grouped_rows[0].get("Text", {}).get("value", "")
                    page_count = text.count("\n\n") + 1 if text else len(grouped_rows)
                    label += f" ({page_count} pages)"

                with st.expander(label):
                    st.markdown(f"**Uploaded File:** {title}")

                    if is_audio:
                        transcript = grouped_rows[0].get("Text", {}).get("value", "")
                        st.markdown("**Transcript Preview:**")
                        st.markdown(transcript or "_No transcript available._", unsafe_allow_html=True)

                    if st.button(f"❌ Delete '{title}'", key=f"delete_{title}_{grouped_rows[0]['ID']}"):
                        try:
                            # Fetch all rows again to ensure full context
                            all_rows = jamai.table.list_table_rows("knowledge", knowledge_table_id).items

                            # Try to match by File ID (more reliable), fallback to Source or Title
                            target_file_id = grouped_rows[0].get("File ID", {}).get("value", "")
                            matched_rows = []

                            if target_file_id:
                                matched_rows = [r for r in all_rows if r.get("File ID", {}).get("value", "") == target_file_id]

                            if not matched_rows:
                                # Fallback: Try Source field (if it exists)
                                source_name = grouped_rows[0].get("Source", {}).get("value", title)
                                matched_rows = [r for r in all_rows if r.get("Source", {}).get("value", "") == source_name]

                            if not matched_rows:
                                # Final fallback: Match by title
                                matched_rows = [r for r in all_rows if r.get("Title", {}).get("value", "") == title]

                            row_ids = [r["ID"] for r in matched_rows]

                            if row_ids:
                                jamai.table.delete_table_rows(
                                    "knowledge",
                                    p.RowDeleteRequest(
                                        table_id=knowledge_table_id,
                                        row_ids=row_ids
                                    )
                                )
                                st.success(f"✅ Deleted '{title}'")
                                st.rerun()
                            else:
                                st.warning("⚠️ No rows matched for deletion.")
                        except Exception as e:
                            st.warning(f"⚠️ Failed to delete file: {e}")

    except Exception as e:
        st.warning(f"⚠️ Failed to load knowledge uploads: {e}")

with tab3:
    st.header("Generate Full Guide")
    st.divider()
    st.subheader("Step 1: Select a Product Blueprint")

    try:
        rows = jamai.table.list_table_rows("action", "action-product-blueprint").items
        if not rows:
            st.warning("❗ Please create a product blueprint first in Tab 1.")
            st.stop()

        options = {row["title"]["value"]: row for row in rows if "title" in row and row.get("title", {}).get("value")}
        selected_title = st.selectbox("Pick a blueprint to use:", list(options.keys()))
        selected_row = options[selected_title]

        # Pre-fill all values
        user_topic = selected_row["title"]["value"]
        audience = selected_row.get("audience", {}).get("value", "")
        promise = selected_row.get("promise", {}).get("value", "")
        delivery = selected_row.get("delivery_method", {}).get("value", "")
        pitch = selected_row.get("pitch", {}).get("value", "")

    except Exception as e:
        st.error(f"Failed to load product blueprints: {e}")
        st.stop()

    with st.form("generation_form"):
        st.markdown(f"**Selected Topic:** `{user_topic}`")
        col1, col2 = st.columns(2)
        with col1:
            basic_guide_submitted = st.form_submit_button("📝 Generate Essential Guide")
        with col2:
            premium_guide_submitted = st.form_submit_button("💎 Generate Premium Guide (Pro Only)")


    if basic_guide_submitted and user_topic:
        st.session_state.user_topic = user_topic
        knowledge_table_id = selected_row.get("knowledge_table_id", {}).get("value", None)

        with st.spinner("🧠 Generating Guide..."):
            prompt = f"""
                Create a detailed, premium-quality guide titled '{user_topic}', with expert positioning, persuasive copy, and complete monetization structure.
                """

            completion = jamai.generate_chat_completions(
                p.ChatRequest(
                    model="ellm/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                    messages=[
                        p.ChatEntry.system("You are a professional ghostwriter. Write a long, motivational, structured full guide for a digital product based on the user's instruction and related documents."),
                        p.ChatEntry.user(prompt)
                    ],
                    rag_params=p.RAGParams(table_id=knowledge_table_id, k=10),
                    stream=True
                )
            )

            st.session_state.guide = "".join(chunk.text for chunk in completion if hasattr(chunk, "text"))
            log_guide_history(user_topic, st.session_state.guide, is_guide=False)

    if premium_guide_submitted and user_topic:
        if is_paid_user:
            st.session_state.user_topic = user_topic
            knowledge_table_id = selected_row.get("knowledge_table_id", {}).get("value", None)

            with st.spinner("📚 Writing Full Guide..."):
                # Step 1: Generate custom module titles
                module_plan_prompt = f"""
                You're a course strategist. Generate a list of 10 high-impact module titles for a premium digital product guide titled '{user_topic}'.

                The guide is for:
                - Audience: {audience}
                - Promise: {promise}
                - Delivery: {delivery}

                Return only a numbered list of concise, engaging module titles that are clear and cover all critical aspects of the product.
                """

                module_response = jamai.generate_chat_completions(
                    p.ChatRequest(
                        model="ellm/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                        messages=[
                            p.ChatEntry.system("You are a digital product curriculum architect."),
                            p.ChatEntry.user(module_plan_prompt)
                        ],
                        rag_params=p.RAGParams(table_id=knowledge_table_id, k=10),
                        stream=True
                    )
                )

                module_text = "".join(chunk.text for chunk in module_response if hasattr(chunk, "text"))
                modules = [re.sub(r"^\d+\.\s*", "", line.strip()) for line in module_text.splitlines() if line.strip()]

                # Step 2: Generate detailed content per module
                full_guide = ""
                for i, module_name in enumerate(modules, 1):
                    module_prompt = f"""
                    You're a professional ghostwriter writing an **in-depth premium module** for a digital product guide titled '{user_topic}'.

                    Write full content for the chapter: **{module_name}** (Module {i} of {len(modules)}).

                    Guidelines:
                    - At least 1,200+ words
                    - Include practical lessons, examples, and case studies
                    - Use Markdown formatting:
                        - `#`, `##`, or `###` for headings
                        - `-` for bullets, `1.` for numbered lists
                        - `**bold**`, `*italic*`, and `` `code` `` for inline styling
                    - Do not use emojis
                    - Include optional frameworks or checklists
                    - Vary structure across modules to keep it fresh
                    - No repetition from earlier modules

                    Tone: inspiring, clear, and expert-level.
                    """

                    response = jamai.generate_chat_completions(
                        p.ChatRequest(
                            model="ellm/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                            messages=[
                                p.ChatEntry.system("You are a professional ghostwriter."),
                                p.ChatEntry.user(module_prompt)
                            ],
                            rag_params=p.RAGParams(table_id=knowledge_table_id, k=30),
                            stream=True
                        )
                    )

                    section_text = "".join(chunk.text for chunk in response if hasattr(chunk, "text"))
                    full_guide += f"## {module_name}\n\n{section_text.strip()}\n\n"

                # Final save
                st.session_state.guide = full_guide
                log_guide_history(user_topic, full_guide, is_guide=True, is_premium=True)
        else:
            st.warning("🚀 Upgrade to Pro to generate premium guides!")

    if "guide" in st.session_state:
        is_premium = is_paid_user
        guide_type = "Premium Guide" if is_premium else "Essential Guide"
        download_name = "premium_guide.txt" if is_premium else "essential_guide.txt"

        st.subheader(f"📘 Your {guide_type}")

        # Optional preview of just the first ~1000 characters
        with st.expander("🔍 Preview First Section"):
            preview = st.session_state.guide.strip()[:1500]
            st.markdown(preview + "...\n\n_(Full content available via Notion or download)_")

        # Download as text
        st.download_button(
            label="📥 Download Full Guide",
            data=st.session_state.guide,
            file_name=download_name,
            mime="text/plain"
        )

        # Auto-upload or show Notion upload
        if st.session_state.notion_api_key and st.session_state.notion_parent_page_id:
            if st.button(f"📤 Upload {guide_type} to Notion", key=f"upload_notion_{guide_type}_1"):
                success = create_notion_page(
                    st.session_state.notion_api_key,
                    st.session_state.notion_parent_page_id,
                    f"{guide_type} - {st.session_state.user_topic}",
                    st.session_state.guide
                )
                if success:
                    st.success(f"✅ {guide_type} uploaded to Notion!")
                else:
                    st.error(f"❌ Failed to upload {guide_type} to Notion.")

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
    display_guide_history()
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

    st.markdown("Upload a video or paste a TikTok/Instagram link:")
    video_url = st.text_input("Paste a video URL (TikTok, IG Reels, etc.)")
    uploaded_video = st.file_uploader("Or upload a .mp4 file", type=["mp4"], key="video_upload")

    video_path = None
    caption = "Uploaded file"

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

    caption = caption if video_url else "Uploaded file"

    st.session_state.video_ready = True
    st.session_state.video_path = "temp_video.mp4"
    st.session_state.caption = caption


    # Process the video once available
    if video_path:
        st.session_state.video_ready = True
        st.session_state.video_path = video_path
        st.session_state.caption = caption

    # Step 2: Remix Options & Trigger
    if st.session_state.get("video_ready"):
        st.subheader("🎛 Step 2: Choose Remix Type and Generate Script")

        # Select Remix Style
        remix_type = st.selectbox("🎛 Remix Style", ["Growth Content", "Nurture Content", "Story / Personal"])

        remix_style_instructions = {
            "Growth Content": """
        Create a short-form script that is optimized to go viral.

        - Use a punchy hook in the first line (something surprising, funny, or highly relatable).
        - Keep the delivery energetic and concise.
        - End with a strong CTA (e.g. comment, follow, “grab the guide”).
        """,
            "Nurture Content": """
        Write an educational short-form script that builds trust.

        - Start with a clear pain point or insight.
        - Provide 2–3 actionable takeaways.
        - End with a CTA like “save this” or “check the link in bio”.
        """,
            "Story / Personal": """
        Create a relatable, first-person script based on personal experience.

        - Start with an authentic moment of struggle, change, or insight.
        - Describe what happened, ideally tied to the digital product.
        - End with a reflective or emotional takeaway and soft CTA.
        """
        }

        if st.button("🎨 Generate Remix"):
            with st.spinner("🔊 Transcribing audio..."):
                transcript = transcribe_video_audio(st.session_state.video_path)

            with st.spinner("📝 Extracting text from visuals..."):
                screen_text = extract_text_from_video(st.session_state.video_path)

            selected_prompt = remix_style_instructions.get(remix_type, "")

            remix_prompt = f"""
    You're a short-form creator. Write a {remix_type.lower()} video script for TikTok, Reels, or Shorts based on this video input.

    Original Caption:
    \"\"\"{caption}\"\"\"

    Transcript:
    \"\"\"{transcript}\"\"\"

    On-Screen Captions:
    \"\"\"{screen_text}\"\"\"

    {selected_prompt}
    Output:
    - 4–6 lines max
    - Format it as a script ready for voiceover or face-to-camera
    - Start with a strong hook
    - End with a CTA
    - Use simple, casual, punchy language

    Only return the script — no titles, no intro, no markdown.
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

            st.subheader("📜 Remix Script")
            st.markdown(clean_remix_output(remix_output))
            st.download_button("📥 Download Script", remix_output, file_name="remix_script.txt", mime="text/plain")

            source_url = video_url if video_url else "Uploaded file"
            log_remix_to_jamai(source_url, [remix_output], st.session_state.caption)

    st.divider()
    display_remix_history()
with tab6:
    st.header("📦 Course Launch Kit Generator")

    try:
        # Load product blueprints
        rows = jamai.table.list_table_rows("action", "action-product-blueprint").items
        if not rows:
            st.warning("❗ Please create a product blueprint first.")
            st.stop()

        options = {row["title"]["value"]: row for row in rows if "title" in row and row.get("title", {}).get("value")}
        selected_title = st.selectbox("Pick a product blueprint:", list(options.keys()))
        selected_row = options[selected_title]

        # Extract values
        title = selected_row["title"]["value"]
        audience = selected_row.get("audience", {}).get("value", "")
        promise = selected_row.get("promise", {}).get("value", "")
        delivery = selected_row.get("delivery_method", {}).get("value", "")
        pitch = selected_row.get("pitch", {}).get("value", "")
    except Exception as e:
        st.error(f"❌ Failed to load blueprints: {e}")
        st.stop()

    # Generate assets on click
    if st.button("🚀 Generate All Course Assets"):
        with st.spinner("Generating slides, workbook, emails, and more..."):
            asset_prompts = [
                ("Slides", f"""
For a course titled '{title}' for an audience of {audience}, generate 3 teaching slide titles and 2–3 bullet points for each module in the course.

Each module should include:
- Slide titles
- Bullet points under each
Use a teaching tone suitable for a video course.
"""),
                ("Workbook", f"""
Create one practical workbook exercise per module for a course called '{title}'.

Each exercise should:
- Reinforce that module's learning
- Be actionable
- Be student-friendly (no jargon)

List as:
Module Title → Exercise → Instructions
"""),
                ("Email Launch Sequence", f"""
Generate a 4-part email sequence to launch a digital product titled '{title}' for {audience}.

Tone: professional, clear, value-driven.

Each email should have:
- Subject line
- Body content (~150–200 words)

Focus on benefits, objections, and motivating action.
"""),
                ("Launch Checklist", f"""
Write a digital course launch checklist.

Split into:
- Pre-Launch
- Launch Week
- Post-Launch

Include items like email setup, social media content, Discord prep, etc.
"""),
                ("Discord Welcome Message", f"""
Write a welcome message for a private Discord for a course titled '{title}'.

Include:
- Friendly welcome
- Course access tips
- Rules
- Support instructions
""")
            ]

            generated_assets = {}

            for label, prompt in asset_prompts:
                try:
                    response = jamai.table.add_table_rows(
                        "action",
                        p.RowAddRequest(
                            table_id="action-course-assets-generator",
                            data=[{"prompt": prompt}],
                            stream=True
                        )
                    )
                    full_text = "".join(chunk.text for chunk in response if hasattr(chunk, "text"))
                    generated_assets[label] = full_text.strip()
                except Exception as e:
                    st.warning(f"⚠️ Failed to generate {label}: {e}")

            st.session_state.course_assets = generated_assets
            st.success("✅ Assets generated!")

    # Display + Download
    if "course_assets" in st.session_state:
        st.subheader("📂 Course Assets Preview")
        for section, content in st.session_state.course_assets.items():
            with st.expander(f"📘 {section}"):
                st.markdown(content)

        full_export = "\n\n".join([f"## {k}\n\n{v}" for k, v in st.session_state.course_assets.items()])
        st.download_button("📄 Download All as Text", full_export, file_name="course_assets.txt", mime="text/plain")

    st.divider()
    display_course_assets_history()