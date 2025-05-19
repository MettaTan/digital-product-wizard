
# -*- coding: utf-8 -*-

import streamlit as st
import extra_streamlit_components as stx
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
from collections import defaultdict
from slugify import slugify  # Add this import at the top if not already
import uuid
from supabase_client import supabase
from supabase_client import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client
from supabase.client import ClientOptions
from datetime import datetime, timedelta, timezone
import stripe

file_map = defaultdict(list)

# Load environment variables
load_dotenv()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

def start_checkout(price_id, mode="payment"):
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        mode=mode,
        line_items=[{"price": price_id, "quantity": 1}],
        customer_email=st.session_state["user_email"],
        success_url="https://digital-product-wizard.onrender.com/?session=success",
        cancel_url="https://digital-product-wizard.onrender.com/?session=cancel"
    )
    st.markdown(f"""<meta http-equiv="refresh" content="0; url={session.url}">""", unsafe_allow_html=True)
    st.stop()

st.set_page_config(page_title="Digital Product Creator", layout="wide")

# --- Handle Stripe redirect status ---
query_params = st.experimental_get_query_params()
if "session" in query_params:
    status = query_params["session"][0]
    if status == "success":
        st.success("🎉 Payment successful. You've been upgraded to Pro!")
    elif status == "cancel":
        st.info("❌ Payment was cancelled.")

@st.fragment
def get_cookie_manager():
    return stx.CookieManager()

cookie_manager = get_cookie_manager()
cookies = cookie_manager.get_all(key="auth_restore")

def restore_session_from_cookies():
    if not cookies:
        return

    access_token = cookies.get("access_token")
    refresh_token = cookies.get("refresh_token")

    if access_token and refresh_token and "user_id" not in st.session_state:
        try:
            supabase.auth.set_session(access_token, refresh_token)
            user = supabase.auth.get_user()

            if user and user.user:
                st.session_state["access_token"] = access_token
                st.session_state["refresh_token"] = refresh_token
                st.session_state["user_id"] = user.user.id
                st.session_state["user_email"] = user.user.email
                st.session_state["user_name"] = user.user.email.split("@")[0]

                user_client = get_user_client()
                user_record = user_client.table("users").select("paid").eq("id", user.user.id).maybe_single().execute()
                st.session_state["is_paid_user"] = user_record.data.get("paid", False) if user_record and user_record.data else False

        except Exception as e:
            st.warning("⚠️ Failed to restore session: " + str(e))
            cookie_manager.delete("access_token", key="wipe1")
            cookie_manager.delete("refresh_token", key="wipe2")

def logout_user():
    for key in ["access_token", "refresh_token", "user_id", "user_email", "user_name", "is_paid_user"]:
        st.session_state.pop(key, None)

    st.session_state["just_logged_out"] = True
    st.rerun()

def get_user_client():
    token = st.session_state.get("access_token")
    if not token:
        st.error("Missing auth token.")
        st.stop()

    options = ClientOptions(headers={"Authorization": f"Bearer {token}"})
    return create_client(SUPABASE_URL, SUPABASE_KEY, options)

# --- Handle logout or restore session ---
if st.session_state.get("just_logged_out"):
    st.session_state.pop("just_logged_out")
    st.success("🔒 You’ve been logged out.")

    # ✅ Cookies MUST be deleted here where the component is rendered
    cookie_manager.set("access_token", "", expires_at=datetime.now(timezone.utc) - timedelta(days=1), key="expire_access_token")
    cookie_manager.set("refresh_token", "", expires_at=datetime.now(timezone.utc) - timedelta(days=1), key="expire_refresh_token")
else:
    restore_session_from_cookies()

# --- Auth UI (Login + Signup) ---
if "user_id" not in st.session_state:
    st.title("🔐 Log In or Sign Up")

    mode = st.radio("Choose an option:", ["Log In", "Sign Up"], horizontal=True, key="auth_mode")
    email = st.text_input("Email", key="auth_email")
    password = st.text_input("Password", type="password", key="auth_password")

    if mode == "Sign Up":
        if st.button("Create Account", key="signup_btn"):
            try:
                result = supabase.auth.sign_up({"email": email, "password": password})
                if result.user:
                    st.success("✅ Account created! Please check your email to confirm before logging in.")
                    st.stop()
                else:
                    st.error("Signup failed. Please try again.")
            except Exception as e:
                st.error(f"Signup error: {e}")

    elif mode == "Log In":
        if st.button("Log In", key="login_btn"):
            try:
                result = supabase.auth.sign_in_with_password({"email": email, "password": password})

                if result.user and result.session:
                    st.session_state["access_token"] = result.session.access_token
                    st.session_state["refresh_token"] = result.session.refresh_token
                    st.session_state["user_id"] = result.user.id
                    st.session_state["user_email"] = result.user.email
                    st.session_state["user_name"] = result.user.email.split("@")[0]

                    cookie_manager.set("access_token", result.session.access_token,
                        expires_at=datetime.now(timezone.utc) + timedelta(days=7), key="set_access_token")
                    cookie_manager.set("refresh_token", result.session.refresh_token,
                        expires_at=datetime.now(timezone.utc) + timedelta(days=30), key="set_refresh_token")

                    user_client = get_user_client()
                    user_record = user_client.table("users").select("paid").eq("id", result.user.id).maybe_single().execute()
                    st.session_state["is_paid_user"] = user_record.data.get("paid", False) if user_record and user_record.data else False

                    st.rerun()
                else:
                    st.error("❌ Invalid credentials or email not confirmed.")
            except Exception as e:
                st.error(f"Login error: {e}")

    st.stop()

# --- Authenticated UI ---
name = st.session_state.get("user_name", "User")
is_paid_user = st.session_state.get("is_paid_user", False)

with st.sidebar:
    st.divider()
    if not is_paid_user:
        st.markdown("### 🔓 Upgrade to Pro")

        if st.button("💳 $9.99/mo"):
            start_checkout("price_1RItpZ4C6tsP4JlLmzqK9IoP", mode="subscription")
        if st.button("💳 $99/year"):
            start_checkout("price_1RQL7E4C6tsP4JlL0psNbSjX", mode="subscription")
        if st.button("🏆 $199 lifetime"):
            start_checkout("price_1RQL7E4C6tsP4JlL0H9rzDtO", mode="payment")

        if "checkout_url" in st.session_state:
            st.markdown(f"[👉 Complete Payment →]({st.session_state['checkout_url']})", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://your-logo-url.png", use_container_width=True)
    st.divider()
    st.markdown(f"👤 **{name}**")
    st.markdown(f"💼 **{'Premium' if is_paid_user else 'Free'}**")
    st.divider()

    if not is_paid_user:
        st.markdown("### 🔓 Upgrade to Pro")
        if st.button("💳 $9.99/mo"):
            start_checkout("price_month", mode="subscription")
        if st.button("💳 $99/year"):
            start_checkout("price_annual", mode="subscription")
        if st.button("🏆 $199 lifetime"):
            start_checkout("price_lifetime", mode="payment")

    if st.button("🚪 Log Out"):
        logout_user()

# --- Main UI ---
st.title("Digital Product Creator - Audio to Guide")
st.success(f"Welcome {name}! Your paid status: {'Premium' if is_paid_user else 'Free'}")

# Rest of your application...
if "audio_upload_count" not in st.session_state:
    st.session_state.audio_upload_count = 0

if "pdf_upload_count" not in st.session_state:
    st.session_state.pdf_upload_count = 0

if "remix_upload_count" not in st.session_state:
    st.session_state.remix_upload_count = 0

CONNECTION_FILE = ".notion_connection.json"
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Product Blueprint",
    "Upload + Transcribe",
    "Connect to Notion",
    "Create Guides",
    "Course Launch Kit",
    "Remix Existing Video",
    "Full Product History"
])

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

def clean_remix_output(text: str) -> str:
    if not text:
        return ""

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove duplicate headers
    text = re.sub(r"(Remix\s+Idea\s*\d+:?|Script\s*[:\-]?)", "", text, flags=re.IGNORECASE)

    # Remove excessive triple+ newlines but keep doubles
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # Don't strip out single blank lines
    return text.strip()

def format_remix_script(script_text: str) -> str:
    import re

    # Normalize
    text = script_text.strip().replace('\r\n', '\n').replace('\r', '\n')

    # Remove model-added labels (if any)
    text = re.sub(r'^(Hook|Main Message|CTA)\s*[:\-]?', '', text, flags=re.IGNORECASE | re.MULTILINE)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) >= 3:
        hook = lines[0]
        cta = lines[-1]
        body = "\n".join(lines[1:-1])
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) >= 3:
            hook = sentences[0]
            cta = sentences[-1]
            body = " ".join(sentences[1:-1])
        else:
            return text

    return f"""🎯 **Hook**  
{hook}

📘 **Main Message**  
{body}

📣 **CTA**  
{cta}
"""


def extract_lesson_titles(text):
    matches = []
    for line in text.splitlines():
        clean_line = re.sub(r"[*_#>`~-]", "", line).strip()
        if re.match(r"^(Module|Lesson|Week)\s+(?:\d+|[IVXLCDM]+):", clean_line, re.IGNORECASE):
            matches.append(clean_line)
    return matches

def generate_video_scripts(guide_text, user_topic=None, guide_type=None):
    lessons = extract_lesson_titles(guide_text)
    if not lessons:
        return [], "❌ No lesson titles found. Try using headings like 'Lesson 1:' or 'Module 2:' in your guide."

    scripts = []
    previous_script_summary = ""

    for i, lesson in enumerate(lessons):
        next_lesson = lessons[i + 1] if i + 1 < len(lessons) else None
        context = f"Previous Summary: {previous_script_summary}\n\n" if previous_script_summary else ""

        prompt = f"""
        You are a video scriptwriter for short-form educational content. Write a high-converting video script for this lesson title:

        **"{lesson}"**

        This is part of a guide titled **"{st.session_state.user_topic}"**.

        Avoid repeating previous lessons or saying "you've got the basics". Each script must feel **fresh and focused only on this topic**.

        Include:
        - 1–2 sentence **hook** to open the video (no repeating "you’ve already done X")
        - 3–5 key **teaching points** or insights, with natural transitions
        - 1 quick **example, analogy, or framework**
        - 1 motivating **takeaway** or mindset shift
        - 1-sentence **preview of the next lesson**: "{next_lesson}"{'' if not next_lesson else ' (or note it’s the final lesson)'}

        No fluff. Avoid generalizations. Keep it real and relevant to the topic: "{st.session_state.user_topic}".

        Use second-person voice ("you"). Format clearly with line breaks for each paragraph.
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

def log_remix_to_jamai(video_url, remix_text, caption="", linked_product="", visual_idea="", remix_type=""):
    try:
        jamai.table.add_table_rows(
            "action",
            p.RowAddRequest(
                table_id="action-remix-history",
                data=[{
                    "video_link": video_url,
                    "video_caption": caption,
                    "remix_1": remix_text,
                    "linked_product": linked_product,
                    "visual_idea": visual_idea,
                    "remix_type": remix_type,
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
            raw_remix = row.get("remix_1", {}).get("value", "")  # We're only using remix_1 now
            row_id = row.get("ID")

            # Clean formatting artifacts
            remix_text = clean_generated_markdown(raw_remix)
            remix_text = remix_text.replace("<b>", "**").replace("</b>", "**")

            label = (
                "📁 Local Upload" if not video_url or video_url == "Uploaded file"
                else shorten_url(video_url)
            )

            with st.expander(f"🎬 {label}"):
                if video_url and video_url != "Uploaded file":
                    st.markdown(f"[🔗 View Original Video]({video_url})", unsafe_allow_html=True)

                if video_caption:
                    st.markdown(f"📝 **Caption:** {video_caption}")

                if remix_text:
                    st.markdown(remix_text)

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
        
def fix_numbered_lists(text):
    lines = text.split("\n")
    fixed_lines = []
    number = 1
    inside_list = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"^1\.\s+", stripped):
            fixed_lines.append(re.sub(r"^1\.\s+", f"{number}. ", line))
            number += 1
            inside_list = True
        elif inside_list and stripped.startswith("- "):
            # Sub-bullet, keep as-is
            fixed_lines.append(line)
        elif stripped == "":
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
            inside_list = False
            number = 1  # Reset when list ends

    return "\n".join(fixed_lines)

def add_lesson_prefix_to_headings(text):
    lines = text.splitlines()
    count = 1
    updated = []
    for line in lines:
        if line.strip().startswith("## "):
            updated.append(f"## Lesson {count}: {line.strip()[3:].strip()}")
            count += 1
        else:
            updated.append(line)
    return "\n".join(updated)

def add_module_prefix_to_headings(text):
    lines = text.splitlines()
    count = 1
    updated = []
    for line in lines:
        if line.strip().startswith("## "):
            updated.append(f"## Module {count}: {line.strip()[3:].strip()}")
            count += 1
        else:
            updated.append(line)
    return "\n".join(updated)

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
                        cols=[
                            p.ColumnSchemaCreate(id="Linked Blueprint", dtype="str"),  # ✅ Allowed
                            p.ColumnSchemaCreate(id="Source", dtype="str")             # ✅ Allowed
                        ],
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
    import uuid

    MAX_UPLOADS_FREE = 5
    MAX_FILE_SIZE_MB = 25

    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = set()

    if "upload_count" not in st.session_state:
        st.session_state.upload_count = 0

    st.header("\U0001F4C2 Upload for PDF → Course or Podcast → Course")

    st.markdown("""
    Upload content you'd like included in your course:

    - **Podcast**: Upload a podcast episode or voice note
    - **PDF**: Upload a PDF study, framework, or research file

    _(Optional: You can skip this step to use our expert-built knowledge base instead.)_
    """)

    audio_files = st.file_uploader("\U0001F3A7 Upload a podcast or voice note", type=["mp3", "wav"], accept_multiple_files=True, key="audio_upload")
    pdf_files = st.file_uploader("\U0001F4C4 Upload a PDF document", type=["pdf"], accept_multiple_files=True, key="pdf_upload")

    if not is_paid_user:
        total_used = st.session_state.upload_count
        uploads_remaining = MAX_UPLOADS_FREE - total_used

        st.info(f"\U0001F4E5 Uploads used: {total_used}/{MAX_UPLOADS_FREE}")
        st.caption("Limit includes both audio and PDF uploads.")

        if uploads_remaining < 0:
            st.warning("❌ Free tier upload limit reached. [Upgrade to Pro](#) to unlock unlimited uploads.")
            audio_files = []
            pdf_files = []
        else:
            audio_files = audio_files[:uploads_remaining] if audio_files else []
            pdf_files = pdf_files[:uploads_remaining - len(audio_files)] if pdf_files else []

    process_disabled = not (audio_files or pdf_files)
    knowledge_table_id = None

    try:
        blueprint_rows = jamai.table.list_table_rows("action", "action-product-blueprint").items
        if not blueprint_rows:
            st.warning("❗ No product blueprints found. Please create one in Tab 1 first.")
            st.stop()

        blueprint_options = {row["title"]["value"]: row.get("title", {}).get("value") for row in blueprint_rows if "title" in row}
        selected_blueprint = st.selectbox("\U0001F4CC Link uploads to a product blueprint:", list(blueprint_options.keys()))
        selected_row = next(row for row in blueprint_rows if row["title"]["value"] == selected_blueprint)
        knowledge_table_id = selected_row.get("knowledge_table_id", {}).get("value")

    except Exception as e:
        st.error(f"❌ Failed to load product blueprints: {e}")
        st.stop()

    if st.button("⏭️ Skip and Use Built-In Knowledge Only"):
        st.session_state.skipped_upload = True
        st.success("✅ You can now proceed without uploading. The guide will use our expert knowledge base.")

    if (audio_files or pdf_files) and st.button("▶️ Start Processing Uploaded Files", disabled=process_disabled):
        st.info("🚀 Uploading and embedding files...")
        total_files = len(audio_files) + len(pdf_files)
        current_step = 0
        progress_bar = st.progress(0)

        def is_valid_size(file):
            return file.size <= MAX_FILE_SIZE_MB * 1024 * 1024

        for file in audio_files:
            if file.name in st.session_state.uploaded_filenames:
                st.warning(f"⛔ '{file.name}' has already been uploaded. Skipping.")
                continue
            if not is_valid_size(file):
                st.warning(f"❌ Skipped '{file.name}' — exceeds {MAX_FILE_SIZE_MB}MB.")
                continue

            with open("temp_audio.wav", "wb") as f:
                f.write(file.read())

            transcription = transcribe_audio_whisper("temp_audio.wav")
            file_id = str(uuid.uuid4())
            upload_transcription_to_knowledge(transcription, title=file.name, blueprint=selected_blueprint, table_id=knowledge_table_id, file_id=file_id)
            clean_temp_files()

            st.session_state.uploaded_filenames.add(file.name)
            st.session_state.upload_count += 1
            current_step += 1
            progress_bar.progress(current_step / total_files)

        for file in pdf_files:
            if file.name in st.session_state.uploaded_filenames:
                st.warning(f"⛔ '{file.name}' has already been uploaded. Skipping.")
                continue
            if not is_valid_size(file):
                st.warning(f"❌ Skipped '{file.name}' — exceeds {MAX_FILE_SIZE_MB}MB.")
                continue

            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(file.read())

            jamai.table.embed_file(temp_path, knowledge_table_id)
            jamai.table.add_table_rows(
                "knowledge",
                p.RowAddRequest(
                    table_id=knowledge_table_id,
                    data=[{
                        "Title": file.name,
                        "Text": "",
                        "Source": file.name,
                        "Linked Blueprint": selected_blueprint
                    }],
                    stream=False
                )
            )

            os.remove(temp_path)
            st.session_state.uploaded_filenames.add(file.name)
            st.session_state.upload_count += 1
            current_step += 1
            progress_bar.progress(current_step / total_files)

        progress_bar.empty()
        st.success("✅ All files processed and embedded!")
        st.rerun()


    if not knowledge_table_id:
        st.warning("⚠️ Please select a valid product blueprint to view uploads.")
        st.stop()

    if selected_blueprint:
        selected_row = next(row for row in blueprint_rows if row["title"]["value"] == selected_blueprint)
        kt_field = selected_row.get("knowledge_table_id")
        if isinstance(kt_field, dict) and "value" in kt_field:
            knowledge_table_id = str(kt_field["value"])

    # --- Display Uploaded Files ---
    st.divider()
    st.subheader("📚 Uploaded Files (Knowledge)")

    try:
        rows = jamai.table.list_table_rows("knowledge", knowledge_table_id)

        # Auto-delete orphaned title-only rows (created during upload but not linked)
        orphaned_rows = [
            r for r in rows.items
            if r.get("File ID", {}).get("value", "") is None
            and r.get("Title", {}).get("value", "") != ""
            and r.get("Text", {}).get("value", "") is None
        ]
        if orphaned_rows:
            row_ids = [r["ID"] for r in orphaned_rows]
            jamai.table.delete_table_rows(
                "knowledge",
                p.RowDeleteRequest(table_id=knowledge_table_id, row_ids=row_ids)
            )

        # Refresh rows after deletion
        rows = jamai.table.list_table_rows("knowledge", knowledge_table_id)

        # Unique file tracking
        processed_files = set()
        unique_files = []

        if not rows.items:
            st.info("No uploaded files yet.")
        else:
            # Group and deduplicate files
            for row in rows.items:
                title = row.get("Title", {}).get("value", "")
                file_id = row.get("File ID", {}).get("value", "")

                if file_id and file_id not in processed_files:
                    file_rows = [r for r in rows.items if r.get("File ID", {}).get("value", "") == file_id]
                    best_row = max(file_rows, key=lambda r: bool(r.get("Title", {}).get("value", "")))
                    processed_files.add(file_id)
                    unique_files.append(best_row)

            for row in unique_files:
                title = row.get("Title", {}).get("value", "")
                file_id = row.get("File ID", {}).get("value", "")

                if not title:
                    continue

                is_audio = title.lower().endswith((".mp3", ".wav", ".m4a"))
                is_pdf = title.lower().endswith(".pdf")
                icon = "🎵" if is_audio else "📄"

                file_chunks = [r for r in rows.items if r.get("File ID", {}).get("value", "") == file_id]
                page_count = len(file_chunks) if is_pdf else 0

                label = f"{icon} {title}"
                if is_pdf:
                    label += f" ({page_count} pages)"

                with st.expander(label):
                    st.markdown(f"**Uploaded File:** {title}")

                    if is_audio:
                        transcripts = [
                            r.get("Text", {}).get("value", "")
                            for r in file_chunks
                            if r.get("Text", {}).get("value", "")
                        ]
                        transcript = transcripts[0] if transcripts else ""
                        st.markdown("**Transcript Preview:**")
                        st.markdown(transcript or "_No transcript available._", unsafe_allow_html=True)

                    if st.button(f"❌ Delete '{title}'", key=f"delete_{title}_{file_id}"):
                        try:
                            if file_id:
                                # File with embedded chunks — delete by file_id
                                matching_rows = [
                                    r for r in rows.items
                                    if r.get("File ID", {}).get("value") == file_id
                                ]
                            else:
                                # Title-only row — delete by matching title
                                matching_rows = [
                                    r for r in rows.items
                                    if r.get("File ID", {}).get("value") is None and (
                                        r.get("Source", {}).get("value", "") == title or
                                        r.get("Title", {}).get("value", "") == title
                                    )
                                ]

                            row_ids = [r["ID"] for r in matching_rows]
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

with tab4:
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
        st.session_state.guide_type = "essential"
        knowledge_table_id = selected_row.get("knowledge_table_id", {}).get("value", None)

        with st.spinner("🧠 Generating Guide..."):
            prompt = f"""
            You're a professional ghostwriter creating a valuable, free-tier digital guide titled "{user_topic}" for beginners.

            This is the **Essential (Free) Guide**, not the paid version.

            It must:
            - Provide 6–8 well-structured sections with practical advice
            - Include detailed explanations and beginner-level examples
            - Offer value (checklists, frameworks, tips) but **not full-depth strategies**
            - Encourage readers to consider the Premium version for deeper transformation

            Tone: Motivating, helpful, beginner-friendly, and credible.

            Write the full guide with Markdown formatting (## for section headers, - for bullets, etc.). Avoid emojis, repetition, or filler content.
            """

            completion = jamai.generate_chat_completions(
                p.ChatRequest(
                    model="ellm/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                    messages=[
                        p.ChatEntry.system("You are a professional ghostwriter creating a compact, valuable digital guide."),
                        p.ChatEntry.user(prompt)
                    ],
                    rag_params=p.RAGParams(table_id=knowledge_table_id, k=10),
                    stream=True
                )
            )

            raw_output = "".join(chunk.text for chunk in completion if hasattr(chunk, "text")).strip()
            cleaned_output = re.sub(r"^#+\s*.*\n+", "", raw_output)
            fixed = fix_numbered_lists(cleaned_output)

            with_lessons = add_lesson_prefix_to_headings(fixed)
            st.session_state.guide = with_lessons
            log_guide_history(user_topic, with_lessons, is_guide=True, is_premium=False)


    if premium_guide_submitted and user_topic:
        if is_paid_user:
            st.session_state.user_topic = user_topic
            st.session_state.guide_type = "premium"
            knowledge_table_id = selected_row.get("knowledge_table_id", {}).get("value", None)

            with st.spinner("📚 Writing Full Guide..."):
                progress = st.progress(0)

                # Step 1: Generate module titles
                module_plan_prompt = f"""
                You're a course strategist. Generate a list of 8 actionable, high-impact module titles for a premium digital product guide titled "{user_topic}".

                This product is designed to:
                - Help: {audience}
                - Achieve: {promise}
                - Delivery method: {delivery}

                Do **not** default to general education, Wikipedia-style content, or historical overviews.

                Instead, focus on:
                - Tangible, monetizable skills
                - Lessons that help someone build, launch, grow, or sell a digital product
                - Practical transformation and real-world progress

                Return only a numbered list of module titles. No extra commentary.
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

                # Step 2: Generate modules
                full_guide = ""
                for i, module_name in enumerate(modules, 1):
                    progress.progress(int((i - 1) / len(modules) * 100))

                    module_prompt = f"""
                    You're a professional ghostwriter writing an **in-depth premium module** for a digital product guide titled "{user_topic}".

                    Write a full module for a digital product guide. This is Module {i} of {len(modules)} titled: **{module_name}**.

                    Do NOT repeat the module title in the first heading or paragraph. Start directly with engaging educational content. Use appropriate Markdown formatting (##, ###) where helpful.

                    This guide is meant for people who:
                    - Want to turn knowledge or skills into digital products
                    - Are looking for practical, business-oriented steps
                    - Prefer actionable content over theory

                    Your writing must:
                    - Be at least 1,200 words
                    - Include examples, case studies, or step-by-step advice
                    - Use Markdown formatting (headings, bullets, etc.)
                    - Avoid fluff, emojis, and repetition
                    - Include templates, frameworks, or next steps if relevant

                    Tone: clear, expert, motivating — written for someone ready to take action.
                    """

                    response = jamai.generate_chat_completions(
                        p.ChatRequest(
                            model="ellm/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                            messages=[
                                p.ChatEntry.system("You are a professional ghostwriter."),
                                p.ChatEntry.user(module_prompt)
                            ],
                            rag_params=p.RAGParams(table_id=knowledge_table_id, k=10),
                            stream=True
                        )
                    )

                    raw_text = "".join(chunk.text for chunk in response if hasattr(chunk, "text")).strip()
                    cleaned = re.sub(r"^#+\s*.*\n+", "", raw_text)  # Remove first heading if repeated
                    cleaned = re.sub(rf"(?i)^{re.escape(module_name)}\s*", "", cleaned).strip()

                    # ✅ Fix numbered list formatting
                    fixed = fix_numbered_lists(cleaned)

                    full_guide += f"## {module_name}\n\n{fixed}\n\n"

                progress.progress(100)
                with_modules = add_module_prefix_to_headings(full_guide)
                st.session_state.guide = with_modules
                log_guide_history(user_topic, full_guide, is_guide=True, is_premium=True)
        else:
            st.warning("🚀 Upgrade to Pro to generate premium guides!")

    if "guide" in st.session_state:
        guide_type = "Premium Guide" if st.session_state.get("guide_type") == "premium" else "Essential Guide"
        download_name = "premium_guide.txt" if st.session_state.get("guide_type") == "premium" else "essential_guide.txt"

        st.subheader(f"📘 Your {guide_type}")

        with st.expander("🔍 Preview First Section"):
            preview = st.session_state.guide.strip()[:1500]
            st.markdown(preview + "...\n\n_(Full content available via Notion or download)_")

        st.download_button(
            label="📥 Download Full Guide",
            data=st.session_state.guide,
            file_name=download_name,
            mime="text/plain"
        )

        if st.session_state.notion_api_key and st.session_state.notion_parent_page_id:
            if st.button(f"📤 Upload {guide_type} to Notion"):
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
                scripts, error = generate_video_scripts(st.session_state.guide, st.session_state.user_topic, st.session_state.guide_type)

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

with tab5:
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

            try:
                jamai.table.add_table_rows(
                    "action",
                    p.RowAddRequest(
                        table_id="action-course-assets-history",
                        data=[{
                            "title": title,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "slides": generated_assets.get("Slides", ""),
                            "workbook": generated_assets.get("Workbook", ""),
                            "emails": generated_assets.get("Email Launch Sequence", ""),
                            "checklist": generated_assets.get("Launch Checklist", ""),
                            "discord": generated_assets.get("Discord Welcome Message", "")
                        }],
                        stream=False
                    )
                )
            except Exception as e:
                st.warning(f"⚠️ Failed to save asset bundle to history: {e}")
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

with tab6:
    st.header("🎬 Remix an Existing Video")

    # Blueprint selector for remix tracking
    try:
        remix_blueprint_rows = jamai.table.list_table_rows("action", "action-product-blueprint").items
        if remix_blueprint_rows:
            remix_options = [row["title"]["value"] for row in remix_blueprint_rows if "title" in row and row["title"]["value"]]
            selected_blueprint = st.selectbox("📌 Link this remix to a product blueprint:", remix_options)

            # Get full blueprint row
            blueprint_row = next(row for row in remix_blueprint_rows if row.get("title", {}).get("value") == selected_blueprint)

            product_title = blueprint_row.get("title", {}).get("value", "")
            product_pitch = blueprint_row.get("pitch", {}).get("value", "")
            product_promise = blueprint_row.get("promise", {}).get("value", "")
            product_delivery = blueprint_row.get("delivery_method", {}).get("value", "")

        else:
            st.warning("❗ No product blueprints found. Please create one in Tab 1 first.")
            selected_blueprint = None
    except Exception as e:
        st.error(f"❌ Failed to load blueprints: {e}")
        selected_blueprint = None

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
        You're creating viral short-form content.

        Follow this structure:
        1. **Hook** — Start with a bold, surprising, or relatable statement (max 1 line).
        2. **Main Point** — Deliver the key message with confidence.
        3. **CTA** — End with an energetic prompt like “Comment below”, “Follow for more”, or “Grab the guide”.

        Use punchy, casual language. Write it like you'd say it in a 15-second TikTok. No emojis, no markdown — just clear line breaks between parts.
        """,
            "Nurture Content": """
        You're creating educational short-form content designed to build trust.

        Follow this structure:
        1. **Hook** — Start with a pain point or relatable insight.
        2. **Main Takeaway** — Share 2–3 helpful tips, steps, or mindset shifts.
        3. **CTA** — End with a soft call-to-action like “Save this”, “Try it out”, or “Check the link in bio”.

        Keep it calm, credible, and helpful. Use natural, friendly language. No emojis or markdown — just line breaks.
        """,
            "Story / Personal": """
        You're telling a first-person story to create emotional connection.

        Follow this structure:
        1. **Hook** — Open with a personal realization, question, or vulnerable moment.
        2. **Story** — Briefly describe the situation or transformation.
        3. **Takeaway + CTA** — End with a reflection and a gentle prompt like “Tag someone who relates” or “Let me know if you’ve been there too”.

        Write it like you're talking to a friend on camera. No hashtags, no formatting — just plain, relatable storytelling.
        """
        }


        if st.button("🎨 Generate Remix"):
            with st.spinner("🔊 Transcribing audio..."):
                transcript = transcribe_video_audio(st.session_state.video_path)

            with st.spinner("📝 Extracting text from visuals..."):
                screen_text = extract_text_from_video(st.session_state.video_path)

            selected_prompt = remix_style_instructions.get(remix_type, "")

            remix_prompt = f"""
            You are a short-form content creator hired to generate **marketing content** for a digital product.

            You are rewriting a real video script to align with this product:

            🛍️ **Product Info**
            Title: {product_title}  
            Promise: {product_promise}  
            Delivery Method: {product_delivery}  
            Pitch: {product_pitch}

            ---

            🎬 **Script Format (Must-Follow)**

            Your output MUST contain exactly **three sections**, each separated by a line break:

            1. Hook — 1 line, catchy and scroll-stopping
            2. Main Message — **3 to 4 complete lines** (not sentences) that clearly explain the value of this product. Include transformation, who it helps, what it includes, and what makes it special.
            3. CTA — 1 final line with a strong, unique call-to-action that includes the product name.

            ✅ Use line breaks between each section  
            ❌ Do NOT label the sections (no “Hook:”, “Main Message:”, etc.)  
            ❌ Do NOT include markdown, formatting, or extra script variations

            ---

            📽️ **Visual Direction (AFTER the script)**

            After the script, skip a line and write exactly:
            Suggested visuals —
            
            Then describe 1–2 short sentences of visual or b-roll ideas that directly support **this specific script**.

            ---

            🚫 Do not include multiple remix ideas. Only return one final script + visuals.

            ---

            🎥 Video Source Info:

            Original Caption:
            \"\"\"{caption}\"\"\"

            Transcript:
            \"\"\"{transcript}\"\"\"

            On-Screen Captions:
            \"\"\"{screen_text}\"\"\"

            {selected_prompt}
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

                raw_response = "".join(chunk.text for chunk in remix_response if hasattr(chunk, "text")).strip()

                # Look for suggested visuals case-insensitively and strip cleanly
                split_match = re.split(r'suggested visuals\s*[\-–—:]', raw_response, flags=re.IGNORECASE)

                if len(split_match) == 2:
                    script_part = split_match[0].strip()
                    visual_note = split_match[1].strip()
                else:
                    script_part = raw_response
                    visual_note = ""

                # Format the main script
                remix_output = format_remix_script(script_part.strip())

            st.session_state.remix_upload_count += 1

            st.subheader("📜 Remix Script")
            st.markdown(clean_remix_output(remix_output))

            if visual_note.strip():
                st.markdown("🎞️ **Suggested Visuals / B-Roll:**")
                st.markdown(clean_remix_output(visual_note.strip()))

            # ✅ Allow download of script only
            st.download_button("📥 Download Script", remix_output, file_name="remix_script.txt", mime="text/plain")

            # ✅ Log both script and visuals (optional enhancement if you expand schema later)
            source_url = video_url if video_url else "Uploaded file"
            log_remix_to_jamai(
                video_url if video_url else "Uploaded file",
                remix_output,
                caption=st.session_state.caption,
                linked_product=selected_blueprint,
                visual_idea=visual_note.strip(),
                remix_type=remix_type
            )

    st.divider()
    display_remix_history()

with tab7:
    st.header("📦 Full Product History")

    try:
        blueprints = jamai.table.list_table_rows("action", "action-product-blueprint").items
        guides = jamai.table.list_table_rows("action", "action-guide-history").items
        remixes = jamai.table.list_table_rows("action", "action-remix-history").items
        assets = jamai.table.list_table_rows("action", "action-course-assets-history").items

        if not blueprints:
            st.info("No product blueprints found.")
            st.stop()

        for bp in blueprints:
            title = bp.get("title", {}).get("value", "")
            blueprint_text = bp.get("product_blueprint", {}).get("value", "")
            timestamp = bp.get("timestamp", {}).get("value", "")

            with st.expander(f"🧠 {title} — {timestamp}"):
                st.markdown("### 🧱 Product Blueprint")
                st.markdown(blueprint_text or "_Not available._")

                # Guides
                guide_matches = [g for g in guides if g.get("user_input", {}).get("value") == title]
                for gtype in ["essential", "premium"]:
                    guide = next((g for g in guide_matches if g.get("type", {}).get("value") == gtype), None)
                    if guide:
                        label = "📝 Essential Guide" if gtype == "essential" else "💎 Premium Guide"
                        st.markdown(f"### {label}")
                        st.markdown(guide.get("content", {}).get("value", "") or "_Missing._")

                # Remixes
                remix_matches = [r for r in remixes if r.get("linked_product", {}).get("value") == title]
                if remix_matches:
                    st.markdown("### 🎬 Video Remixes")
                    for remix in remix_matches:
                        caption = remix.get("video_caption", {}).get("value", "")
                        remix_text = remix.get("remix_1", {}).get("value", "")
                        visuals = remix.get("visual_idea", {}).get("value", "")
                        remix_type = remix.get("remix_type", {}).get("value", "")
                        st.markdown(f"**{remix_type or 'Remix'} — {caption or 'No caption'}**")
                        st.markdown(remix_text or "_No script._")
                        if visuals:
                            st.markdown(f"**🎞️ Suggested Visuals:**\n{visuals.strip()}")

                # Course Assets
                kit = next((a for a in assets if a.get("title", {}).get("value") == title), None)
                if kit:
                    st.markdown("### 🧰 Course Launch Kit")
                    for field, label in {
                        "slides": "🖥️ Slides",
                        "workbook": "📓 Workbook",
                        "emails": "✉️ Emails",
                        "checklist": "✅ Checklist",
                        "discord": "💬 Discord"
                    }.items():
                        content = kit.get(field, {}).get("value", "")
                        if content:
                            st.markdown(f"#### {label}")
                            st.markdown(content.strip())

                st.divider()


    except Exception as e:
        st.error(f"⚠️ Failed to load product history: {e}")
