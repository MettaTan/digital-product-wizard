from jamaibase import JamAI, protocol as p
import os
from dotenv import load_dotenv

load_dotenv()

jamai = JamAI(
    project_id=os.getenv("JAMAI_PROJECT_ID"),
    token=os.getenv("JAMAI_API_KEY")
)

def recreate_table(func, table_type, table_id):
    try:
        jamai.table.delete_table(table_type, table_id)
        print(f"🗑️ Deleted table: {table_id}")
    except Exception as e:
        print(f"⚠️ Could not delete {table_id} (maybe doesn't exist): {e}")
    func()
    print(f"✅ Recreated table: {table_id}")

def create_tables():
    # 🔥 Delete all existing knowledge tables first
    try:
        existing_knowledge_tables = jamai.table.list_tables("knowledge").items
        for kt in existing_knowledge_tables:
            jamai.table.delete_table("knowledge", kt.id)
            print(f"🗑️ Deleted knowledge table: {kt.id}")
    except Exception as e:
        print(f"⚠️ Failed to delete knowledge tables: {e}")

    recreate_table(lambda: jamai.table.create_action_table(
        p.ActionTableSchemaCreate(
            id="action-guide-history",
            cols=[
                p.ColumnSchemaCreate(id="user_input", dtype="str"),
                p.ColumnSchemaCreate(id="type", dtype="str"),  # "guide"
                p.ColumnSchemaCreate(id="content", dtype="str"),
                p.ColumnSchemaCreate(id="timestamp", dtype="str"),
            ]
        )
    ), "action", "action-guide-history")

    recreate_table(lambda: jamai.table.create_action_table(
        p.ActionTableSchemaCreate(
            id="action-video-script-generator",
            cols=[
                p.ColumnSchemaCreate(id="lesson_topic", dtype="str"),
                p.ColumnSchemaCreate(
                    id="video_script", dtype="str",
                    gen_config=p.LLMGenConfig(
                        model="ellm/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                        system_prompt="You're a professional course presenter and curriculum designer.\n\nWrite a 2–3 minute video script for a course lesson titled \"${lesson_topic}\". This is part of a larger video series — each module should build upon the previous one, not repeat the same hook or opening.\n\nAvoid generic intros like “You're interested in...” or “Let’s dive into…” — make each lesson feel fresh and connected to the journey.\n\nInclude:\n- A unique, engaging hook that references previous content if applicable.\n- Clear and conversational explanation of the lesson topic.\n- Optional visual cues for the presenter (e.g. [On screen:], [Cut to:]).\n\nSpeak directly to the viewer using “you” and “we”. Make it motivating and practical, like a mentor teaching on camera.\n\nAvoid repeating structure or wording from earlier modules. End the script naturally on a full sentence if approaching the token limit.",
                        prompt="""
                        You are a professional course presenter and curriculum designer.

                        Write a 2–3 minute speaking script for a course lesson titled "${lesson_topic}". This is part of a larger video series — assume the viewer has watched the previous lessons.

                        Make each module unique and connected — do NOT repeat hooks or phrasing from previous scripts.

                        Avoid common openers like:
                        - "Alright, listen up!"
                        - "You're interested in..."
                        - "Let's dive in."

                        Instead:
                        - Start with a unique hook or insight that naturally transitions from the last module.
                        - Use subtle references to earlier content if helpful.
                        - Speak directly to the viewer using “you” and “we” — conversational, confident, motivating.

                        Structure:
                        1. A one-line hook or insight.
                        2. Clear, tactical explanation of the topic.
                        3. Occasional rhetorical questions or analogies.
                        4. Conclude with a motivating line that tees up the next lesson.

                        Optional: Add cues like [Visuals: show X], [Pause], [Cut to whiteboard].

                        End on a clean, complete sentence. Do not say “that’s it” or “you got this” every time. Avoid filler.
                        """,
                        temperature=0.5,
                        top_p=0.9,
                        max_tokens=1200,
                    ),
                ),
            ]
        )
    ), "action", "action-video-script-generator")

    # --- Content Remix Table ---
    recreate_table(lambda: jamai.table.create_action_table(
        p.ActionTableSchemaCreate(
            id="action-content-remix-generator",
            cols=[
                p.ColumnSchemaCreate(id="video_analysis_prompt", dtype="str"),
                p.ColumnSchemaCreate(
                    id="remix_ideas", dtype="str",
                    gen_config=p.LLMGenConfig(
                        model="ellm/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                        system_prompt="You are a short-form content creator. Your job is to generate one engaging, social-native video script for Instagram Reels or TikTok, based on existing video content. The goal is to promote a specific digital product. Include one script only.",
                        prompt="""
Analyze the following video and generate 3 creative remix ideas for short-form platforms (Reels, TikTok):

${video_analysis_prompt}

Each idea should include:
1. A new angle or hook
2. Optional caption or visual suggestion
3. Format suggestion (e.g., voiceover skit, reaction, storytelling)

The tone should be fun, practical, and performance-oriented.
""",
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=1000,
                    )
                ),
            ]
        )
    ), "action", "action-content-remix-generator")

    recreate_table(lambda: jamai.table.create_action_table(
        p.ActionTableSchemaCreate(
            id="action-remix-history",
            cols=[
                p.ColumnSchemaCreate(id="video_link", dtype="str"),
                p.ColumnSchemaCreate(id="video_caption", dtype="str"),
                p.ColumnSchemaCreate(id="remix_1", dtype="str"),
                p.ColumnSchemaCreate(id="linked_product", dtype="str"),
                p.ColumnSchemaCreate(id="visual_idea", dtype="str"),  # ✅ NEW FIELD
                p.ColumnSchemaCreate(id="remix_type", dtype="str"),
                p.ColumnSchemaCreate(id="timestamp", dtype="str"),
            ]
        )
    ), "action", "action-remix-history")

    recreate_table(lambda: jamai.table.create_action_table(
        p.ActionTableSchemaCreate(
            id="action-product-blueprint",
            cols=[
                p.ColumnSchemaCreate(id="title", dtype="str"),  # 🔹 New: for display
                p.ColumnSchemaCreate(id="timestamp", dtype="str"),  # 🔹 New: human-readable time
                p.ColumnSchemaCreate(id="user_instruction", dtype="str"),
                p.ColumnSchemaCreate(id="delivery_method", dtype="str"),
                p.ColumnSchemaCreate(id="knowledge_table_id", dtype="str"),  # ✅ Add this line
                p.ColumnSchemaCreate(
                    id="product_blueprint", dtype="str",
                    gen_config=p.LLMGenConfig(
                        model="ellm/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                        system_prompt="You're a digital product strategist who helps creators build, name, and position their online offers.",
                        prompt="""

    
You are a digital product strategist helping creators build online products.

The user wants to build a product on:
"${user_instruction}"

Delivery Method: ${delivery_method}

Please generate:
- A benefit-driven product title
- The big promise (what transformation does it deliver?)
- Delivery method (course, ebook, template bundle, etc.)
- A one-sentence core transformation
- A persuasive 1-paragraph sales pitch
- 3–5 sales bullet points

Keep it clear, modern, and value-focused.
""",
                        temperature=0.4,
                        top_p=0.95,
                        max_tokens=1500,
                    )
                ),
            ]
        )
    ), "action", "action-product-blueprint")

    recreate_table(lambda: jamai.table.create_action_table(
        p.ActionTableSchemaCreate(
            id="action-course-assets-generator",
            cols=[
                p.ColumnSchemaCreate(id="prompt", dtype="str"),
                p.ColumnSchemaCreate(
                    id="output", dtype="str",
                    gen_config=p.LLMGenConfig(
                        model="ellm/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                        system_prompt="You're a professional course creation strategist. You help creators generate teaching assets and marketing material for digital products.",
                        prompt="${prompt}",
                        temperature=0.4,
                        top_p=0.95,
                        max_tokens=3000,
                    )
                )
            ]
        )
    ), "action", "action-course-assets-generator")
    recreate_table(lambda: jamai.table.create_action_table(
        p.ActionTableSchemaCreate(
            id="action-course-assets-history",
            cols=[
                p.ColumnSchemaCreate(id="title", dtype="str"),
                p.ColumnSchemaCreate(id="timestamp", dtype="str"),
                p.ColumnSchemaCreate(id="slides", dtype="str"),
                p.ColumnSchemaCreate(id="workbook", dtype="str"),
                p.ColumnSchemaCreate(id="emails", dtype="str"),
                p.ColumnSchemaCreate(id="checklist", dtype="str"),
                p.ColumnSchemaCreate(id="discord", dtype="str"),
            ]
        )
    ), "action", "action-course-assets-history")


if __name__ == "__main__":
    create_tables()
    print("✅ All tables created and up to date!")
