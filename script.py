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
    recreate_table(lambda: jamai.table.create_knowledge_table(
        p.KnowledgeTableSchemaCreate(
            id="knowledge-digital-products",
            cols=[
                p.ColumnSchemaCreate(id="Source", dtype="str"),
            ],
            embedding_model="ellm/BAAI/bge-m3",
        )
    ), "knowledge", "knowledge-digital-products")

    recreate_table(lambda: jamai.table.create_action_table(
        p.ActionTableSchemaCreate(
            id="action-outline-generator",
            cols=[
                p.ColumnSchemaCreate(id="user_instruction", dtype="str"),
                p.ColumnSchemaCreate(
                    id="outline", dtype="str",
                    gen_config=p.LLMGenConfig(
                        system_prompt="You are an expert course creator. Generate a clear, structured outline for a new digital product based on the user's instruction and related knowledge.",
                        prompt="Create a detailed outline for: ${user_instruction}",
                        rag_params=p.RAGParams(table_id="knowledge-digital-products", k=10),
                        temperature=0.2,
                        top_p=0.95,
                        max_tokens=2000,
                    )
                ),
            ]
        )
    ), "action", "action-outline-generator")

    recreate_table(lambda: jamai.table.create_action_table(
        p.ActionTableSchemaCreate(
            id="action-full-guide-generator",
            cols=[
                p.ColumnSchemaCreate(id="user_instruction", dtype="str"),
                p.ColumnSchemaCreate(
                    id="guide", dtype="str",
                    gen_config=p.LLMGenConfig(
                        system_prompt="You are a professional ghostwriter. Write a long, motivational, structured full guide for a digital product based on the user's instruction and related documents.",
                        prompt="Create a full guide titled '${user_instruction}' using all available source documents. Format it with clear numbered modules (e.g. Module I, Module II, etc) and detailed sections.",
                        rag_params=p.RAGParams(table_id="knowledge-digital-products", k=20),
                        temperature=0.2,
                        top_p=0.95,
                        max_tokens=6000,
                    )
                ),
            ]
        )
    ), "action", "action-full-guide-generator")

    recreate_table(lambda: jamai.table.create_action_table(
        p.ActionTableSchemaCreate(
            id="action-video-script-generator",
            cols=[
                p.ColumnSchemaCreate(id="lesson_topic", dtype="str"),
                p.ColumnSchemaCreate(
                    id="video_script", dtype="str",
                    gen_config=p.LLMGenConfig(
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
                        rag_params=p.RAGParams(table_id="knowledge-digital-products", k=5),
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
                        system_prompt="You’re a viral short-form content strategist. Your job is to take existing video content and generate new, high-performing variations for Instagram Reels and TikTok.",
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
                p.ColumnSchemaCreate(id="remix_2", dtype="str"),
                p.ColumnSchemaCreate(id="remix_3", dtype="str"),
            ]
        )
    ), "action", "action-remix-history")

if __name__ == "__main__":
    create_tables()
    print("✅ All tables created and up to date!")
