import os
import logging
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from starlette.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# Environment
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "test_database")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY is not set. AI routes will fail until it is configured.")

# Database
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# App
app = FastAPI(title="Mini Tutor API")
api_router = APIRouter(prefix="/api")


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_base64: Optional[str] = None
    step_number: Optional[int] = None


class ChatRequest(BaseModel):
    message: str
    session_id: str
    grade_level: str
    image_base64: Optional[str] = None


class HintRequest(BaseModel):
    session_id: str
    grade_level: str


class SimplifyRequest(BaseModel):
    session_id: str
    grade_level: str


class QuizRequest(BaseModel):
    session_id: str
    grade_level: str
    topic: str


def get_tutor_system_prompt(grade_level: str) -> str:
    grade_descriptions = {
        "elementary": "elementary school student (grades K-5)",
        "middle": "middle school student (grades 6-8)",
        "high": "high school student (grades 9-12)",
    }
    student_level = grade_descriptions.get(grade_level, "student")

    return f"""You are Mini Tutor, a friendly and patient math tutor helping a {student_level}.

Your teaching principles:
1. Never give direct answers immediately. Guide the student to discover the solution.
2. Break problems into manageable steps.
3. Ask guiding questions to help the student think through problems.
4. Use simple, age-appropriate language.
5. Encourage and praise effort, not just correctness.
6. If a student is stuck, offer hints before revealing more.
7. Explain concepts using real-world examples when useful.
8. Be warm, encouraging, and supportive.
9. Focus on understanding, not just getting the right answer.
10. Adapt your explanation complexity based on the student's responses.

Important formatting rules:
- Never use LaTeX notation.
- Use plain text symbols: × ÷ ² ³ ≈ ≤ ≥ ≠ √
- Write fractions as 1/2
- Write exponents as 2^3
- Keep all math expressions readable on a mobile screen

When presented with a math problem:
- Help the student understand what the problem is asking
- Guide them to identify what information they have
- Help them think about what strategy or formula to use
- Let them try to solve each step with your guidance
- Only reveal the answer after they have worked through the process

Keep responses concise but complete. Make learning feel like a conversation, not a lecture."""


def _guess_mime_type(image_base64: str) -> str:
    if image_base64.startswith("/9j/"):
        return "image/jpeg"
    if image_base64.startswith("iVBOR"):
        return "image/png"
    if image_base64.startswith("R0lGOD"):
        return "image/gif"
    if image_base64.startswith("UklGR"):
        return "image/webp"
    return "image/png"


async def generate_ai_response(
    *,
    system_message: str,
    user_text: str,
    image_base64: Optional[str] = None,
) -> str:
    try:
        content_parts: List[types.Part] = [types.Part(text=user_text)]

        if image_base64:
            mime_type = _guess_mime_type(image_base64)
            image_bytes = base64.b64decode(image_base64)
            content_parts.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                )
            )

        response = gemini_client.models.generate_content(
            model="gemini-3-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=content_parts,
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=0.7,
            ),
        )

        text = getattr(response, "text", None)
        if not text:
            raise HTTPException(status_code=500, detail="Gemini returned an empty response.")

        return text.strip()

    except HTTPException:
        raise
    except Exception as e:
        logging.error("Gemini API error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"AI request failed: {str(e)}")


@api_router.get("/")
async def root():
    return {"message": "Mini Tutor API"}


@api_router.post("/chat")
async def chat(request: ChatRequest):
    try:
        student_msg = ChatMessage(
            session_id=request.session_id,
            role="student",
            content=request.message,
            image_base64=request.image_base64,
        )
        await db.messages.insert_one(student_msg.model_dump())

        history = (
            await db.messages.find({"session_id": request.session_id})
            .sort("timestamp", -1)
            .limit(10)
            .to_list(10)
        )
        history.reverse()

        conversation_context = ""
        for msg in history[:-1]:
            role = "Student" if msg["role"] == "student" else "Tutor"
            conversation_context += f"{role}: {msg['content']}\n"

        full_message = f"""Previous conversation:
{conversation_context if conversation_context else 'This is the start of the conversation.'}

Student's current question or response: {request.message}"""

        ai_response = await generate_ai_response(
            system_message=get_tutor_system_prompt(request.grade_level),
            user_text=full_message,
            image_base64=request.image_base64,
        )

        tutor_msg = ChatMessage(
            session_id=request.session_id,
            role="tutor",
            content=ai_response,
        )
        await db.messages.insert_one(tutor_msg.model_dump())

        await db.sessions.update_one(
            {"session_id": request.session_id},
            {
                "$set": {
                    "last_updated": datetime.utcnow(),
                    "grade_level": request.grade_level,
                    "title": request.message[:50] + "..." if len(request.message) > 50 else request.message,
                }
            },
            upsert=True,
        )

        return {
            "response": ai_response,
            "message_id": tutor_msg.id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error("Chat error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/hint")
async def get_hint(request: HintRequest):
    try:
        history = (
            await db.messages.find({"session_id": request.session_id})
            .sort("timestamp", -1)
            .limit(5)
            .to_list(5)
        )

        if not history:
            return {"response": "Let's start with a problem first. What would you like help with?"}

        context = ""
        for msg in reversed(history):
            role = "Student" if msg["role"] == "student" else "Tutor"
            context += f"{role}: {msg['content']}\n"

        hint_prompt = f"""{context}

The student is asking for a hint. Provide a small, helpful hint that guides them toward the next step without giving away the answer. Make it encouraging and focused on their current thinking."""

        hint_response = await generate_ai_response(
            system_message=get_tutor_system_prompt(request.grade_level),
            user_text=hint_prompt,
        )

        tutor_msg = ChatMessage(
            session_id=request.session_id,
            role="tutor",
            content=f"💡 Hint: {hint_response}",
        )
        await db.messages.insert_one(tutor_msg.model_dump())

        return {"response": hint_response}

    except HTTPException:
        raise
    except Exception as e:
        logging.error("Hint error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/simplify")
async def simplify_explanation(request: SimplifyRequest):
    try:
        last_tutor_msg = await db.messages.find_one(
            {"session_id": request.session_id, "role": "tutor"},
            sort=[("timestamp", -1)],
        )

        if not last_tutor_msg:
            return {"response": "I haven't explained anything yet. Ask me a question first."}

        system_message = (
            f"You are Mini Tutor. Your job is to take complex explanations and make them much "
            f"simpler and easier to understand for a {request.grade_level} school student. "
            f"Use everyday language, simple examples, and break things down into the most basic steps."
        )

        simplify_prompt = f"""Please explain this in much simpler terms:

{last_tutor_msg['content']}

Make it really easy to understand, like you're explaining to a younger student. Use simple words and clear examples."""

        simple_response = await generate_ai_response(
            system_message=system_message,
            user_text=simplify_prompt,
        )

        tutor_msg = ChatMessage(
            session_id=request.session_id,
            role="tutor",
            content=f"📚 Simpler explanation: {simple_response}",
        )
        await db.messages.insert_one(tutor_msg.model_dump())

        return {"response": simple_response}

    except HTTPException:
        raise
    except Exception as e:
        logging.error("Simplify error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/quiz")
async def generate_quiz(request: QuizRequest):
    try:
        history = (
            await db.messages.find({"session_id": request.session_id})
            .sort("timestamp", -1)
            .limit(10)
            .to_list(10)
        )

        if not history:
            return {"response": "Let's solve a problem together first, then I can create practice questions for you."}

        context = ""
        for msg in reversed(history):
            context += f"{msg['content']}\n"

        quiz_prompt = f"""Based on our conversation about {request.topic}, create 3 similar practice problems that will help the student master this concept.

Make sure the problems:
1. Are at the same difficulty level
2. Use similar problem-solving strategies
3. Have different numbers or contexts to keep it interesting
4. Are clearly numbered (Problem 1, Problem 2, Problem 3)

Do not provide answers, only the problems.

Recent conversation:
{context}"""

        quiz_response = await generate_ai_response(
            system_message=get_tutor_system_prompt(request.grade_level),
            user_text=quiz_prompt,
        )

        tutor_msg = ChatMessage(
            session_id=request.session_id,
            role="tutor",
            content=f"📝 Practice Problems:\n\n{quiz_response}",
        )
        await db.messages.insert_one(tutor_msg.model_dump())

        return {"response": quiz_response}

    except HTTPException:
        raise
    except Exception as e:
        logging.error("Quiz error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/history")
async def get_history():
    try:
        sessions = await db.sessions.find().sort("last_updated", -1).to_list(100)
        return [
            {
                "session_id": s["session_id"],
                "title": s.get("title", "Untitled"),
                "last_updated": s["last_updated"].isoformat() if isinstance(s.get("last_updated"), datetime) else str(s.get("last_updated")),
                "grade_level": s.get("grade_level", "unknown"),
            }
            for s in sessions
        ]
    except Exception as e:
        logging.error("History error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/messages/{session_id}")
async def get_messages(session_id: str):
    try:
        messages = (
            await db.messages.find({"session_id": session_id})
            .sort("timestamp", 1)
            .to_list(1000)
        )

        return [
            {
                "id": msg["id"],
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"].isoformat() if isinstance(msg.get("timestamp"), datetime) else str(msg.get("timestamp")),
                "image_base64": msg.get("image_base64"),
            }
            for msg in messages
        ]
    except Exception as e:
        logging.error("Messages error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
