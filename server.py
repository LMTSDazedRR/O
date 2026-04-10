from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import base64
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Models
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str  # 'student' or 'tutor'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_base64: Optional[str] = None
    step_number: Optional[int] = None

class ChatRequest(BaseModel):
    message: str
    session_id: str
    grade_level: str  # 'elementary', 'middle', 'high'
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

class StepRevealRequest(BaseModel):
    session_id: str
    grade_level: str
    current_step: int

class SessionHistory(BaseModel):
    session_id: str
    title: str
    last_updated: datetime
    grade_level: str

# Helper function to get tutor system prompt
def get_tutor_system_prompt(grade_level: str) -> str:
    grade_descriptions = {
        'elementary': 'elementary school student (grades K-5)',
        'middle': 'middle school student (grades 6-8)',
        'high': 'high school student (grades 9-12)'
    }
    
    student_level = grade_descriptions.get(grade_level, 'student')
    
    return f"""You are Mini Tutor, a friendly and patient math tutor helping a {student_level}.

Your teaching principles:
1. NEVER give direct answers immediately - guide the student to discover the solution
2. Break down problems into manageable steps
3. Ask guiding questions to help students think through problems
4. Use simple, age-appropriate language
5. Encourage and praise effort, not just correct answers
6. If a student is stuck, offer hints before revealing more
7. Explain concepts using real-world examples when possible
8. Be warm, encouraging, and supportive
9. Focus on understanding, not just getting the right answer
10. Adapt your explanation complexity based on the student's responses

IMPORTANT FORMATTING RULES:
- NEVER use LaTeX notation like \\( \\), \\[ \\], $, or $$
- Use plain text symbols: × for multiplication, ÷ for division, ² for squared, ³ for cubed
- Write fractions as "1/2" or "one half"
- Write exponents as "2^3" or "2 to the power of 3"
- Use simple Unicode symbols when needed: ≈, ≤, ≥, ≠, √
- Keep all math expressions readable on a mobile screen

When presented with a math problem:
- First, help the student understand what the problem is asking
- Guide them to identify what information they have
- Help them think about what strategy or formula to use
- Let them try to solve each step with your guidance
- Only reveal the answer after they've worked through the process

Keep your responses concise but complete. Make learning feel like a conversation, not a lecture."""

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Mini Tutor API"}

@api_router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Save student message
        student_msg = ChatMessage(
            session_id=request.session_id,
            role="student",
            content=request.message,
            image_base64=request.image_base64
        )
        await db.messages.insert_one(student_msg.dict())
        
        # Get conversation history for context
        history = await db.messages.find(
            {"session_id": request.session_id}
        ).sort("timestamp", -1).limit(10).to_list(10)
        history.reverse()
        
        # Initialize AI chat
        api_key = os.environ.get('EMERGENT_LLM_KEY', '')
        system_message = get_tutor_system_prompt(request.grade_level)
        
        chat = LlmChat(
            api_key=api_key,
            session_id=request.session_id,
            system_message=system_message
        ).with_model("openai", "gpt-5.2")
        
        # Build conversation context
        conversation_context = ""
        for msg in history[:-1]:  # Exclude the current message
            role = "Student" if msg['role'] == 'student' else "Tutor"
            conversation_context += f"{role}: {msg['content']}\n"
        
        # Create message with context
        full_message = f"""Previous conversation:
{conversation_context if conversation_context else 'This is the start of the conversation.'}

Student's current question/response: {request.message}"""
        
        # Handle image if present
        file_contents = []
        if request.image_base64:
            image_content = ImageContent(image_base64=request.image_base64)
            file_contents.append(image_content)
        
        user_message = UserMessage(
            text=full_message,
            file_contents=file_contents if file_contents else None
        )
        
        # Get AI response
        ai_response = await chat.send_message(user_message)
        
        # Save tutor response
        tutor_msg = ChatMessage(
            session_id=request.session_id,
            role="tutor",
            content=ai_response
        )
        await db.messages.insert_one(tutor_msg.dict())
        
        # Update session history
        await db.sessions.update_one(
            {"session_id": request.session_id},
            {
                "$set": {
                    "last_updated": datetime.utcnow(),
                    "grade_level": request.grade_level,
                    "title": request.message[:50] + "..." if len(request.message) > 50 else request.message
                }
            },
            upsert=True
        )
        
        return {
            "response": ai_response,
            "message_id": tutor_msg.id
        }
        
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/hint")
async def get_hint(request: HintRequest):
    try:
        # Get last few messages for context
        history = await db.messages.find(
            {"session_id": request.session_id}
        ).sort("timestamp", -1).limit(5).to_list(5)
        
        if not history:
            return {"response": "Let's start with a problem first! What would you like help with?"}
        
        # Build context
        context = ""
        for msg in reversed(history):
            role = "Student" if msg['role'] == 'student' else "Tutor"
            context += f"{role}: {msg['content']}\n"
        
        # Create hint request
        api_key = os.environ.get('EMERGENT_LLM_KEY', '')
        system_message = get_tutor_system_prompt(request.grade_level)
        
        chat = LlmChat(
            api_key=api_key,
            session_id=request.session_id + "_hint",
            system_message=system_message
        ).with_model("openai", "gpt-5.2")
        
        hint_prompt = f"""{context}

The student is asking for a hint. Provide a small, helpful hint that guides them toward the next step without giving away the answer. Make it encouraging and focused on their current thinking."""
        
        user_message = UserMessage(text=hint_prompt)
        hint_response = await chat.send_message(user_message)
        
        # Save hint as tutor message
        tutor_msg = ChatMessage(
            session_id=request.session_id,
            role="tutor",
            content=f"💡 Hint: {hint_response}"
        )
        await db.messages.insert_one(tutor_msg.dict())
        
        return {"response": hint_response}
        
    except Exception as e:
        logging.error(f"Hint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/simplify")
async def simplify_explanation(request: SimplifyRequest):
    try:
        # Get last tutor response
        last_tutor_msg = await db.messages.find_one(
            {"session_id": request.session_id, "role": "tutor"},
            sort=[("timestamp", -1)]
        )
        
        if not last_tutor_msg:
            return {"response": "I haven't explained anything yet. Ask me a question first!"}
        
        # Create simplification request
        api_key = os.environ.get('EMERGENT_LLM_KEY', '')
        system_message = f"""You are Mini Tutor. Your job is to take complex explanations and make them much simpler and easier to understand for a {request.grade_level} school student. Use everyday language, simple examples, and break things down into the most basic steps."""
        
        chat = LlmChat(
            api_key=api_key,
            session_id=request.session_id + "_simplify",
            system_message=system_message
        ).with_model("openai", "gpt-5.2")
        
        simplify_prompt = f"""Please explain this in much simpler terms:

{last_tutor_msg['content']}

Make it really easy to understand, like you're explaining to a younger student. Use simple words and clear examples."""
        
        user_message = UserMessage(text=simplify_prompt)
        simple_response = await chat.send_message(user_message)
        
        # Save simplified explanation
        tutor_msg = ChatMessage(
            session_id=request.session_id,
            role="tutor",
            content=f"📚 Simpler explanation: {simple_response}"
        )
        await db.messages.insert_one(tutor_msg.dict())
        
        return {"response": simple_response}
        
    except Exception as e:
        logging.error(f"Simplify error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/quiz")
async def generate_quiz(request: QuizRequest):
    try:
        # Get recent conversation for context
        history = await db.messages.find(
            {"session_id": request.session_id}
        ).sort("timestamp", -1).limit(10).to_list(10)
        
        if not history:
            return {"response": "Let's solve a problem together first, then I can create practice questions for you!"}
        
        # Build context
        context = ""
        for msg in reversed(history):
            context += f"{msg['content']}\n"
        
        # Create quiz generation request
        api_key = os.environ.get('EMERGENT_LLM_KEY', '')
        system_message = get_tutor_system_prompt(request.grade_level)
        
        chat = LlmChat(
            api_key=api_key,
            session_id=request.session_id + "_quiz",
            system_message=system_message
        ).with_model("openai", "gpt-5.2")
        
        quiz_prompt = f"""Based on our conversation about {request.topic}, create 3 similar practice problems that will help the student master this concept.

Make sure the problems:
1. Are at the same difficulty level
2. Use similar problem-solving strategies
3. Have different numbers/contexts to keep it interesting
4. Are clearly numbered (Problem 1, Problem 2, Problem 3)

Don't provide answers - just the problems. The student will work through them.

Recent conversation:
{context}"""
        
        user_message = UserMessage(text=quiz_prompt)
        quiz_response = await chat.send_message(user_message)
        
        # Save quiz
        tutor_msg = ChatMessage(
            session_id=request.session_id,
            role="tutor",
            content=f"📝 Practice Problems:\n\n{quiz_response}"
        )
        await db.messages.insert_one(tutor_msg.dict())
        
        return {"response": quiz_response}
        
    except Exception as e:
        logging.error(f"Quiz error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/history")
async def get_history():
    try:
        sessions = await db.sessions.find().sort("last_updated", -1).to_list(100)
        return [{"session_id": s['session_id'], "title": s.get('title', 'Untitled'), 
                 "last_updated": s['last_updated'].isoformat(), "grade_level": s.get('grade_level', 'unknown')} 
                for s in sessions]
    except Exception as e:
        logging.error(f"History error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/messages/{session_id}")
async def get_messages(session_id: str):
    try:
        messages = await db.messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list(1000)
        
        return [{"id": msg['id'], "role": msg['role'], "content": msg['content'],
                 "timestamp": msg['timestamp'].isoformat(), 
                 "image_base64": msg.get('image_base64')} 
                for msg in messages]
    except Exception as e:
        logging.error(f"Messages error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
