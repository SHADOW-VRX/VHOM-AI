# backend.py
import os
import uuid
import json
import re
import hashlib
import tempfile
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI with better error handling
openai.api_key = os.getenv("OPENAI_API_KEY", "")
openai_client = None

# Try to initialize OpenAI client if API key exists
if openai.api_key and openai.api_key != "":
    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=openai.api_key)
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client: {e}")
        openai_client = None
else:
    logger.info("OpenAI API key not found, running in mock mode")

app = FastAPI(title="VHOM AI Backend", docs_url="/api/docs", redoc_url="/api/redoc")

# Enable CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Create temp directory for uploads
TEMP_DIR = tempfile.gettempdir()
UPLOAD_DIR = os.path.join(TEMP_DIR, "vhom_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Thread pool for CPU-intensive operations
thread_pool = ThreadPoolExecutor(max_workers=2)

# Data Models with validation
class ProcessRequest(BaseModel):
    text: Optional[str] = Field(None, max_length=50000)
    document_id: Optional[str] = None
    content_type: str = "text"

class YouTubeRequest(BaseModel):
    url: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=10000)
    document_id: Optional[str] = None
    conversation_id: Optional[str] = None

# In-memory storage with size limits
MAX_DOCUMENTS = 20
MAX_CONVERSATIONS = 50
MAX_CONVERSATION_MESSAGES = 100

knowledge_base = {}  # Stores document content by ID
conversations = {}   # Stores chat history by conversation ID

# Helper Functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file with error handling"""
    try:
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        logger.error("PyPDF2 not installed")
        return ""
    except Exception as e:
        logger.error(f"Error extracting PDF: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file with error handling"""
    try:
        import docx
        text = ""
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text += paragraph.text + "\n"
        return text
    except ImportError:
        logger.error("python-docx not installed")
        return ""
    except Exception as e:
        logger.error(f"Error extracting DOCX: {e}")
        return ""

async def save_upload_file(upload_file: UploadFile) -> tuple:
    """Save uploaded file to temp directory"""
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(upload_file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    try:
        content = await upload_file.read()
        
        # Validate file size (10MB limit)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Write file
        with open(file_path, 'wb') as out_file:
            out_file.write(content)
        
        return file_path, file_id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
    return text.strip()[:50000]  # Limit text length

def get_document_fingerprint(text: str) -> str:
    """Create a unique fingerprint of the document"""
    if not text:
        return "empty"
    sample = text[:500]
    return hashlib.md5(sample.encode()).hexdigest()[:8]

def generate_document_based_mock(prompt_type: str, document_text: str) -> str:
    """Generate mock responses based on document content"""
    if not document_text:
        return "No document content available."
    
    fingerprint = get_document_fingerprint(document_text)
    
    # Extract words from document
    words = re.findall(r'\b[a-zA-Z]{4,}\b', document_text.lower())
    unique_words = list(set(words))[:10]
    
    # Get sentences
    sentences = re.split(r'[.!?]+', document_text)
    first_sentence = sentences[0][:100] if sentences else ""
    second_sentence = sentences[1][:100] if len(sentences) > 1 else ""
    
    seed = int(fingerprint, 16) % 1000 if fingerprint != "empty" else 42
    
    if prompt_type == "summary":
        topics = unique_words[:5] if unique_words else ["topics", "concepts", "ideas"]
        topics_str = ", ".join(topics)
        
        summaries = [
            f"""This document discusses {topics_str} as its main themes. The content begins by introducing {first_sentence[:50]}... It then explores various aspects including {', '.join(unique_words[:3])}. The key arguments presented focus on understanding these concepts in depth. The document concludes with important insights about how these ideas connect to broader contexts.""",
            
            f"""Based on the document analysis, the primary focus is on {topics_str}. The text explains that {second_sentence[:60]}... Several important points are made about {', '.join(unique_words[:4])}. The author provides detailed explanations and supporting evidence throughout. The conclusion ties together the main themes and suggests practical applications.""",
            
            f"""The document provides comprehensive coverage of {topics_str}. It starts with {first_sentence[:70]}... The middle sections delve into specific aspects including {', '.join(unique_words[:5])}. The writing style is informative and well-structured. The final sections summarize the key takeaways and their implications."""
        ]
        return summaries[seed % len(summaries)]
    
    elif prompt_type == "concepts":
        base_concepts = [word.title() for word in unique_words[:15]] if unique_words else []
        
        if len(document_text) > 1000:
            base_concepts.append("In-depth Analysis")
            base_concepts.append("Detailed Explanation")
        if "example" in document_text.lower():
            base_concepts.append("Practical Examples")
        if "conclusion" in document_text.lower():
            base_concepts.append("Key Findings")
        
        while len(base_concepts) < 5:
            base_concepts.append(f"Concept {len(base_concepts) + 1}")
        
        return ",".join(base_concepts[:15])
    
    elif prompt_type == "flashcards":
        word_count = len(document_text.split())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        flashcards = []
        
        main_topic = unique_words[0] if unique_words else "main topic"
        flashcards.append({
            "question": f"What is the main topic discussed in this document?",
            "answer": f"The document primarily discusses {main_topic} and related concepts. {first_sentence[:100]}"
        })
        
        if len(unique_words) > 2:
            flashcards.append({
                "question": f"What are the key points about {unique_words[1]}?",
                "answer": f"The document explains that {unique_words[1]} is important because {second_sentence[:100] if second_sentence else 'it relates to various aspects of the main topic'}."
            })
        else:
            flashcards.append({
                "question": "What are the main arguments presented?",
                "answer": f"The document presents several arguments including: {', '.join(unique_words[:3])}. These points are supported by evidence and examples throughout the text."
            })
        
        flashcards.append({
            "question": "How is the document structured?",
            "answer": f"The document contains approximately {word_count} words and is organized into sections that cover {', '.join(unique_words[:4])}. It begins with an introduction, presents detailed information, and concludes with key takeaways."
        })
        
        if len(sentences) > 3:
            flashcards.append({
                "question": f"What does the document say about {unique_words[2] if len(unique_words) > 2 else 'the subject'}?",
                "answer": f"According to the document, {sentences[3][:150]}"
            })
        else:
            flashcards.append({
                "question": f"What is the significance of {unique_words[0] if unique_words else 'the topic'}?",
                "answer": f"The document emphasizes that {unique_words[0] if unique_words else 'the topic'} plays a crucial role in understanding the broader context. It provides examples and explanations to support this."
            })
        
        if len(sentences) > 2:
            flashcards.append({
                "question": "What are the main conclusions?",
                "answer": f"The document concludes that {sentences[-2][:150] if len(sentences) > 2 else 'the topic has important implications for understanding the subject matter.'}"
            })
        else:
            flashcards.append({
                "question": "What are the key takeaways?",
                "answer": f"The key takeaways include understanding {', '.join(unique_words[:3])} and their interconnections. The document emphasizes the importance of these concepts in practical applications."
            })
        
        return json.dumps(flashcards)
    
    return ""

async def generate_with_openai(prompt: str, document_text: str, max_tokens: int = 500) -> str:
    """Generate text using OpenAI or fallback to document-based mock"""
    
    # If OpenAI is available, use it
    if openai_client:
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that analyzes documents and provides accurate, detailed responses based strictly on the provided content. Keep responses concise and informative."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
    
    # Fallback to document-based mock
    logger.info("Using mock response generation")
    
    if "summary" in prompt.lower():
        return generate_document_based_mock("summary", document_text)
    elif "concepts" in prompt.lower():
        return generate_document_based_mock("concepts", document_text)
    elif "flashcards" in prompt.lower():
        return generate_document_based_mock("flashcards", document_text)
    else:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', document_text.lower())
        unique_words = list(set(words))[:5]
        if unique_words:
            return f"Based on your document, I can see it discusses topics related to {', '.join(unique_words)}. What specific aspect would you like to know more about?"
        else:
            return "I've analyzed your document. Please ask me specific questions about its content, and I'll do my best to answer them."

def manage_memory():
    """Manage in-memory storage to prevent memory issues"""
    # Limit documents
    if len(knowledge_base) > MAX_DOCUMENTS:
        oldest_doc = min(knowledge_base.keys(), 
                        key=lambda x: knowledge_base[x].get('uploaded_at', ''))
        del knowledge_base[oldest_doc]
        logger.info(f"Removed old document {oldest_doc} to free memory")
    
    # Limit conversations
    if len(conversations) > MAX_CONVERSATIONS:
        oldest_conv = min(conversations.keys(),
                         key=lambda x: conversations[x][0].get('timestamp', '') if conversations[x] else '')
        del conversations[oldest_conv]
        logger.info(f"Removed old conversation {oldest_conv} to free memory")
    
    # Limit conversation messages
    for conv_id, messages in conversations.items():
        if len(messages) > MAX_CONVERSATION_MESSAGES:
            conversations[conv_id] = messages[-MAX_CONVERSATION_MESSAGES:]

# API Endpoints
@app.get("/")
async def serve_frontend():
    """Serve the main frontend"""
    try:
        if os.path.exists("index.html"):
            return FileResponse("index.html")
        else:
            return HTMLResponse("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>VHOM AI</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                        h1 { color: #333; }
                        .status { color: green; }
                    </style>
                </head>
                <body>
                    <h1>VHOM AI Backend</h1>
                    <p>Status: <span class="status">Running</span></p>
                    <p>API is operational. Please ensure the frontend is properly configured.</p>
                    <hr>
                    <h2>Available Endpoints:</h2>
                    <ul>
                        <li><code>GET /api</code> - API status</li>
                        <li><code>POST /api/upload/file</code> - Upload files</li>
                        <li><code>POST /api/upload/youtube</code> - Process YouTube videos</li>
                        <li><code>POST /api/process/text</code> - Process pasted text</li>
                        <li><code>POST /api/generate/summary</code> - Generate summaries</li>
                        <li><code>POST /api/generate/concepts</code> - Extract concepts</li>
                        <li><code>POST /api/generate/flashcards</code> - Generate flashcards</li>
                        <li><code>POST /api/chat</code> - Chat with AI</li>
                        <li><code>GET /api/documents</code> - List documents</li>
                    </ul>
                    <p><a href="/api/docs">API Documentation</a></p>
                </body>
                </html>
            """)
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return HTMLResponse("<h1>VHOM AI Backend Running</h1>", status_code=200)

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "documents": len(knowledge_base),
        "conversations": len(conversations)
    }

@app.get("/api")
async def api_root():
    return {
        "message": "VHOM AI API is running",
        "status": "active",
        "version": "1.0.0",
        "openai_enabled": openai_client is not None
    }

@app.post("/api/upload/file")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file"""
    file_path = None
    try:
        # Validate file type
        allowed_extensions = ('.pdf', '.docx', '.txt')
        if not file.filename.lower().endswith(allowed_extensions):
            raise HTTPException(status_code=400, detail="Invalid file type. Allowed: PDF, DOCX, TXT")
        
        # Save file
        file_path, file_id = await save_upload_file(file)
        
        # Extract text based on file type
        text = ""
        
        if file.filename.lower().endswith('.pdf'):
            text = await asyncio.get_event_loop().run_in_executor(
                thread_pool, extract_text_from_pdf, file_path
            )
        elif file.filename.lower().endswith('.docx'):
            text = await asyncio.get_event_loop().run_in_executor(
                thread_pool, extract_text_from_docx, file_path
            )
        elif file.filename.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Validate extracted text
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Manage memory
        manage_memory()
        
        # Store in knowledge base
        knowledge_base[file_id] = {
            'filename': file.filename,
            'full_text': cleaned_text,
            'preview': cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
            'uploaded_at': datetime.now().isoformat(),
            'type': 'file',
            'word_count': len(cleaned_text.split()),
            'fingerprint': get_document_fingerprint(cleaned_text)
        }
        
        logger.info(f"File uploaded: {file.filename} (ID: {file_id}, Words: {len(cleaned_text.split())})")
        
        return JSONResponse({
            'success': True,
            'file_id': file_id,
            'filename': file.filename,
            'text_preview': knowledge_base[file_id]['preview'],
            'message': 'File uploaded successfully'
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.post("/api/upload/youtube")
async def process_youtube(request: YouTubeRequest):
    """Process YouTube video transcript"""
    try:
        # Extract video ID
        video_id = None
        if 'youtube.com' in request.url:
            match = re.search(r'v=([^&]+)', request.url)
            if match:
                video_id = match.group(1)
        elif 'youtu.be' in request.url:
            video_id = request.url.split('/')[-1].split('?')[0]
        
        # Get transcript
        transcript_text = ""
        if video_id and len(video_id) == 11:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = ' '.join([item['text'] for item in transcript_list])
                logger.info(f"Retrieved transcript for video: {video_id}")
            except Exception as e:
                logger.warning(f"Could not get transcript: {e}")
                transcript_text = f"Video content from URL: {request.url}"
        else:
            transcript_text = f"Video content description for: {request.url}"
        
        # Clean the text
        cleaned_text = clean_text(transcript_text)
        
        # Manage memory
        manage_memory()
        
        # Store in knowledge base
        file_id = str(uuid.uuid4())
        knowledge_base[file_id] = {
            'filename': f"YouTube Video",
            'full_text': cleaned_text,
            'preview': cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
            'url': request.url,
            'uploaded_at': datetime.now().isoformat(),
            'type': 'youtube',
            'word_count': len(cleaned_text.split()),
            'fingerprint': get_document_fingerprint(cleaned_text)
        }
        
        logger.info(f"YouTube processed: {video_id} (ID: {file_id}, Words: {len(cleaned_text.split())})")
        
        return JSONResponse({
            'success': True,
            'file_id': file_id,
            'text_preview': knowledge_base[file_id]['preview'],
            'message': 'YouTube video processed successfully'
        })
    
    except Exception as e:
        logger.error(f"YouTube error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process/text")
async def process_text(request: ProcessRequest):
    """Process pasted text"""
    try:
        if not request.text or len(request.text.strip()) == 0:
            return JSONResponse({
                'success': False,
                'message': 'No text provided'
            })
        
        # Clean the text
        cleaned_text = clean_text(request.text)
        
        # Manage memory
        manage_memory()
        
        # Store in knowledge base
        file_id = str(uuid.uuid4())
        knowledge_base[file_id] = {
            'filename': 'Pasted Text',
            'full_text': cleaned_text,
            'preview': cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
            'uploaded_at': datetime.now().isoformat(),
            'type': 'text',
            'word_count': len(cleaned_text.split()),
            'fingerprint': get_document_fingerprint(cleaned_text)
        }
        
        logger.info(f"Text processed (ID: {file_id}, Words: {len(cleaned_text.split())})")
        
        return JSONResponse({
            'success': True,
            'file_id': file_id,
            'message': 'Text processed successfully'
        })
    
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/summary")
async def generate_summary(request: ProcessRequest):
    """Generate summary from document"""
    try:
        if not request.document_id or request.document_id not in knowledge_base:
            return JSONResponse({
                'success': False,
                'message': 'Document not found'
            })
        
        document = knowledge_base[request.document_id]
        document_text = document['full_text']
        
        logger.info(f"Generating summary for: {document['filename']}")
        
        # Truncate if too long
        if len(document_text) > 4000:
            document_text = document_text[:4000]
        
        prompt = f"""Please provide a comprehensive summary of the following text. 
        Focus on the main points, key arguments, and important details.
        
        TEXT:
        {document_text}
        
        SUMMARY:"""
        
        summary = await generate_with_openai(prompt, document['full_text'], max_tokens=800)
        
        return JSONResponse({
            'success': True,
            'summary': summary
        })
    
    except Exception as e:
        logger.error(f"Summary error: {e}")
        return JSONResponse({
            'success': False,
            'message': str(e)
        })

@app.post("/api/generate/concepts")
async def generate_concepts(request: ProcessRequest):
    """Extract key concepts from document"""
    try:
        if not request.document_id or request.document_id not in knowledge_base:
            return JSONResponse({
                'success': False,
                'message': 'Document not found'
            })
        
        document = knowledge_base[request.document_id]
        document_text = document['full_text']
        
        logger.info(f"Generating concepts for: {document['filename']}")
        
        # Truncate if too long
        if len(document_text) > 3000:
            document_text = document_text[:3000]
        
        prompt = f"""Extract the 10-15 most important key concepts, topics, and terms from this text.
        Return them as a comma-separated list.
        
        TEXT:
        {document_text}
        
        KEY CONCEPTS (comma-separated):"""
        
        concepts_text = await generate_with_openai(prompt, document['full_text'], max_tokens=300)
        
        # Parse concepts
        concepts = [c.strip() for c in concepts_text.split(',') if c.strip()]
        
        return JSONResponse({
            'success': True,
            'concepts': concepts[:15]
        })
    
    except Exception as e:
        logger.error(f"Concepts error: {e}")
        return JSONResponse({
            'success': False,
            'message': str(e)
        })

@app.post("/api/generate/flashcards")
async def generate_flashcards(request: ProcessRequest):
    """Generate flashcards from document"""
    try:
        if not request.document_id or request.document_id not in knowledge_base:
            return JSONResponse({
                'success': False,
                'message': 'Document not found'
            })
        
        document = knowledge_base[request.document_id]
        document_text = document['full_text']
        
        logger.info(f"Generating flashcards for: {document['filename']}")
        
        # Truncate if too long
        if len(document_text) > 3000:
            document_text = document_text[:3000]
        
        prompt = f"""Create 5-7 flashcards from this text for learning purposes.
        Each flashcard should have a question and answer that tests understanding.
        Return as a JSON array with 'question' and 'answer' fields.
        
        TEXT:
        {document_text}
        
        FLASHCARDS (JSON array):"""
        
        flashcards_text = await generate_with_openai(prompt, document['full_text'], max_tokens=1000)
        
        # Parse flashcards
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', flashcards_text, re.DOTALL)
            if json_match:
                flashcards = json.loads(json_match.group())
            else:
                flashcards = json.loads(flashcards_text)
            
            if isinstance(flashcards, dict) and 'flashcards' in flashcards:
                flashcards = flashcards['flashcards']
        except:
            # Fallback to mock generation
            flashcards = json.loads(generate_document_based_mock("flashcards", document['full_text']))
        
        return JSONResponse({
            'success': True,
            'flashcards': flashcards[:7]
        })
    
    except Exception as e:
        logger.error(f"Flashcards error: {e}")
        return JSONResponse({
            'success': False,
            'message': str(e)
        })

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with AI about document"""
    try:
        # Get document context
        context = ""
        doc_info = "No document selected"
        
        if request.document_id and request.document_id in knowledge_base:
            document = knowledge_base[request.document_id]
            context = document['full_text']
            doc_info = document['filename']
            logger.info(f"Chat using: {doc_info}")
        
        # Manage conversation
        conversation_id = request.conversation_id or str(uuid.uuid4())
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Add user message with timestamp
        conversations[conversation_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Manage memory
        manage_memory()
        
        # Prepare context (limit length)
        context_preview = context[:3000] if context else "No document context available."
        
        # Generate response
        prompt = f"""You are a helpful AI assistant answering questions about a document.

DOCUMENT CONTENT:
{context_preview}

USER QUESTION: {request.message}

Answer based strictly on the document content. If the answer isn't in the document, say so politely. Keep responses concise and informative."""

        response_text = await generate_with_openai(prompt, context if context else "", max_tokens=500)
        
        # Add response to conversation
        conversations[conversation_id].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        return JSONResponse({
            'success': True,
            'response': response_text,
            'conversation_id': conversation_id
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse({
            'success': False,
            'message': str(e),
            'conversation_id': request.conversation_id or str(uuid.uuid4())
        })

@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents"""
    docs = []
    for doc_id, doc in knowledge_base.items():
        docs.append({
            'id': doc_id,
            'filename': doc['filename'],
            'uploaded_at': doc['uploaded_at'],
            'type': doc.get('type', 'unknown'),
            'word_count': doc.get('word_count', 0),
            'preview': doc['preview'][:100] + "..." if len(doc['preview']) > 100 else doc['preview']
        })
    return JSONResponse({'documents': docs})

@app.get("/api/document/{doc_id}")
async def get_document(doc_id: str):
    """Get document details"""
    if doc_id in knowledge_base:
        doc_copy = knowledge_base[doc_id].copy()
        # Don't send full text if it's too large
        if len(doc_copy.get('full_text', '')) > 5000:
            doc_copy['full_text'] = doc_copy['full_text'][:5000] + "..."
        return JSONResponse({
            'success': True,
            'document': doc_copy
        })
    raise HTTPException(status_code=404, detail="Document not found")

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document"""
    if doc_id in knowledge_base:
        del knowledge_base[doc_id]
        logger.info(f"Deleted document: {doc_id}")
        return JSONResponse({'success': True, 'message': 'Document deleted'})
    raise HTTPException(status_code=404, detail="Document not found")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    thread_pool.shutdown(wait=True)
    logger.info("Cleanup complete")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info("="*60)
    logger.info("VHOM AI Backend Server - PRODUCTION MODE")
    logger.info(f"Starting on port {port}")
    logger.info(f"OpenAI Enabled: {openai_client is not None}")
    logger.info("="*60)
    
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False  # Disable auto-reload in production
    )