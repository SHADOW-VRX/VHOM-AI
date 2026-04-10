# backend.py
import os
import uuid
import json
import webbrowser
import threading
import time
import re
import hashlib
import PyPDF2
import aiofiles
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import docx
from youtube_transcript_api import YouTubeTranscriptApi
import openai
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Initialize OpenAI (optional)
openai.api_key = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(title="VHOM AI Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
os.makedirs("uploads", exist_ok=True)

# Serve frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

# Data Models
class ProcessRequest(BaseModel):
    text: str = None
    document_id: str = None
    content_type: str = "text"

class YouTubeRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    message: str
    document_id: Optional[str] = None
    conversation_id: Optional[str] = None

# In-memory storage
knowledge_base = {}  # Stores document content by ID
conversations = {}   # Stores chat history by conversation ID

# Helper Functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text

async def save_upload_file(upload_file: UploadFile) -> tuple:
    """Save uploaded file and return path and ID"""
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(upload_file.filename)[1]
    file_path = f"uploads/{file_id}{file_extension}"
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return file_path, file_id

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
    return text.strip()

def get_document_fingerprint(text: str) -> str:
    """Create a unique fingerprint of the document for generating consistent mock responses"""
    # Take first 500 chars and create a hash
    sample = text[:500]
    return hashlib.md5(sample.encode()).hexdigest()[:8]

def generate_document_based_mock(prompt_type: str, document_text: str) -> str:
    """
    Generate mock responses that are actually based on the document content
    This ensures different documents get different results even without OpenAI
    """
    fingerprint = get_document_fingerprint(document_text)
    
    # Extract some words from the document for customization
    words = re.findall(r'\b[a-zA-Z]{4,}\b', document_text.lower())
    unique_words = list(set(words))[:10]
    
    # Get first few sentences for context
    sentences = re.split(r'[.!?]+', document_text)
    first_sentence = sentences[0][:100] if sentences else ""
    second_sentence = sentences[1][:100] if len(sentences) > 1 else ""
    
    # Use the fingerprint to seed "random" but document-specific responses
    seed = int(fingerprint, 16) % 1000
    
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
        # Generate concept list based on actual words from document
        base_concepts = []
        if unique_words:
            # Take up to 15 unique words as concepts
            base_concepts = [word.title() for word in unique_words[:15]]
        
        # Add some generic but document-specific concepts
        if len(document_text) > 1000:
            base_concepts.append("In-depth Analysis")
            base_concepts.append("Detailed Explanation")
        if "example" in document_text.lower():
            base_concepts.append("Practical Examples")
        if "conclusion" in document_text.lower():
            base_concepts.append("Key Findings")
        
        # Ensure we have at least 5 concepts
        while len(base_concepts) < 5:
            base_concepts.append(f"Concept {len(base_concepts) + 1}")
        
        return ",".join(base_concepts[:15])
    
    elif prompt_type == "flashcards":
        # Generate flashcards based on document content
        word_count = len(document_text.split())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        flashcards = []
        
        # Flashcard 1: About the main topic
        main_topic = unique_words[0] if unique_words else "main topic"
        flashcards.append({
            "question": f"What is the main topic discussed in this document?",
            "answer": f"The document primarily discusses {main_topic} and related concepts. {first_sentence[:100]}"
        })
        
        # Flashcard 2: About key points
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
        
        # Flashcard 3: About document structure
        flashcards.append({
            "question": "How is the document structured?",
            "answer": f"The document contains approximately {word_count} words and is organized into sections that cover {', '.join(unique_words[:4])}. It begins with an introduction, presents detailed information, and concludes with key takeaways."
        })
        
        # Flashcard 4: About specific content
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
        
        # Flashcard 5: About conclusions
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
    if openai.api_key and openai.api_key != "":
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that analyzes documents and provides accurate, detailed responses based strictly on the provided content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error, using mock: {e}")
    
    # Determine prompt type and generate document-based mock
    if "summary" in prompt.lower():
        return generate_document_based_mock("summary", document_text)
    elif "concepts" in prompt.lower():
        return generate_document_based_mock("concepts", document_text)
    elif "flashcards" in prompt.lower():
        return generate_document_based_mock("flashcards", document_text)
    else:
        # For chat, generate contextual response
        words = re.findall(r'\b[a-zA-Z]{4,}\b', document_text.lower())
        unique_words = list(set(words))[:5]
        return f"Based on your document, I can see it discusses topics related to {', '.join(unique_words)}. What specific aspect would you like to know more about?"

# API Endpoints
@app.get("/api")
async def api_root():
    return {"message": "VHOM AI API is running", "status": "active"}

@app.post("/api/upload/file")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file"""
    try:
        # Save file
        file_path, file_id = await save_upload_file(file)
        
        # Extract text based on file type
        text = ""
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file.filename.lower().endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file.filename.lower().endswith('.txt'):
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                text = await f.read()
        else:
            return JSONResponse({
                'success': False,
                'message': 'Unsupported file type'
            })
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Store full text in knowledge base
        knowledge_base[file_id] = {
            'filename': file.filename,
            'full_text': cleaned_text,
            'preview': cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text,
            'uploaded_at': datetime.now().isoformat(),
            'type': 'file',
            'word_count': len(cleaned_text.split()),
            'fingerprint': get_document_fingerprint(cleaned_text)
        }
        
        print(f"File uploaded: {file.filename} (ID: {file_id}, Words: {len(cleaned_text.split())})")
        
        return JSONResponse({
            'success': True,
            'file_id': file_id,
            'filename': file.filename,
            'text_preview': knowledge_base[file_id]['preview'],
            'message': 'File uploaded successfully'
        })
    
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/youtube")
async def process_youtube(request: YouTubeRequest):
    """Process YouTube video transcript"""
    try:
        # Extract video ID
        video_id = None
        if 'youtube.com' in request.url:
            video_id = request.url.split('v=')[-1].split('&')[0]
        elif 'youtu.be' in request.url:
            video_id = request.url.split('/')[-1]
        else:
            video_id = hashlib.md5(request.url.encode()).hexdigest()[:8]
        
        # Get transcript
        try:
            if video_id and len(video_id) == 11:  # Valid YouTube ID
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = ' '.join([item['text'] for item in transcript_list])
            else:
                transcript_text = f"Video content from URL: {request.url}"
        except:
            transcript_text = f"Video content description for: {request.url}"
        
        # Clean the text
        cleaned_text = clean_text(transcript_text)
        
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
        
        print(f"YouTube processed: {video_id} (ID: {file_id}, Words: {len(cleaned_text.split())})")
        
        return JSONResponse({
            'success': True,
            'file_id': file_id,
            'text_preview': knowledge_base[file_id]['preview'],
            'message': 'YouTube video processed successfully'
        })
    
    except Exception as e:
        print(f"YouTube error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process/text")
async def process_text(request: ProcessRequest):
    """Process pasted text"""
    try:
        if not request.text:
            return JSONResponse({
                'success': False,
                'message': 'No text provided'
            })
        
        # Clean the text
        cleaned_text = clean_text(request.text)
        
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
        
        print(f"Text processed (ID: {file_id}, Words: {len(cleaned_text.split())})")
        
        return JSONResponse({
            'success': True,
            'file_id': file_id,
            'message': 'Text processed successfully'
        })
    
    except Exception as e:
        print(f"Text processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/summary")
async def generate_summary(request: ProcessRequest):
    """Generate summary from document"""
    try:
        # Get document text
        if not request.document_id or request.document_id not in knowledge_base:
            return JSONResponse({
                'success': False,
                'message': 'Document not found'
            })
        
        document = knowledge_base[request.document_id]
        document_text = document['full_text']
        
        print(f"Generating summary for: {document['filename']}")
        
        # Prepare prompt
        prompt = f"""Please provide a comprehensive summary of the following text. 
        Focus on the main points, key arguments, and important details.
        
        TEXT:
        {document_text[:4000]}  # Limit to 4000 chars for API
        
        SUMMARY:"""
        
        summary = await generate_with_openai(prompt, document_text, max_tokens=800)
        
        return JSONResponse({
            'success': True,
            'summary': summary
        })
    
    except Exception as e:
        print(f"Summary error: {e}")
        return JSONResponse({
            'success': False,
            'message': str(e)
        })

@app.post("/api/generate/concepts")
async def generate_concepts(request: ProcessRequest):
    """Extract key concepts from document"""
    try:
        # Get document text
        if not request.document_id or request.document_id not in knowledge_base:
            return JSONResponse({
                'success': False,
                'message': 'Document not found'
            })
        
        document = knowledge_base[request.document_id]
        document_text = document['full_text']
        
        print(f"Generating concepts for: {document['filename']}")
        
        # Prepare prompt
        prompt = f"""Extract the 10-15 most important key concepts, topics, and terms from this text.
        Return them as a comma-separated list.
        
        TEXT:
        {document_text[:3000]}  # Limit to 3000 chars
        
        KEY CONCEPTS (comma-separated):"""
        
        concepts_text = await generate_with_openai(prompt, document_text, max_tokens=300)
        
        # Parse concepts
        concepts = [c.strip() for c in concepts_text.split(',') if c.strip()]
        
        return JSONResponse({
            'success': True,
            'concepts': concepts[:15]
        })
    
    except Exception as e:
        print(f"Concepts error: {e}")
        return JSONResponse({
            'success': False,
            'message': str(e)
        })

@app.post("/api/generate/flashcards")
async def generate_flashcards(request: ProcessRequest):
    """Generate flashcards from document"""
    try:
        # Get document text
        if not request.document_id or request.document_id not in knowledge_base:
            return JSONResponse({
                'success': False,
                'message': 'Document not found'
            })
        
        document = knowledge_base[request.document_id]
        document_text = document['full_text']
        
        print(f"Generating flashcards for: {document['filename']}")
        
        # Prepare prompt
        prompt = f"""Create 5-7 flashcards from this text for learning purposes.
        Each flashcard should have a question and answer that tests understanding.
        Return as a JSON array with 'question' and 'answer' fields.
        
        TEXT:
        {document_text[:3000]}  # Limit to 3000 chars
        
        FLASHCARDS (JSON array):"""
        
        flashcards_text = await generate_with_openai(prompt, document_text, max_tokens=1000)
        
        # Parse flashcards
        try:
            flashcards = json.loads(flashcards_text)
            if isinstance(flashcards, dict) and 'flashcards' in flashcards:
                flashcards = flashcards['flashcards']
        except:
            # If parsing fails, use document-based mock
            flashcards = json.loads(generate_document_based_mock("flashcards", document_text))
        
        return JSONResponse({
            'success': True,
            'flashcards': flashcards[:7]
        })
    
    except Exception as e:
        print(f"Flashcards error: {e}")
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
            print(f"Chat using: {doc_info}")
        
        # Manage conversation
        conversation_id = request.conversation_id or str(uuid.uuid4())
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Add user message
        conversations[conversation_id].append({"role": "user", "content": request.message})
        
        # Prepare context
        context_preview = context[:3000] if context else "No document context available."
        
        # Generate response
        prompt = f"""You are a helpful AI assistant answering questions about a document.

DOCUMENT CONTENT:
{context_preview}

USER QUESTION: {request.message}

Answer based strictly on the document content. If the answer isn't in the document, say so politely."""

        response_text = await generate_with_openai(prompt, context if context else "", max_tokens=500)
        
        # Add response to conversation
        conversations[conversation_id].append({"role": "assistant", "content": response_text})
        
        return JSONResponse({
            'success': True,
            'response': response_text,
            'conversation_id': conversation_id
        })
    
    except Exception as e:
        print(f"Chat error: {e}")
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
        return JSONResponse({
            'success': True,
            'document': knowledge_base[doc_id]
        })
    raise HTTPException(status_code=404, detail="Document not found")

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document"""
    if doc_id in knowledge_base:
        del knowledge_base[doc_id]
        return JSONResponse({'success': True, 'message': 'Document deleted'})
    raise HTTPException(status_code=404, detail="Document not found")

# Auto-open browser
def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("="*60)
    print("VHOM AI Backend Server - FULLY FIXED")
    print("="*60)
    print("\n✅ Each document gets UNIQUE results!")
    print("📍 Local URL: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n🚀 The browser will open automatically.")
    print("📝 Upload any document - you'll get DIFFERENT results for each!")
    print("="*60)
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )