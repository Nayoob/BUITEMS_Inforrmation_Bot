import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from retriever import retrieve
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Question(BaseModel):
    question: str
    history: list = []


@app.post("/ask")
def ask(body: Question):
    question = body.question

    chunks = retrieve(question)
    context = "\n\n".join(chunks)

    system_prompt = """You are BUITEMS Assistant, an official AI chatbot for 
Balochistan University of Information Technology, Engineering and Management Sciences.

You help students and faculty with:
- Admissions requirements and procedures
- Department and program information
- Fee structure and scholarships
- Rules, regulations and policies
- Campus facilities and services
- Important dates and deadlines
- Contact information

Rules:
- Answer ONLY from the provided context
- If not in context say: 'I don't have that information. Please contact BUITEMS directly at info@buitms.edu.pk'
- Answer in the same language the user asks in (Urdu or English)
- Keep answers clear and concise
- Format answers with bullet points where appropriate"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=512,
        temperature=0.5
    )

    answer = response.choices[0].message.content

    return {"answer": answer}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")