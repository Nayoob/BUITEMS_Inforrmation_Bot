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

chat_history = []

@app.post("/ask")
def ask(body: Question):
    question = body.question

    chunks = retrieve(question)
    context = "\n\n".join(chunks)

    # Keep only last 4 messages to avoid token overflow
    if len(chat_history) > 4:
        chat_history.clear()

    system_prompt = """You are BUITEMS Assistant, an official AI chatbot for
Balochistan University of Information Technology, Engineering and Management Sciences (BUITEMS).

You have knowledge about the following areas:
- Admissions: admission process, merit criteria, entry tests, important dates
- Fee Structure: semester fees, registration fees, security deposits
- Faculties and Programs:
    * FICT - Faculty of Information & Communication Technology (CS, SE, IT, Computer Engineering, Electrical, Electronic, Telecom Engineering)
    * FOE  - Faculty of Engineering (Civil, Mechanical, Chemical, Geological, Mining, Petroleum & Gas, Textile Engineering, Architecture)
    * FMS  - Faculty of Management Sciences (Management Sciences, Public Administration)
    * FABS - Faculty of Applied Biosciences (Chemistry, Physics, Mathematics, Biology)
    * FLSI - Faculty of Life Sciences & Informatics (Biotechnology, Microbiology, Environmental Science)
    * FSSH - Faculty of Social Sciences & Humanities (Education, English, Psychology, Mass Communication, International Relations)
- Scholarships: HEC scholarships, need based, merit based
- Sub Campuses: Muslim Bagh campus, Zhob campus
- General: university overview, history, facilities, contacts

Strict Rules:
- Answer ONLY from the provided context
- If the answer is not in context say exactly:
  "I don't have that information right now. Please contact BUITEMS admissions office directly at admissions@buitms.edu.pk"
- Answer in the same language the user asks in (Urdu or English)
- Keep answers clear, structured and to the point
- Use bullet points for lists
- Use bold text for important information
- Never make up information that is not in the context"""

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