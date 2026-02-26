from fastapi import FastAPI
from pydantic import BaseModel
from rag.retrieve import retrieve

app = FastAPI(title="Healthcare AI (Privacy Safe)")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Query):
    try:
        answer = retrieve(q.question)
        return {
            "status": "success",
            "answer": answer
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
