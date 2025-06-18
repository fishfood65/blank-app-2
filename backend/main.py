from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
from utils.docx_generator import generate_docx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunbookRequest(BaseModel):
    section: str
    inputs: dict

@app.post("/generate-runbook/")
def generate_runbook(data: RunbookRequest):
    docx_file = generate_docx(data.section, data.inputs)
    return StreamingResponse(
        docx_file,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename={data.section}_runbook.docx"}
    )
