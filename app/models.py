# api/models.py
from pydantic import BaseModel, Field

class TashkeelRequest(BaseModel):
    text: str = Field(..., description="Undiacritized Arabic text", min_length=1)

class TashkeelResponse(BaseModel):
    original_text: str
    diacritized_text: str
    status: str