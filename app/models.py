from pydantic import BaseModel, Field
from typing import Optional

class TashkeelRequest(BaseModel):
    text: str = Field(..., description="Undiacritized Arabic text", min_length=1)
    use_beam:Optional[bool]=False
class TashkeelResponse(BaseModel):
    original_text: str
    diacritized_text: str
    status: str