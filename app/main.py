from fastapi import FastAPI

from src.inference.predictor import TashkeelPredictor
from .models import TashkeelRequest,TashkeelResponse
app= FastAPI()

predictor= TashkeelPredictor
@app.on_event("startup")
def load_model():
    global predictor
    predictor = TashkeelPredictor(
        
        )

@app.post("/v1/diacritize",response_model=TashkeelResponse)
def diacritize_text(req: TashkeelRequest):
    """
    Endpoint to convert undiacritized Arabic text to fully diacritized text.
    """
    print(f"Req Text: {req.text} ")
    result_text=predictor.diacritize(text=req.text,
                         use_beam=True, beam_size=4)
    print(f"Response Text: {result_text}")
    return {
        "original_text": req.text,
        "diacritized_text": result_text,
        "status": "success"
    }
