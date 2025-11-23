from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import shared_state
import uvicorn
import threading

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stats")
def get_stats():
    return {
        "stats": shared_state.state.get_stats(),
        "controls": shared_state.state.get_controls()
    }

@app.post("/control/{key}/{value}")
def update_control(key: str, value: str):
    # Convert value if necessary
    if value.lower() == "true": val = True
    elif value.lower() == "false": val = False
    else:
        try:
            val = float(value)
        except:
            val = value
            
    shared_state.state.set_control(key, val)
    return {"status": "ok", "updated": {key: val}}

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

def start_dashboard_thread():
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
