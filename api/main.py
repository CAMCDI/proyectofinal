import os
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI(title="Proxy API (API2)", description="Lightweight Gateway to ML Backend")

# CONFIGURACIÓN: Pega aquí tu URL de Cloudflare Tunnel
# Ejemplo: "https://mi-tunel.trycloudflare.com"
TUNNEL_URL = "https://staff-bonds-bend-packets.trycloudflare.com" 

# El sistema usará la URL del túnel si la pones arriba, 
# de lo contrario buscará una variable de entorno o usará localhost.
API1_BASE_URL = TUNNEL_URL or os.getenv("API1_BASE_URL", "http://127.0.0.1:8000")

# CORS Setup
origins = [
    "*", # Allow all for now to ensure frontend works from any device/origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_client():
    return httpx.AsyncClient(base_url=API1_BASE_URL, timeout=60.0)

@app.get("/")
async def health_check():
    return {"status": "ok", "proxy_target": API1_BASE_URL}

@app.get("/tasks/")
async def get_tasks():
    async with await get_client() as client:
        try:
            response = await client.get("/tasks/")
            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Error communicating with API1: {exc}")

@app.post("/tasks/{task_id}/")
async def proxy_upload(task_id: str, file: UploadFile = File(...)):
    """
    Proxies the file upload to API1 using streaming to avoid loading the entire file into RAM.
    """
    
    files = {'file': (file.filename, file.file, file.content_type)}
    
    async with await get_client() as client:
        try:
            # Note: We need a long timeout for large file uploads/processing if it's synchronous
            # But the requirement says API1 does ML processing, so we validat connection.
            # Usually API1 should accept the file and return quickly if processing is async,
            # but if it blocks, we need timeout.
            response = await client.post(f"/tasks/{task_id}/", files=files, timeout=300.0)
            
            if response.status_code != 200:
                 try:
                     error_detail = response.json()
                 except:
                     error_detail = response.text
                 raise HTTPException(status_code=response.status_code, detail=error_detail)
            
            return response.json()
            
        except httpx.RequestError as exc:
             raise HTTPException(status_code=502, detail=f"Error uploading to API1: {exc}")

@app.get("/tasks/{task_id}/result/{file_id}/")
async def get_result(task_id: str, file_id: str):
    async with await get_client() as client:
        try:
            response = await client.get(f"/tasks/{task_id}/result/{file_id}/")
            
            if response.status_code != 200:
                # Pass through the error status code
                raise HTTPException(status_code=response.status_code, detail="Error fetching result from API1")

            return response.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Error communicating with API1: {exc}")

if __name__ == "__main__":
    import uvicorn
    # Local development run
    uvicorn.run(app, host="0.0.0.0", port=8001)
