import os
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Proxy API (API2)", description="Pasarela ligera hacia el backend de ML")

# Configuración del Túnel Cloudflare (Opcional para prod)
# Si estás en local, usa la URL de localhost para velocidad máxima
API1_BASE_URL = os.getenv("API1_BASE_URL", "http://127.0.0.1:8000")
TUNNEL_URL = os.getenv("TUNNEL_URL", "https://metals-consistently-sheet-xml.trycloudflare.com")

# Si se proporciona una URL de túnel por env, se usa, de lo contrario localhost
if os.getenv("USE_TUNNEL") == "true":
    API1_BASE_URL = TUNNEL_URL

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    """Obtiene la lista de tareas desde la API1"""
    async with await get_client() as client:
        try:
            response = await client.get("/tasks/")
            if response.is_success:
                return response.json()
            raise HTTPException(status_code=response.status_code, detail="Error en API1")
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Error de comunicación con API1: {exc}")

@app.post("/tasks/{task_id}/")
async def proxy_upload(task_id: str, file: UploadFile = File(...)):
    """Reenvía la subida de archivos a la API1 usando streaming para ahorrar RAM"""
    files = {'file': (file.filename, file.file, file.content_type)}
    
    async with await get_client() as client:
        try:
            # Timeout largo para procesamientos sincronos pesados
            response = await client.post(f"/tasks/{task_id}/", files=files, timeout=300.0)
            
            if response.is_success:
                return response.json()
            
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            raise HTTPException(status_code=response.status_code, detail=error_detail)
            
        except httpx.RequestError as exc:
             raise HTTPException(status_code=502, detail=f"Error subiendo a API1: {exc}")

@app.get("/tasks/{task_id}/result/{file_id}/")
async def get_result(task_id: str, file_id: str):
    """Consulta el estado o resultado de una tarea en la API1"""
    async with await get_client() as client:
        try:
            response = await client.get(f"/tasks/{task_id}/result/{file_id}/")
            if response.is_success:
                return response.json()
            raise HTTPException(status_code=response.status_code, detail="Error obteniendo resultado de API1")
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Error de comunicación con API1: {exc}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
