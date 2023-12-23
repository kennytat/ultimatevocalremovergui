import os
import json
# from gradio_client import Client
from fastapi import FastAPI, Form, HTTPException, Request, Response, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
# from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import yt_dlp
import subprocess
import tempfile
# import zipfile
# import hashlib
import shutil
from pathlib import Path
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor
from datetime import datetime
# from urllib.parse import urljoin

temp_dir = os.path.join(tempfile.gettempdir(), "ultimatevocalremover")
youtube_temp_dir = os.path.join(temp_dir, 'youtube')
shutil.rmtree(youtube_temp_dir, ignore_errors=True)
Path(youtube_temp_dir).mkdir(parents=True, exist_ok=True) 

dcp = DeepChromaProcessor()
dccrp = DeepChromaChordRecognitionProcessor()
process_files = []
ydl = yt_dlp.YoutubeDL()
class LinkInput(BaseModel):
    link: str
class FilePath(BaseModel):
    path: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_json(self, message: object, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

## Init websocket manager
manager = ConnectionManager()

def new_dir_now():
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M")
    return date_time
  
def youtube_download(url, output_path):
    ydl_opts = {
        'format': 'bestaudio',
        'force_overwrites': True,
        'max_downloads': 5,
        'no_warnings': True,
        'ignore_no_formats_error': True,
        'restrictfilenames': True,
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
        ydl_download.download([url])
        
def get_final_redirected_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        return response.url
    except requests.RequestException as e:
        print(f"Error: {e}")
        return "invalid_url"

def chord_recognition(file_path):
  try:
    chroma = dcp(file_path)
    chord_data = dccrp(chroma)
    chord_data = [{'start': start, 'end': end, 'name': name} for start, end, name in chord_data]
    return chord_data
  except:
    print('chord_recognition failed::')
    return []

def uvr(file_path, output_dir, model_name, model_dir=os.path.join("models","Demucs_Models","v3_v4_repo")):
    subprocess.run(['demucs', file_path, '-o', output_dir, '-n', model_name, '--repo', model_dir])
  
# ## Call api to gradio for file processing
# def media_split(file_path="", link_url=""):
#   # auth_user = os.getenv('AUTH_USER', '')
#   # auth_pass = os.getenv('AUTH_PASS', '')
#   # client = Client("http://localhost:6870/", auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None)
#   client = Client("http://localhost:6870/")
#   result = client.predict(
#       [file_path] if file_path else [],	# List[str] (List of filepath(s) or URL(s) to files) in 'VIDEO|AUDIO' File component
#       link_url if link_url else "",	# str  in 'Youtube Link' Textbox component ## https://www.youtube.com/watch?v=-biOGdYiF-I
#       False,	# bool  in 'Enable' video_burn Checkbox component
#       False,	# bool  in 'Enable' export_subtitle Checkbox component
#       "Karaoke",	# str (Option from: [('Normal', 'Normal'), ('Karaoke', 'Karaoke')]) in 'Subtitle Mode' Dropdown component
#       "large-v3",	# str (Option from: [('tiny', 'tiny'), ('base', 'base'), ('small', 'small'), ('medium', 'medium'), ('large-v1', 'large-v1'), ('large-v2', 'large-v2'), ('large-v3', 'large-v3')]) in 'Whisper model' Dropdown component
#       "Automatic detection",	# str (Option from: [('Automatic detection', 'Automatic detection'), ('Arabic (ar)', 'Arabic (ar)'), ('Cantonese (yue)', 'Cantonese (yue)'), ('Chinese (zh)', 'Chinese (zh)'), ('Czech (cs)', 'Czech (cs)'), ('Danish (da)', 'Danish (da)'), ('Dutch (nl)', 'Dutch (nl)'), ('English (en)', 'English (en)'), ('Finnish (fi)', 'Finnish (fi)'), ('French (fr)', 'French (fr)'), ('German (de)', 'German (de)'), ('Greek (el)', 'Greek (el)'), ('Hebrew (he)', 'Hebrew (he)'), ('Hindi (hi)', 'Hindi (hi)'), ('Hungarian (hu)', 'Hungarian (hu)'), ('Italian (it)', 'Italian (it)'), ('Japanese (ja)', 'Japanese (ja)'), ('Korean (ko)', 'Korean (ko)'), ('Persian (fa)', 'Persian (fa)'), ('Polish (pl)', 'Polish (pl)'), ('Portuguese (pt)', 'Portuguese (pt)'), ('Russian (ru)', 'Russian (ru)'), ('Spanish (es)', 'Spanish (es)'), ('Turkish (tr)', 'Turkish (tr)'), ('Ukrainian (uk)', 'Ukrainian (uk)'), ('Urdu (ur)', 'Urdu (ur)'), ('Vietnamese (vi)', 'Vietnamese (vi)')]) in 'Target language' Dropdown component
#       False,	# bool  in 'Enable' Burn subtitle into video Checkbox component 
#       40,	# int | float (numeric value between 2 and 50) in 'Batch Size' Slider component
#       10,	# int | float (numeric value between 5 and 50) in 'Chuck Size' Slider component
#       25,	# int | float (numeric value between 10 and 30) in 'Font Size' Slider component
#       "Demucs",	# str (Option from: [('MDX-Net', 'MDX-Net'), ('Demucs', 'Demucs'), ('VR Arc', 'VR Arc')]) in 'AI Tech' Dropdown component
#       "v4 | htdemucs",	# str (Option from: [('v1 | Tasnet', 'v1 | Tasnet'), ('v1 | Tasnet_extra', 'v1 | Tasnet_extra'), ('v1 | Demucs', 'v1 | Demucs'), ('v1 | Demucs_extra', 'v1 | Demucs_extra'), ('v1 | Light', 'v1 | Light'), ('v1 | Light_extra', 'v1 | Light_extra'), ('v1 | Tasnet.gz', 'v1 | Tasnet.gz'), ('v1 | Tasnet_extra.gz', 'v1 | Tasnet_extra.gz'), ('v1 | Demucs_extra.gz', 'v1 | Demucs_extra.gz'), ('v1 | Light.gz', 'v1 | Light.gz'), ('v1 | Light_extra.gz', 'v1 | Light_extra.gz'), ('v2 | Tasnet', 'v2 | Tasnet'), ('v2 | Tasnet_extra', 'v2 | Tasnet_extra'), ('v2 | Demucs48_hq', 'v2 | Demucs48_hq'), ('v2 | Demucs', 'v2 | Demucs'), ('v2 | Demucs_extra', 'v2 | Demucs_extra'), ('v2 | Demucs_unittest', 'v2 | Demucs_unittest'), ('v3 | mdx', 'v3 | mdx'), ('v3 | mdx_extra', 'v3 | mdx_extra'), ('v3 | mdx_extra_q', 'v3 | mdx_extra_q'), ('v3 | mdx_q', 'v3 | mdx_q'), ('v3 | repro_mdx_a', 'v3 | repro_mdx_a'), ('v3 | repro_mdx_a_hybrid', 'v3 | repro_mdx_a_hybrid'), ('v3 | repro_mdx_a_time', 'v3 | repro_mdx_a_time'), ('v3 | UVR_Model_1', 'v3 | UVR_Model_1'), ('v3 | UVR_Model_2', 'v3 | UVR_Model_2'), ('v3 | UVR_Model_Bag', 'v3 | UVR_Model_Bag'), ('v4 | hdemucs_mmi', 'v4 | hdemucs_mmi'), ('v4 | htdemucs', 'v4 | htdemucs'), ('v4 | htdemucs_ft', 'v4 | htdemucs_ft'), ('v4 | htdemucs_6s', 'v4 | htdemucs_6s'), ('v4 | UVR_Model_ht', 'v4 | UVR_Model_ht')]) in 'UVR Model' Dropdown component
#       api_name="/convert"
#   )
#   zip_path = result[0]
#   extract_path, _ = os.path.splitext(zip_path)
#   print("prepare extract file::")
#   with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)
#   return extract_path

## Execute directly for file processing
def media_split(file_path="", link_url=""):
    if link_url:
      media_info =  ydl.extract_info(link_url, download=False)
      download_path = f"{os.path.join(youtube_temp_dir, media_info['title'])}.webm"
      youtube_download(link_url, download_path)
      file_path = download_path
    base_name, _ = os.path.splitext(file_path)
    file_name = os.path.basename(base_name)
    output_dir = os.path.join(temp_dir, new_dir_now())
    model_name = "htdemucs"
    uvr(file_path, output_dir, model_name, model_dir=os.path.join("models","Demucs_Models","v3_v4_repo"))
    return os.path.join(output_dir, model_name, file_name)
   
# Define function for processing files
def process_media(file_path='', link_url='') -> str:
    # url = hashlib.md5(open(file_path, "rb").read()).hexdigest() if file_path and os.path.exists(file_path) and os.path.isfile(file_path) else (link_url if link_url else None)
    # found_items = [item for item in process_files if "url" in item and item["url"] == url] if url else None
    # if found_items and found_items[0] and "processed_data" in found_items[0] and "wavFiles" in found_items[0]["processed_data"] and found_items[0]["processed_data"]["wavFiles"][0] and os.path.exists(found_items[0]["processed_data"]["wavFiles"][0]):
    #   return found_items[0]["processed_data"]
    # else:
      wav_dir = media_split(file_path=file_path, link_url=link_url)
      file_name = os.path.basename(wav_dir)
      wav_files = [os.path.join(wav_dir,wav) for wav in os.listdir(wav_dir) if wav.endswith('.wav')]
      print("extracted file::", wav_files)
      os.remove(file_path) if file_path and os.path.exists(file_path) else None
      print(f"process_media done::", wav_dir, file_name)
      # Placeholder processing logic
      processed_data = {
          "fileName": file_name,
          "wavFiles": wav_files
      }
      # process_files.append({
      #   "url": url,
      #   "processed_data": processed_data
      # })
      return processed_data

## Start fast api server  
app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="templates/assets"), name="assets")
app.mount("/icons", StaticFiles(directory="templates/icons"), name="icons")
# Add middleware to manage sessions
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")
origins = [
    "http://localhost",
    "http://localhost:5173",
    "https://ms.kentco.xyz",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Jinja2 template renderer
templates = Jinja2Templates(directory="templates")

def is_logged_in(request: Request):
    return "user_logged_in" in request.session

@app.get("/login")
async def login_page(request: Request):
    if is_logged_in(request):
        return RedirectResponse(url="/", status_code=303)
    # if is_logged_in(request):
    #     base_url = str(request.base_url)
    #     redirect_url = urljoin(base_url, '/')
    #     return RedirectResponse(url=redirect_url, status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == "admin" and password == "mypassword":
        request.session["user_logged_in"] = True
        return RedirectResponse(url="/", status_code=303)
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/")
async def root(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/{file_name}.{extension}")
async def handle_file(file_name: str, extension: str, request: Request):
    mime_types = {
        "js": "application/javascript",
        "map": "application/javascript",
        "webmanifest": "application/manifest+json",
        "html": "text/html",
        "css": "text/css",
        "svg": "image/svg+xml",
        "ico": "image/x-icon",
    }
    if extension in mime_types:
        content = templates.get_template(f"{file_name}.{extension}").render({"request": request})
        return Response(content=content, media_type=mime_types[extension])
    else:
        raise HTTPException(status_code=404, detail="file not found")

@app.get("/{_:path}")
async def catch_all(request: Request):
    return RedirectResponse(url="/", status_code=303)

@app.post("/media-upload")
async def media_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        tmp_file_path = os.path.join(temp_dir,file.filename)
        with open(tmp_file_path, "wb") as temp_file:
            temp_file.write(contents)
        # response = process_media(file_path=tmp_file_path)
        return {"url": tmp_file_path}                                   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
             
@app.post("/link-upload")
async def link_upload(link_input: LinkInput):
    final_url = get_final_redirected_url(link_input.url)
    if final_url.startswith("https://www.youtube.com"):
      try:
        response = process_media(link_url=final_url)
        return response
      except Exception as e:
          raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    else:
      raise HTTPException(status_code=500, detail=f"Link input is not valid URL: {link_input.url}")

@app.websocket("/ws/media-upload")
async def websocket_link_upload(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
              await manager.send_message("pong", websocket)
            else:
              try:
                  link_input = json.loads(data)
                  link_input["url"] = get_final_redirected_url(link_input["url"]) if link_input["type"] == "online" else link_input["url"]
                  if link_input == "invalid_url":
                    await manager.send_message("invalid_url", websocket)
                  else:
                    response = process_media(link_url=link_input["url"]) if link_input["type"] == "online" else process_media(file_path=link_input["url"])
                    await manager.send_json(response, websocket)
              except Exception as e:
                  await manager.send_json('{"error": "process_media_error"}', websocket)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Any cleanup code can go here
        pass
                  
@app.post("/file")
async def get_file(file_path: FilePath) -> FileResponse:
    if os.path.exists(file_path.path):
        return FileResponse(file_path.path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.post("/chord")
async def get_file(file_path: FilePath):
    if os.path.exists(file_path.path):
        respone = chord_recognition(file_path.path)
        return respone
    else:
        raise HTTPException(status_code=404, detail="File not found")
          
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
