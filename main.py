import os
from gradio_client import Client
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
import subprocess
import shutil
import tempfile
import zipfile

temp_dir = os.path.join(tempfile.gettempdir(), "ultimatevocalremover")

class LinkInput(BaseModel):
    link: str
class FileName(BaseModel):
    name: str

## Call api to gradio for file processing
def media_split(file_path="", link_url=""):
  client = Client("http://localhost:6870/")
  return client.predict(
  		[file_path] if file_path else [],	# List[str] (List of filepath(s) or URL(s) to files) in 'VIDEO|AUDIO' File component
  		link_url if link_url else "",	# str  in 'Youtube Link' Textbox component ## https://www.youtube.com/watch?v=-biOGdYiF-I
  		False,	# bool  in 'Enable' Checkbox component
  		"Karaoke",	# str (Option from: [('Normal', 'Normal'), ('Karaoke', 'Karaoke')]) in 'Subtitle Mode' Dropdown component
  		"large-v3",	# str (Option from: [('tiny', 'tiny'), ('base', 'base'), ('small', 'small'), ('medium', 'medium'), ('large-v1', 'large-v1'), ('large-v2', 'large-v2'), ('large-v3', 'large-v3')]) in 'Whisper model' Dropdown component
  		"Automatic detection",	# str (Option from: [('Automatic detection', 'Automatic detection'), ('Arabic (ar)', 'Arabic (ar)'), ('Cantonese (yue)', 'Cantonese (yue)'), ('Chinese (zh)', 'Chinese (zh)'), ('Czech (cs)', 'Czech (cs)'), ('Danish (da)', 'Danish (da)'), ('Dutch (nl)', 'Dutch (nl)'), ('English (en)', 'English (en)'), ('Finnish (fi)', 'Finnish (fi)'), ('French (fr)', 'French (fr)'), ('German (de)', 'German (de)'), ('Greek (el)', 'Greek (el)'), ('Hebrew (he)', 'Hebrew (he)'), ('Hindi (hi)', 'Hindi (hi)'), ('Hungarian (hu)', 'Hungarian (hu)'), ('Italian (it)', 'Italian (it)'), ('Japanese (ja)', 'Japanese (ja)'), ('Korean (ko)', 'Korean (ko)'), ('Persian (fa)', 'Persian (fa)'), ('Polish (pl)', 'Polish (pl)'), ('Portuguese (pt)', 'Portuguese (pt)'), ('Russian (ru)', 'Russian (ru)'), ('Spanish (es)', 'Spanish (es)'), ('Turkish (tr)', 'Turkish (tr)'), ('Ukrainian (uk)', 'Ukrainian (uk)'), ('Urdu (ur)', 'Urdu (ur)'), ('Vietnamese (vi)', 'Vietnamese (vi)')]) in 'Target language' Dropdown component
  		False,	# bool  in 'Enable' Burn subtitle into video Checkbox component 
  		40,	# int | float (numeric value between 2 and 50) in 'Batch Size' Slider component
  		10,	# int | float (numeric value between 5 and 50) in 'Chuck Size' Slider component
  		25,	# int | float (numeric value between 10 and 30) in 'Font Size' Slider component
  		"Demucs",	# str (Option from: [('MDX-Net', 'MDX-Net'), ('Demucs', 'Demucs'), ('VR Arc', 'VR Arc')]) in 'AI Tech' Dropdown component
  		"v4 | htdemucs",	# str (Option from: [('v1 | Tasnet', 'v1 | Tasnet'), ('v1 | Tasnet_extra', 'v1 | Tasnet_extra'), ('v1 | Demucs', 'v1 | Demucs'), ('v1 | Demucs_extra', 'v1 | Demucs_extra'), ('v1 | Light', 'v1 | Light'), ('v1 | Light_extra', 'v1 | Light_extra'), ('v1 | Tasnet.gz', 'v1 | Tasnet.gz'), ('v1 | Tasnet_extra.gz', 'v1 | Tasnet_extra.gz'), ('v1 | Demucs_extra.gz', 'v1 | Demucs_extra.gz'), ('v1 | Light.gz', 'v1 | Light.gz'), ('v1 | Light_extra.gz', 'v1 | Light_extra.gz'), ('v2 | Tasnet', 'v2 | Tasnet'), ('v2 | Tasnet_extra', 'v2 | Tasnet_extra'), ('v2 | Demucs48_hq', 'v2 | Demucs48_hq'), ('v2 | Demucs', 'v2 | Demucs'), ('v2 | Demucs_extra', 'v2 | Demucs_extra'), ('v2 | Demucs_unittest', 'v2 | Demucs_unittest'), ('v3 | mdx', 'v3 | mdx'), ('v3 | mdx_extra', 'v3 | mdx_extra'), ('v3 | mdx_extra_q', 'v3 | mdx_extra_q'), ('v3 | mdx_q', 'v3 | mdx_q'), ('v3 | repro_mdx_a', 'v3 | repro_mdx_a'), ('v3 | repro_mdx_a_hybrid', 'v3 | repro_mdx_a_hybrid'), ('v3 | repro_mdx_a_time', 'v3 | repro_mdx_a_time'), ('v3 | UVR_Model_1', 'v3 | UVR_Model_1'), ('v3 | UVR_Model_2', 'v3 | UVR_Model_2'), ('v3 | UVR_Model_Bag', 'v3 | UVR_Model_Bag'), ('v4 | hdemucs_mmi', 'v4 | hdemucs_mmi'), ('v4 | htdemucs', 'v4 | htdemucs'), ('v4 | htdemucs_ft', 'v4 | htdemucs_ft'), ('v4 | htdemucs_6s', 'v4 | htdemucs_6s'), ('v4 | UVR_Model_ht', 'v4 | UVR_Model_ht')]) in 'UVR Model' Dropdown component
  		api_name="/convert"
  )
   
# Define function for processing files
def process_media(file_path='', link_url='') -> str:
    result = media_split(file_path=file_path, link_url=link_url)
    zip_path = result[0]
    extract_path, _ = os.path.splitext(zip_path)
    file_name = os.path.basename(extract_path)
    print("prepare extract file::")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      zip_ref.extractall(extract_path)
    wav_files = [os.path.join(extract_path,wav) for wav in os.listdir(extract_path) if wav.endswith('.wav')]
    print("extracted file::", wav_files)
    os.remove(file_path) if file_path and os.path.exists(file_path) else None
    print(f"process_media done::", extract_path, file_name)
    # Placeholder processing logic
    processed_data = {
        "fileName": file_name,
        "wavFiles": ",".join(wav_files)
    }
    return processed_data

## Start fast api server  
app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# Add middleware to manage sessions
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Jinja2 template renderer
templates = Jinja2Templates(directory="templates")

def is_logged_in(request: Request):
    return "user_logged_in" in request.session

@app.get("/login")
async def login_page(request: Request):
    if is_logged_in(request):
        return RedirectResponse(url="/home", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == "admin" and password == "password":
        request.session["user_logged_in"] = True
        return RedirectResponse(url="/home", status_code=303)
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/home")
async def home(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/")
async def root(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url="/home", status_code=303)

@app.get("/{_:path}")
async def catch_all(request: Request):
    return RedirectResponse(url="/home", status_code=303)

@app.post("/media-upload")
async def media_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        tmp_file_path = os.path.join(temp_dir,file.filename)
        with open(tmp_file_path, "wb") as temp_file:
            temp_file.write(contents)
        response = process_media(file_path=tmp_file_path)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
      
@app.post("/link-upload")
async def link_upload(link_input: LinkInput):
    if link_input.link.startswith("https://www.youtube.com"):
      try:
        response = process_media(link_url=link_input.link)
        return response
      except Exception as e:
          raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    else:
      raise HTTPException(status_code=500, detail=f"Link input is not valid URL: {link_input.link}")
      
@app.post("/file")
async def get_file(file_path: FileName) -> FileResponse:
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")
     
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


