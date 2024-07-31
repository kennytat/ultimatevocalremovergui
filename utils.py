import srt
import re
import os
import yt_dlp
import ffmpeg
import fnmatch
from natsort import natsorted
from datetime import timedelta, datetime
import shutil
# from .logging_setup import logger
# from urllib.parse import urlparse
# from IPython.utils import capture
# from vietnam_number import n2w


video_patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
audio_patterns = ['*.mp3', '*.wav', '*.aac', '*.flac', '*.ogg']

def find_files(directory, patterns):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(root, filename))
    matches = natsorted(matches, key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x)))
    return matches
  
def find_all_media_files(directory):
    video_files = find_files(directory, video_patterns)
    audio_files = find_files(directory, audio_patterns)
    media_files = video_files + audio_files
    media_files = natsorted(media_files, key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x)))
    return media_files

def find_most_matching_prefix(path_list, path):
    matching_prefix = ""
    for prefix in path_list:
        if path.startswith(prefix) and len(prefix) > len(matching_prefix):
            matching_prefix = prefix
    return matching_prefix
  
def is_video_or_audio(file_path):
    try:
        info = ffmpeg.probe(file_path, select_streams='v:0', show_entries='stream=codec_type')
        if len(info["streams"]) > 0 and info["streams"][0]["codec_type"] == "video":
            return "video"
    except ffmpeg.Error:
        print("ffmpeg error:")
        pass

    try:
        info = ffmpeg.probe(file_path, select_streams='a:0', show_entries='stream=codec_type')
        if len(info["streams"]) > 0 and info["streams"][0]["codec_type"] == "audio":
            return "audio"
    except ffmpeg.Error:
        print("ffmpeg error:")
        pass
    return "Unknown"

def is_windows_path(path):
    # Use a regular expression to check for a Windows drive letter and separator
    return re.match(r'^[a-zA-Z]:\\', path) is not None

def convert_to_wsl_path(path):
    # Convert Windows path to WSL path
    drive_letter, rest_of_path = path.split(':\\', 1)
    wsl_path = "/".join(['/mnt', drive_letter.lower(), rest_of_path.replace('\\', '/')])
    return wsl_path.rstrip("/")
    
def youtube_download(url, output_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'force_overwrites': True,
        'max_downloads': 5,
        'no_warnings': True,
        'ignore_no_formats_error': True,
        'restrictfilenames': True,
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
        ydl_download.download([url])

def segments_to_srt(segments, output_path):
	# print("segments_to_srt::", type(segments[0]), segments)
	shutil.rmtree(output_path, ignore_errors=True)
	def srt_time(str):
		return re.sub(r"\.",",",re.sub(r"0{3}$","",str)) if re.search(r"\.\d{6}", str) else f'{str},000'
	for index, segment in enumerate(segments):
			startTime = srt_time(str(0)+str(timedelta(seconds=segment['start'])))
			endTime = srt_time(str(0)+str(timedelta(seconds=segment['end'])))
			text = segment['text']
			segmentId = index+1
			segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text and text[0] == ' ' else text}\n\n"
			with open(output_path, 'a', encoding='utf-8') as srtFile:
					srtFile.write(segment)
					
def srt_to_segments(srt_input:str):
  srt_list = list(srt.parse(srt_input))
  srt_segments = list([vars(obj) for obj in srt_list])
  srt_segments = [ segment for segment in srt_segments if segment['start'].total_seconds() > 0]
  for i, segment in enumerate(srt_segments):
    text = str(srt_segments[i]['content'])
    speaker = re.findall(r"SPEAKER_\d+", text)[0] if re.search(r"SPEAKER_\d+", text) else "SPEAKER_00"
    srt_segments[i]['speaker'] = speaker
    srt_segments[i]['start'] = srt_segments[i]['start'].total_seconds()
    srt_segments[i]['end'] = srt_segments[i]['end'].total_seconds()
    srt_segments[i]['text'] = re.sub(r"SPEAKER_\d+\:", "", text)
    srt_segments[i]['index'] = i + 1
    del srt_segments[i]['content']
  return srt_segments

def srt_to_lrc(srt_path:str):
  def srt_block_to_lrc(block):
      SRT_BLOCK_REGEX = re.compile(
              r'(\d+)[^\S\r\n]*[\r\n]+'
              r'(\d{2}:\d{2}:\d{2},\d{3,4})[^\S\r\n]*-->[^\S\r\n]*(\d{2}:\d{2}:\d{2},\d{3,4})[^\S\r\n]*[\r\n]+'
              r'([\s\S]*)')
      match = SRT_BLOCK_REGEX.search(block)
      if not match:
          return None
      num, ts, te, content = match.groups()
      ts = ts[3:-1].replace(',', '.')
      # te = te[3:-1].replace(',', '.')
      co = content.replace('\n', ' ')
      return '[%s]%s\n' % (ts, co)
  with open(srt_path, encoding='utf8') as file_in:
    str_in = file_in.read()
    blocks_in = str_in.replace('\r\n', '\n').split('\n\n')
    blocks_out = [srt_block_to_lrc(block) for block in blocks_in]
    blocks_out = filter(None, blocks_out)
    str_out = ''.join(blocks_out)
    print("str_out::",str_out)
    with open(srt_path.replace('.srt', '.lrc'), 'w', encoding='utf8') as file_out:
        file_out.write(str_out)