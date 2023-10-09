# GUI modules
from dotenv import load_dotenv
import hashlib
import json
import librosa
import logging
import os
import pickle  # Save Data
import psutil
import gradio as gr
import shutil
import sys
import yt_dlp
import ffmpeg
# import soundfile as sf
import time
import torch
import traceback
from gui_data.app_size_values import ImagePath, AdjustedValues as av
from gui_data.constants import *
from gui_data.error_handling import error_text, error_dialouge
from gui_data.old_data_check import file_check, remove_unneeded_yamls, remove_temps
from gui_data.tkinterdnd2 import TkinterDnD, DND_FILES
from lib_v5.vr_network.model_param_init import ModelParameters
from pathlib  import Path
from separate import SeperateDemucs, SeperateMDX, SeperateVR, save_format
from typing import List
import sys
import tempfile
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('UVR BEGIN')

load_dotenv()
temp_dir = os.path.join(tempfile.gettempdir(), "ultimatevocalremover")
ydl = yt_dlp.YoutubeDL()

def new_dir_now():
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M")
    return date_time
  
def save_data(data):
    """
    Saves given data as a .pkl (pickle) file

    Paramters:
        data(dict):
            Dictionary containing all the necessary data to save
    """
    # Open data file, create it if it does not exist
    with open('data.pkl', 'wb') as data_file:
        pickle.dump(data, data_file)

def load_data() -> dict:
    """
    Loads saved pkl file and returns the stored data

    Returns(dict):
        Dictionary containing all the saved data
    """
    try:
        with open('data.pkl', 'rb') as data_file:  # Open data file
            data = pickle.load(data_file)

        return data
    except (ValueError, FileNotFoundError):
        # Data File is corrupted or not found so recreate it

        save_data(data=DEFAULT_DATA)

        return load_data()

def load_model_hash_data(dictionary):
    '''Get the model hash dictionary'''

    with open(dictionary) as d:
        data = d.read()

    return json.loads(data)
  
# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_PATH)  # Change the current working directory to the base path

debugger = []

#--Constants--
#Models
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
MDX_MIXER_PATH = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
VR_MODEL_NAME_SELECT = os.path.join(VR_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_data.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
DEMUCS_MODEL_NAME_SELECT = os.path.join(DEMUCS_MODELS_DIR, 'model_data', 'model_name_mapper.json')
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_ensembles')
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, 'gui_data', 'saved_settings')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, 'temp_sample_clips')
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, 'ensemble_temps')


#Style
ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-Icon.ico')
MAIN_ICON_IMG_PATH = os.path.join(BASE_PATH, 'gui_data', 'img', 'GUI-Icon.png')
FONT_PATH = os.path.join(BASE_PATH, 'gui_data', 'fonts', 'centurygothic', 'GOTHIC.TTF')#ensemble_temps

#Other
COMPLETE_CHIME = os.path.join(BASE_PATH, 'gui_data', 'complete_chime.wav')
FAIL_CHIME = os.path.join(BASE_PATH, 'gui_data', 'fail_chime.wav')
CHANGE_LOG = os.path.join(BASE_PATH, 'gui_data', 'change_log.txt')
SPLASH_DOC = os.path.join(BASE_PATH, 'tmp', 'splash.txt')

file_check(os.path.join(MODELS_DIR, 'Main_Models'), VR_MODELS_DIR)
file_check(os.path.join(DEMUCS_MODELS_DIR, 'v3_repo'), DEMUCS_NEWER_REPO_DIR)
remove_unneeded_yamls(DEMUCS_MODELS_DIR)

remove_temps(ENSEMBLE_TEMP_PATH)
remove_temps(SAMPLE_CLIP_PATH)
remove_temps(os.path.join(BASE_PATH, 'img'))

if not os.path.isdir(ENSEMBLE_TEMP_PATH):
    os.mkdir(ENSEMBLE_TEMP_PATH)
    
if not os.path.isdir(SAMPLE_CLIP_PATH):
    os.mkdir(SAMPLE_CLIP_PATH)

model_hash_table = {}
data = load_data()

class ModelData():
    def __init__(self, model_name: str, 
                 selected_process_method=ENSEMBLE_MODE, 
                 is_secondary_model=False, 
                 primary_model_primary_stem=None, 
                 is_primary_model_primary_stem_only=False, 
                 is_primary_model_secondary_stem_only=False, 
                 is_pre_proc_model=False,
                 is_dry_check=False):

        self.is_gpu_conversion = 0 if root.is_gpu_conversion_var else -1
        self.is_normalization = root.is_normalization_var
        self.is_primary_stem_only = root.is_primary_stem_only_var
        self.is_secondary_stem_only = root.is_secondary_stem_only_var
        self.is_denoise = root.is_denoise_var
        self.mdx_batch_size =  1 if root.mdx_batch_size_var == DEF_OPT else int(root.mdx_batch_size_var)
        self.is_mdx_ckpt = False
        self.wav_type_set = root.wav_type_set
        self.mp3_bit_set =  root.mp3_bit_set_var
        self.save_format = root.save_format_var
        self.is_invert_spec = root.is_invert_spec_var
        self.is_mixer_mode = root.is_mixer_mode_var
        self.demucs_stems = root.demucs_stems_var
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name
        self.process_method = selected_process_method
        self.model_status = False if self.model_name == CHOOSE_MODEL or self.model_name == NO_MODEL else True
        self.primary_stem = None
        self.secondary_stem = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = is_secondary_model
        self.secondary_model = None
        self.secondary_model_scale = None
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = is_pre_proc_model
        self.is_dry_check = is_dry_check
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.manual_download_Button = None
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None

        if selected_process_method == ENSEMBLE_MODE:
            partitioned_name = model_name.partition(ENSEMBLE_PARTITION)
            self.process_method = partitioned_name[0]
            self.model_name = partitioned_name[2]
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem, self.ensemble_secondary_stem = root.return_ensemble_stems()
            self.is_ensemble_mode = True if not is_secondary_model and not is_pre_proc_model else False
            self.is_4_stem_ensemble = True if root.ensemble_main_stem_var == FOUR_STEM_ENSEMBLE and self.is_ensemble_mode else False
            self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var if not self.ensemble_primary_stem == VOCAL_STEM else False

        if self.process_method == VR_ARCH_TYPE:
            self.is_secondary_model_activated = root.vr_is_secondary_model_activate_var if not self.is_secondary_model else False
            self.aggression_setting = float(int(root.aggression_setting_var)/100)
            self.is_tta = root.is_tta_var
            self.is_post_process = root.is_post_process_var
            self.window_size = int(root.window_size_var)
            self.batch_size = 1 if root.batch_size_var == DEF_OPT else int(root.batch_size_var)
            self.crop_size = int(root.crop_size_var)
            self.is_high_end_process = 'mirroring' if root.is_high_end_process_var else 'None'
            self.post_process_threshold = float(root.post_process_threshold_var)
            self.model_capacity = 32, 128
            self.model_path = os.path.join(VR_MODELS_DIR, f"{self.model_name}.pth")
            self.get_model_hash()
            if self.model_hash:
                self.model_data = self.get_model_data(VR_HASH_DIR, root.vr_hash_MAPPER) if not self.model_hash == WOOD_INST_MODEL_HASH else WOOD_INST_PARAMS
                if self.model_data:
                    vr_model_param = os.path.join(VR_PARAM_DIR, "{}.json".format(self.model_data["vr_model_param"]))
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
                    self.vr_model_param = ModelParameters(vr_model_param)
                    self.model_samplerate = self.vr_model_param.param['sr']
                    if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
                        self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
                        self.is_vr_51_model = True
                else:
                    self.model_status = False
                
        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = root.mdx_is_secondary_model_activate_var if not is_secondary_model else False
            self.margin = int(root.margin_var)
            self.chunks = root.determine_auto_chunks(root.chunks_var, self.is_gpu_conversion) if root.is_chunk_mdxnet_var else 0
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_data = self.get_model_data(MDX_HASH_DIR, root.mdx_hash_MAPPER)
                if self.model_data:
                    self.compensate = self.model_data["compensate"] if root.compensate_var == AUTO_SELECT else float(root.compensate_var)
                    self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                    self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                    self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
                else:
                    self.model_status = False

        if self.process_method == DEMUCS_ARCH_TYPE:
            self.is_secondary_model_activated = root.demucs_is_secondary_model_activate_var if not is_secondary_model else False
            if not self.is_ensemble_mode:
                self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var if not root.demucs_stems_var in [VOCAL_STEM, INST_STEM] else False
            self.overlap = float(root.overlap_var)
            self.margin_demucs = int(root.margin_demucs_var)
            self.chunks_demucs = root.determine_auto_chunks(root.chunks_demucs_var, self.is_gpu_conversion)
            self.shifts = int(root.shifts_var)
            self.is_split_mode = root.is_split_mode_var
            self.segment = root.segment_var
            self.is_chunk_demucs = root.is_chunk_demucs_var
            self.is_demucs_combine_stems = root.is_demucs_combine_stems_var
            self.is_primary_stem_only = root.is_primary_stem_only_var if self.is_ensemble_mode else root.is_primary_stem_only_Demucs_var 
            self.is_secondary_stem_only = root.is_secondary_stem_only_var if self.is_ensemble_mode else root.is_secondary_stem_only_Demucs_var
            self.get_demucs_model_path()
            self.get_demucs_model_data()

        self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0] if self.model_status else None
        self.pre_proc_model_activated = self.pre_proc_model_activated if not self.is_secondary_model else False
        
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        if self.is_secondary_model_activated and self.model_status:
            if (not self.is_ensemble_mode and root.demucs_stems_var == ALL_STEMS and self.process_method == DEMUCS_ARCH_TYPE) or self.is_4_stem_ensemble:
                for key in DEMUCS_4_SOURCE_LIST:
                    self.secondary_model_data(key)
                    self.secondary_model_4_stem.append(self.secondary_model)
                    self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                    self.secondary_model_4_stem_names.append(key)
                self.demucs_4_stem_added_count = sum(i is not None for i in self.secondary_model_4_stem)
                self.is_secondary_model_activated = False if all(i is None for i in self.secondary_model_4_stem) else True
                self.demucs_4_stem_added_count = self.demucs_4_stem_added_count - 1 if self.is_secondary_model_activated else self.demucs_4_stem_added_count
                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [None if i is None else i.model_basename for i in self.secondary_model_4_stem]
                    self.is_demucs_4_stem_secondaries = True 
            else:
                primary_stem = self.ensemble_primary_stem if self.is_ensemble_mode and self.process_method == DEMUCS_ARCH_TYPE else self.primary_stem
                self.secondary_model_data(primary_stem)
                
        if self.process_method == DEMUCS_ARCH_TYPE and not is_secondary_model:
            if self.demucs_stem_count >= 3 and self.pre_proc_model_activated:
                self.pre_proc_model_activated = True
                self.pre_proc_model = root.process_determine_demucs_pre_proc_model(self.primary_stem)
                self.is_demucs_pre_proc_model_inst_mix = root.is_demucs_pre_proc_model_inst_mix_var if self.pre_proc_model else False

    def secondary_model_data(self, primary_stem):
        secondary_model_data = root.process_determine_secondary_model(self.process_method, primary_stem, self.is_primary_stem_only, self.is_secondary_stem_only)
        self.secondary_model = secondary_model_data[0]
        self.secondary_model_scale = secondary_model_data[1]
        self.is_secondary_model_activated = False if not self.secondary_model else True
        if self.secondary_model:
            self.is_secondary_model_activated = False if self.secondary_model.model_basename == self.model_basename else True
              
    def get_mdx_model_path(self):
        
        if self.model_name.endswith(CKPT):
            # self.chunks = 0
            # self.is_mdx_batch_mode = True
            self.is_mdx_ckpt = True
            
        ext = '' if self.is_mdx_ckpt else ONNX
        
        for file_name, chosen_mdx_model in root.mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")
            
        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")
    
    def get_demucs_model_path(self):
        
        demucs_newer = [True for x in DEMUCS_NEWER_TAGS if x in self.model_name]
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR
        
        for file_name, chosen_model in root.demucs_name_select_MAPPER.items():
            if self.model_name in chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(DEMUCS_NEWER_REPO_DIR, f'{self.model_name}.yaml')

    def get_demucs_model_data(self):

        self.demucs_version = DEMUCS_V4

        for key, value in DEMUCS_VERSION_MAPPER.items():
            if value in self.model_name:
                self.demucs_version = key

        self.demucs_source_list = DEMUCS_2_SOURCE if DEMUCS_UVR_MODEL in self.model_name else DEMUCS_4_SOURCE
        self.demucs_source_map = DEMUCS_2_SOURCE_MAPPER if DEMUCS_UVR_MODEL in self.model_name else DEMUCS_4_SOURCE_MAPPER
        self.demucs_stem_count = 2 if DEMUCS_UVR_MODEL in self.model_name else 4
        
        if not self.is_ensemble_mode:
            self.primary_stem = PRIMARY_STEM if self.demucs_stems == ALL_STEMS else self.demucs_stems
            self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]

    def get_model_data(self, model_hash_dir, hash_mapper):

        model_settings_json = os.path.join(model_hash_dir, "{}.json".format(self.model_hash))

        if os.path.isfile(model_settings_json):
            return json.load(open(model_settings_json))
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings
                else:
                    return self.get_model_data_from_popup()

    def get_model_data_from_popup(self):
            return None

    def get_model_hash(self):
        self.model_hash = None
        
        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if model_hash_table:
                for (key, value) in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break
                    
            if not self.model_hash:
                try:
                    with open(self.model_path, 'rb') as f:
                        f.seek(- 10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(open(self.model_path,'rb').read()).hexdigest()
                    
                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)
                
class UVR():
    def __init__(self):
        # Initialize an instance variable
        super().__init__()
        
        #Placeholders
        self.error_log_var = ""
        self.vr_secondary_model_names = []
        self.mdx_secondary_model_names = []
        self.demucs_secondary_model_names = []
        self.vr_primary_model_names = []
        self.mdx_primary_model_names = []
        self.demucs_primary_model_names = []
        
        self.vr_cache_source_mapper = {}
        self.mdx_cache_source_mapper = {}
        self.demucs_cache_source_mapper = {}
        
        # -Tkinter Value Holders-
        
        try:
            self.load_saved_vars(data)
        except Exception as e:
            # self.error_log_var.set(error_text('Loading Saved Variables', e))
            self.load_saved_vars(DEFAULT_DATA)
            
        self.cached_sources_clear()
        
        self.method_mapper = {
            VR_ARCH_PM: self.vr_model_var,
            MDX_ARCH_TYPE: self.mdx_net_model_var,
            DEMUCS_ARCH_TYPE: self.demucs_model_var}

        self.vr_secondary_model_vars = {'voc_inst_secondary_model': self.vr_voc_inst_secondary_model_var,
                                        'other_secondary_model': self.vr_other_secondary_model_var,
                                        'bass_secondary_model': self.vr_bass_secondary_model_var,
                                        'drums_secondary_model': self.vr_drums_secondary_model_var,
                                        'is_secondary_model_activate': self.vr_is_secondary_model_activate_var,
                                        'voc_inst_secondary_model_scale': self.vr_voc_inst_secondary_model_scale_var,
                          'other_secondary_model_scale': self.vr_other_secondary_model_scale_var,
                          'bass_secondary_model_scale': self.vr_bass_secondary_model_scale_var,
                          'drums_secondary_model_scale': self.vr_drums_secondary_model_scale_var}
        
        self.demucs_secondary_model_vars = {'voc_inst_secondary_model': self.demucs_voc_inst_secondary_model_var,
                                        'other_secondary_model': self.demucs_other_secondary_model_var,
                                        'bass_secondary_model': self.demucs_bass_secondary_model_var,
                                        'drums_secondary_model': self.demucs_drums_secondary_model_var,
                                        'is_secondary_model_activate': self.demucs_is_secondary_model_activate_var,
                                        'voc_inst_secondary_model_scale': self.demucs_voc_inst_secondary_model_scale_var,
                          'other_secondary_model_scale': self.demucs_other_secondary_model_scale_var,
                          'bass_secondary_model_scale': self.demucs_bass_secondary_model_scale_var,
                          'drums_secondary_model_scale': self.demucs_drums_secondary_model_scale_var}
        
        self.mdx_secondary_model_vars = {'voc_inst_secondary_model': self.mdx_voc_inst_secondary_model_var,
                                        'other_secondary_model': self.mdx_other_secondary_model_var,
                                        'bass_secondary_model': self.mdx_bass_secondary_model_var,
                                        'drums_secondary_model': self.mdx_drums_secondary_model_var,
                                        'is_secondary_model_activate': self.mdx_is_secondary_model_activate_var,
                                        'voc_inst_secondary_model_scale': self.mdx_voc_inst_secondary_model_scale_var,
                          'other_secondary_model_scale': self.mdx_other_secondary_model_scale_var,
                          'bass_secondary_model_scale': self.mdx_bass_secondary_model_scale_var,
                          'drums_secondary_model_scale': self.mdx_drums_secondary_model_scale_var}

        #Main Application Vars
        self.progress_bar_main_var = 0
        self.inputPathsEntry_var = ""
        self.conversion_Button_Text_var = START_PROCESSING
        self.chosen_ensemble_var = CHOOSE_ENSEMBLE_OPTION
        self.ensemble_main_stem_var = CHOOSE_STEM_PAIR
        self.ensemble_type_var = MAX_MIN
        self.save_current_settings_var = SELECT_SAVED_SET
        self.demucs_stems_var = ALL_STEMS
        self.is_primary_stem_only_Text_var = ""
        self.is_secondary_stem_only_Text_var = ""
        self.is_primary_stem_only_Demucs_Text_var = ""
        self.is_secondary_stem_only_Demucs_Text_var = ""
        self.scaling_var = 1.0
        self.active_processing_thread = None
        self.verification_thread = None
        self.is_menu_settings_open = False
        
        self.is_open_menu_advanced_vr_options = False
        self.is_open_menu_advanced_demucs_options = False
        self.is_open_menu_advanced_mdx_options = False
        self.is_open_menu_advanced_ensemble_options = False
        self.is_open_menu_view_inputs = False
        self.is_open_menu_help = False
        self.is_open_menu_error_log = False

        self.mdx_model_params = None
        self.vr_model_params = None
        self.current_text_box = None
        self.wav_type_set = None
        self.is_online_model_menu = None
        self.progress_bar_var = 0
        self.is_confirm_error_var = False
        self.clear_cache_torch = False
        self.vr_hash_MAPPER = load_model_hash_data(VR_HASH_JSON)
        self.vr_name_select_MAPPER = load_model_hash_data(VR_MODEL_NAME_SELECT)
        self.mdx_hash_MAPPER = load_model_hash_data(MDX_HASH_JSON)
        self.mdx_name_select_MAPPER = load_model_hash_data(MDX_MODEL_NAME_SELECT)
        self.demucs_name_select_MAPPER = load_model_hash_data(DEMUCS_MODEL_NAME_SELECT)
        self.is_gpu_available = torch.cuda.is_available() if not OPERATING_SYSTEM == 'Darwin' else torch.backends.mps.is_available()
        self.is_process_stopped = False
        self.inputs_from_dir = []
        self.iteration = 0
        self.vr_primary_source = None
        self.vr_secondary_source = None
        self.mdx_primary_source = None
        self.mdx_secondary_source = None
        self.demucs_primary_source = None
        self.demucs_secondary_source = None

        #Download Center Vars
        self.online_data = {}
        self.is_online = False
        self.lastest_version = ''
        self.model_download_demucs_var = ""
        self.model_download_mdx_var = ""
        self.model_download_vr_var = ""
        self.selected_download_var = NO_MODEL
        self.select_download_var = ""
        self.download_progress_info_var = ""
        self.download_progress_percent_var = ""
        self.download_progress_bar_var = 0
        self.download_stop_var = "" 
        self.app_update_status_Text_var = ""
        self.app_update_button_Text_var = ""
        self.user_code_validation_var = ""
        self.download_link_path_var = "" 
        self.download_save_path_var = ""
        self.download_update_link_var = "" 
        self.download_update_path_var = "" 
        self.download_demucs_models_list = []
        self.download_demucs_newer_models = []
        self.refresh_list_Button = None
        self.stop_download_Button_DISABLE = None
        self.enable_tabs = None
        self.is_download_thread_active = False
        self.is_process_thread_active = False
        self.is_active_processing_thread = False
        self.active_download_thread = None

        #Model Update
        self.last_found_ensembles = ENSEMBLE_OPTIONS
        self.last_found_settings = ENSEMBLE_OPTIONS
        self.last_found_models = ()
        self.model_data_table = ()
        self.ensemble_model_list = ()

        self.chosen_process_method_var = MDX_ARCH_TYPE # MDX_ARCH_TYPE|VR_ARCH_TYPE|DEMUCS_ARCH_TYPE

    def cached_sources_clear(self):

        self.vr_cache_source_mapper = {}
        self.mdx_cache_source_mapper = {}
        self.demucs_cache_source_mapper = {}

    def cached_model_source_holder(self, process_method, sources, model_name=None):
        
        if process_method == VR_ARCH_TYPE:
            self.vr_cache_source_mapper = {**self.vr_cache_source_mapper, **{model_name: sources}}
        if process_method == MDX_ARCH_TYPE:
            self.mdx_cache_source_mapper = {**self.mdx_cache_source_mapper, **{model_name: sources}}
        if process_method == DEMUCS_ARCH_TYPE:
            self.demucs_cache_source_mapper = {**self.demucs_cache_source_mapper, **{model_name: sources}}
                         
    def cached_source_callback(self, process_method, model_name=None):
        
        model, sources = None, None
        
        if process_method == VR_ARCH_TYPE:
            mapper = self.vr_cache_source_mapper
        if process_method == MDX_ARCH_TYPE:
            mapper = self.mdx_cache_source_mapper
        if process_method == DEMUCS_ARCH_TYPE:
            mapper = self.demucs_cache_source_mapper
        
        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value
        
        return model, sources
                
    def cached_source_model_list_check(self, model_list: list[ModelData]):

        model: ModelData
        primary_model_names = lambda process_method:[model.model_basename if model.process_method == process_method else None for model in model_list]
        secondary_model_names = lambda process_method:[model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == process_method else None for model in model_list]

        self.vr_primary_model_names = primary_model_names(VR_ARCH_TYPE)
        self.mdx_primary_model_names = primary_model_names(MDX_ARCH_TYPE)
        self.demucs_primary_model_names = primary_model_names(DEMUCS_ARCH_TYPE)
        self.vr_secondary_model_names = secondary_model_names(VR_ARCH_TYPE)
        self.mdx_secondary_model_names = secondary_model_names(MDX_ARCH_TYPE)
        self.demucs_secondary_model_names = [model.secondary_model.model_basename if model.is_secondary_model_activated and model.process_method == DEMUCS_ARCH_TYPE and not model.secondary_model is None else None for model in model_list]
        self.demucs_pre_proc_model_name = [model.pre_proc_model.model_basename if model.pre_proc_model else None for model in model_list]#list(dict.fromkeys())
        
        for model in model_list:
            if model.process_method == DEMUCS_ARCH_TYPE and model.is_demucs_4_stem_secondaries:
                if not model.is_4_stem_ensemble:
                    self.demucs_secondary_model_names = model.secondary_model_4_stem_model_names_list
                    break
                else:
                    for i in model.secondary_model_4_stem_model_names_list:
                        self.demucs_secondary_model_names.append(i)
        
        self.all_models = self.vr_primary_model_names + self.mdx_primary_model_names + self.demucs_primary_model_names + self.vr_secondary_model_names + self.mdx_secondary_model_names + self.demucs_secondary_model_names + self.demucs_pre_proc_model_name

    def determine_auto_chunks(self, chunks, gpu):
        """Determines appropriate chunk size based on user computer specs"""
        
        if OPERATING_SYSTEM == 'Darwin':
            gpu = -1

        if chunks == BATCH_MODE:
            chunks = 0
            #self.chunks_var.set(AUTO_SELECT)

        if chunks == 'Full':
            chunk_set = 0
        elif chunks == 'Auto':
            if gpu == 0:
                gpu_mem = round(torch.cuda.get_device_properties(0).total_memory/1.074e+9)
                if gpu_mem <= int(6):
                    chunk_set = int(5)
                if gpu_mem in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
                    chunk_set = int(10)
                if gpu_mem >= int(16):
                    chunk_set = int(40)
            if gpu == -1:
                sys_mem = psutil.virtual_memory().total >> 30
                if sys_mem <= int(4):
                    chunk_set = int(1)
                if sys_mem in [5, 6, 7, 8]:
                    chunk_set = int(10)
                if sys_mem in [9, 10, 11, 12, 13, 14, 15, 16]:
                    chunk_set = int(25)
                if sys_mem >= int(17):
                    chunk_set = int(60) 
        elif chunks == '0':
            chunk_set = 0
        else:
            chunk_set = int(chunks)
                 
        return chunk_set
      
    def assemble_model_data(self, model=None, arch_type=ENSEMBLE_MODE, is_dry_check=False):
        
        if arch_type == ENSEMBLE_STEM_CHECK:
            
            model_data = self.model_data_table
            missing_models = [model.model_status for model in model_data if not model.model_status]
            
            if missing_models or not model_data:
                model_data: List[ModelData] = [ModelData(model_name, is_dry_check=is_dry_check) for model_name in self.ensemble_model_list]
                self.model_data_table = model_data

        if arch_type == ENSEMBLE_MODE:
            model_data: List[ModelData] = [ModelData(model_name) for model_name in self.ensemble_listbox_get_all_selected_models()]
        if arch_type == ENSEMBLE_CHECK:
            model_data: List[ModelData] = [ModelData(model)]
        if arch_type == VR_ARCH_TYPE or arch_type == VR_ARCH_PM:
            model_data: List[ModelData] = [ModelData(model, VR_ARCH_TYPE)]
        if arch_type == MDX_ARCH_TYPE:
            model_data: List[ModelData] = [ModelData(model, MDX_ARCH_TYPE)]
        if arch_type == DEMUCS_ARCH_TYPE:
            model_data: List[ModelData] = [ModelData(model, DEMUCS_ARCH_TYPE)]#
        return model_data
 
    def load_saved_vars(self, data):
        """Initializes primary Tkinter vars"""
        
        for key, value in DEFAULT_DATA.items():
            if not key in data.keys():
                data = {**data, **{key:value}}
                data['batch_size'] = DEF_OPT

        ## ADD_BUTTON
        self.chosen_process_method_var = data['chosen_process_method']
        
        #VR Architecture Vars
        self.vr_model_var = data['vr_model']
        self.aggression_setting_var = data['aggression_setting']
        self.window_size_var = data['window_size']
        self.batch_size_var = data['batch_size']
        self.crop_size_var = data['crop_size']
        self.is_tta_var = data['is_tta']
        self.is_output_image_var = data['is_output_image']
        self.is_post_process_var = data['is_post_process']
        self.is_high_end_process_var = data['is_high_end_process']
        self.post_process_threshold_var = data['post_process_threshold']
        self.vr_voc_inst_secondary_model_var = data['vr_voc_inst_secondary_model']
        self.vr_other_secondary_model_var = data['vr_other_secondary_model']
        self.vr_bass_secondary_model_var = data['vr_bass_secondary_model']
        self.vr_drums_secondary_model_var = data['vr_drums_secondary_model']
        self.vr_is_secondary_model_activate_var = data['vr_is_secondary_model_activate']
        self.vr_voc_inst_secondary_model_scale_var = data['vr_voc_inst_secondary_model_scale']
        self.vr_other_secondary_model_scale_var = data['vr_other_secondary_model_scale']
        self.vr_bass_secondary_model_scale_var = data['vr_bass_secondary_model_scale']
        self.vr_drums_secondary_model_scale_var = data['vr_drums_secondary_model_scale']

        #Demucs Vars
        self.demucs_model_var = data['demucs_model']
        self.segment_var = data['segment']
        self.overlap_var = data['overlap']
        self.shifts_var = data['shifts']
        self.chunks_demucs_var = data['chunks_demucs']
        self.margin_demucs_var = data['margin_demucs']
        self.is_chunk_demucs_var = data['is_chunk_demucs']
        self.is_chunk_mdxnet_var = data['is_chunk_mdxnet']
        self.is_primary_stem_only_Demucs_var = data['is_primary_stem_only_Demucs']
        self.is_secondary_stem_only_Demucs_var = data['is_secondary_stem_only_Demucs']
        self.is_split_mode_var = data['is_split_mode']
        self.is_demucs_combine_stems_var = data['is_demucs_combine_stems']
        self.demucs_voc_inst_secondary_model_var = data['demucs_voc_inst_secondary_model']
        self.demucs_other_secondary_model_var = data['demucs_other_secondary_model']
        self.demucs_bass_secondary_model_var = data['demucs_bass_secondary_model']
        self.demucs_drums_secondary_model_var = data['demucs_drums_secondary_model']
        self.demucs_is_secondary_model_activate_var = data['demucs_is_secondary_model_activate']
        self.demucs_voc_inst_secondary_model_scale_var = data['demucs_voc_inst_secondary_model_scale']
        self.demucs_other_secondary_model_scale_var = data['demucs_other_secondary_model_scale']
        self.demucs_bass_secondary_model_scale_var = data['demucs_bass_secondary_model_scale']
        self.demucs_drums_secondary_model_scale_var = data['demucs_drums_secondary_model_scale']
        self.demucs_pre_proc_model_var = data['demucs_pre_proc_model']
        self.is_demucs_pre_proc_model_activate_var = data['is_demucs_pre_proc_model_activate']
        self.is_demucs_pre_proc_model_inst_mix_var = data['is_demucs_pre_proc_model_inst_mix']
      
        #MDX-Net Vars
        self.mdx_net_model_var = data['mdx_net_model']
        self.chunks_var = data['chunks']
        self.margin_var = data['margin']
        self.compensate_var = data['compensate']
        self.is_denoise_var = data['is_denoise']
        self.is_invert_spec_var = data['is_invert_spec']
        self.is_mixer_mode_var = data['is_mixer_mode']
        self.mdx_batch_size_var = data['mdx_batch_size']
        self.mdx_voc_inst_secondary_model_var = data['mdx_voc_inst_secondary_model']
        self.mdx_other_secondary_model_var = data['mdx_other_secondary_model']
        self.mdx_bass_secondary_model_var = data['mdx_bass_secondary_model']
        self.mdx_drums_secondary_model_var = data['mdx_drums_secondary_model']
        self.mdx_is_secondary_model_activate_var = data['mdx_is_secondary_model_activate']
        self.mdx_voc_inst_secondary_model_scale_var = data['mdx_voc_inst_secondary_model_scale']
        self.mdx_other_secondary_model_scale_var = data['mdx_other_secondary_model_scale']
        self.mdx_bass_secondary_model_scale_var = data['mdx_bass_secondary_model_scale']
        self.mdx_drums_secondary_model_scale_var = data['mdx_drums_secondary_model_scale']
    
        #Ensemble Vars
        self.is_save_all_outputs_ensemble_var = data['is_save_all_outputs_ensemble']
        self.is_append_ensemble_name_var = data['is_append_ensemble_name']

        #Audio Tool Vars
        self.chosen_audio_tool_var = data['chosen_audio_tool']
        self.choose_algorithm_var = data['choose_algorithm']
        self.time_stretch_rate_var = data['time_stretch_rate']
        self.pitch_rate_var = data['pitch_rate']

        #Shared Vars
        self.mp3_bit_set_var = data['mp3_bit_set']
        self.save_format_var = data['save_format']
        self.wav_type_set_var = data['wav_type_set']
        self.user_code_var = data['user_code']
        self.is_gpu_conversion_var = data['is_gpu_conversion']
        self.is_primary_stem_only_var = data['is_primary_stem_only']
        self.is_secondary_stem_only_var = data['is_secondary_stem_only']
        self.is_testing_audio_var = data['is_testing_audio']
        self.is_add_model_name_var = data['is_add_model_name']
        self.is_accept_any_input_var = data['is_accept_any_input']
        self.is_task_complete_var = data['is_task_complete']
        self.is_normalization_var = data['is_normalization']
        self.is_create_model_folder_var = data['is_create_model_folder']
        self.help_hints_var = data['help_hints_var']
        self.model_sample_mode_var = data['model_sample_mode']
        self.model_sample_mode_duration_var = data['model_sample_mode_duration']
        self.model_sample_mode_duration_checkbox_var = SAMPLE_MODE_CHECKBOX(self.model_sample_mode_duration_var)
        
        #Path Vars
        self.export_path_var = data['export_path']
        self.inputPaths = data['input_paths']
        self.lastDir = data['lastDir']

    def load_saved_settings(self, loaded_setting: dict, process_method=None):
        """Loads user saved application settings or resets to default"""
        
        for key, value in DEFAULT_DATA.items():
            if not key in loaded_setting.keys():
                loaded_setting = {**loaded_setting, **{key:value}}
                loaded_setting['batch_size'] = DEF_OPT
        
        is_ensemble = True if process_method == ENSEMBLE_MODE else False
        
        if not process_method or process_method == VR_ARCH_PM or is_ensemble:
            self.vr_model_var = loaded_setting['vr_model']
            self.aggression_setting_var = loaded_setting['aggression_setting']
            self.window_size_var = loaded_setting['window_size']
            self.batch_size_var = loaded_setting['batch_size']
            self.crop_size_var = loaded_setting['crop_size']
            self.is_tta_var = loaded_setting['is_tta']
            self.is_output_image_var = loaded_setting['is_output_image']
            self.is_post_process_var = loaded_setting['is_post_process']
            self.is_high_end_process_var = loaded_setting['is_high_end_process']
            self.post_process_threshold_var = loaded_setting['post_process_threshold']
            self.vr_voc_inst_secondary_model_var = loaded_setting['vr_voc_inst_secondary_model']
            self.vr_other_secondary_model_var = loaded_setting['vr_other_secondary_model']
            self.vr_bass_secondary_model_var = loaded_setting['vr_bass_secondary_model']
            self.vr_drums_secondary_model_var = loaded_setting['vr_drums_secondary_model']
            self.vr_is_secondary_model_activate_var = loaded_setting['vr_is_secondary_model_activate']
            self.vr_voc_inst_secondary_model_scale_var = loaded_setting['vr_voc_inst_secondary_model_scale']
            self.vr_other_secondary_model_scale_var = loaded_setting['vr_other_secondary_model_scale']
            self.vr_bass_secondary_model_scale_var = loaded_setting['vr_bass_secondary_model_scale']
            self.vr_drums_secondary_model_scale_var = loaded_setting['vr_drums_secondary_model_scale']
        
        if not process_method or process_method == DEMUCS_ARCH_TYPE or is_ensemble:
            self.demucs_model_var = loaded_setting['demucs_model']
            self.segment_var = loaded_setting['segment']
            self.overlap_var = loaded_setting['overlap']
            self.shifts_var = loaded_setting['shifts']
            self.chunks_demucs_var = loaded_setting['chunks_demucs']
            self.margin_demucs_var = loaded_setting['margin_demucs']
            self.is_chunk_demucs_var = loaded_setting['is_chunk_demucs']
            self.is_chunk_mdxnet_var = loaded_setting['is_chunk_mdxnet']
            self.is_primary_stem_only_Demucs_var = loaded_setting['is_primary_stem_only_Demucs']
            self.is_secondary_stem_only_Demucs_var = loaded_setting['is_secondary_stem_only_Demucs']
            self.is_split_mode_var = loaded_setting['is_split_mode']
            self.is_demucs_combine_stems_var = loaded_setting['is_demucs_combine_stems']
            self.demucs_voc_inst_secondary_model_var = loaded_setting['demucs_voc_inst_secondary_model']
            self.demucs_other_secondary_model_var = loaded_setting['demucs_other_secondary_model']
            self.demucs_bass_secondary_model_var = loaded_setting['demucs_bass_secondary_model']
            self.demucs_drums_secondary_model_var = loaded_setting['demucs_drums_secondary_model']
            self.demucs_is_secondary_model_activate_var = loaded_setting['demucs_is_secondary_model_activate']
            self.demucs_voc_inst_secondary_model_scale_var = loaded_setting['demucs_voc_inst_secondary_model_scale']
            self.demucs_other_secondary_model_scale_var = loaded_setting['demucs_other_secondary_model_scale']
            self.demucs_bass_secondary_model_scale_var = loaded_setting['demucs_bass_secondary_model_scale']
            self.demucs_drums_secondary_model_scale_var = loaded_setting['demucs_drums_secondary_model_scale']
            self.demucs_stems_var = loaded_setting['demucs_stems']
            self.update_stem_checkbox_labels(self.demucs_stems_var, demucs=True)
            self.demucs_pre_proc_model_var = data['demucs_pre_proc_model']
            self.is_demucs_pre_proc_model_activate_var = data['is_demucs_pre_proc_model_activate']
            self.is_demucs_pre_proc_model_inst_mix_var = data['is_demucs_pre_proc_model_inst_mix']
        
        if not process_method or process_method == MDX_ARCH_TYPE or is_ensemble:
            self.mdx_net_model_var = loaded_setting['mdx_net_model']
            self.chunks_var = loaded_setting['chunks']
            self.margin_var = loaded_setting['margin']
            self.compensate_var = loaded_setting['compensate']
            self.is_denoise_var = loaded_setting['is_denoise']
            self.is_invert_spec_var = loaded_setting['is_invert_spec']
            self.is_mixer_mode_var = loaded_setting['is_mixer_mode']
            self.mdx_batch_size_var = loaded_setting['mdx_batch_size']
            self.mdx_voc_inst_secondary_model_var = loaded_setting['mdx_voc_inst_secondary_model']
            self.mdx_other_secondary_model_var = loaded_setting['mdx_other_secondary_model']
            self.mdx_bass_secondary_model_var = loaded_setting['mdx_bass_secondary_model']
            self.mdx_drums_secondary_model_var = loaded_setting['mdx_drums_secondary_model']
            self.mdx_is_secondary_model_activate_var = loaded_setting['mdx_is_secondary_model_activate']
            self.mdx_voc_inst_secondary_model_scale_var = loaded_setting['mdx_voc_inst_secondary_model_scale']
            self.mdx_other_secondary_model_scale_var = loaded_setting['mdx_other_secondary_model_scale']
            self.mdx_bass_secondary_model_scale_var = loaded_setting['mdx_bass_secondary_model_scale']
            self.mdx_drums_secondary_model_scale_var = loaded_setting['mdx_drums_secondary_model_scale']
        
        if not process_method or is_ensemble:
            self.is_save_all_outputs_ensemble_var = loaded_setting['is_save_all_outputs_ensemble']
            self.is_append_ensemble_name_var = loaded_setting['is_append_ensemble_name']
            self.chosen_audio_tool_var = loaded_setting['chosen_audio_tool']
            self.choose_algorithm_var = loaded_setting['choose_algorithm']
            self.time_stretch_rate_var = loaded_setting['time_stretch_rate']
            self.pitch_rate_var = loaded_setting['pitch_rate']
            self.is_primary_stem_only_var = loaded_setting['is_primary_stem_only']
            self.is_secondary_stem_only_var = loaded_setting['is_secondary_stem_only']
            self.is_testing_audio_var = loaded_setting['is_testing_audio']
            self.is_add_model_name_var = loaded_setting['is_add_model_name']
            self.is_accept_any_input_var = loaded_setting["is_accept_any_input"]
            self.is_task_complete_var = loaded_setting['is_task_complete']
            self.is_create_model_folder_var = loaded_setting['is_create_model_folder']
            self.mp3_bit_set_var = loaded_setting['mp3_bit_set']
            self.save_format_var = loaded_setting['save_format']
            self.wav_type_set_var = loaded_setting['wav_type_set']
            self.user_code_var = loaded_setting['user_code']
            
        self.is_gpu_conversion_var = loaded_setting['is_gpu_conversion']
        self.is_normalization_var = loaded_setting['is_normalization']
        self.help_hints_var = loaded_setting['help_hints_var']
        
        self.model_sample_mode_var = loaded_setting['model_sample_mode']
        self.model_sample_mode_duration_var = loaded_setting['model_sample_mode_duration']
        self.model_sample_mode_duration_checkbox_var = SAMPLE_MODE_CHECKBOX(self.model_sample_mode_duration_var)

    def verify_audio(self, audio_file, is_process=True, sample_path=None):
        is_good = False
        error_data = ''
        
        if os.path.isfile(audio_file):
            try:
                librosa.load(audio_file, duration=3, mono=False, sr=44100) if not type(sample_path) is str else self.create_sample(audio_file, sample_path)
                is_good = True
            except Exception as e:
                error_name = f'{type(e).__name__}'
                traceback_text = ''.join(traceback.format_tb(e.__traceback__))
                message = f'{error_name}: "{e}"\n{traceback_text}"'
                if is_process:
                    audio_base_name = os.path.basename(audio_file)
                    # self.error_log_var.set(f'Error Loading the Following File:\n\n\"{audio_base_name}\"\n\nRaw Error Details:\n\n{message}')
                else:
                    error_data = AUDIO_VERIFICATION_CHECK(audio_file, message)

        if is_process:
            return is_good
        else:
            return is_good, error_data
                
    def process_iteration(self):
        self.iteration = self.iteration + 1
        
    def process_get_baseText(self, total_files, file_num):
        """Create the base text for the command widget"""    
        text = 'File {file_num}/{total_files} '.format(file_num=file_num,
                                                    total_files=total_files) 
        return text     

    def process_update_progress(self, model_count, total_files, step: float = 1):
        """Calculate the progress for the progress widget in the GUI"""
        
        total_count = model_count * total_files
        base = (100 / total_count)
        progress = base * self.iteration - base
        progress += base * step

        self.progress_bar_main_var = progress
        
        # self.conversion_Button_Text_var.set(f'Process Progress: {int(progress)}%')
                      
    def process_start(self, inputPaths, uvr_method, choosen_model, progress=gr.Progress()):
        """Start the conversion for all the given mp3 and wav files"""
        
        print("process_start::")
        final_output = []
        self.chosen_process_method_var = uvr_method
        self.vr_model_var = choosen_model if uvr_method == VR_ARCH_TYPE else None
        self.mdx_net_model_var = choosen_model if uvr_method == MDX_ARCH_TYPE else None
        self.demucs_model_var = choosen_model if uvr_method == DEMUCS_ARCH_TYPE else None
        
        stime = time.perf_counter()
        time_elapsed = lambda:f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}'
        is_ensemble = False
        true_model_count = 0
        iteration = 0
        is_verified_audio = True
        inputPath_total_len = len(inputPaths)
        is_model_sample_mode = False
        
        try:
            if self.chosen_process_method_var == VR_ARCH_TYPE:
                model = self.assemble_model_data(self.vr_model_var, VR_ARCH_TYPE)
            if self.chosen_process_method_var == MDX_ARCH_TYPE:
                model = self.assemble_model_data(self.mdx_net_model_var, MDX_ARCH_TYPE)
            if self.chosen_process_method_var == DEMUCS_ARCH_TYPE:
                model = self.assemble_model_data(self.demucs_model_var, DEMUCS_ARCH_TYPE)

            self.cached_source_model_list_check(model)
            
            true_model_4_stem_count = sum(m.demucs_4_stem_added_count if m.process_method == DEMUCS_ARCH_TYPE else 0 for m in model)
            true_model_pre_proc_model_count = sum(2 if m.pre_proc_model_activated else 0 for m in model)
            true_model_count = sum(2 if m.is_secondary_model_activated else 1 for m in model) + true_model_4_stem_count + true_model_pre_proc_model_count

            for file_num, media_file in enumerate(inputPaths, start=1):
                export_path = os.path.join(temp_dir, new_dir_now())
                Path(export_path).mkdir(parents=True, exist_ok=True)
                is_audio = True if self.is_video_or_audio(media_file) == 'audio' else False
                audio_file = ''
                video_file = ''
                if is_audio:
                  audio_file = media_file
                else:
                  video_file = media_file
                  audio_file = f'{os.path.splitext(media_file)[0]}.wav'
                  for i in range (120):
                      time.sleep(1)
                      print('process media...')
                      if os.path.exists(video_file):
                          time.sleep(1)
                          os.system(f"ffmpeg -y -i '{video_file}' -vn -acodec pcm_s16le -ar 44100 -ac 2 '{audio_file}'")
                          time.sleep(1)
                          break
                      if i == 119:
                        print('Error processing media')
                        return
                  for i in range (120):
                      time.sleep(1)
                      print('process audio...')
                      if os.path.exists(audio_file):
                          break
                      if i == 119:
                        print("Error can't create the audio")
                        return
                             
                self.cached_sources_clear()
                base_text = self.process_get_baseText(total_files=inputPath_total_len, file_num=file_num)

                if self.verify_audio(audio_file):
                    audio_file = self.create_sample(audio_file) if is_model_sample_mode else audio_file
                    print(f'{NEW_LINE if not file_num ==1 else NO_LINE}{base_text}"{os.path.basename(audio_file)}\".{NEW_LINES}')
                    is_verified_audio = True
                else:
                    error_text_console = f'{base_text}"{os.path.basename(audio_file)}\" is missing or currupted.\n'
                    print(f'\n{error_text_console}') if inputPath_total_len >= 2 else None
                    iteration += true_model_count
                    is_verified_audio = False
                    continue

                for current_model_num, current_model in enumerate(model, start=1):
                    iteration += 1

                    model_name_text = f'({current_model.model_basename})' if not is_ensemble else ''
                    print(base_text + f'Loading model {model_name_text}...')

                    progress_kwargs = {'model_count': true_model_count,
                                        'total_files': inputPath_total_len}

                    set_progress_bar = lambda step, inference_iterations=0:self.process_update_progress(**progress_kwargs, step=(step + (inference_iterations)))
                    write_to_console = lambda progress_text, base_text=base_text:print(base_text + progress_text)

                    audio_file_base = f"{os.path.splitext(os.path.basename(audio_file))[0]}"
                    audio_file_base = audio_file_base if not self.is_testing_audio_var or is_ensemble else f"{round(time.time())}_{audio_file_base}"
                    audio_file_base = audio_file_base if not is_ensemble else f"{audio_file_base}_{current_model.model_basename}"
                    if not is_ensemble:
                        audio_file_base = audio_file_base if not self.is_add_model_name_var else f"{audio_file_base}_{current_model.model_basename}"

                    process_data = {
                                    'model_data': current_model, 
                                    'export_path': export_path,
                                    'audio_file_base': audio_file_base,
                                    'audio_file': audio_file,
                                    'set_progress_bar': set_progress_bar,
                                    'write_to_console': write_to_console,
                                    'process_iteration': self.process_iteration,
                                    'cached_source_callback': self.cached_source_callback,
                                    'cached_model_source_holder': self.cached_model_source_holder,
                                    'list_all_models': self.all_models,
                                    'is_ensemble_master': is_ensemble,
                                    'is_4_stem_ensemble': True if self.ensemble_main_stem_var == FOUR_STEM_ENSEMBLE and is_ensemble else False}
                    
                    if current_model.process_method == VR_ARCH_TYPE:
                        seperator = SeperateVR(current_model, process_data)
                    if current_model.process_method == MDX_ARCH_TYPE:
                        seperator = SeperateMDX(current_model, process_data)
                    if current_model.process_method == DEMUCS_ARCH_TYPE:
                        seperator = SeperateDemucs(current_model, process_data)
                    seperator.seperate()
                    
                ## merge video with converted audio
                audio_converted_file = os.path.join(export_path, f'{audio_file_base}_({INST_STEM}).wav')
                if not is_audio and os.path.exists(video_file):
                  media_output_file = os.path.join(export_path, os.path.basename(video_file))
                  os.system(f"ffmpeg -i '{video_file}' -i '{audio_converted_file}' -c:v copy -c:a aac -map 0:v -map 1:a -shortest '{media_output_file}'")
                else:
                  media_output_file = audio_converted_file
                  
                ## Copy to custom output directory if specify 
                COPY_OUTPUT_DIR = os.environ.get('COPY_OUTPUT_DIR', "")
                if COPY_OUTPUT_DIR != "": 
                  shutil.copy(media_output_file, os.path.join(COPY_OUTPUT_DIR, os.path.basename(media_output_file)))
                # Archive final output folder when done
                archive_path = os.path.join(Path(temp_dir).absolute(), os.path.splitext(os.path.basename(audio_file))[0])
                shutil.make_archive(archive_path, 'zip', export_path)   
                final_output.append(f"{archive_path}.zip")
                
                # Remove input file when done
                if is_model_sample_mode:
                    if os.path.isfile(audio_file):
                        os.remove(audio_file)
                
                # Release torch cache    
                torch.cuda.empty_cache()
                
            shutil.rmtree(export_path) if is_ensemble and len(os.listdir(export_path)) == 0 else None

            if inputPath_total_len == 1 and not is_verified_audio:
                print(f'{error_text_console}\n{PROCESS_FAILED}')
                print(time_elapsed())
            else:
                set_progress_bar(1.0)
                print('\nProcess Complete\n')
                print(time_elapsed())
                
            self.process_end()
        except Exception as e:
            print(f"{self.chosen_process_method_var}:: {e}")
            print(f'\n\n{PROCESS_FAILED}')
            print(time_elapsed())
            self.process_end(error=e)
        return final_output                
            
    def process_end(self, error=None):
        """End of process actions"""
        self.cached_sources_clear()
        torch.cuda.empty_cache()
        self.progress_bar_main_var = 0
        if error:
          print(f'{error_dialouge(error)}{ERROR_OCCURED[1]}')

    def is_video_or_audio(self, file_path):
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
      
    def youtube_download(self, url, output_path):
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

    def preprocess(self, media_inputs, link_inputs, uvr_method, uvr_model, progress=gr.Progress()):
      media_inputs = media_inputs if media_inputs is not None else []
      media_inputs = [media_input if isinstance(media_input, str) else media_input.name for media_input in media_inputs]
      youtube_temp_dir = os.path.join(temp_dir, 'youtube')
      Path(youtube_temp_dir).mkdir(parents=True, exist_ok=True)
      os.system(f"rm -rf {youtube_temp_dir}/*")
      link_inputs = link_inputs.split(',')
      print("link_inputs::", link_inputs)
      if link_inputs is not None and len(link_inputs) > 0 and link_inputs[0] != '':
        for url in link_inputs:
          url = url.strip()
          if url.startswith('https://www.youtube.com'):
            media_info =  ydl.extract_info(url, download=False)
            download_path = f"{os.path.join(youtube_temp_dir, media_info['title'])}.mp4"
            self.youtube_download(url, download_path)
            media_inputs.append(download_path) 
      print(media_inputs, link_inputs, uvr_method, uvr_model)
      if media_inputs is not None and len(media_inputs) > 0 and media_inputs[0] != '':
        output = root.process_start(media_inputs, uvr_method, uvr_model)
        return output
      else:
        raise gr.Error("Input not valid!!")
      
    def start_webui(self):
        title = "<center><strong><font size='7'>Ultimate Vocal Remover</font></strong></center>"
        description = """
        ### 🎥 **Ultimate Vocal Remover Tool** 📽️
        🎥 Upload a video or provide a video link. 📽️
        """
        theme = gr.themes.Base.load(os.path.join('themes','taithrah-minimal@0.0.1.json')).set(
            background_fill_primary ="#171717",
            panel_background_fill = "transparent"
        )
        with gr.Blocks(title="UVR",theme=theme) as demo:
            gr.Markdown(title)
            gr.Markdown(description)

        #### video
            with gr.Tab("Input Video|Audio for UVR"):
                with gr.Row():
                    with gr.Column():
                        #media_input = gr.UploadButton("Click to Upload a video", file_types=["video"], file_count="single") #gr.Video() # height=300,width=300
                        media_input = gr.Files(label="VIDEO|AUDIO", file_types=['audio','video'])
                        link_input = gr.Textbox(label="Youtube Link",info="Example: https://www.youtube.com/watch?v=-biOGdYiF-I,https://www.youtube.com/watch?v=-biOGdYiF-I", placeholder="URL goes here, seperate by comma...")        
                        gr.ClearButton(components=[media_input,link_input], size='sm')
                        with gr.Row():
                          uvr_type_option = [str(MDX_ARCH_TYPE),str(DEMUCS_ARCH_TYPE),str(VR_ARCH_TYPE)]
                          uvr_type = gr.Dropdown(choices=uvr_type_option, value=str(DEMUCS_ARCH_TYPE), label='AI Tech', info="Choose AI Tech for UVR", interactive=True)
                          uvr_model_option = [str(value) for key, value in self.demucs_name_select_MAPPER.items()]
                          uvr_model = gr.Dropdown(choices=uvr_model_option,value="v3 | UVR_Model_1",label='UVR Model', info="Choose UVR Model", interactive=True)
                          ## media_input change function
                          def update_model(uvr_type):
                            values = []
                            print("uvr_type::",uvr_type)
                            if uvr_type == MDX_ARCH_TYPE:
                              data = self.mdx_name_select_MAPPER
                            elif uvr_type ==  DEMUCS_ARCH_TYPE:
                              data = self.demucs_name_select_MAPPER
                            elif uvr_type ==  VR_ARCH_TYPE:
                              data = self.vr_name_select_MAPPER
                            else:
                              pass
                            for key, value in data.items():
                              values.append(str(value))
                            print("update model:",values)
                            return gr.update(choices=values, value=values[0])
                          uvr_type.change(update_model, uvr_type, [uvr_model], show_progress='full')

                    with gr.Column(variant='compact'):
                        with gr.Row():
                            media_button = gr.Button("CONVERT", )
                        with gr.Row():
                            media_output = gr.Files(label="DOWNLOAD CONVERTED VIDEO")
                        # gr.Examples(
                        #     examples=[
                        #         [
                        #             "",
                        #             "https://www.youtube.com/watch?v=-biOGdYiF-I",
                        #             "Demucs",
                        #             "v3 | UVR_Model_1",
                        #         ],
                        #     ],
                        #     fn=self.preprocess,
                        #     inputs=[
                        #       media_input,
                        #       link_input,
                        #       uvr_type,
                        #       uvr_model
                        #     ],
                        #     outputs=[media_output],
                        #     cache_examples=True,
                        # )

            # run
            media_button.click(self.preprocess, inputs=[
                media_input,
                link_input,
                uvr_type,
                uvr_model
                ], outputs=media_output)

    
        auth_user = os.getenv('AUTH_USER', '')
        auth_pass = os.getenv('AUTH_PASS', '')
        demo.launch(
          auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None,
          share=False,     
          server_name="0.0.0.0",
          server_port=6870,
          enable_queue=True,
          quiet=True, 
          debug=False)

if __name__ == "__main__":
  Path(temp_dir).mkdir(parents=True, exist_ok=True)
  os.system(f'rm -rf {temp_dir}/*')
  root = UVR()
  root.start_webui()