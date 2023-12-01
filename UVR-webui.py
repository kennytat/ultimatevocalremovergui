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
import gc
import re
import subprocess
import ass
# import soundfile as sf
import time
import torch
import traceback
# from gui_data.app_size_values import ImagePath, AdjustedValues as av
from gui_data.constants import *
from gui_data.error_handling import error_text, error_dialouge
from gui_data.old_data_check import file_check, remove_unneeded_yamls, remove_temps
from gui_data.tkinterdnd2 import TkinterDnD, DND_FILES
from lib_v5.vr_network.model_param_init import ModelParameters
from pathlib  import Path
from separate import SeperateDemucs, SeperateMDX, SeperateVR, save_format
import whisperx
from whisperx.utils import LANGUAGES as LANG_TRANSCRIPT
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH as DAMT, DEFAULT_ALIGN_MODELS_HF as DAMHF
from typing import List
from ml_collections import ConfigDict
import yaml
import sys
import tempfile
from datetime import datetime, timedelta

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('UVR BEGIN')

load_dotenv()
temp_dir = os.path.join(tempfile.gettempdir(), "ultimatevocalremover")
ydl = yt_dlp.YoutubeDL()

CUDA_MEM = int(torch.cuda.get_device_properties(0).total_memory)
print("CUDA_MEM::", CUDA_MEM)
if torch.cuda.is_available():
    device = "cuda"
    list_compute_type = ['float16', 'float32']
    compute_type_default = 'float16'
    whisper_model_default = 'large-v3' if CUDA_MEM > 9000000000 else 'medium'
else:
    device = "cpu"
    list_compute_type = ['float32']
    compute_type_default = 'float32'
    whisper_model_default = 'medium'
print('Working in: ', device)

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
DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')

#Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, 'model_data')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
VR_MODEL_NAME_SELECT = os.path.join(VR_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_data.json')
MDX_MODEL_NAME_SELECT = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_name_mapper.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')
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
    
LANGUAGES = {
    'Automatic detection': None,
    'Arabic (ar)': 'ar',
    'Cantonese (yue)': 'yue',
    'Chinese (zh)': 'zh',
    'Czech (cs)': 'cs',
    'Danish (da)': 'da',
    'Dutch (nl)': 'nl',
    'English (en)': 'en',
    'Finnish (fi)': 'fi',
    'French (fr)': 'fr',
    'German (de)': 'de',
    'Greek (el)': 'el',
    'Hebrew (he)': 'he',
    'Hungarian (hu)': 'hu',
    'Italian (it)': 'it',
    'Japanese (ja)': 'ja',
    'Korean (ko)': 'ko',
    'Persian (fa)': 'fa',
    'Polish (pl)': 'pl',
    'Portuguese (pt)': 'pt',
    'Russian (ru)': 'ru',
    'Spanish (es)': 'es',
    'Turkish (tr)': 'tr',
    'Ukrainian (uk)': 'uk',
    'Urdu (ur)': 'ur',
    'Vietnamese (vi)': 'vi',
    'Hindi (hi)': 'hi',
}

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
                 is_dry_check=False,
                 is_change_def=False,
                 is_get_hash_dir_only=False,
                 is_vocal_split_model=False):

        device_set = "Default"
        self.DENOISER_MODEL = DENOISER_MODEL_PATH
        self.DEVERBER_MODEL = DEVERBER_MODEL_PATH
        self.is_deverb_vocals = root.is_deverb_vocals_var if os.path.isfile(DEVERBER_MODEL_PATH) else False
        self.deverb_vocal_opt = DEVERB_MAPPER[root.deverb_vocal_opt_var]
        self.is_denoise_model = True if root.denoise_option_var == DENOISE_M and os.path.isfile(DENOISER_MODEL_PATH) else False
        self.is_gpu_conversion = 0 if root.is_gpu_conversion_var else -1
        self.is_normalization = root.is_normalization_var#
        self.is_use_opencl = False#True if is_opencl_only else root.is_use_opencl_var
        self.is_primary_stem_only = root.is_primary_stem_only_var
        self.is_secondary_stem_only = root.is_secondary_stem_only_var
        self.is_denoise = True if not root.denoise_option_var == DENOISE_NONE else False
        self.is_mdx_c_seg_def = root.is_mdx_c_seg_def_var#
        self.mdx_batch_size = 1 if root.mdx_batch_size_var == DEF_OPT else int(root.mdx_batch_size_var)
        self.mdxnet_stem_select = root.mdxnet_stems_var 
        self.overlap = float(root.overlap_var) if not root.overlap_var == DEFAULT else 0.25
        self.overlap_mdx = float(root.overlap_mdx_var) if not root.overlap_mdx_var == DEFAULT else root.overlap_mdx_var
        self.overlap_mdx23 = int(float(root.overlap_mdx23_var))
        self.semitone_shift = float(root.semitone_shift_var)
        self.is_pitch_change = False if self.semitone_shift == 0 else True
        self.is_match_frequency_pitch = root.is_match_frequency_pitch_var
        self.is_mdx_ckpt = False
        self.is_mdx_c = False
        self.is_mdx_combine_stems = root.is_mdx23_combine_stems_var#
        self.mdx_c_configs = None
        self.mdx_model_stems = []
        self.mdx_dim_f_set = None
        self.mdx_dim_t_set = None
        self.mdx_stem_count = 1
        self.compensate = None
        self.mdx_n_fft_scale_set = None
        self.wav_type_set = root.wav_type_set#
        self.device_set = device_set.split(':')[-1].strip() if ':' in device_set else device_set
        self.mp3_bit_set = root.mp3_bit_set_var
        self.save_format = root.save_format_var
        self.is_invert_spec = root.is_invert_spec_var#
        self.is_mixer_mode = False#
        self.demucs_stems = root.demucs_stems_var
        self.is_demucs_combine_stems = root.is_demucs_combine_stems_var
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name
        self.process_method = selected_process_method
        self.model_status = False if self.model_name == CHOOSE_MODEL or self.model_name == NO_MODEL else True
        self.primary_stem = None
        self.secondary_stem = None
        self.primary_stem_native = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = True if is_vocal_split_model else is_secondary_model
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
        self.is_multi_stem_ensemble = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.is_change_def = is_change_def
        self.model_hash_dir = None
        self.is_get_hash_dir_only = is_get_hash_dir_only
        self.is_secondary_model_activated = False
        self.vocal_split_model = None
        self.is_vocal_split_model = is_vocal_split_model
        self.is_vocal_split_model_activated = False
        self.is_save_inst_vocal_splitter = root.is_save_inst_set_vocal_splitter_var
        self.is_inst_only_voc_splitter = root.check_only_selection_stem(INST_STEM_ONLY)
        self.is_save_vocal_only = root.check_only_selection_stem(IS_SAVE_VOC_ONLY)

        if selected_process_method == ENSEMBLE_MODE:
            self.process_method, _, self.model_name = model_name.partition(ENSEMBLE_PARTITION)
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem, self.ensemble_secondary_stem = root.return_ensemble_stems()
            
            is_not_secondary_or_pre_proc = not is_secondary_model and not is_pre_proc_model
            self.is_ensemble_mode = is_not_secondary_or_pre_proc
            
            if root.ensemble_main_stem_var == FOUR_STEM_ENSEMBLE:
                self.is_4_stem_ensemble = self.is_ensemble_mode
            elif root.ensemble_main_stem_var == MULTI_STEM_ENSEMBLE and root.chosen_process_method_var == ENSEMBLE_MODE:
                self.is_multi_stem_ensemble = True

            is_not_vocal_stem = self.ensemble_primary_stem != VOCAL_STEM
            self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var if is_not_vocal_stem else False

        if self.process_method == VR_ARCH_TYPE:
            self.is_secondary_model_activated = root.vr_is_secondary_model_activate_var if not is_secondary_model else False
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
                self.model_hash_dir = os.path.join(VR_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(VR_HASH_DIR, root.vr_hash_MAPPER) if not self.model_hash == WOOD_INST_MODEL_HASH else WOOD_INST_PARAMS
                if self.model_data:
                    vr_model_param = os.path.join(VR_PARAM_DIR, "{}.json".format(self.model_data["vr_model_param"]))
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = secondary_stem(self.primary_stem)
                    self.vr_model_param = ModelParameters(vr_model_param)
                    self.model_samplerate = self.vr_model_param.param['sr']
                    self.primary_stem_native = self.primary_stem
                    if "nout" in self.model_data.keys() and "nout_lstm" in self.model_data.keys():
                        self.model_capacity = self.model_data["nout"], self.model_data["nout_lstm"]
                        self.is_vr_51_model = True
                    self.check_if_karaokee_model()
   
                else:
                    self.model_status = False
                
        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = root.mdx_is_secondary_model_activate_var if not is_secondary_model else False
            self.margin = int(root.margin_var)
            self.chunks = 0
            self.mdx_segment_size = int(root.mdx_segment_size_var)
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(MDX_HASH_DIR, f"{self.model_hash}.json")
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(MDX_HASH_DIR, root.mdx_hash_MAPPER)
                if self.model_data:
                    
                    if "config_yaml" in self.model_data:
                        self.is_mdx_c = True
                        config_path = os.path.join(MDX_C_CONFIG_PATH, self.model_data["config_yaml"])
                        if os.path.isfile(config_path):
                            with open(config_path) as f:
                                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

                            self.mdx_c_configs = config
                                
                            if self.mdx_c_configs.training.target_instrument:
                                # Use target_instrument as the primary stem and set 4-stem ensemble to False
                                target = self.mdx_c_configs.training.target_instrument
                                self.mdx_model_stems = [target]
                                self.primary_stem = target
                            else:
                                # If no specific target_instrument, use all instruments in the training config
                                self.mdx_model_stems = self.mdx_c_configs.training.instruments
                                self.mdx_stem_count = len(self.mdx_model_stems)
                                
                                # Set primary stem based on stem count
                                if self.mdx_stem_count == 2:
                                    self.primary_stem = self.mdx_model_stems[0]
                                else:
                                    self.primary_stem = self.mdxnet_stem_select
                                
                                # Update mdxnet_stem_select based on ensemble mode
                                if self.is_ensemble_mode:
                                    self.mdxnet_stem_select = self.ensemble_primary_stem
                        else:
                            self.model_status = False
                    else:
                        self.compensate = self.model_data["compensate"] if root.compensate_var == AUTO_SELECT else float(root.compensate_var)
                        self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                        self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                        self.mdx_n_fft_scale_set = self.model_data["mdx_n_fft_scale_set"]
                        self.primary_stem = self.model_data["primary_stem"]
                        self.primary_stem_native = self.model_data["primary_stem"]
                        self.check_if_karaokee_model()
                        
                    self.secondary_stem = secondary_stem(self.primary_stem)
                else:
                    self.model_status = False

        if self.process_method == DEMUCS_ARCH_TYPE:
            self.is_secondary_model_activated = root.demucs_is_secondary_model_activate_var if not is_secondary_model else False
            if not self.is_ensemble_mode:
                self.pre_proc_model_activated = root.is_demucs_pre_proc_model_activate_var if not root.demucs_stems_var in [VOCAL_STEM, INST_STEM] else False
            self.margin_demucs = int(root.margin_demucs_var)
            self.chunks_demucs = 0
            self.shifts = int(root.shifts_var)
            self.is_split_mode = root.is_split_mode_var
            self.segment = root.segment_var
            self.is_chunk_demucs = root.is_chunk_demucs_var
            self.is_primary_stem_only = root.is_primary_stem_only_var if self.is_ensemble_mode else root.is_primary_stem_only_Demucs_var 
            self.is_secondary_stem_only = root.is_secondary_stem_only_var if self.is_ensemble_mode else root.is_secondary_stem_only_Demucs_var
            self.get_demucs_model_data()
            self.get_demucs_model_path()
            
        if self.model_status:
            self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        else:
            self.model_basename = None
            
        self.pre_proc_model_activated = self.pre_proc_model_activated if not self.is_secondary_model else False
        
        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        is_secondary_activated_and_status = self.is_secondary_model_activated and self.model_status
        is_demucs = self.process_method == DEMUCS_ARCH_TYPE
        is_all_stems = root.demucs_stems_var == ALL_STEMS
        is_valid_ensemble = not self.is_ensemble_mode and is_all_stems and is_demucs
        is_multi_stem_ensemble_demucs = self.is_multi_stem_ensemble and is_demucs

        if is_secondary_activated_and_status:
            if is_valid_ensemble or self.is_4_stem_ensemble or is_multi_stem_ensemble_demucs:
                for key in DEMUCS_4_SOURCE_LIST:
                    self.secondary_model_data(key)
                    self.secondary_model_4_stem.append(self.secondary_model)
                    self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                    self.secondary_model_4_stem_names.append(key)
                
                self.demucs_4_stem_added_count = sum(i is not None for i in self.secondary_model_4_stem)
                self.is_secondary_model_activated = any(i is not None for i in self.secondary_model_4_stem)
                self.demucs_4_stem_added_count -= 1 if self.is_secondary_model_activated else 0
                
                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [i.model_basename if i is not None else None for i in self.secondary_model_4_stem]
                    self.is_demucs_4_stem_secondaries = True
            else:
                primary_stem = self.ensemble_primary_stem if self.is_ensemble_mode and is_demucs else self.primary_stem
                self.secondary_model_data(primary_stem)

        if self.process_method == DEMUCS_ARCH_TYPE and not is_secondary_model:
            if self.demucs_stem_count >= 3 and self.pre_proc_model_activated:
                self.pre_proc_model = root.process_determine_demucs_pre_proc_model(self.primary_stem)
                self.pre_proc_model_activated = True if self.pre_proc_model else False
                self.is_demucs_pre_proc_model_inst_mix = root.is_demucs_pre_proc_model_inst_mix_var if self.pre_proc_model else False

        if self.is_vocal_split_model and self.model_status:
            self.is_secondary_model_activated = False
            if self.is_bv_model:
                primary = BV_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else LEAD_VOCAL_STEM
            else:
                primary = LEAD_VOCAL_STEM if self.primary_stem_native == VOCAL_STEM else BV_VOCAL_STEM
            self.primary_stem, self.secondary_stem = primary, secondary_stem(primary)
            
        self.vocal_splitter_model_data()
               
    def vocal_splitter_model_data(self):
        if not self.is_secondary_model and self.model_status:
            self.vocal_split_model = root.process_determine_vocal_split_model()
            self.is_vocal_split_model_activated = True if self.vocal_split_model else False
            
            if self.vocal_split_model:
                if self.vocal_split_model.bv_model_rebalance:
                    self.is_sec_bv_rebalance = True
            
    def secondary_model_data(self, primary_stem):
        secondary_model_data = root.process_determine_secondary_model(self.process_method, primary_stem, self.is_primary_stem_only, self.is_secondary_stem_only)
        self.secondary_model = secondary_model_data[0]
        self.secondary_model_scale = secondary_model_data[1]
        self.is_secondary_model_activated = False if not self.secondary_model else True
        if self.secondary_model:
            self.is_secondary_model_activated = False if self.secondary_model.model_basename == self.model_basename else True
            
        #print("self.is_secondary_model_activated: ", self.is_secondary_model_activated)
              
    def check_if_karaokee_model(self):
        if IS_KARAOKEE in self.model_data.keys():
            self.is_karaoke = self.model_data[IS_KARAOKEE]
        if IS_BV_MODEL in self.model_data.keys():
            self.is_bv_model = self.model_data[IS_BV_MODEL]#
        if IS_BV_MODEL_REBAL in self.model_data.keys() and self.is_bv_model:
            self.bv_model_rebalance = self.model_data[IS_BV_MODEL_REBAL]#
   
    def get_mdx_model_path(self):
        
        if self.model_name.endswith(CKPT):
            self.is_mdx_ckpt = True

        ext = '' if self.is_mdx_ckpt else ONNX
        
        for file_name, chosen_mdx_model in root.mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                if file_name.endswith(CKPT):
                    ext = ''
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")
            
        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")
    
    def get_demucs_model_path(self):
        
        demucs_newer = self.demucs_version in {DEMUCS_V3, DEMUCS_V4}
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR
        
        for file_name, chosen_model in root.demucs_name_select_MAPPER.items():
            if self.model_name == chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(DEMUCS_NEWER_REPO_DIR, f'{self.model_name}.yaml')

    def get_demucs_model_data(self):

        self.demucs_version = DEMUCS_V4

        for key, value in DEMUCS_VERSION_MAPPER.items():
            if value in self.model_name:
                self.demucs_version = key

        if DEMUCS_UVR_MODEL in self.model_name:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_2_SOURCE, DEMUCS_2_SOURCE_MAPPER, 2
        else:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = DEMUCS_4_SOURCE, DEMUCS_4_SOURCE_MAPPER, 4

        if not self.is_ensemble_mode:
            self.primary_stem = PRIMARY_STEM if self.demucs_stems == ALL_STEMS else self.demucs_stems
            self.secondary_stem = secondary_stem(self.primary_stem)
            
    def get_model_data(self, model_hash_dir, hash_mapper:dict):
        model_settings_json = os.path.join(model_hash_dir, f"{self.model_hash}.json")

        if os.path.isfile(model_settings_json):
            with open(model_settings_json, 'r') as json_file:
                return json.load(json_file)
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings

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
        #print(self.model_name," - ", self.model_hash)
        
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
        self.mdxnet_stems_var = ALL_STEMS
        self.is_primary_stem_only_Text_var = ""
        self.is_secondary_stem_only_Text_var = ""
        self.is_primary_stem_only_Demucs_Text_var = ""
        self.is_secondary_stem_only_Demucs_Text_var = ""
        self.scaling_var = 1.0
        self.active_processing_thread = None
        self.verification_thread = None
        self.is_menu_settings_open = False
        self.is_root_defined_var = False
        self.is_check_splash = False
        
        self.is_open_menu_advanced_vr_options = False
        self.is_open_menu_advanced_demucs_options = False
        self.is_open_menu_advanced_mdx_options = False
        self.is_open_menu_advanced_ensemble_options = False
        self.is_open_menu_view_inputs = False
        self.is_open_menu_help = False
        self.is_open_menu_error_log = False
        self.is_open_menu_advanced_align_options = False

        self.menu_advanced_vr_options_close_window = None
        self.menu_advanced_demucs_options_close_window = None
        self.menu_advanced_mdx_options_close_window = None
        self.menu_advanced_ensemble_options_close_window = None
        self.menu_help_close_window = None
        self.menu_error_log_close_window = None
        self.menu_view_inputs_close_window = None
        self.menu_advanced_align_options_close_window = None

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
        self.true_model_count = 0
        self.vr_primary_source = None
        self.vr_secondary_source = None
        self.mdx_primary_source = None
        self.mdx_secondary_source = None
        self.demucs_primary_source = None
        self.demucs_secondary_source = None
        self.toplevels = []

        #Download Center Vars
        self.online_data = {}
        self.bulletin_data = INFO_UNAVAILABLE_TEXT
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
        self.pre_proc_model_toggle = None
        self.change_state_lambda = None
        self.file_one_sub_var = FILE_ONE_MAIN_LABEL
        self.file_two_sub_var = FILE_TWO_MAIN_LABEL
        self.cuda_device_list = GPU_DEVICE_NUM_OPTS
        self.opencl_list = GPU_DEVICE_NUM_OPTS
        
        #Model Update
        self.last_found_ensembles = ENSEMBLE_OPTIONS
        self.last_found_settings = ENSEMBLE_OPTIONS
        self.last_found_models = ()
        self.model_data_table = ()
        self.ensemble_model_list = ()
        self.default_change_model_list = ()

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

    def check_only_selection_stem(self, checktype):
        
        chosen_method = self.chosen_process_method_var
        is_demucs = chosen_method == DEMUCS_ARCH_TYPE#

        stem_primary_label = self.is_primary_stem_only_Demucs_Text_var if is_demucs else self.is_primary_stem_only_Text_var
        stem_primary_bool = self.is_primary_stem_only_Demucs_var if is_demucs else self.is_primary_stem_only_var
        stem_secondary_label = self.is_secondary_stem_only_Demucs_Text_var if is_demucs else self.is_secondary_stem_only_Text_var
        stem_secondary_bool = self.is_secondary_stem_only_Demucs_var if is_demucs else self.is_secondary_stem_only_var

        if checktype == VOCAL_STEM_ONLY:
            return not (
                (not VOCAL_STEM_ONLY == stem_primary_label and stem_primary_bool) or 
                (not VOCAL_STEM_ONLY in stem_secondary_label and stem_secondary_bool)
            )
        elif checktype == INST_STEM_ONLY:
            return (
                (INST_STEM_ONLY == stem_primary_label and stem_primary_bool and self.is_save_inst_set_vocal_splitter_var and self.set_vocal_splitter_var != NO_MODEL) or 
                (INST_STEM_ONLY == stem_secondary_label and stem_secondary_bool and self.is_save_inst_set_vocal_splitter_var and self.set_vocal_splitter_var != NO_MODEL)
            )
        elif checktype == IS_SAVE_VOC_ONLY:
            return (
                (VOCAL_STEM_ONLY == stem_primary_label and stem_primary_bool) or 
                (VOCAL_STEM_ONLY == stem_secondary_label and stem_secondary_bool)
            )
        elif checktype == IS_SAVE_INST_ONLY:
            return (
                (INST_STEM_ONLY == stem_primary_label and stem_primary_bool) or 
                (INST_STEM_ONLY == stem_secondary_label and stem_secondary_bool)
            )

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
        self.mdx_segment_size_var = data['mdx_segment_size']
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
        self.overlap_mdx_var = data['overlap_mdx']
        self.overlap_mdx23_var = data['overlap_mdx23']  
        self.shifts_var = data['shifts']
        self.chunks_demucs_var = data['chunks_demucs']
        self.margin_demucs_var = data['margin_demucs']
        self.is_chunk_demucs_var = data['is_chunk_demucs']
        self.is_chunk_mdxnet_var = data['is_chunk_mdxnet']
        self.is_primary_stem_only_Demucs_var = data['is_primary_stem_only_Demucs']
        self.is_secondary_stem_only_Demucs_var = data['is_secondary_stem_only_Demucs']
        self.is_split_mode_var = data['is_split_mode']
        self.is_demucs_combine_stems_var = data['is_demucs_combine_stems']
        self.is_mdx23_combine_stems_var = data['is_mdx23_combine_stems']
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
        self.denoise_option_var = data['denoise_option']
        self.phase_option_var = data['phase_option']
        self.phase_shifts_var = data['phase_shifts']
        self.is_save_align_var = data['is_save_align']
        self.is_match_silence_var = data['is_match_silence']
        self.is_spec_match_var = data['is_spec_match']
        self.is_match_frequency_pitch_var = data['is_match_frequency_pitch']
        self.is_mdx_c_seg_def_var = data['is_mdx_c_seg_def']
        self.is_invert_spec_var = data['is_invert_spec']
        self.is_deverb_vocals_var = data['is_deverb_vocals']
        self.deverb_vocal_opt_var = data['deverb_vocal_opt']
        self.voc_split_save_opt_var = data['voc_split_save_opt']

        self.mdx_voc_inst_secondary_model_var = data['mdx_voc_inst_secondary_model']
        self.mdx_other_secondary_model_var = data['mdx_other_secondary_model']
        self.mdx_bass_secondary_model_var = data['mdx_bass_secondary_model']
        self.mdx_drums_secondary_model_var = data['mdx_drums_secondary_model']
        self.mdx_is_secondary_model_activate_var = data['mdx_is_secondary_model_activate']
        self.mdx_voc_inst_secondary_model_scale_var = data['mdx_voc_inst_secondary_model_scale']
        self.mdx_other_secondary_model_scale_var = data['mdx_other_secondary_model_scale']
        self.mdx_bass_secondary_model_scale_var = data['mdx_bass_secondary_model_scale']
        self.mdx_drums_secondary_model_scale_var = data['mdx_drums_secondary_model_scale']
        self.is_mdxnet_c_model_var = False
        #Ensemble Vars
        self.is_save_all_outputs_ensemble_var = data['is_save_all_outputs_ensemble']
        self.is_append_ensemble_name_var = data['is_append_ensemble_name']

        #Audio Tool Vars
        self.chosen_audio_tool_var = data['chosen_audio_tool']
        self.choose_algorithm_var = data['choose_algorithm']
        self.time_stretch_rate_var = data['time_stretch_rate']
        self.pitch_rate_var = data['pitch_rate']
        self.is_time_correction_var = data['is_time_correction']

        #Shared Vars
        self.semitone_shift_var = data['semitone_shift']
        self.mp3_bit_set_var = data['mp3_bit_set']
        self.save_format_var = data['save_format']
        self.wav_type_set_var = data['wav_type_set']
        self.device_set_var = data['device_set']
        self.user_code_var = data['user_code']
        self.is_gpu_conversion_var = data['is_gpu_conversion']
        self.is_primary_stem_only_var = data['is_primary_stem_only']
        self.is_secondary_stem_only_var = data['is_secondary_stem_only']
        self.is_testing_audio_var = data['is_testing_audio']
        self.is_auto_update_model_params_var = True
        self.is_auto_update_model_params = data['is_auto_update_model_params']
        self.is_add_model_name_var = data['is_add_model_name']
        self.is_accept_any_input_var = data['is_accept_any_input']
        self.is_task_complete_var = data['is_task_complete']
        self.is_normalization_var = data['is_normalization']
        self.is_use_opencl_var = False #True if is_opencl_only else data['is_use_opencl'])#
        self.is_wav_ensemble_var = data['is_wav_ensemble']
        self.is_create_model_folder_var = data['is_create_model_folder']
        self.help_hints_var = data['help_hints_var']
        self.model_sample_mode_var = data['model_sample_mode']
        self.model_sample_mode_duration_var = data['model_sample_mode_duration']
        self.model_sample_mode_duration_checkbox_var = SAMPLE_MODE_CHECKBOX(self.model_sample_mode_duration_var)
        self.model_sample_mode_duration_label_var = f'{self.model_sample_mode_duration_var} Seconds'
        self.set_vocal_splitter_var = data['set_vocal_splitter']
        self.is_set_vocal_splitter_var = data['is_set_vocal_splitter']
        self.is_save_inst_set_vocal_splitter_var = data['is_save_inst_set_vocal_splitter']
        
        #Path Vars
        self.export_path_var = data['export_path']
        self.inputPaths = data['input_paths']
        self.lastDir = data['lastDir']
        
        #DualPaths-Align
        self.time_window_var = data['time_window']
        self.intro_analysis_var = data['intro_analysis']
        self.db_analysis_var = data['db_analysis']
        
        self.fileOneEntry_var = data['fileOneEntry']
        self.fileOneEntry_Full_var = data['fileOneEntry_Full']
        self.fileTwoEntry_var = data['fileTwoEntry']
        self.fileTwoEntry_Full_var = data['fileTwoEntry_Full']
        self.DualBatch_inputPaths = data['DualBatch_inputPaths']

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

    def speech_to_segments(self, audio_wav, language=None, WHISPER_MODEL_SIZE=whisper_model_default, batch_size=8, chunk_size=10):
      print("Start speech_to_segments::")
      device = "cuda" if torch.cuda.is_available() else "cpu"
      model = whisperx.load_model(
          WHISPER_MODEL_SIZE,
          device,
          compute_type=compute_type_default,
          language=language,
          )
      audio = whisperx.load_audio(audio_wav)
      print("Transcribing::")
      result = model.transcribe(WHISPER_MODEL_SIZE, audio, batch_size=batch_size, chunk_size=chunk_size)
      gc.collect(); torch.cuda.empty_cache(); del model
      print("Aligning::")
      model_a, metadata = whisperx.load_align_model(
          language_code=result["language"],
          device=device,
          model_name=None
          )
      result = whisperx.align(
          result["segments"],
          model_a,
          metadata,
          audio,
          device,
          return_char_alignments=True,
          )
      gc.collect(); torch.cuda.empty_cache(); del model_a
      return result
    
    def segments_to_srt(self, segments, output_path):
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

    def modify_ass(self, segments, ass_path, font_size=25):
      print("segments length:: ",len(segments), ass_path)
      with open(ass_path, encoding='utf_8_sig') as f:
        _ass = ass.parse(f)
        print("ass length", len(_ass.events))
        print("ass style", _ass.styles)
        print("ass keys", list(_ass.sections.keys()))
        
      subtitle_style = f"{'{'+chr(92)+'fs'+str(font_size)+'}'+'{'+chr(92)+'b1&'+chr(92)+'c&HA95C21&'+chr(92)+'2c&HC8C8C8&'+chr(92)+'3c&HFFFFFF&'+chr(92)+'4c&H000000'+'}'}"
      even_pos = f"{'{'+chr(92)+'an1&'+chr(92)+'pos(15,245)'+'}'}"
      odd_pos = f"{'{'+chr(92)+'an3&'+chr(92)+'pos(370,275)'+'}'}"
      ## Calculate time for each word
      for i in range(len(segments)):
        for index, item in enumerate(segments[i]["words"]):
          if 'start' in item:
            if index < len(segments[i]["words"]) - 1 and 'start' in segments[i]['words'][index+1]:
              item['word'] = f"{'{'+chr(92)+'K'+str(round(timedelta(seconds=(segments[i]['words'][index+1]['start']-item['start'])).total_seconds()*100))+'}'+item['word']}"
            else:
              item['word'] = f"{'{'+chr(92)+'K'+str(round(timedelta(seconds=(item['end']-item['start'])).total_seconds()*100))+'}'+item['word']}"
          else:
            item['word'] = item['word']
        # _ass.events[i].text = " ".join([ f"{'{'+chr(92)+'K'+str(round(timedelta(seconds=(item['end']-item['start'])).total_seconds()*100))+'}'+item['word']}" if 'start' in item else item['word'] for index, item in enumerate(segments[i]["words"])])
        _ass.events[i].text = " ".join([ item['word'] for item in segments[i]['words']])
        ## Modify pre time of subtitle
        if i == 0:
          pre_time = 3 ## seconds
          _ass.events[i].start = timedelta(seconds=(segments[i]["start"] - pre_time)) ## Add 3 seconds to the beginning
          _ass.events[i].text = f"{'{'+chr(92)+'K'+str(round(pre_time*100))+'}'+' ♪ ♪ ♪ '}{_ass.events[i].text}"
        else:
          pre_time = segments[i]["start"] - segments[i-1]["start"] ## seconds
          _ass.events[i].start = timedelta(seconds=(segments[i]["start"] - pre_time)) ## Add 3 seconds to the beginning
          _ass.events[i].text = f"{'{'+chr(92)+'K'+str(round(pre_time*100))+'}'}{_ass.events[i].text}"
        ## Modify post time of subtitle
        if i == len(segments) - 1:
          post_time = 3
          _ass.events[i].end = timedelta(seconds=(segments[i]["end"] + post_time)) ## Add 3 seconds to the end
        else:
          _ass.events[i].end = timedelta(seconds=(segments[i+1]["start"])) ## Add 3 seconds to the end
        ## Add style to subtitles
        _ass.events[i].text = f"{subtitle_style}{_ass.events[i].text}"
        ## Add position to odd and even sub
        if i % 2 == 0:
          _ass.events[i].text = f"{even_pos}{_ass.events[i].text}"
        else:
          _ass.events[i].text = f"{odd_pos}{_ass.events[i].text}"
      with open(ass_path, "w", encoding='utf_8_sig') as f:
        _ass.dump_file(f)
                           
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

    def process_determine_vocal_split_model(self):
        """Obtains the correct vocal splitter secondary model data for conversion."""
        
        # Check if a vocal splitter model is set and if it's not the 'NO_MODEL' value
        if self.set_vocal_splitter_var != NO_MODEL and self.is_set_vocal_splitter_var:
            vocal_splitter_model = ModelData(self.set_vocal_splitter_var, is_vocal_split_model=True)
            
            # Return the model if it's valid
            if vocal_splitter_model.model_status:
                return vocal_splitter_model
                
        return None
                           
    def process_start(self, inputPaths, video_burn, stt, stt_mode, stt_model, stt_language, stt_burn, stt_batch_size,stt_chuck_size,stt_font_size,uvr_method, choosen_model, progress=gr.Progress()):
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

            progress(0.15, desc=f"Splitting media... 0/{len(inputPaths)}")
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
                          # os.system(f"ffmpeg -y -i '{video_file}' -vn -acodec pcm_s16le -ar 44100 -ac 2 '{audio_file}'")
                          subprocess.run(['ffmpeg', '-y', '-i', video_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_file])
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

                ## Start splitting vocals and instrument
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
                    print("start seperating voices...")
                    if current_model.process_method == VR_ARCH_TYPE:
                        seperator = SeperateVR(current_model, process_data)
                    if current_model.process_method == MDX_ARCH_TYPE:
                        seperator = SeperateMDX(current_model, process_data)
                    if current_model.process_method == DEMUCS_ARCH_TYPE:
                        seperator = SeperateDemucs(current_model, process_data)
                    seperator.seperate()
                
                ## Define path
                inst_path = os.path.join(export_path, f'{audio_file_base}_({INST_STEM}).wav')
                other_path = os.path.join(export_path, f'{audio_file_base}_({OTHER_STEM}).wav')
                bgmusic_path = inst_path if os.path.exists(inst_path) else other_path
                vocal_path = os.path.join(export_path, f'{audio_file_base}_({VOCAL_STEM}).wav')
                srt_path = os.path.join(export_path, f'{audio_file_base}.srt')
                ass_path = os.path.join(export_path, f'{audio_file_base}.ass')
                json_path = os.path.join(export_path, f'{audio_file_base}.json')
                
                ## export srt,ass with timestamp
                if stt and os.path.exists(vocal_path):
                  stt_language = LANGUAGES[stt_language]
                  result_segments = self.speech_to_segments(audio_wav=vocal_path,language=stt_language, WHISPER_MODEL_SIZE=stt_model, batch_size=stt_batch_size,chunk_size=stt_chuck_size)
                  print("dumping speech_to_segments::")
                  with open(json_path, 'a', encoding='utf-8') as jsonFile:
                    json.dump(result_segments['segments'], jsonFile, indent=4)
                  self.segments_to_srt(result_segments['segments'], srt_path)
                  subprocess.run(['ffmpeg', '-i', srt_path, ass_path])
                  self.modify_ass(result_segments['segments'], ass_path, stt_font_size)    
                              
                ## merge video with split audio
                if video_burn and not is_audio and os.path.exists(video_file):
                  media_output_file = os.path.join(export_path, os.path.basename(video_file))
                  if stt and stt_mode == 'Karaoke':
                    ## create video file with dual mono of original audio and removed vocals
                    ffmpeg_command = [
                        "ffmpeg", "-i", video_file, "-i", bgmusic_path,
                        "-filter_complex", "[0:a]pan=mono|c0=c0[a1];[1:a]pan=mono|c0=c1[a2]", "-map", "0:v", "-map", "[a1]", "-map", "[a2]", "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", media_output_file
                    ]
                  else:
                    ## create video file with stereo removed vocals
                    ffmpeg_command = [
                        "ffmpeg", "-i", video_file, "-i", bgmusic_path,
                        "-c:v", "copy", "-c:a", "aac", "-map", "0:v", "-map", "1:a", "-shortest", media_output_file
                    ]
                  if stt and stt_burn and os.path.exists(ass_path):
                    if stt_mode == 'Karaoke':
                      ffmpeg_command[5:5] = ["-vf", f"ass='{ass_path}'"]
                    else:
                      ffmpeg_command[5:5] = ["-vf", f"subtitles='{srt_path}'"]
                  print("merging video::", ffmpeg_command)
                  subprocess.run(ffmpeg_command)
                else:
                  media_output_file = bgmusic_path
                  
                ## Copy to custom output directory if specify 
                COPY_OUTPUT_DIR = os.environ.get('COPY_OUTPUT_DIR', "")
                if COPY_OUTPUT_DIR != "":
                  print("copy video to COPY_OUTPUT_DIR::")
                  shutil.copy(media_output_file, os.path.join(COPY_OUTPUT_DIR, os.path.basename(media_output_file)))
                # Archive final output folder when done
                archive_path = os.path.join(Path(temp_dir).absolute(), os.path.splitext(os.path.basename(audio_file))[0])
                shutil.make_archive(archive_path, 'zip', export_path)   
                final_output.append(f"{archive_path}.zip")
                progress(file_num/len(inputPaths), desc=f"Splitting media... {file_num}/{len(inputPaths)}")
                
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

    def preprocess(self, media_inputs, link_inputs, video_burn, stt, stt_mode, stt_model, stt_language, stt_burn,stt_batch_size,stt_chuck_size, stt_font_size,uvr_method, uvr_model, progress=gr.Progress()):
      progress(0.05, desc="Processing media...")
      print(media_inputs, link_inputs, video_burn, stt, stt_mode, stt_model, stt_language, stt_burn,stt_batch_size,stt_chuck_size, stt_font_size,uvr_method, uvr_model)
      media_inputs = media_inputs if media_inputs is not None else []
      media_inputs = media_inputs if isinstance(media_inputs, list) else [media_inputs]
      media_inputs = [media_input if isinstance(media_input, str) else media_input.name for media_input in media_inputs]
      youtube_temp_dir = os.path.join(temp_dir, 'youtube')
      shutil.rmtree(youtube_temp_dir, ignore_errors=True)
      Path(youtube_temp_dir).mkdir(parents=True, exist_ok=True)   
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
        output = root.process_start(media_inputs, video_burn, stt, stt_mode, stt_model, stt_language, stt_burn, stt_batch_size,stt_chuck_size, stt_font_size,uvr_method, uvr_model)
        return output
      else:
        raise gr.Error("Input not valid!!")
      
    def init_webui(self):
        title = "<center><strong><font size='7'>Ultimate Vocal Remover</font></strong></center>"
        description = """
        ### 🎥 **Ultimate Vocal Remover Tool** 📽️
        🎥 Upload a video or provide a video link. 📽️
        """
        theme = gr.themes.Base.load(os.path.join('themes','taithrah-minimal@0.0.1.json')).set(
            background_fill_primary ="#171717",
            panel_background_fill = "transparent",
            body_text_color = "#afafaf"
        )
        with gr.Blocks(title="UVR",theme=theme) as self.demo:
            gr.Markdown(title)
            gr.Markdown(description)

        #### video
            with gr.Tab("Input Video|Audio for UVR"):
                with gr.Row():
                    with gr.Column():
                        #media_input = gr.UploadButton("Click to Upload a video", file_types=["video"], file_count="single") #gr.Video() # height=300,width=300
                        media_input = gr.File(label="VIDEO|AUDIO",file_count='multiple', file_types=['audio','video'])
                        link_input = gr.Textbox(label="Youtube Link",info="Example: https://www.youtube.com/watch?v=-biOGdYiF-I,https://www.youtube.com/watch?v=-biOGdYiF-I", type="text", placeholder="URL goes here, seperate by comma...")        
                        with gr.Row():
                          video_burn = gr.Checkbox(label="Enable",  value=True, info='Export with video', visible=False,scale=1)
                          stt = gr.Checkbox(label="Enable",  value=False,info='Export subtitle with timestamp',scale=1)
                        with gr.Accordion(label="Subtitle Option", visible=False) as stt_option:
                          with gr.Row():
                            stt_mode = gr.Dropdown(['Normal', 'Karaoke'], label='Subtitle Mode', value='Normal',scale=1)
                            stt_model = gr.Dropdown(['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'], value=whisper_model_default, label="Whisper model", scale=1)
                            stt_language = gr.Dropdown(['Automatic detection', 'Arabic (ar)', 'Cantonese (yue)', 'Chinese (zh)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Russian (ru)', 'Spanish (es)', 'Turkish (tr)', 'Ukrainian (uk)', 'Urdu (ur)', 'Vietnamese (vi)'], label='Target language', value='Automatic detection',scale=1)
                            stt_burn = gr.Checkbox(label="Enable",  value=False, info='Burn subtitle into video',scale=1)
                          with gr.Row():  
                            stt_batch_size =gr.Slider(minimum=2, maximum=50, value=round(int(torch.cuda.get_device_properties(0).total_memory)*1.6/1000000000), label="Batch Size", step=1,scale=1)
                            stt_chuck_size = gr.Slider(minimum=5, maximum=50, value=10, label="Chuck Size", step=1,scale=1)
                            stt_font_size = gr.Slider(minimum=10, maximum=30, value=25, label="Font Size", step=1,scale=1)
                          def update_visible(stt_check):
                            return  gr.update(visible=stt_check)
                          stt.change(update_visible, stt, [stt_option])
                        gr.ClearButton(components=[media_input,link_input], size='sm')
                        with gr.Row():
                          uvr_type_option = [str(MDX_ARCH_TYPE),str(DEMUCS_ARCH_TYPE),str(VR_ARCH_TYPE)]
                          uvr_type = gr.Dropdown(choices=uvr_type_option, value=str(DEMUCS_ARCH_TYPE), label='AI Tech', info="Choose AI Tech for UVR")
                          uvr_model_option = [str(value) for key, value in self.demucs_name_select_MAPPER.items()]
                          uvr_model = gr.Dropdown(choices=uvr_model_option,value="v3 | UVR_Model_1",label='UVR Model', info="Choose UVR Model")
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
            media_button.click(self.preprocess, inputs=[
                media_input,
                link_input,
                video_burn,
                stt,
                stt_mode,
                stt_model,
                stt_language,
                stt_burn,
                stt_batch_size,
                stt_chuck_size,
                stt_font_size,
                uvr_type,
                uvr_model,
                
                ], outputs=media_output, api_name="convert")

    def start_webui(self):
        proxy = os.getenv('API_ENABLE', '')
        if proxy and proxy == 'true':
          auth_user = ''
          auth_pass = ''
        else:
          auth_user = os.getenv('AUTH_USER', '')
          auth_pass = os.getenv('AUTH_PASS', '')
        self.demo.queue(concurrency_count=1).launch(
          auth=(auth_user, auth_pass) if auth_user != '' and auth_pass != '' else None,
          show_api=True,
          debug=True,
          inbrowser=True,
          show_error=True,
          server_name="0.0.0.0",
          server_port=6870,
          # quiet=True, 
          share=False   
          )

if __name__ == "__main__":
  # shutil.rmtree(os.path.join(tempfile.gettempdir(), "gradio"),ignore_errors=True)
  shutil.rmtree(temp_dir,ignore_errors=True)
  Path(temp_dir).mkdir(parents=True, exist_ok=True)
  root = UVR()
  root.init_webui()
  root.start_webui()