from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import sounddevice as sd
import numpy as np
import librosa
import argparse
import torch
import sys


def load_model(in_fpath, parser):

	parser.add_argument("-e", "--enc_model_fpath", type=Path, 
		        default="encoder/saved_models/pretrained.pt",
		        help="Path to a saved encoder")
	parser.add_argument("-s", "--syn_model_dir", type=Path, 
		        default="synthesizer/saved_models/logs-pretrained/",
		        help="Directory containing the synthesizer model")
	parser.add_argument("-v", "--voc_model_fpath", type=Path, 
		        default="vocoder/saved_models/pretrained/pretrained.pt",
		        help="Path to a saved vocoder")
	parser.add_argument("--low_mem", action="store_true", help=\
	"If True, the memory used by the synthesizer will be freed after each use. Adds large "
	"overhead but allows to save some GPU memory for lower-end GPUs.")
	parser.add_argument("--no_sound", action="store_true", help=\
	"If True, audio won't be played.")
	args = parser.parse_args()
	encoder.load_model(args.enc_model_fpath)
	synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
	vocoder.load_model(args.voc_model_fpath)

	preprocessed_wav = encoder.preprocess_wav(in_fpath)
	original_wav, sampling_rate = librosa.load(in_fpath)
	preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
	embed = encoder.embed_utterance(preprocessed_wav)
	
	return synthesizer, sampling_rate, embed

def generate_wav(text, num_generated, synthesizer, sampling_rate, embed, debug = False):
	texts = [text]
	embeds = [embed]
	specs = synthesizer.synthesize_spectrograms(texts, embeds)
	spec = specs[0]
	generated_wav = vocoder.infer_waveform(spec, True, False)
	generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
	print("zeros=", np.count_nonzero(generated_wav==0), "\n")
	fpath = "output_%02d.wav" % num_generated
	librosa.output.write_wav(fpath, generated_wav.astype(np.float32), 
			     synthesizer.sample_rate)
	if debug:
		sd.stop()
		sd.play(generated_wav, synthesizer.sample_rate)