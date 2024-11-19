from .model import VoiceConvertModel
import sys
import logging
from pydub import AudioSegment
from tqdm import tqdm
from typing import Optional
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import os
import shutil
import argparse
import concurrent.futures
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def convert_folder(
    root_path: Path | str,
    model_path: Path | str,
    model_name: str,
    f0_up_key: int,
    f0_method: str = 'rmvpe',
    overwrite_converted: bool = False,
    format: str = 'ogg',
    target_sr: int = 16000,
    converted_path: Optional[Path | str] = None
):
    """
    Convert a folder of audio files using a trained VoiceConvertModel.
    """
    model = VoiceConvertModel(
        model_name, torch.load(model_path, map_location="cpu"))

    root_path = Path(root_path)

    # Safety checks
    if not root_path.exists():
        raise FileNotFoundError(f'{root_path} does not exist')
    if root_path.is_file():
        raise ValueError(f'{root_path} is a file. Please provide a folder')

    # if converted_path is None, save the converted files in the same directory as the wav files
    if converted_path is None:
        converted_path = root_path
    else:
        converted_path = Path(converted_path)
        converted_path.mkdir(exist_ok=True)

    # (you like that walrus operator? pretty cool huh)
    files  = sum([list(root_path.glob(f'*.{ext}')) for ext in ['wav', 'flac', 'ogg']], [])
    
    # do not convert if the file is already converted unless overwrite_converted is True
    files = list(filter(lambda wav: ('_original' in wav.name) and ((not '_converted' in wav.name) or (overwrite_converted)), files))
    
    for wav in (progress := tqdm(files, unit="file")):

        # if ('_converted' in wav.name) and (not overwrite_converted):
        #     progress.set_description(
        #         f'{wav.name} is already converted. Skipping...')
        #     continue

        # never overwrite resampled files
        # if '_original' not in wav.name:
        #     progress.set_description(f'{wav.name} not original. Skipping...')
        #     continue

        # get the extension and create the converted path
        extension = wav.name.split('.')[-1]
        converted_wav = converted_path / \
            wav.name.replace(f'.{extension}', f'_converted.{format}').replace(
                '_original', '')

        # this is the case where the file is already converted and is not contained in the root_path
        if converted_wav.exists() and not overwrite_converted:
            # progress.set_description(
            #     f'{converted_wav} already exists. Skipping...')
            continue

        # convert and save (explicitly typed to avoid misconstruing the audio type for something else)
        out: np.ndarray = np.array(model.single(sid=1,
                                         input_audio=str(wav),
                                         embedder_model_name='hubert_base',
                                         embedding_output_layer='auto',
                                         f0_up_key=f0_up_key,
                                         f0_file=None,
                                         f0_method=f0_method,
                                         auto_load_index=False,
                                         faiss_index_file=None,
                                         index_rate=None,
                                         f0_relative=True,
                                         output_dir='out')
                                            .get_array_of_samples())
        # out = out.unsqueeze(0)    
        # from Int16 to float32               
        # out = out.astype(np.float32) / 32768.0   
        out = librosa.util.buf_to_float(out)
        out = librosa.resample(out, orig_sr=model.tgt_sr, target_sr=target_sr)
        subtype = "vorbis" if format == "ogg" else None
        sf.write(converted_wav, out, target_sr, format=format, subtype=subtype)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", help="Path to dataset to be processed", type=str, required=True, default='.LibriSpeech/')
    # parser.add_argument("--train_set_path", help="Path to dataset to be processed",
    #                     type=str, default='./LibriSpeech/train-clean-100')
    # parser.add_argument("--dev_set_path", help="path to the dev set",
    #                     type=str, default='./LibriSpeech/dev-clean/')
    # parser.add_argument(
    #     '--flatten', help='Flatten LibriSpeech datasets before converting. This will also loop through each speaker in the dataset instead of a single folder', action='store_true')
    # parser.add_argument(
    #     '--clean_flattened', help='Remove the flattened LibriSpeech dataset after processing TBA', action='store_true')
    parser.add_argument("--out_path", help="Path to save the new dataset folders",
                        type=str, required=True, default='./LibriSpeech_processed/')
    parser.add_argument(
        "--model_path", help="Path to RVC model to create `_converted` data", type=str)
    parser.add_argument(
        "--f0_up_key", help="Pitch adjust for conversion", type=int, default=12)
    parser.add_argument(
        "--f0_method", help="f0 method for pitch extraction", type=str)
    parser.add_argument("--model_name", help="Name to be added to output filenames. If `None`, the filename of --model_path will be used",
                        type=str, required=False, default=None)
    # parser.add_argument(
    #     "--target_sr", help="Sample rate to resample the dataset into", type=int, default=16000)
    # parser.add_argument(
    #     "--val_percent", help="Percent of the dataset to use for validation", type=float, default=0.1)
    # parser.add_argument(
    #     "--random_seed", help="Random seed for splitting the dataset", type=int, default=42)
    
    args = parser.parse_args()

    root_path = Path(args.root_path)
    out_path = Path(args.out_path)
    model_path = Path(args.model_path)
    model_name = args.model_name
    if model_name is None:
        model_name = model_path.name.split('.')[0]

    f0_up_key: int = args.f0_up_key
    f0_method: str = args.f0_method

    out_path.mkdir(exist_ok=True)
    for folder in ['dev', 'val', 'train']:
        convert_folder(
            root_path=Path(f'./{root_path}/{folder}'),
            model_path=Path(f'./{model_path}'),
            model_name=model_name,
            f0_up_key=f0_up_key,
            f0_method=f0_method,
            overwrite_converted=False,
            format='ogg',
            converted_path=Path(f'./{out_path}/{folder}')
        )


if __name__ == "__main__":
    main()