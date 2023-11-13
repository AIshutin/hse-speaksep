import logging
import random
from typing import List
import os
import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path
from speaksep.utils.parse_config import ConfigParser
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MixtureDataset(Dataset):
    def __init__(
            self,
            path: Path,
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None,
            max_audio_length=None,
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.log_spec = config_parser["preprocessing"]["log_spec"]
        
        path = Path(path)
        if os.path.exists(path / "index.json"):
            with open(path / "index.json") as file:
                index = json.load(file)
        else:
            index = self.make_index(path)
            with open(path / "index.json", 'w') as file:
                json.dump(index, file)

        index = self._filter_records_from_dataset(index, max_audio_length, limit)
        self._index: List[dict] = index
        logger.info(f"N_classes ({path}): {self.get_n_classes()} Length: {len(self)}")

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["target_path"]
        speaker_id = data_dict["speaker_id"]
        target_wave = self.load_audio(data_dict['target_path'])
        ref_wave = self.load_audio(data_dict['ref_path'])
        mixed_wave = self.load_audio(data_dict['mixed_path'])
        
        t_wave = self.process_wave(target_wave)
        m_wave = self.process_wave(mixed_wave)
        r_wave = self.process_wave(ref_wave)

        return {
            "audio_path": audio_path,
            "target_wave": t_wave,
        #    "target_spectrogram": t_spectrogram,
            "duration": t_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "ref_wave": r_wave,
        #    "ref_spectrogram": r_spectrogram,
            "mixed_wave": m_wave,
        #    "mixed_spectrogram": m_spectrogram,
            "speaker_id": speaker_id
        }
    
    def get_n_classes(self):
        mx = 0
        for el in self._index:
            mx = max(mx, el['speaker_id'])
        return mx + 1

    def make_index(self, path):
        index = []
        for audio in tqdm(os.listdir(path)):
            try:
                meta, atype = audio.split('-')
                dtype = atype[:atype.find('.')]
                
                if dtype == "target":
                    audio_wave = self.load_audio(path / audio)
                    audio_length = audio_wave.view(-1).shape[0] / self.config_parser["preprocessing"]["sr"]
                    index.append(
                        {
                            "duration":    audio_length,
                            "target_path": str(path / audio),
                            "ref_path":    str(path / (meta + '-' + atype.replace('target', 'ref'))),
                            "mixed_path":  str(path / (meta + '-' + atype.replace('target', 'mixed'))),
                            "speaker_id":  int(audio.split('_')[0])
                        }
                    )
            except:
                pass # probably index file file
        speaker_ids = set()
        for el in index:
            speaker_ids.add(el['speaker_id'])
        speaker_ids = sorted(speaker_ids)
        speaker_ids = {el: i for i, el in enumerate(speaker_ids)}
        for i, el in enumerate(index):
            index[i]['speaker_id'] = speaker_ids[el['speaker_id']]
        return index

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            return audio_tensor_wave
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            if self.spec_augs is not None:
                audio_tensor_spec = self.spec_augs(audio_tensor_spec)

        with torch.no_grad():
            if self.log_spec:
                audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
            return audio_tensor_wave, audio_tensor_spec
            

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["duration"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)

        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index