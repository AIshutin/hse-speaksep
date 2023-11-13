from pathlib import Path
from speaksep.utils.parse_config import ConfigParser
from .mixture_dataset import *


class CustomDirDataset(MixtureDataset):
    def __init__(self, path: Path, config_parser: ConfigParser, wave_augs=None, spec_augs=None, limit=None, max_audio_length=None):
        
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
        
        super().__init__(path, config_parser, wave_augs, spec_augs, limit, max_audio_length)

    def make_index(self, path):
        index = []
        for audio in tqdm(os.listdir(path / "targets")):
            try:
                meta, atype = audio.split('-')
                dtype = atype[:atype.find('.')]
                
                if dtype == "target":
                    audio_wave = self.load_audio(path / "targets" / audio)
                    audio_length = audio_wave.view(-1).shape[0] / self.config_parser["preprocessing"]["sr"]
                    index.append(
                        {
                            "duration":    audio_length,
                            "target_path": str(path / "targets" / audio),
                            "ref_path":    str(path / "refs" / (meta + '-' + atype.replace('target', 'ref'))),
                            "mixed_path":  str(path / "mix" / (meta + '-' + atype.replace('target', 'mixed'))),
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