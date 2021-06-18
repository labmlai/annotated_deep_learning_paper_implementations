import json
import pathlib
from typing import Dict

from labml import experiment
from labml_nn.cfr import InfoSet


class InfoSetSaver(experiment.ModelSaver):
    def __init__(self, infosets: Dict[str, InfoSet]):
        self.infosets = infosets

    def save(self, checkpoint_path: pathlib.Path) -> any:
        data = {key: infoset.to_dict() for key, infoset in self.infosets.items()}
        file_name = f"infosets.json"

        with open(str(checkpoint_path / file_name), 'w') as f:
            f.write(json.dumps(data))

        return file_name

    def load(self, checkpoint_path: pathlib.Path, file_name: str):
        with open(str(checkpoint_path / file_name), 'w') as f:
            data = json.loads(f.read())

        for key, d in data.items():
            self.infosets[key] = InfoSet.from_dict(d)
