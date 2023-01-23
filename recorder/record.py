from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import time
import wave
from pvrecorder import PvRecorder
import IPython
import struct
import ipywidgets as widgets
import wandb
import os
os.environ["WANDB_SILENT"] = "true"


class Response:
    def __init__(self, frame):
        self.frame_length = frame
        self.path = './test.wav'
        self.t_delta = 1
        self.fs = 16000
        self.root = Path('./')
        self.run_id = None

    def set_run(self, project: str, entity: str, group: str, name: str):
        self.record_group = group
        self.name = name
        api = wandb.Api()
        for run in api.runs(f'{entity}/{project}'):
            if run.name == name:
                self.run_id = run.id
        if self.run_id:
            self.run = wandb.init(resume='must',
                                  id=self.run_id,
                                  entity=entity,
                                  project=project,
                                  group=self.record_group,
                                  name=self.name)
        else:
            self.run = wandb.init(entity=entity,
                                  project=project,
                                  group=self.record_group,
                                  name=self.name)
            self.run_id = self.run.id

    def set_time(self, delta_sec: int):
        self.t_delta = delta_sec

    def createdirs(self):
        if not self.dir.exists():
            self.dir.mkdir(exist_ok=True)
        for sound_class in self.class_dirs:
            sound_dir = self.dir/sound_class
            sound_dir.mkdir(exist_ok=True)

    def get_classes(self, sound_classes: tuple[str, str]):
        self.dir = self.root/'data'
        self.paths = [f for f in self.dir.rglob("*.wav")]
        self.class_dirs = sound_classes
        self.createdirs()
        self.classes = {sound: {'class': idx, 'count': len(
            list((self.dir/sound).iterdir()))} for idx, sound in enumerate(sound_classes)}
        self.set_widgets()

    def set_widgets(self):

        actions = [
            widgets.Button(description=f'record {name}') for name, entry in self.classes.items()]
        for act in actions:
            act.on_click(self.record)

        save = widgets.Button(description='save \U0001F4BE')
        play = widgets.Button(description='play ▶️')

        delete = widgets.Button(description='delete \U0000274C')
        save.on_click(self.save)
        play.on_click(self.play)
        delete.on_click(self.drop_recording)
        self.out = widgets.Output()
        actions += [play, save, delete, self.out]
        acts = tuple(actions)
        self.vbox = widgets.VBox(children=acts)
        display(self.vbox)

    def record(self, button: widgets.Button):
        with self.out:
            key = button.description.split(' ')[1]
            self.state = key
            self.classes[key]['count'] += 1
            self.classes[key][f'record {self.classes[key]["count"]}'] = np.array([
            ])
            self.recorder = PvRecorder(
                device_index=0, frame_length=self.frame_length)
            self.recorder.start()
            t_0 = time.time()
            record = np.array([]).astype(np.int16)
            while time.time()-t_0 < self.t_delta:
                frame = self.recorder.read()
                record = np.append(record, np.array(frame)).astype(np.int16)
            self.recorder.stop()
            self.recorder.delete()
            self.classes[key][f'record {self.classes[key]["count"]}'] = record

    def save(self, _):
        path = self.root/'data'
        path.mkdir(exist_ok=True)
        with self.out:
            for state in self.classes:
                self.class_dir = path/state
                self.class_dir.mkdir(exist_ok=True)
                for key in self.classes[state]:
                    if 'record' in key:
                        rec = self.classes[state][key]
                        fid = self.class_dir/f'{state}_{key}.wav'
                        with wave.open(str(fid), 'w') as f:
                            f.setparams((1, 2, self.fs, 512, "NONE", "NONE"))
                            f.writeframes(struct.pack("h" * len(rec), *rec))
                self.log_wandb_sound()

    def log_wandb_sound(self):
        artifact = wandb.Artifact(
            name=self.class_dir.name, type='recorded_sound_data')
        artifact.add_dir(self.class_dir)
        self.run.log_artifact(artifact)
        path = self.root/'data'
        new_paths = [f for f in path.rglob("*.wav") if f not in self.paths]
        for pth in new_paths:
            self.run.log({pth.parent.name: wandb.Audio(str(pth))})
            self.paths.append(new_paths)
        print(
            f'logging to run name {self.run.name}, id {self.run.name}, project = {self.run.project}')

    def play(self, _):
        try:
            count = self.classes[self.state]['count']
            for key in self.classes[self.state]:
                if 'record' in key:
                    sd.play(self.classes[self.state][key], self.fs)
                    sd.wait()
        except (KeyError, AttributeError):
            print('no recordings')

    def drop_recording(self, _):
        try:
            if self.classes[self.state]['count'] != 0:
                self.classes[self.state].popitem()
                self.classes[self.state]['count'] -= 1
        except (KeyError, AttributeError):
            print('no recordings')

    def finish(self):
        self.run.finish()
