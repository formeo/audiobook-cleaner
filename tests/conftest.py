"""
Фейковый audio_separator для тестов: без нейросетей, но с тем же поведением,
на котором ломался скрипт.

- output_dir фиксируется в экземпляре модели при load_model (как в реальной
  библиотеке), поэтому присваивание separator.output_dir после загрузки
  ничего не меняет;
- separate() возвращает голые имена файлов, без директории;
- имена по умолчанию: <input>_(<Stem>)_<model>.wav, как в библиотеке;
- custom_output_names и output_single_stem сравниваются без учёта регистра.

Стемы — масштабированный вход, чтобы по громкости результата было видно,
какой именно стем попал в итоговый файл.
"""
# Фейки повторяют интерфейс библиотеки, а не документируют его.
# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (основной стем, второй стем) — как primary/secondary_stem_name у модели
MODEL_STEMS = {
    'UVR_MDXNET_KARA_2.onnx': ('Instrumental', 'Vocals'),
    'UVR-DeNoise.pth': ('Noise', 'No Noise'),
    'weird.ckpt': ('Bass', 'Drums'),
}
DEFAULT_STEMS = ('Vocals', 'Instrumental')

# Во сколько раз стем тише входа — по этому тесты узнают стем в результате
STEM_GAIN = {
    'vocals': 0.5,
    'instrumental': 0.25,
    'no noise': 0.8,
    'noise': 0.2,
}


class FakeModel:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        stems = MODEL_STEMS.get(model_name, DEFAULT_STEMS)
        self.primary_stem_name, self.secondary_stem_name = stems


class FakeSeparator:
    calls = []          # (model_name, input_path)
    fail_on = None      # подстрока пути, на которой separate() падает
    write_nothing = False  # имитация «Failed to process file» внутри библиотеки

    init_kwargs = []    # параметры конструктора по каждому созданному сепаратору

    def __init__(self, output_dir=None, output_format='WAV', output_single_stem=None, **kwargs):
        FakeSeparator.init_kwargs.append(kwargs)
        self.output_dir = output_dir
        self.output_format = output_format
        self.output_single_stem = output_single_stem
        self.model_instance = None

    def load_model(self, model_filename):
        self.model_instance = FakeModel(model_filename, self.output_dir)

    def separate(self, audio_file_path, custom_output_names=None):
        model = self.model_instance
        FakeSeparator.calls.append((model.model_name, str(audio_file_path)))
        if FakeSeparator.fail_on and FakeSeparator.fail_on in str(audio_file_path):
            raise RuntimeError(f"fake failure on {audio_file_path}")
        if FakeSeparator.write_nothing:
            return []

        data, sr = sf.read(str(audio_file_path))
        names = {k.lower(): v for k, v in (custom_output_names or {}).items()}
        base = Path(audio_file_path).stem
        model_stem = Path(model.model_name).stem
        ext = self.output_format.lower()

        written = []
        for stem in (model.primary_stem_name, model.secondary_stem_name):
            if self.output_single_stem and self.output_single_stem.lower() != stem.lower():
                continue
            name = names.get(stem.lower(), f"{base}_({stem})_{model_stem}")
            filename = f"{name}.{ext}"
            gain = STEM_GAIN.get(stem.lower(), 0.1)
            sf.write(str(Path(model.output_dir) / filename), data * gain, sr)
            written.append(filename)
        return written


@pytest.fixture(autouse=True)
def fake_separator(monkeypatch):
    package = types.ModuleType('audio_separator')
    module = types.ModuleType('audio_separator.separator')
    module.Separator = FakeSeparator
    package.separator = module
    monkeypatch.setitem(sys.modules, 'audio_separator', package)
    monkeypatch.setitem(sys.modules, 'audio_separator.separator', module)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')

    FakeSeparator.calls = []
    FakeSeparator.init_kwargs = []
    FakeSeparator.fail_on = None
    FakeSeparator.write_nothing = False

    import audiobook_cleaner
    cache = getattr(audiobook_cleaner, '_SEPARATORS', None)
    if cache is not None:
        cache.clear()
    yield FakeSeparator
    if cache is not None:
        cache.clear()


def make_wav(path, seconds, sr=8000):
    """Синус с нарастающей амплитудой: у каждого куска своя громкость."""
    t = np.arange(int(seconds * sr)) / sr
    data = 0.5 * np.sin(2 * np.pi * 220 * t) * (0.2 + 0.8 * t / seconds)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr)
    return path
