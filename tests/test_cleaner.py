"""Регрессионные тесты на баги из ревью 2026-10-16 (нумерация как в ревью)."""
# Хелперы тестов говорят сами за себя; фикстура fake_separator в аргументах
# иногда нужна только ради подмены audio_separator.
# pylint: disable=missing-function-docstring,unused-argument
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from conftest import make_wav
import audiobook_cleaner as ac

DUR = 5.0      # секунд во входном файле
CHUNK = 2      # нарезка по 2 с → 3 части


def gain(output, source):
    """Отношение RMS результата к RMS входа: по нему видно, какой стем склеен."""
    out, _ = sf.read(str(output))
    src, _ = sf.read(str(source))
    n = min(len(out), len(src))
    return float(np.sqrt(np.mean(out[:n] ** 2)) / np.sqrt(np.mean(src[:n] ** 2)))


def duration(path):
    return sf.info(str(path)).duration


def run(src, **kwargs):
    params = {'chunk_duration': CHUNK, 'output_format': 'wav', 'use_gpu': False}
    params.update(kwargs)
    return ac.clean_audiobook(str(src), **params)


def vocals_of(results):
    return next(f for f in results if '(Vocals)' in f.name)


def instrumental_of(results):
    return next(f for f in results if '(Instrumental)' in f.name)


# --- P0 ---------------------------------------------------------------------

def test_chunked_sequential_gives_full_denoised_book(tmp_path):
    """#1: последовательный режим по частям работает и применяет шумоподавление."""
    src = make_wav(tmp_path / 'book.wav', DUR)
    results = run(src)
    vocals = vocals_of(results)
    assert duration(vocals) == pytest.approx(DUR, abs=0.01)
    assert gain(vocals, src) == pytest.approx(0.5 * 0.8, rel=0.01)   # Vocals → No Noise
    assert gain(instrumental_of(results), src) == pytest.approx(0.25, rel=0.01)


def test_single_file_mode_applies_denoise(tmp_path):
    """#2: без нарезки шумоподавление тоже выполняется, независимо от текущей папки."""
    src = make_wav(tmp_path / 'in' / 'book.wav', DUR)
    results = run(src, chunk_duration=0)
    vocals = vocals_of(results)
    assert duration(vocals) == pytest.approx(DUR, abs=0.01)
    assert gain(vocals, src) == pytest.approx(0.5 * 0.8, rel=0.01)


def test_resume_reuses_clean_parts_and_gives_same_file(tmp_path, fake_separator):
    """#3: повторный запуск ничего не пересчитывает и не подмешивает стем Noise."""
    src = make_wav(tmp_path / 'book.wav', DUR)
    first = vocals_of(run(src)).read_bytes()
    fake_separator.calls.clear()

    second = vocals_of(run(src))
    assert not fake_separator.calls
    assert second.read_bytes() == first
    assert gain(second, src) == pytest.approx(0.4, rel=0.01)


def test_failed_part_aborts_instead_of_leaving_a_gap(tmp_path, fake_separator):
    """#4: упавшая часть не даёт склеить книгу с дырой; перезапуск досчитывает только её."""
    src = make_wav(tmp_path / 'book.wav', DUR)
    fake_separator.fail_on = 'chunk_0001'
    with pytest.raises(RuntimeError, match='[Нн]е обработан'):
        run(src)
    assert not list(tmp_path.glob('book_cleaned/*(Vocals)*'))

    fake_separator.fail_on = None
    fake_separator.calls.clear()
    vocals = vocals_of(run(src))
    assert duration(vocals) == pytest.approx(DUR, abs=0.01)
    # разделение + шумоподавление только для упавшей части
    assert len(fake_separator.calls) == 2
    assert all('chunk_0001' in path for _, path in fake_separator.calls)


def test_leftover_of_interrupted_split_is_ignored(tmp_path):
    """#4: недорезанная папка от прерванного запуска не считается готовой нарезкой."""
    src = make_wav(tmp_path / 'book.wav', DUR)
    chunks_dir = tmp_path / 'book_cleaned' / '_chunks'
    for leftover in (chunks_dir / f'c{CHUNK}.tmp', chunks_dir / f'c{CHUNK}'):
        leftover.mkdir(parents=True, exist_ok=True)
    make_wav(chunks_dir / f'c{CHUNK}.tmp' / 'chunk_0000.wav', 1.0)

    vocals = vocals_of(run(src))
    assert duration(vocals) == pytest.approx(DUR, abs=0.01)


def test_empty_result_from_library_is_an_error(tmp_path, fake_separator):
    """#14: библиотека «обработала» файл, но ничего не записала → ошибка, а не «Готово!»."""
    src = make_wav(tmp_path / 'book.wav', DUR)
    fake_separator.write_nothing = True
    with pytest.raises(RuntimeError):
        run(src)


def test_cli_exits_nonzero_on_failure(tmp_path, monkeypatch, fake_separator):
    src = make_wav(tmp_path / 'book.wav', DUR)
    fake_separator.fail_on = 'chunk_0002'
    monkeypatch.setattr(sys, 'argv', ['audiobook_cleaner.py', str(src), '-c', str(CHUNK), '--cpu'])
    with pytest.raises(SystemExit) as exc:
        ac.main()
    assert exc.value.code == 1


def test_concat_handles_apostrophes_in_path(tmp_path):
    """#6: апостроф в пути не ломает склейку."""
    folder = tmp_path / "O'Brien books"
    parts = [make_wav(folder / f"Book's part {i}.wav", 1.0) for i in (1, 2)]
    out = folder / "Book's Title — Глава 1.wav"
    ac.concat_audio(parts, out, 'wav')
    assert duration(out) == pytest.approx(2.0, abs=0.01)


# --- P1 ---------------------------------------------------------------------

@pytest.mark.parametrize('model, clean_gain, rest_gain', [
    ('Kim_Vocal_2.onnx', 0.5, 0.25),          # «Vocal» в имени модели
    ('UVR_MDXNET_KARA_2.onnx', 0.5, 0.25),    # основной стем — Instrumental
    ('UVR-DeNoise.pth', 0.8, 0.2),            # стемы No Noise / Noise
])
def test_clean_stem_is_chosen_by_role_not_by_filename(tmp_path, model, clean_gain, rest_gain):
    """#7, #8: стемы выбираются по роли, а не по подстроке в имени файла."""
    src = make_wav(tmp_path / 'book.wav', DUR)
    results = run(src, model_name=model, denoise=False)
    assert gain(vocals_of(results), src) == pytest.approx(clean_gain, rel=0.01)
    assert gain(instrumental_of(results), src) == pytest.approx(rest_gain, rel=0.01)


def test_unknown_stems_fail_loudly(tmp_path, caplog):
    src = make_wav(tmp_path / 'book.wav', DUR)
    with pytest.raises(RuntimeError):
        run(src, model_name='weird.ckpt', denoise=False)
    assert "('Bass', 'Drums')" in caplog.text


def test_changing_model_reprocesses_parts(tmp_path, fake_separator):
    """#9: результаты другой модели не переиспользуются."""
    src = make_wav(tmp_path / 'book.wav', DUR)
    run(src, model_name='model_a.onnx', denoise=False)
    fake_separator.calls.clear()
    run(src, model_name='model_b.onnx', denoise=False)
    assert [m for m, _ in fake_separator.calls] == ['model_b.onnx'] * 3


def test_same_stem_files_in_directory_do_not_collide(tmp_path, fake_separator):
    """#10: book.wav и book.flac в одной папке обрабатываются оба и раздельно."""
    src_dir = tmp_path / 'in'
    make_wav(src_dir / 'book.wav', DUR)
    sf.write(str(src_dir / 'book.flac'), sf.read(str(src_dir / 'book.wav'))[0] * 0.5, 8000)

    results = ac.process_directory(str(src_dir), output_dir=str(tmp_path / 'out'),
                                   chunk_duration=CHUNK, output_format='wav', use_gpu=False,
                                   denoise=False)
    outputs = [vocals_of(r) for r in results.values()]
    assert len(outputs) == 2 and len({o.parent for o in outputs}) == 2
    gains = sorted(round(gain(o, src_dir / 'book.wav'), 2) for o in outputs)
    assert gains == [0.25, 0.5]


def test_cpu_flag_hides_gpu(tmp_path, monkeypatch):
    """#11: --cpu реально прячет CUDA от torch и onnxruntime."""
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    run(make_wav(tmp_path / 'book.wav', DUR), use_gpu=False, denoise=False)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''


def test_default_segment_size_matches_mdx_dim_t(tmp_path, monkeypatch, fake_separator):
    """#12: 64 заставляло библиотеку конвертировать ONNX-модель в PyTorch."""
    src = make_wav(tmp_path / 'book.wav', DUR)
    monkeypatch.setattr(sys, 'argv', ['audiobook_cleaner.py', str(src), '--cpu', '--no-denoise'])
    ac.main()
    assert {kw['mdx_params']['segment_size'] for kw in fake_separator.init_kwargs} == {256}


def test_m4b_is_picked_up_in_directory_mode(tmp_path):
    """#13: .m4b — основной формат аудиокниг."""
    src_dir = tmp_path / 'in'
    wav = make_wav(tmp_path / 'book.wav', DUR)
    src_dir.mkdir()
    subprocess.run(['ffmpeg', '-v', 'quiet', '-i', str(wav), '-c:a', 'aac', '-f', 'ipod',
                    str(src_dir / 'book.m4b')], check=True)
    results = ac.process_directory(str(src_dir), chunk_duration=CHUNK, output_format='wav',
                                   use_gpu=False, denoise=False)
    assert [len(r) for r in results.values()] == [2]


def test_import_error_keeps_original_reason(tmp_path, monkeypatch):
    """#14: транзитивная ошибка импорта не выдаётся за «не установлен»."""
    monkeypatch.setitem(sys.modules, 'audio_separator.separator', None)
    with pytest.raises(RuntimeError, match='audio_separator.separator'):
        run(make_wav(tmp_path / 'book.wav', DUR))


# --- гигиена ----------------------------------------------------------------

@pytest.mark.parametrize('missing', ['ffmpeg', 'ffprobe'])
def test_missing_ffmpeg_is_reported_before_any_work(tmp_path, monkeypatch, fake_separator, missing):
    """I6: без ffmpeg/ffprobe — понятная ошибка до нарезки, а не FileNotFoundError из subprocess."""
    src = make_wav(tmp_path / 'book.wav', DUR)
    real_which = ac.shutil.which
    monkeypatch.setattr(ac.shutil, 'which', lambda cmd: None if cmd == missing else real_which(cmd))
    with pytest.raises(RuntimeError, match=f'{missing}.*не найден'):
        run(src)
    assert not (tmp_path / 'book_cleaned').exists()
    assert not fake_separator.calls
