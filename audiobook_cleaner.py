#!/usr/bin/env python3
"""
Очистка аудиокниг от фоновой музыки и шумов с использованием MDX-Net моделей.
Использует audio-separator — Python-враппер над моделями UVR5/MDX-Net.

Установка зависимостей:
    pip install audio-separator[gpu]  # для GPU (CUDA)
    pip install audio-separator[cpu]  # для CPU

Модели скачиваются автоматически при первом запуске (~100-300 MB каждая).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import subprocess
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Популярные модели для разных задач
MODELS = {
    # Для отделения вокала/речи от музыки (MDX-Net, .onnx)
    'vocals': 'UVR-MDX-NET-Voc_FT.onnx',
    'vocals_kim': 'Kim_Vocal_2.onnx',
    'vocals_kim1': 'Kim_Vocal_1.onnx',

    # Для удаления шумов (VR, .pth)
    'denoise': 'UVR-DeNoise.pth',
    'denoise_lite': 'UVR-DeNoise-Lite.pth',

    # Для удаления реверберации/эха (VR, .pth)
    'dereverb': 'UVR-DeEcho-DeReverb.pth',
    'dereverb_normal': 'UVR-De-Echo-Normal.pth',

    # Универсальные MDX-Net модели
    'mdx_main': 'UVR_MDXNET_Main.onnx',
    'mdx_kara': 'UVR_MDXNET_KARA_2.onnx',

    # Roformer — лучшее качество, но медленнее
    'roformer': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    'roformer_kim': 'vocals_mel_band_roformer.ckpt',
    'roformer_denoise': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
}

AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.m4a', '.m4b', '.ogg', '.opus')

# Второй шаг (шумоподавление) и его «чистый» стем
DENOISE_MODEL = 'UVR-DeNoise.pth'
DENOISE_CLEAN_STEM = 'No Noise'

# Сепараторы с загруженными моделями: по одному на роль в каждом процессе,
# чтобы модель не грузилась заново на каждую часть.
_SEPARATORS: dict = {}


def format_time(seconds: float) -> str:
    """Форматирует секунды в человекочитаемый вид."""
    if seconds < 60:
        return f"{seconds:.0f}с"
    if seconds < 3600:
        return f"{seconds // 60:.0f}м {seconds % 60:.0f}с"
    return f"{seconds // 3600:.0f}ч {(seconds % 3600) // 60:.0f}м"


def _load_separator_class():
    """Импортирует Separator, сохраняя настоящую причину ошибки импорта."""
    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        raise RuntimeError(
            f"Не удалось импортировать audio-separator: {e}\n"
            "Установите зависимости: pip install -r requirements.txt"
        ) from e
    return Separator


def _get_separator(role: str, model_name: str, output_dir: str, use_gpu: bool,
                   sample_rate: int, segment_size: int, single_stem: Optional[str] = None):
    """
    Сепаратор с загруженной моделью, закэшированный на процесс.

    output_dir задаётся только при создании: библиотека запоминает его
    в экземпляре модели при load_model, поэтому менять его потом бесполезно.
    """
    key = (model_name, output_dir, use_gpu, sample_rate, segment_size, single_stem)
    cached = _SEPARATORS.get(role)
    if cached is None or cached[0] != key:
        separator_cls = _load_separator_class()
        separator = separator_cls(
            output_dir=output_dir,
            output_format='wav',
            sample_rate=sample_rate,
            use_autocast=use_gpu,
            output_single_stem=single_stem,
            mdx_params={
                "hop_length": 1024,
                "segment_size": segment_size,
                "overlap": 0.25,
                "batch_size": 1,
                "enable_denoise": False,
            }
        )
        separator.load_model(model_name)
        _SEPARATORS[role] = (key, separator)
    return _SEPARATORS[role][1]


def _split_stems(model_name: str, model_instance) -> tuple:
    """
    Возвращает (стем с чистым голосом, остальное) для загруженной модели.

    Чистый голос — Vocals, dry или No <что-то> (No Noise, No Echo, No Reverb).
    Имя файла не используется: в нём бывает имя модели вроде Kim_Vocal_2.
    """
    stems = (model_instance.primary_stem_name, model_instance.secondary_stem_name)
    for i, stem in enumerate(stems):
        name = stem.lower()
        if name in ('vocals', 'dry') or name.startswith('no '):
            return stem, stems[1 - i]
    raise RuntimeError(
        f"Модель {model_name}: не понимаю, какой из стемов {stems} содержит голос"
    )


def _check_ffmpeg():
    """Падает с понятной ошибкой, если ffmpeg или ffprobe не в PATH."""
    for tool in ('ffmpeg', 'ffprobe'):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"{tool} не найден в PATH. Установите FFmpeg (см. README): "
                "winget install ffmpeg / sudo apt install ffmpeg / brew install ffmpeg"
            )


def _require(path: Path) -> Path:
    if not path.exists():
        raise RuntimeError(f"audio-separator не создал {path.name} — смотрите его лог выше")
    return path


def _process_chunk_worker(args: tuple) -> dict:
    """
    Обрабатывает одну часть. Используется и в параллельном,
    и в последовательном режиме, и для файла без нарезки.

    Результаты: <name>.clean.wav (голос, после шумоподавления если включено)
    и <name>.instrumental.wav. Часть считается готовой, когда есть clean.wav:
    он появляется последним, переименованием.
    """
    (
        chunk_idx, name, chunk_path, work_dir,
        model_name, denoise, use_gpu, sample_rate, segment_size
    ) = args

    work_dir = Path(work_dir)
    clean_path = work_dir / f"{name}.clean.wav"
    if clean_path.exists():
        return {'idx': chunk_idx, 'skipped': True}

    try:
        separator = _get_separator('main', model_name, str(work_dir),
                                   use_gpu, sample_rate, segment_size)
        voice_stem, rest_stem = _split_stems(model_name, separator.model_instance)
        voice_path = work_dir / f"{name}.voice.wav"
        inst_path = work_dir / f"{name}.instrumental.wav"
        separator.separate(str(chunk_path), custom_output_names={
            voice_stem: voice_path.stem,
            rest_stem: inst_path.stem,
        })
        result_path = _require(voice_path)
        _require(inst_path)

        if denoise:
            denoiser = _get_separator('denoise', DENOISE_MODEL, str(work_dir), use_gpu,
                                      sample_rate, segment_size, single_stem=DENOISE_CLEAN_STEM)
            denoised_path = work_dir / f"{name}.denoised.wav"
            denoiser.separate(str(voice_path),
                              custom_output_names={DENOISE_CLEAN_STEM: denoised_path.stem})
            result_path = _require(denoised_path)

        result_path.replace(clean_path)
        return {'idx': chunk_idx}

    except Exception as e:  # pylint: disable=broad-exception-caught
        return {'idx': chunk_idx, 'error': str(e)}


def get_audio_duration(file_path: str) -> float:
    """Получает длительность аудиофайла в секундах через ffprobe."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', str(file_path)
            ],
            capture_output=True, text=True, check=True
        )
        info = json.loads(result.stdout)
        return float(info['format']['duration'])
    except Exception as e:
        logger.warning(f"Не удалось получить длительность: {e}")
        return 0


def split_audio(input_file: str, output_dir: Path, chunk_duration: int) -> list[Path]:
    """
    Нарезает аудиофайл на части указанной длительности.

    Нарезка идёт во временную папку и переименовывается только целиком,
    поэтому прерванный запуск не оставляет «готовую» неполную нарезку.
    """
    # Проверяем, есть ли уже нарезанные чанки
    existing_chunks = sorted(output_dir.glob("chunk_*.wav"))
    if existing_chunks:
        logger.info(f"Найдено {len(existing_chunks)} ранее нарезанных частей")
        return existing_chunks

    logger.info(f"Нарезка на части по {chunk_duration} секунд...")

    tmp_dir = output_dir.with_name(output_dir.name + '.tmp')
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    subprocess.run([
        'ffmpeg', '-y', '-i', str(input_file),
        '-f', 'segment', '-segment_time', str(chunk_duration),
        '-c:a', 'pcm_s16le',
        str(tmp_dir / "chunk_%04d.wav")
    ], capture_output=True, check=True)
    if output_dir.exists():
        output_dir.rmdir()  # пустая: иначе вернули бы существующие части выше
    tmp_dir.replace(output_dir)

    chunks = sorted(output_dir.glob("chunk_*.wav"))
    logger.info(f"Создано {len(chunks)} частей")
    return chunks


def concat_audio(input_files: list[Path], output_file: Path, output_format: str):
    """Склеивает аудиофайлы в один."""
    if not input_files:
        return

    list_file = output_file.parent / "concat_list.txt"

    # Создаём файл со списком для ffmpeg с абсолютными путями (UTF-8 без BOM!)
    with open(list_file, 'w', encoding='utf-8') as f:
        for file in input_files:
            # Используем прямые слэши для совместимости с ffmpeg
            abs_path = str(file.absolute()).replace('\\', '/')
            # Апостроф внутри '...' экранируется по правилам concat: ' → '\''
            abs_path = abs_path.replace("'", "'\\''")
            f.write(f"file '{abs_path}'\n")

    logger.info(f"Склейка {len(input_files)} файлов...")

    # Параметры кодирования в зависимости от формата
    codec_args = []
    if output_format == 'mp3':
        codec_args = ['-c:a', 'libmp3lame', '-q:a', '2']
    elif output_format == 'flac':
        codec_args = ['-c:a', 'flac']
    else:  # wav
        codec_args = ['-c:a', 'pcm_s16le']

    result = subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(list_file.absolute()),
        *codec_args,
        str(output_file)
    ], capture_output=True, text=True, check=False)  # код проверяем ниже, чтобы показать stderr

    if result.returncode != 0:
        logger.error(f"Ошибка ffmpeg: {result.stderr}")
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

    list_file.unlink()
    logger.info(f"Результат сохранён: {output_file}")


def clean_audiobook(
        input_file: str,
        output_dir: Optional[str] = None,
        model_name: str = 'UVR-MDX-NET-Voc_FT.onnx',
        denoise: bool = True,
        use_gpu: bool = True,
        output_format: str = 'mp3',
        sample_rate: int = 44100,
        segment_size: int = 256,
        chunk_duration: int = 0,
        workers: int = 1,
) -> list[Path]:
    """
    Очищает аудиокнигу от фоновой музыки и шумов.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")

    return clean_audiobook_chunked(
        input_file=input_file,
        output_dir=output_dir,
        model_name=model_name,
        denoise=denoise,
        use_gpu=use_gpu,
        output_format=output_format,
        sample_rate=sample_rate,
        segment_size=segment_size,
        chunk_duration=chunk_duration,
        workers=workers,
    )


def _run_parallel(tasks: list, workers: int):
    """
    Обрабатывает части в отдельных процессах, отдавая результаты по порядку.

    spawn вместо fork: дочерние процессы не наследуют состояние torch/CUDA.
    Если воркер умер (обычно OOM), ProcessPoolExecutor бросает
    BrokenProcessPool — в отличие от mp.Pool, который ждёт вечно.
    """
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        try:
            yield from pool.map(_process_chunk_worker, tasks)
        except BrokenProcessPool:
            logger.error(
                "Процесс-воркер аварийно завершился — чаще всего из-за нехватки памяти. "
                "Уменьшите --workers, --segment-size или --chunk-duration."
            )


def clean_audiobook_chunked(
        input_file: str,
        output_dir: Optional[str] = None,
        model_name: str = 'UVR-MDX-NET-Voc_FT.onnx',
        denoise: bool = True,
        use_gpu: bool = True,
        output_format: str = 'mp3',
        sample_rate: int = 44100,
        segment_size: int = 256,
        chunk_duration: int = 600,
        workers: int = 1,
) -> list[Path]:
    """
    Обрабатывает файл с сохранением прогресса: по частям или целиком
    (chunk_duration=0). Повторный запуск пропускает готовые части.
    """

    # GPU + несколько процессов = OOM на одной карте.
    if workers > 1 and use_gpu:
        logger.warning(
            "--workers > 1 несовместимо с GPU; переключаюсь на CPU. "
            "Для GPU оставьте workers=1."
        )
        use_gpu = False

    # Библиотека сама выбирает CUDA, если она видна, поэтому CPU — это
    # скрыть видеокарту до импорта torch/onnxruntime (воркеры наследуют).
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Понятные ошибки до нарезки, если чего-то не хватает
    _check_ffmpeg()
    _load_separator_class()

    input_path = Path(input_file)

    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_cleaned"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Промежуточные файлы (сохраняются при падении). Папка результатов
    # зависит от параметров, влияющих на результат, чтобы прогон с другой
    # моделью не подхватил чужие части.
    run_tag = f"{Path(model_name).stem}{'+denoise' if denoise else ''}_c{chunk_duration}"
    work_dir = output_path / "_processed" / run_tag
    work_dir.mkdir(parents=True, exist_ok=True)

    duration = get_audio_duration(input_file)
    logger.info(f"Входной файл: {input_path}")
    logger.info(f"Длительность: {duration / 60:.1f} минут")
    logger.info(f"Модель: {model_name}")
    logger.info(f"Промежуточные файлы: {output_path}")

    if chunk_duration > 0:
        logger.info(f"Режим: нарезка по {chunk_duration} секунд")
        chunks = split_audio(input_file, output_path / "_chunks" / f"c{chunk_duration}",
                             chunk_duration)
        if not chunks:
            raise RuntimeError("Не удалось нарезать файл")
        parts = [(chunk.stem, chunk) for chunk in chunks]
    else:
        parts = [("full", input_path)]

    tasks = [
        (i, name, str(path), str(work_dir),
         model_name, denoise, use_gpu, sample_rate, segment_size)
        for i, (name, path) in enumerate(parts, 1)
    ]
    total_chunks = len(tasks)

    start_time = time.time()
    processed_count = 0
    skipped_count = 0

    if workers > 1:
        logger.info(f"Параллельная обработка: {workers} воркеров")
        results = _run_parallel(tasks, workers)
    else:
        results = map(_process_chunk_worker, tasks)

    for result in results:
        idx = result['idx']

        if result.get('skipped'):
            logger.info(f"[{idx}/{total_chunks}] Пропущен (уже обработан)")
            skipped_count += 1
        elif result.get('error'):
            logger.error(f"[{idx}/{total_chunks}] Ошибка: {result['error']}")
        else:
            processed_count += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count
            remaining = avg_time * (total_chunks - processed_count - skipped_count)
            logger.info(f"[{idx}/{total_chunks}] Готово | ETA: {format_time(remaining)}")

    total_time = time.time() - start_time
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Обработка завершена за {format_time(total_time)}")
    logger.info(f"Обработано: {processed_count}, пропущено: {skipped_count}")

    # Готовность проверяем по диску, а не по результатам: так учитываются
    # и упавшие части, и части, чей воркер умер.
    missing = [idx for idx, name, *_ in tasks if not (work_dir / f"{name}.clean.wav").exists()]
    if missing:
        raise RuntimeError(
            f"Не обработаны части {missing} из {total_chunks}. Склейка отменена, "
            "чтобы в книге не было дыр. Перезапустите ту же команду — "
            "готовые части будут пропущены."
        )

    logger.info("Склейка результатов...")
    output_files = []
    vocals_output = output_path / f"{input_path.stem}_(Vocals).{output_format}"
    concat_audio([work_dir / f"{name}.clean.wav" for _, name, *_ in tasks],
                 vocals_output, output_format)
    output_files.append(vocals_output)

    inst_output = output_path / f"{input_path.stem}_(Instrumental).{output_format}"
    concat_audio([work_dir / f"{name}.instrumental.wav" for _, name, *_ in tasks],
                 inst_output, output_format)
    output_files.append(inst_output)

    logger.info("\nГотово!")
    logger.info("Результаты:")
    for f in output_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  - {f} ({size_mb:.1f} MB)")

    logger.info(f"\nПромежуточные файлы сохранены в: {output_path}")
    logger.info("Для очистки удалите папки _chunks и _processed")

    return output_files


def process_directory(
        input_dir: str,
        output_dir: Optional[str] = None,
        extensions: tuple = AUDIO_EXTENSIONS,
        **kwargs
) -> dict[str, list[Path]]:
    """Обрабатывает все аудиофайлы в директории."""
    input_path = Path(input_dir)
    results = {}

    audio_files = [f for f in input_path.iterdir()
                   if f.is_file() and f.suffix.lower() in extensions]
    stems = [f.stem for f in audio_files]

    logger.info(f"Найдено {len(audio_files)} аудиофайлов")

    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Обработка [{i}/{len(audio_files)}]: {audio_file.name}")

        # book.mp3 и book.flac не должны делить одну папку (и её прогресс)
        name = audio_file.stem
        if stems.count(name) > 1:
            name = f"{name}_{audio_file.suffix.lstrip('.').lower()}"
        if output_dir:
            file_output_dir = Path(output_dir) / name
        else:
            file_output_dir = input_path / f"{name}_cleaned"

        try:
            result = clean_audiobook(
                str(audio_file),
                output_dir=str(file_output_dir),
                **kwargs
            )
            results[str(audio_file)] = result

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Ошибка: {e}")
            results[str(audio_file)] = []

    return results


def list_models():
    """Выводит список доступных моделей."""
    print("\nПредустановленные модели:")
    print("-" * 60)
    for key, model in MODELS.items():
        print(f"  {key:15} -> {model}")

    print("\nРекомендации:")
    print("-" * 60)
    print("  Аудиокниги с фоновой музыкой:")
    print("    --model vocals  (UVR-MDX-NET-Voc_FT.onnx)")
    print("    --model vocals_kim  (Kim_Vocal_2.onnx)")
    print()
    print("  Удаление шумов (шипение, гул):")
    print("    --model denoise  (UVR-DeNoise.pth)")
    print()
    print("  Удаление эха/реверберации:")
    print("    --model dereverb  (UVR-DeEcho-DeReverb.pth)")
    print()
    print("  Лучшее качество (медленнее):")
    print("    --model roformer")


def main():
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(
        description="Очистка аудиокниг от фоновой музыки и шумов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python audiobook_cleaner.py audiobook.mp3
  python audiobook_cleaner.py audiobook.mp3 --model vocals_kim
  python audiobook_cleaner.py audiobook.mp3 --no-denoise
  python audiobook_cleaner.py audiobook.mp3 --chunk-duration 300  # большие файлы
  python audiobook_cleaner.py --dir /path/to/audiobooks
  python audiobook_cleaner.py audiobook.mp3 --cpu
  python audiobook_cleaner.py --list-models
        """
    )

    parser.add_argument('input', nargs='?', help='Входной аудиофайл')
    parser.add_argument('--dir', '-d', help='Обработать директорию')
    parser.add_argument('--output', '-o', help='Выходная директория')
    parser.add_argument('--model', '-m', default='vocals',
                        help='Модель (default: vocals)')
    parser.add_argument('--no-denoise', action='store_true',
                        help='Без дополнительного шумоподавления')
    parser.add_argument('--cpu', action='store_true',
                        help='Использовать CPU')
    parser.add_argument('--format', '-f', default='mp3',
                        choices=['mp3', 'wav', 'flac'],
                        help='Формат выхода (default: mp3)')
    parser.add_argument('--sample-rate', '-sr', type=int, default=44100,
                        help='Sample rate (default: 44100)')
    parser.add_argument('--segment-size', '-s', type=int, default=256,
                        help='Размер сегмента MDX-модели (default: 256; другое значение '
                             'переводит ONNX-модель на более медленный PyTorch)')
    parser.add_argument('--chunk-duration', '-c', type=int, default=0,
                        help='Нарезать на части по N секунд (default: 0 = выкл)')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Количество параллельных воркеров для CPU (default: 1)')
    parser.add_argument('--list-models', '-l', action='store_true',
                        help='Список моделей')

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.input and not args.dir:
        parser.print_help()
        sys.exit(1)

    model = MODELS.get(args.model, args.model)

    common_args = {
        'model_name': model,
        'denoise': not args.no_denoise,
        'use_gpu': not args.cpu,
        'output_format': args.format,
        'sample_rate': args.sample_rate,
        'segment_size': args.segment_size,
        'chunk_duration': args.chunk_duration,
        'workers': args.workers,
    }

    if args.dir:
        results = process_directory(args.dir, output_dir=args.output, **common_args)
        failed = [f for f, outputs in results.items() if not outputs]
        if failed:
            logger.error(f"Не обработано файлов: {len(failed)} из {len(results)}")
            sys.exit(1)
    else:
        try:
            clean_audiobook(args.input, output_dir=args.output, **common_args)
        except (RuntimeError, FileNotFoundError) as e:
            logger.error(str(e))
            sys.exit(1)


if __name__ == '__main__':
    main()
