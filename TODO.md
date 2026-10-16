# TODO — улучшения

Несрочные улучшения качества кода и UX. Критичные баги (B1–B5) уже пофикшены,
здесь — то, что стоит сделать, но не горит.

## I4. Гигиена pylint вместо band-aid
Сейчас CI зелёный за счёт `--fail-under=8.0` (сейчас 8.5). Лучше:
- добавить `.pylintrc` с `disable=logging-fstring-interpolation` (W1203 —
  осознанный стиль логов) и `import-outside-toplevel` (C0415 — намеренный
  ленивый импорт опциональной `audio-separator`);
- поднять `--fail-under` до ~9.0.

## I6. Проверка ffmpeg
В рантайме проверять наличие `ffmpeg`/`ffprobe` (`shutil.which`) и падать
с понятной ошибкой, а не с сырым `FileNotFoundError` из subprocess.
