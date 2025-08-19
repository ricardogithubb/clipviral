# ViralClip Generator (Flask, single-file backend)

App web para gerar clipes 9:16 de 60s usando IA leve para detecção de refrões (áudio) e auto-reframe para o protagonista (face tracking).

## Requisitos
- Python 3.10+
- FFmpeg instalado (ffmpeg/ffprobe no PATH)
- Pip install:
  ```bash
  pip install -r requirements.txt
  ```

## Executar
```bash
python app.py
# abre em http://localhost:8000
```

## Fluxo
1. Faça upload do vídeo
2. Clique em "Gerar cortes" para obter propostas (60s)
3. Ajuste inícios/fins na lista (se desejar)
4. Clique em "Renderizar 9:16" para processar
5. Baixe o ZIP

## Pastas
- `uploads/` – vídeos enviados
- `outputs/<video_id>/` – clipes renderizados
- `outputs/<video_id>_clips.zip` – pacote ZIP

## Observações
- Se `librosa`/`opencv` não estiverem instalados, o app ainda funciona com heurísticas básicas (cortes sequenciais e centro fixo). 
- Para produção, use um servidor WSGI (gunicorn/uwsgi) e um worker de fila (RQ/Celery) se precisar escalar.
