FROM deeplabcut/deeplabcut:3.0.0-core-cuda11.8

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

COPY train_dlc.py /app/train_dlc.py

ENTRYPOINT ["python", "/app/train_dlc.py"]
