ARG DLC_BASE_IMAGE=deeplabcut/deeplabcut:3.0.0-core-cuda11.8
FROM ${DLC_BASE_IMAGE}
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY train_dlc.py processor_utils.py /app/
COPY entrypoint.sh /app/entrypoint.sh
ENTRYPOINT ["sh", "/app/entrypoint.sh"]
