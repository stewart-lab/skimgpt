FROM python:3.11.3

WORKDIR /app

COPY requirements.txt /app
COPY main.py /app/
COPY config.json /app
COPY ./src/km_output.tsv /app
COPY ~/.cache/huggingface/hub /app

RUN pip install -r /requirements.txt
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

ENTRYPOINT [ "executable" ] ["python", "./main.py"]
