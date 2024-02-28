FROM python:3.11.3

WORKDIR /app

COPY requirements.txt /app
# COPY main.py /app/
# COPY config.json /app
# COPY ./src /app/src
# COPY ./test /app/test
COPY ./.cache/huggingface/hub/* /app/.cache/huggingface/hub

RUN pip install -r /app/requirements.txt
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub
# Need to edit library file
COPY ./guidance/_model.py /usr/local/lib/python3.11/site-packages/guidance/models

# ENTRYPOINT ["python", "/app/main.py"]
# CMD ["--km_output", "/app/src/km_output.tsv", "--config", "/app/config.json", "--output_file", "/app/output.tsv"]