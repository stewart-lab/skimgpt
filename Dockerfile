FROM python:3.11.3

WORKDIR /app

COPY --chmod=777 requirements.txt /app
# COPY main.py /app/
# COPY config.json /app
# COPY ./src /app/src
# COPY ./test /app/test
# COPY ./src/km_output.tsv /app
COPY --chmod=777 ./Mistral-7B-OpenOrca /app/Mistral-7B-OpenOrca

RUN pip install -r /app/requirements.txt
# ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub
# Need to edit library file
COPY --chmod=777 ./guidance/_model.py /usr/local/lib/python3.11/site-packages/guidance/models

# ENTRYPOINT ["python", "/app/main.py"]
# CMD ["--km_output", "/app/src/km_output.tsv", "--config", "/app/config.json", "--output_file", "/app/output.tsv"]