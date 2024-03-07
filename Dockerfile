FROM jfreeman88/kmtest:latest

WORKDIR /app

COPY --chmod=777 ./Mistral-7B-OpenOrca /app/Mistral-7B-OpenOrca

COPY --chmod=777 requirements.txt /app
COPY main.py /app/
COPY config.json /app
COPY ./src /app/src
COPY ./test /app/test
COPY ./data.tsv /app

RUN pip install -r /app/requirements.txt

# ENTRYPOINT ["python", "/app/main.py"]
# CMD ["--km_output", "/app/src/km_output.tsv", "--config", "/app/config.json", "--output_file", "/app/output.tsv"]