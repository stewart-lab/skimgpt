FROM jfreeman88/kmtest:latest

WORKDIR /app

COPY --chmod=777 requirements.txt /app

RUN pip install -r /app/requirements.txt
