FROM python:3.7

WORKDIR /usr/src/app

COPY src/. /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "Main.py"]
