FROM python:3.9

WORKDIR /code

RUN apt-get -y update  && apt-get install -y \
  python3-dev \
  apt-utils \
  python-dev \
  build-essential 

ENV VIRTUAL_ENV=/code/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
#RUN python3 -m venv /code/venv
#ENV PATH="/code/venv/bin/activate"

#RUN code/.venv/bin/pip install --upgrade pip
RUN pip install --upgrade pip
COPY ./requirements.txt /code/requirements.txt
#RUN code/.venv/bin/pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./iris-api /code/iris-api

CMD ["uvicorn", "iris-api.main:app", "--host", "0.0.0.0", "--port", "8080"]

# If running behind a proxy like Nginx or Traefik add --proxy-headers
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]
