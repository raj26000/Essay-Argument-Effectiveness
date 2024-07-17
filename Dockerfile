FROM python:3.8
WORKDIR /essay_argument_effectiveness
COPY . /essay_argument_effectiveness
RUN pip install -r requirements.txt
EXPOSE 8008
CMD ["python", "app.py"]
