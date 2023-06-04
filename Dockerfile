FROM python:3.9-slim
COPY . /dockerapp
WORKDIR /dockerapp
RUN pip install -r requirements.txt 

EXPOSE 5000
CMD ["python", "app.py"]