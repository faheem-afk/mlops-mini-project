# base image
FROM python:3.12

# working dir 
WORKDIR /app

# copy
COPY . /app/

# run 
RUN pip install -r requirements.txt

# ports
EXPOSE 5000

# command
CMD ["python", "flask_app/app.py"]

