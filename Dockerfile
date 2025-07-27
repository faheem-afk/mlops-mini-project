# base image
FROM python:3.12

# working dir 
WORKDIR /app

# copy
COPY .env ./
COPY models/ ./models/
COPY flask_app/ ./flask_app/

# run 
RUN pip install -r ./flask_app/requirements.txt

# ports
EXPOSE 5000

# command
CMD ["gunicorn", "-b", "0.0.0.0:5000", "flask_app.app:app"]

