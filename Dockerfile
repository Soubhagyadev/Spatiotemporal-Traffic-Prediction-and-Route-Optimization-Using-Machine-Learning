FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#Expose The Port
EXPOSE 5001 8050 8000

# Inside Docker, services talk to each other via localhost
ENV TRAFFIC_API_URL=http://127.0.0.1:5001
ENV TRAFFIC_DASHBOARD_URL=http://127.0.0.1:8050
ENV TRAFFIC_WEB_URL=http://127.0.0.1:8000
ENV DJANGO_SECRET_KEY=change-me-in-production
ENV DJANGO_DEBUG=false

ENV DJANGO_ALLOWED_HOSTS=*
ENV RUNNING_IN_DOCKER=1

CMD ["python", "start.py"]
