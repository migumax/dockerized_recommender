FROM ubuntu

# ...
RUN apt-get update && apt-get -y install gcc 
RUN apt-get update && apt-get install --assume-yes --fix-missing python git

RUN apt-get install python3-dev --assume-yes # fixing the LightFM installation issue with gcc
# Clone repository to /app folder in the container image
RUN git clone https://github.com/MaximMigutin/dockerized_recommender /app

#####################################################################################################################
FROM python:3.7

# Mount current directory to /app in the container image
VOLUME ./:app/

# Copy local directory to /app in container

COPY . /app/

# Change WORKDIR
WORKDIR /app

# Install dependencies

RUN pip install -r requirements.txt

# In Docker, the containers themselves can have applications running on ports. To access these applications, we need to expose the containers internal port and bind the exposed port to a specified port on the host.
# Expose port and run the application when the container is started
EXPOSE 8891:8891
ENTRYPOINT python api.py 8891
# CMD ["flask_api.py"]


# docker build
# docker build -t "<app name>" .

# docker run
# docker run ml_app -p 9999 # to make the port externally avaiable for browsers

# show all running containers
# docker ps

# Kill and remove running container
# docker rm <containerid> -f

# open bash in a running docker container
# docker exec -ti <containerid> bash

# docker compose
# run and interact between multiple docker containers