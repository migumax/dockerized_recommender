# Dockerized Recommender system as REST API using Python, LightFM and Flask
This project was created as a follow-up wrapper for the paper that I prepared for the 2020 Lomonosov Conference "Economic overview of 2020s": https://lomonosov-msu.ru/rus/event/6500/page/1743. 

My paper got selected for the "Digital transformation" section (please see p.23: https://www.econ.msu.ru/sys/raw.php?o=66528&p=attachment).

I currently do a lot regarding recommender systems at work (in the domain of food retail) so I decided to replicate some of the results using publicly available data.

To sum up, the end result of this research is a matrix factorisation-based recommender system for retail goods built on Flask and wrapped into a Docker container.
- Data source used: https://www.kaggle.com/carrie1/ecommerce-data

## How to deploy the recommender
The model can be ran either in a regular mode or inside a Docker container.

All the main logic sits inside `api.py` file. It trains and serialises the model using `pickle` module.

To run the API run `python api.py <port_number>` 

### Endpoints
    ### /train (GET)

    ### /wipe (GET)

    ### /items_to_user (POST)

    ### /users_to_item (POST)

    ### /items_to_item (POST)



## Working with Docker
One of the ways to make your Data Science (and, generally, all Software Engineering endeavours) project portable and easily deployable to production is packaging all the environment and your code into a Docker container.

It's somehing I often do at work and aim to achieve with this academic reasearch too.

1. Build docker image from Dockerfile
        
        `docker build -t "recommender" -f Dockerfile . `

2. Run docker container from image
        `docker run -p 9999:9999 recommender -p`

## Sending real requests to recommender
You can send requests to the recommnder model either using Python `requests` module or straight from the command line using `CURL`.
