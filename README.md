# Movie Recommendation System

This repository contains the source code for a movie recommendation system. The system collects data from Kafka streams, cleans and processes the data, and then employs two machine learning models—collaborative filtering and matrix factorisation—to generate movie recommendations. A detailed comparison of these two models is provided in the project documentation.

For serving the application, we use [Flask](https://flask.palletsprojects.com/). The server is implemented in `app.py` and listens for HTTP requests at:

http://team23@128.2.205.124:8082/recommend/userid

The API returns the top 20 recommended movie IDs for a given user ID as comma-separated values.

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo

	2.	Install Docker
Docker is a platform that simplifies the process of building, running, and shipping applications. To install Docker on your machine, follow these general steps:
	•	Visit the official Docker installation guide for instructions specific to your operating system.
	•	For Linux users, you might typically run:

sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker


	•	For macOS and Windows, download the Docker Desktop application from the Docker website and follow the installation wizard.
These steps ensure that Docker is set up correctly on your system and that you can run Docker commands from your terminal.

	3.	Build the Docker Image
Build the Docker image for the application using:

docker build -t reco_app:v1 .


	4.	Run the Docker Container
Start the container, forward port 8082, and run it in detached mode (in the background) using:

docker run -d -p 8082:8082 reco_app:v1

This command maps the container’s port 8082 to your local machine’s port 8082.

	5.	Send a Request to the Application
To test the recommendation system, send a curl request. Replace <userid> with the actual user ID:

curl -X GET http://team23@128.2.205.124:8082/recommend/<userid>

This command makes an HTTP GET request to the service and returns the top 20 recommended movie IDs in comma-separated format.

Contents
	•	app.py
The main server application that handles HTTP requests and returns movie recommendations.
	•	service/
Contains the business logic for the movie recommendation models.
	•	Dockerfile
The Dockerfile used to build the Docker image for the service.
	•	requirements.txt
Lists the Python dependencies required for the service.
	•	model/
Contains code for training experimental models.
	•	movie_logs.csv
The dataset fetched from the Kafka stream, which has been cleaned and processed.
	•	utils/
Utility scripts and functions supporting the service.

Contributors

Backprop to the Future group!
	•	Changwook Shim
	•	Himansh Mulchandani
	•	Nikhil Reddy
	•	Smit Patel
	•	Vishal Chatterjee

References
	•	Flask Documentation 
	•	Docker Get Started 

---