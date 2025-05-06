#Start Kafka connection, Data loading, Data cleaning
python3 kafka_read.py

#Trains and stores the model
python3 ./model/matrix_factorization.py

# Starts the flask server
python3 app.py

#Offline Evaluation
python3 offline_evaluation.py

#Collect telemetry
python3 online_evaluation.py