
# Dataset: kaggle.com - Give me some credit - https://www.kaggle.com/c/GiveMeSomeCredit/overview

# Problem Description
Banks play a crucial role in market economies. They decide who can get finance and on what terms and can make or break investment decisions. For markets and society to function, individuals and companies need access to credit. 

Credit scoring algorithms, which make a guess at the probability of default, are the method banks use to determine whether or not a loan should be granted. This competition requires participants to improve on the state of the art in credit scoring, by predicting the probability that somebody will experience financial distress in the next two years

# EDA
EDA and Model Training process available in notebook.ipynb or notebook.py

EDA focussed on two areas, the target, in order to best establish what performance metric to use, and then the given features, in order to determine the best way to clean the data. 

The target had a very low positivity rate, so the AUC was selected as the performance metric. 

The following things were looked at for features:
- Missing values
- Expected range of variables, and how to deal with values outside of these ranges
- Distribution of values

# Model Training

Four models were tried, these were:
- Logistic regression
- Decision Tree
- Random Forest
- XGBoost

For each a full parameter selection was performed by fitting the model on various combinations of paramters, and selecting the best one. 
Ultimately, the Random forest was selected due to it's high validation performance, as well as it's relative simplicity vs the other top performing model (XGBoost)

# Exporting notebook to script
The code from the notebook was exported to notebook.py
Training for the final model was isolated to train.py

# Environment
All required packages are available in Pipfile and pipfile.lock, to prepare the environment do the following:
- go to terminal
- cd to the directory of the project content
- run pipenv install

This should check the Pipfile for the required packages, install them
To activate the environment run:
- pipenv shell

# Model Deployment
The model is deployed locally using docker, it can be deployed by following these steps: (take note, these were performed on a Linux machine)
- Build the docker image by running the following from terminal: sudo docker build -t midterm .
- Run the container by running the following in terminal: sudo docker run -it --rm -p 9696:9696 midterm
- Once image is up and running, test it by running the following in a new terminal (don't close the first terminal): python predict_docker_test.py

The steps above should generate a prediction.

