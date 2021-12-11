### MLZoomcamp Capstone Project

## Problem Description and Data

This project uses the flowers-recognition dataset that can be found at this link: https://www.kaggle.com/alxmamaev/flowers-recognition

It's a dataset composed of around 4200 images of 5 different flowers. The images were scraped off of various sites on the internet. 
We will attempt to accurately label the flowers in these images. 

It can be downloaded by clicking on the Data tab and then on the Download button found underneath the banner. It downloads 5 folders, each containing a specific flower's images. 
For the code in this repo to work it's advised that the 'flowers' folder be copied out of the archive into the root directory of this project, the train.py file references 'flowers/' as the path for the input data.

Due to file size limitations the following has to be downloaded or recreated seperately from the repo in order to have everything run succesfully: 
* Data from kaggle link above (remember to only save the flowers folder and all sub folders, or change the train.py code)
* The model file: xception_fin_23-0.880.h5 can be downloaded from here: https://www.kaggle.com/nielkemp/mlzoom-project-2/data?select=xception_fin_23_0.880.h5

## EDA and Model Building Process

The EDA and Model training process can be viewed here: https://www.kaggle.com/nielkemp/mlzoom-project-2/notebook
** Using GPU the above notebook takes more than 3 hours to run, it has been pre-run with all results commited **
** If you desire to run it, it's advised that all model epochs be heavily reduced, or only certin code-blocks should be run to verify that they work **

The process in the above linked notebook is as follows:
* Importing of required packages
* Testing the reading of image data, and displaying an example of each type of flower in the data
* EDA on the class distribution of the data, classes were deemed as being evenly-enough distributed for no resampling to be done. 
* EDA on dimensions of the images, all images were read with their raw dimensions in order to determine to which single dimension they should be resized 
    * The dimensions used in the final model were detemined by the constraints of the pre-trained model used.
* A variety of models were applied to the data, the data was fed to each model using the ImageDataGenerator in keras.
    * Due to limited GPU time available, each model was only attempted for epochs, this should give enough of an indication of which model is better.
    * The final model will be trained for more epochs
* The different modelling options tried are: 
    * Baseline model with 2 Conv layers, Max pooling, 1 Hidden layer, SGD solver with learning rate = 0.002 and momentum = 0.8
    * Baseline model with augemented data (augementation options can be seen in the notebook linked)
    * Baseline model with augmented data and batchsize changed from 10 to 32
    * Model in previous bullet but added more Conv layers, Dropout and more Hidden layers
    * Pre-trained model: Xception
* From the above options the pre-trained model (Xception) outperformed the model we put together in the previous steps. Xception was selected as the final model architecture
* The 3 different options of learning-rate and momentum of the solver were tried on the data, the final selection was: learning rate = 0.01 and momentum = 0.75
* The final model was trained for 30 epochs and by using Checkpointing the best model (based on validation accuracy) was selected, this model was found at the 28th epoch.

## Train.py
The code to train the final model has been saved in train.py, before running this the data should be downloaded from the kaggle link above. 
train.py is comitted with only 1 epoch and the checkpointing commented out in order to not overwrite the final-best model currently in the repo. 
For the full experience, increase epochs to 30, and uncomment the checkpointint and model-saving code.
Runtims for this file are estimated at:
* CPU: 5 minutes per epoch, total runtime ~3 hours
* Kaggle GPU: 1 minute per epoch, total runtime of ~45 minutes

## Environment
Pipenv was used to manage the environment outside of Kaggle
All required packages are loaded in pipfile and pipfile.lock
To initialize the environment navigate to the folder in terminal and execute: pipenv shell
This should start up the environment and load all required dependencies.
The following code can now be run in this environment: 
* train.py (could take a long time)
* predict.py (requires the rose.jpg file)

## Model Deployment
** All of this was done on an Ubuntu Linux machine, this probably won't work on windows without some tinkering and changes **

The model was deployed using AWS Lambda functionality, but only deployed locally. 

The following steps were followed:
* Convert model from h5 formate to tflite format (using convertToTFLite.py) (This has to be run, currently the tflite model doesn't exist)
* Create lambda_file.py
    * This is the file that has all the functions to call a prediction and will be the main driver of our model once deployed. 
* Create dockerfile
* Build docker image by running the following in terminal: sudo docker build -t capstone-project 
* Run docker image by running the following in terminal: sudo docker run -it --rm -p 8080:8080 capstone-project:latest 

## Testing Deployment
After running the docker image succesfully the model can be tested by running dockerTest.py from a new terminal window (don't close the terminal running the docker image)
