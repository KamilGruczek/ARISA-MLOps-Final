# ARISA MLOPS wind-power-generation-forecasting
This project was created for educational purposes as part of ARISA's AI classes at Warsaw University of Computer Science. It is a credit-based project whose overall goal was to create an architecture for repeatable and reliable training of machine learning models, along with monitoring their performance and changes.

## Prerequisites

Install python 3.11 with py manager on your local machine.  
Install Visual Studio Code on your local machine.  
Create a kaggle account and create a kaggle api key kaggle.json file.  
Move the kaggle.json to   
C:\Users\USERNAME\\.kaggle folder for windows,  
/home/username/.config/kaggle folder for mac or linux.  
### Local
Fork repo to your own github account.  
Clone forked repo to your local machine.  
Open VS Code and open the repo directory.  
In the VS Code terminal run the following to create a new python virtual environment:  
```
py -3.11 -m venv .venv
```
windows
```
.\.venv\Scripts\activate
```
mac or linux  
```
source .venv/bin/activate
```
source .venv/bin/activate
```
and then open up notebook 01 and attempt to run the cells. 

Another way to run project is:

```
python -m ARISA_DSML.preproc
```
python -m ARISA_DSML.train
```
python -m ARISA_DSML.predict
```