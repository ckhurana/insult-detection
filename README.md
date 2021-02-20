# Insult Detection in Social Commentary

Main goal of this project is to detect if a comment or a post online is an insult or
not using various machine learning techniques.

> [__Github Repository__ ](https://github.com/ckhurana/insult-detection/) 

## Working Demo
* [Insult Detection](http://labs.chiragkhurana.com/iiitd/nlp/insult-detection) - Enter a test a sentence to tag it.
* [Insult Detection on Live Tweets](http://labs.chiragkhurana.com/iiitd/nlp/insult-detection-twitter) - Enter a query to search related to it.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## File Structure

- ___root___
    * __data__ - Contains the data sets for the project
    * __src__
        * _main.py_ - Main entry point
        * _ensemble.py_ - Ensembling code
        * _preprocess.py_ - Helper preprocessing code
        * _features.py_ - Helper feature extraction code
    * __interactive__ - Contains the jupyter notebook for interactive project representation
    * __ppt__ - The Presentation (ppt and pdf)
    * __misc__ - Some miscellaneous files (Sample Output.txt)
    * __visualise__ - Various graphs and curves for different Classification techniques used
    * _requirements.txt_ - Requirements file for installed modules.
    * _README.md_ - Readme file in MarkDown format
    * _README.pdf_ - Readme in portable document format

### Prerequisites

- Python 3.5+
- Following Python Modules
  * jupyter==1.0.0
  * scikit-learn==0.19.0
  * scipy==1.0.0
  * nltk==3.2.4
  * numpy==1.13.1
  * pandas==0.20.3
  * matplotlib==2.0.2
  * virtualenv==15.1.0 [Optional]

Or you can directly install all the required modules along with dependencies using requirements.txt file.
```
pip install -r requirements.txt
```

### Installing

Follow the following steps to setup a virtual environment to run the project

0. Install Python 3.5.x
```
Refer the internet for installing python.
```
 
1. Setup virtual environment [Optional]
```
virtualenv -p python3 venv
```

2. Use the virtual env for further work [Optional]
```
# For Ubuntu/Linux
source venv/bin/activate

# For Windows - CommandPrompt
.\venv\Scripts\activate.bat

# For Windows - PowerShell
.\venv\Scripts\activate.ps1]

# The CLI will have a (venv) at the beginning of every line from now on.
```

3. Installing the required modules
```
pip install -r requirements.txt
python -m spacy dowmload en_core_web_sm
```

4. Run the main file for the project
```
cd src
python main.py
```

## Interactive Testing

To test the project and visualize the project more intuitively, try using our jupyter notebook.
Note: Make sure to try the following with environment properly set up.
```
cd interactive
jupyter notebook
```
A brower tab will open with the notebooks listed.
Try the _Presentation.ipynb_ to use the project file.
Then use the the notebook in a standard way.
 
## Running the tests
```
cd src
python main.py
```
The above should provide with all the usefull information neccessary including _Accuracy score, Confusion matrices, ROC Curves, Area Under Curve score_.

### Result Interpretation

* The __confusion matrix__ helps represemt the precision and recall of a classifier.
* The __accuracy score__ gives the percentage of accurate predictions by the model.
* The __ROC AUC__ of a classifier is equal to the probability that the classifier will rank a randomly chosen positive example higher than a randomly chosen negative example, i.e. P(score(x+)>score(x−))

## Train data and Test data

To test a custom set of data, some modifications in the code needs to be done, as the code in its natural form splits the train data in the train and test sets, therefore using a seperate file to test data requires minor configuration changes in the code.
Although this can be done easily in the Jupyter notebook available in the package.

### The training and test data
> Source [Kaggle](https://www.kaggle.com/c/detecting-insults-in-social-commentary)

## Built With

* [Jupyter](http://jupyter.org/) - Interactive computing
* [Scikit-Learn](http://scikit-learn.org/stable/documentation.html) - Machine Learning and Classification Library
* [NLTK](http://www.nltk.org/) - Generic NLP tasks
* [spaCy](https://spacy.io/) - Advanced and intuitive NLP tasks (dependency parsing)
 
## Authors

* [**Chirag Khurana**](http://chiragkhurana.com) - [Github](https://github.com/ckhurana)
* **Shubham Goyal** - [Github](https://github.com/imshubhamgoyal)
* **Pallavi Rawat** - [Github](https://github.com/PallaviSRawat)


## Acknowledgments

* [__Tanmoy Chakraborty__](https://sites.google.com/site/tanmoychakra88/) - Mentor / Instructor
