Requeriments:
- Python 3.7 +

To execute the project we need to create a new virtual env in the project folder:
- python -m venv venv

After that run the venv with and install the libraries:
- . venv/bin/activate

Libraries
- nltk
- pandas
- flask
- tweetpy
- sklearn

To install the libraries we need execute:

    pip install <library_name>

NTLK dependencies:
- stopword
- twitter_samples
- wordnet
- punkt

To install the dependencies execute the python shell and write:
- import nltk
- nltk.download('<dependence_name>')

To execute the app:
- python web_app.py 