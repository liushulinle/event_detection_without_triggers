Event Detection without Triggers (NAACL2019)
============================================

This repository provides the code for the work in NAACL2019: "Event Detection without Triggers".

Because of the copyright issue of ACE2005 Corpus, we can not release the corpus. For test, we give 10 samples in data/test_corpus_10.txt.

We provide a trained model, which can be downloaded here: [model files](https://drive.google.com/open?id=1X9mP8z2mxehxM92VDMQQi4D_HGa4A38U). 

You can run this code to evaluate the trained model using the following command:
```python
python run_model.py evaluation 
```
or train the model using your own traininng corpus:
```python
python run_model.py train 
```

**Required running environment:**  
*1. python 2.7*  
*2. tensorflow 1.4 or higher*  
