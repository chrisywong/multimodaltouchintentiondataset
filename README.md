Final trained model and training dataset for the paper titled "Vision- and tactile-based continuous multimodal intention and attention recognition for safer physical human-robot interaction" 
* Supplementary video: https://youtu.be/r0KHTMqw31s

# File Struture:
* `datasets` folder: Preprocessed training data using the reduced feature set
* `models` folder: Different trained classifier models
* `raw_data` folder: Contains the raw training data
* `main.py` file: Basic Python script to showcase how to use the training data and classifier
* `README.md` file: This readme

# Installation:

Requires python3, scikit-learn & dependencies

# Usage:

**Create a model from a dataset**  
To create a Machine Learning model from a dataset, you can use `create_model` function.
The created model is stored into `models` folder.  
`create_model` inputs:  
* raw_data (str): path of the dataset  
* model (str): type of model you want to create (optionnal)  
* cross_validation (bool): True if you want to run the cross validation (optional)  
* outputflag (bool): True if you want to save the model 
* outputname (str): name of the output file

**Use a created model**  
You can load a model with `model_use` function.  
* model_use inputs:  
* name_model (str): name of the joblib model  
* input_data (list): list of 5 features and return predction.  

# Training Dataset (in `data_raw`)
Please refer to this link for the raw data:
* https://usherbrooke-my.sharepoint.com/:f:/g/personal/wonc2503_usherbrooke_ca/EixcLwGRo9pBiEPZ5rIHRWoBRIp6JieEdX-SQ3n6bOMjiw

The fully preprocessed training data file with all training samples is located in `ReducedFeatureDataset.csv`
The columns are as follows:
1. Touch sensor $\gamma'$
2. Hand distance $d'$
3. Hand speed $\dot{d}'$
4. Gaze angle $\alpha'$
5. Gaze speed $\dot{\alpha}'$
6. Labeled ground truth value

# Contributors/Authors:
Christopher Yee WONG (christopher.wong2 [at] usherbrooke.ca)  
Lucas VERGEZ (lucas.vergez [at] ensam.eu)
Wael SULEIMAN (wael.suleiman [at] usherbrooke.ca)