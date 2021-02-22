# ETA-Model
This is the DoorDash estimation of the arrival time project.

The main code scripts and Jupiter notebooks are included in the src folder.
EDA.ipynb: the Jupiter notebook for feature exploration and engineering.
Process.ipynb: the Jupiter notebook for data pre-processing.
rnn.py: the python script for the RNN model.
ml_algo.py/ml_algo_pca.py: the python script for Ridge regression and Random forest regression model with/without PCA.
result_visual.ipynb: the Jupiter notebook for reading the model results and visualizting them.
Prediction.ipynb:  the Jupiter notebook for doing the final RNN prediction.

The model folder includes the saved models, including the preprocessor, the PCA models, and the RNN model.

The data folder contains the raw data and the processed data.

The result folder contains all the models' results and the final prediction (data_to_predict.csv).
