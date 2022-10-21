# Pulsar-Stars-Prediction
Applying the XGBClassifier machine learning model to predict the 0 and 1 classes in the test dataset, i.e, x_test.

Each candidate is described by 8 continuous variables and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency. The remaining four variables are similarly obtained from the DM-SNR curve. You do not have to worry about what they actually mean. These 8 variables are summarised below:

1. The mean of the integrated profile.
2. The standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. The skewness of the integrated profile.
5. The mean of the DM-SNR curve.
6. The standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. The skewness of the DM-SNR curve.

Source: https://archive.ics.uci.edu/ml/datasets/HTRU2

Courtesy
Dr Robert Lyon
University of Manchester
School of Physics and Astronomy
Alan Turing Building
Manchester M13 9PL
United Kingdom
robert.lyon@manchester.ac.uk
