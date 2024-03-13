import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

variables =  ['Spine_Xrotation','Spine_Yrotation','Spine_Zrotation','Spine1_Xrotation','Spine1_Yrotation','Spine1_Zrotation','Spine2_Xrotation','Spine2_Yrotation','Spine2_Zrotation',
           'Spine3_Xrotation','Spine3_Yrotation','Spine3_Zrotation','Hips_Xrotation','Hips_Yrotation','Hips_Zrotation','Neck_Xrotation','Neck_Yrotation','Neck_Zrotation',
           'Head_Xrotation','Head_Yrotation','Head_Zrotation','LeftArm_Xrotation','LeftArm_Yrotation','LeftArm_Zrotation',
           'LeftForeArm_Xrotation','LeftForeArm_Yrotation','LeftForeArm_Zrotation','RightArm_Xrotation','RightArm_Yrotation','RightArm_Zrotation',
           'RightForeArm_Xrotation','RightForeArm_Yrotation','RightForeArm_Zrotation','LeftShoulder_Xrotation','LeftShoulder_Yrotation','LeftShoulder_Zrotation',
           'LeftShoulder2_Xrotation','LeftShoulder2_Yrotation','LeftShoulder2_Zrotation','RightShoulder_Xrotation','RightShoulder_Yrotation','RightShoulder_Zrotation',
           'RightShoulder2_Xrotation','RightShoulder2_Yrotation','RightShoulder2_Zrotation','LeftUpLeg_Xrotation','LeftUpLeg_Yrotation','LeftUpLeg_Zrotation',
           'LeftLeg_Xrotation','LeftLeg_Yrotation','LeftLeg_Zrotation','RightUpLeg_Xrotation','RightUpLeg_Yrotation','RightUpLeg_Zrotation',
           'RightLeg_Xrotation','RightLeg_Yrotation','RightLeg_Zrotation']

variablesOpt =  ['Spine','Spine1','Spine2','Spine3','Hips','Neck','Head','LeftArm','LeftForeArm','RightArm',
           'RightForeArm','LeftShoulder','LeftShoulder2','RightShoulder','RightShoulder2','LeftUpLeg',
           'LeftLeg','RightUpLeg','RightLeg']


variables_1 = [sub + '(t-1)' for sub in variables]
variables_2 = [sub + '(t-2)' for sub in variables]

coef_labels_kf = np.concatenate((['Bias'], variables_1,variables_2))

class KfGom:
    def __init__(self, variables, coef_labels_kf):
        self.variables = variables
        self.coef_labels = coef_labels_kf
        self.model = None
        self.coef = None
        self.pvalues = None
        self.df_pred = None
        self.df_y = None
        self.lags = 2

    def pred_ang_coef(self, dat_mod, coef_mod):
        df = dat_mod[self.variables] 
        # Create a DataFrame of lagged values for all variables
        lagged_values = pd.concat([df.shift(i) for i in range(1, self.lags + 1)], axis=1)

        # The columns of lagged_values are now the variables and lags in the format (variable, lag)
        lagged_values.columns = pd.MultiIndex.from_product([df.columns, range(1, self.lags + 1)], names=['Variable', 'Lag'])

        # Flatten the DataFrame to get a 2D array of shape (num_time_steps, num_variables * num_lags)
        lagged_values_array = lagged_values.values.reshape(-1, len(self.variables) * self.lags)

        # Initialize a DataFrame to store the predicted values for all variables
        predicted_values = pd.DataFrame(index=df.index, columns=df.columns)

        # Loop over all variables
        for i in range(len(self.variables)):
            # The first parameter is the constant, and the rest are the coefficients of the lagged values
            constant = float(coef_mod.values[0, i])
            coefficients = coef_mod.values[1:, i]
            # Calculate the predicted value for the current variable
            predicted_values.iloc[:, i] = constant + np.dot(lagged_values_array, coefficients)

        df_p_mod = predicted_values.dropna()

        return df_p_mod

    def do_gom(self, eulerAngles):
        df = eulerAngles[self.variables]  #DATAFRAME WITH ANGLES   
        self.model = VAR(df.values).fit(2)
        coef = self.model.params

        self.coef = pd.DataFrame(coef, index=self.coef_labels, columns=self.variables)
        self.pvalues = pd.DataFrame(self.model.pvalues, index=self.coef_labels, columns=self.variables)

        # Create a DataFrame of lagged values for all variables
        lagged_values = pd.concat([df.shift(i) for i in range(1, self.lags + 1)], axis=1)

        # The columns of lagged_values are now the variables and lags in the format (variable, lag)
        lagged_values.columns = pd.MultiIndex.from_product([df.columns, range(1, self.lags + 1)], names=['Variable', 'Lag'])

        # Flatten the DataFrame to get a 2D array of shape (num_time_steps, num_variables * num_lags)
        lagged_values_array = lagged_values.values.reshape(-1, len(self.variables) * self.lags)

        # Initialize a DataFrame to store the predicted values for all variables
        predicted_values = pd.DataFrame(index=df.index, columns=df.columns)

        # Loop over all variables
        for i in range(len(self.variables)):
            # The first parameter is the constant, and the rest are the coefficients of the lagged values
            constant = coef[0, i]
            coefficients = coef[1:, i]

            # Calculate the predicted value for the current variable
            predicted_values.iloc[:, i] = constant + np.dot(lagged_values_array, coefficients)

        self.df_pred = predicted_values.dropna().iloc[:-1,:]
        self.df_y = df.iloc[2:-1,:]

        return self.coef, self.df_pred