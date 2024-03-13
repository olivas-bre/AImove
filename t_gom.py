
import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

model_T = keras.models.load_model("Transformer_GOM_Coef")

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

varSub =  ['sp0x','sp0y','sp0z','sp1x','sp1y','sp1z','sp2x','sp2y','sp2z',
           'sp3x','sp3y','sp3z','hipx','hipy','hipz','nex','ney','nez',
           'hex','hey','hez','lax','lay','laz',
           'lfax','lfay','lfaz','rax','ray','raz',
           'rfax','rfay','rfaz','lshx','lshy','lshz',
           'lsh2x','lsh2y','lsh2z','rshx','rshy','rshz',
           'rsh2x','rsh2y','rsh2z','lupx','lupy','lupz',
           'llx','lly','llz','rupx','rupy','rupz',
           'rlx','rly','rlz']



variables_2 = [sub + '(t-2)' for sub in variables]
variables_3 = [sub + '(t-3)' for sub in variables]

coef_labels = np.concatenate((variables,variables_2,variables_3))

varCoef = []
for j in varSub:
    v = [ j +'_'+ sub for sub in coef_labels]
    varCoef = varCoef+ v

class TGom:
    def __init__(self, variables, varCoef, model_T):
        self.variables = variables
        self.varCoef = varCoef
        self.model_T = model_T
        self.dataX = None
        self.dataY = None
        self.scaler = None
        self.coef_tr = None
        self.df_pred = None
        self.df_yT = None

    def hadamard_product(self, x):
        coeff = keras.backend.expand_dims(x[0], axis=0)
        inp = keras.backend.expand_dims(x[1], axis=1)
        m1 = coeff * inp
        m2 = keras.backend.sum(m1, axis=(2,3)) 
        y = keras.backend.expand_dims(m2, axis=-2)
        return y

    def pred_ang_coef(self, dat_mod, coef_mod):
        val = dat_mod[self.variables].values  
        ned = self.scaler.transform(val)
            
        wx = []
        for w in np.arange(0,len(val)-3, 1):
            wx.append(np.arange(0+w,3+w))

        dX=[]
        for wi in np.arange(len(wx)):
            dX.append(ned[wx[wi],:])

        dataX_mod = np.array(dX)

        coef_modT = np.array(coef_mod).reshape((coef_mod.shape[0], 57, 3, 57)) # 57 joints angles, 3 samples, 57 models
        coef_modT = coef_modT[:, :, ::-1, :]

        p_mod = dataX_mod[0,:,:]
        for t in np.arange(1, len(dataX_mod)-1):
            coeff = keras.backend.constant(coef_modT[t,:,:,:])
            inp = tf.convert_to_tensor(dataX_mod[t,:,:].reshape([1, dataX_mod[t,:,:].shape[0], dataX_mod[t,:,:].shape[1]]), dtype=tf.float32)
            yS = self.hadamard_product([coeff, inp])
            yS = keras.backend.get_value(yS).reshape([yS.shape[1], yS.shape[2]])
            p_mod = np.vstack((p_mod, yS))

        predT = self.scaler.inverse_transform(p_mod)
        df_p_mod = pd.DataFrame(predT, columns =self.variables)

        nedT = self.scaler.inverse_transform(ned)
        df_y = pd.DataFrame(val[3:,:], columns =self.variables)
        df_yT = pd.DataFrame(nedT[3:,:], columns =self.variables)
        offset = df_yT.iloc[0,:] - df_y.iloc[0,:]
        df_p_mod = df_p_mod - offset

        return df_p_mod

    def do_gom(self, eulerAngles):
        val = eulerAngles[self.variables].values  #DATAFRAME WITH ANGLES   
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = self.scaler.fit(val)
        ned = self.scaler.transform(val)
            
        wx = []
        wy = []
        for w in np.arange(0,len(val)-3, 1):
            wx.append(np.arange(0+w,3+w))
            wy.append(np.arange(3+w,4+w))

        dX=[]
        dY = []
        for wi in np.arange(len(wx)):
            dX.append(ned[wx[wi],:])
            dY.append(ned[wy[wi],:])

        self.dataX = np.array(dX)
        self.dataY = np.array(dY)

        outputs = self.model_T.predict([self.dataX, self.dataY])
        d1_pred = outputs[0]
        coef = outputs[1]
        
        d1_pred = keras.backend.get_value(d1_pred)
        self.coef_tr = keras.backend.get_value(coef)
        d1_pred=d1_pred.reshape([d1_pred.shape[0],d1_pred.shape[2]]) 
        predT = self.scaler.inverse_transform(d1_pred)
        self.df_pred = pd.DataFrame(predT, columns =self.variables)

        nedT = self.scaler.inverse_transform(ned)
        df_y = pd.DataFrame(val[3:,:], columns =self.variables)
        self.df_yT = pd.DataFrame(nedT[3:,:], columns =self.variables)
        offset = self.df_yT.iloc[0,:] - df_y.iloc[0,:]
        self.df_pred = self.df_pred - offset

        rc = np.concatenate((self.coef_tr[0,:,2,:],self.coef_tr[0,:,1,:],self.coef_tr[0,:,0,:]), axis=1)
        rc2 = rc.flatten()
        coef_mat_tr = pd.DataFrame(rc2[np.newaxis,:], columns=self.varCoef)
        
        for i in np.arange(1,len(self.coef_tr)):
            rc = np.concatenate((self.coef_tr[i,:,2,:],self.coef_tr[i,:,1,:],self.coef_tr[i,:,0,:]), axis=1)
            rc2 = rc.flatten()
            c_ae = pd.DataFrame(rc2[np.newaxis,:], columns=self.varCoef)
            coef_mat_tr = pd.concat([coef_mat_tr, c_ae], axis = 0) 

        return coef_mat_tr, self.df_pred
    

    