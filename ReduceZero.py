import math
import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData

class ReduceZero(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.rest_columns

    def process(self, components_percent=0.75, eigenvalue_percent=0.75):
        if len(self.label_df) > 1:
            equal_zero_index = (self.label_df != 1).values
            equal_one_index = ~equal_zero_index

           
            fail_feature = np.array(self.feature_df[equal_one_index])

            ex_index=[]
            for temp in fail_feature:
                for i in range(len(temp)):
                    if temp[i]==0:
                        ex_index.append(i)
            select_index=[]
            for i in range(len(self.feature_df.values[0])):
                if i not in ex_index:
                    select_index.append(i)
            ex_index = list(set(ex_index))
           
            select_index=list(set(select_index))
           
            featureold=self.feature_df
            sel_feature = self.feature_df.values.T[select_index].T
           
            columns = self.feature_df.columns[select_index]
            self.feature_df = pd.DataFrame(sel_feature, columns=columns)          
            covMatrix = self.feature_df.cov()

            featValue, featVec = np.linalg.eig(covMatrix)
            index = np.argsort(-featValue)
            eigenvalue_num = math.trunc(len(self.feature_df.values[0]) * eigenvalue_percent)
            selected_values = featValue[index[:eigenvalue_num]]
            selected_vectors = featVec.T[index[:eigenvalue_num]].T

            contri = np.array([sum(v) for v in np.abs(selected_vectors)])
            contri_index = np.argsort(-contri)

            num_components = math.trunc(len(self.feature_df.values[0]) * components_percent)
            selected_index = contri_index[:num_components]
            rest_index = np.hstack((contri_index[num_components:],np.array(ex_index)))
            rest_columns = featureold.columns[rest_index]
            self.rest_columns = list(rest_columns)
            low_features = self.feature_df.values.T[selected_index].T

            columns = self.feature_df.columns[selected_index]
            low_features = pd.DataFrame(low_features, columns=columns)
            low_data = pd.concat([low_features, self.label_df], axis=1)

            self.feature_df = low_features
            self.label_df = self.label_df
            self.data_df = low_data
