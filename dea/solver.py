import numpy as np
import pandas as pd
from scipy.optimize import linprog


class DEA:
    def __init__(self, csv_file:pd.DataFrame, input_cols:tuple,
               output_cols:tuple, header:bool=True)->None:
        
        # Complete dataset 
        self.datas = csv_file

        # Dataframe columns 
        self.datas_in=csv_file.iloc[:, input_cols[0]: input_cols[1]]
        self.datas_out=csv_file.iloc[:, output_cols[0]: output_cols[1]]
    
        # Array columns
        self.x_in = self.datas_in.values
        self.x_out = self.datas_out.values
        
        # Atributes extractions
        self.DMU_size = len(self.x_in)
        self.input_size = self.x_in.shape[1]
        self.output_size = self.x_out.shape[1]
        self.attr_size = self.input_size+self.output_size
        
      
    # Visualize input data
    def show_inputs(self)->pd.DataFrame:
        return self.datas_in


    # Visualize output data
    def show_outputs(self)->pd.DataFrame:
        return self.datas_out


    # Normalize datas 
    def normalize(self)->np.ndarray:
        # Normalize X_in 
        norm_x_in = self.x_in/(
            np.sqrt(
                np.sum(
                    self.x_in**2, axis=0, keepdims=True
                    )))

        # Normalize X_out
        norm_x_out = self.x_out/(
            np.sqrt(
                np.sum(
                    self.x_out**2, axis=0, keepdims=True
                    )))
        return norm_x_in, norm_x_out



# Dea CCR model class      
class DEA_CCR(DEA):
    def __init__(self, csv_file, input_cols, output_cols, header = True):
        super().__init__(csv_file, input_cols, output_cols, header)

    # Solve the problem 
    def solve(self, normalize=True)->np.ndarray:
        if normalize==True:
            data_input, data_output = self.normalize()
        else:
            data_input, data_output = self.x_in, self.x_out

        
        lhs_ineq=np.concatenate((data_output*(-1), data_input),axis=1)
        rhs_ineq=[0]*self.DMU_size
        rhs_eq=np.array([1])
        bounds=[(0, np.inf)]*self.attr_size

        score=[]
        for i in range(self.DMU_size):
            g1=data_input[i] #tem iterador
            obj=np.concatenate(([0]*self.output_size,g1),axis=0)
            
            lhs_eq= np.concatenate((data_output[i], [0]*self.input_size),axis=0).reshape(1,-1)# tem iterador

            res=linprog(c=obj,
                        A_ub=lhs_ineq*(-1), b_ub=rhs_ineq,
                        A_eq=lhs_eq, b_eq=rhs_eq,
                        bounds=bounds)
                        # Multiplicamos por -1 pra mudar a direcao das constraints em A_ub

            score.append(round(res.fun,4))

        # score = np.array([score]).T
        # score = np.round(1/score,4)
        # return np.concatenate((self.x_in,self.x_out, score), axis=1)
        return pd.DataFrame(data=score, columns=['score'])
    

    # Rank the DMUS
    def rank(self, sort='no'):
        score = self.solve()
        ranks = self.datas.join(1/score)

        if sort.lower() == 'no':
            return ranks
        elif sort.lower() in ['ascending', 'asc']:
            return ranks.sort_values(by='score', ascending=True)
        elif sort.lower() in ['descending', 'desc']:
            return ranks.sort_values(by='score', ascending=False)
        else:
            return "Unknown sort method. Please choose between 'no', 'asc', and 'desc'"


  

# Dea BCC model class 
class DEA_BCC(DEA):
    def __init__(self, csv_file, input_cols, output_cols, header = True):
        super().__init__(csv_file, input_cols, output_cols, header)

    # Solve the problem 
    def solve(self, normalize=True)->np.ndarray:
        if normalize==True:
            data_input, data_output = self.normalize()
        else:
            data_input, data_output = self.x_in, self.x_out

        return "BCC model is not implemented in this version yet."
    

    # Rank the DMUS
    def rank(self, sort='no'):
        return "BCC model is not implemented in this version yet."
    