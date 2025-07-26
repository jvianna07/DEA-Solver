import numpy as np
from scipy.optimize import linprog


class DEA_CCR:
    def __init__(self,
                 input_array:np.ndarray,
                 output_array:np.ndarray)->None:
        self.x_in = input_array
        self.x_out = output_array
        
        self.DMU_size = len(self.x_in)
        self.input_size = self.x_in.shape[1]
        self.output_size = self.x_out.shape[1]
        self.attr_size = self.input_size+self.output_size
        
        self.datas = np.concatenate((self.x_in, self.x_out,), axis=1)

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

    # Solve the problem 
    def solve(self)->np.ndarray:
        norm_xi, norm_xo = self.normalize()

        lhs_ineq=np.concatenate((norm_xo*(-1), norm_xi),axis=1)
        rhs_ineq=[0]*self.DMU_size
        rhs_eq=np.array([1])
        bounds=[(0, np.inf)]*self.attr_size

        score=[]
        for i in range(self.DMU_size):
            g1=norm_xi[i] #tem iterador
            obj=np.concatenate(([0]*self.output_size,g1),axis=0)
            
            lhs_eq= np.concatenate((norm_xo[i], [0]*self.input_size),axis=0).reshape(1,-1)# tem iterador

            res=linprog(c=obj,
                        A_ub=lhs_ineq*(-1), b_ub=rhs_ineq,
                        A_eq=lhs_eq, b_eq=rhs_eq,
                        bounds=bounds)
                        # Multiplicamos por -1 pra mudar a direcao das constraints em A_ub

            score.append(round(res.fun,4))

        score = np.array([score]).T
        score = np.round(1/score,4)
        return np.concatenate((self.x_in,self.x_out, score), axis=1)
        
