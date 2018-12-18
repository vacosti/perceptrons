
# coding: utf-8

# In[2]:

#importanto numpy
import numpy as np


# In[76]:

#--------treinamento---------------------------------

#vetor das caracteristicas para treinamento (conjunto de números aleatórios entre 0 e 1):

s=100 #número de elementos do training set
X=np.random.rand(s,2) #cria um training set com s pares de números aleatórios

#"vetor coluna" de bias inicializados com 1:
b=np.ones(s)[:,None]

#unindo o vetor de bias ao das caracteristicas:
X=np.hstack((X,b)) 

#vetor e[i] dos outputs esperados para cada par de entrada X[0][i],x[1][i]: (soma de dois números)
e=X.T[0]+X.T[1]


# In[83]:

#------------------------------------treinamento com laço de for--------------------------------------

#número de iterações, que corresponde ao número de vezes ou rodadas que o training set 
#será passado ao SLP para o treinamento:
num_iter=1000
 
#taxa de aprendizagem: (obtida por tentativa e erro, é a que leva o vetor peso mais rapidamente ao ponto de mínimo sem divergir)
n=0.0065

#vetor de pesos a ser "ajustado" pelo treinamento, inicializados com 0:
w=np.zeros(3) 

#nessa etapa, os pesos devem se ajustar de modo que o gradiente do erro total tenda a zero
#o training set será passado ao perceptron a quantidade de vezes definida por num_iter:
for i in range(num_iter):
        
    #vetor gradiente a ser subtraído de w toda rodada. 
    grad = (((2*n*(np.dot(X,w)-e))[:,None])*X).sum(axis=0)

    #variando o vetor grad na direção oposta ao gradiente, para levá-lo ao ponto de mínimo
    w -= grad
    print('rodada ', i+1,': ','vetor w:', w, ', erro:', (np.dot(X,w)-e).sum(axis=0) )


# In[78]:

#teste
In=np.array([1000,450,1])
y=np.dot(w,In)
y


# In[74]:

#------------------------------------tentativa de treinamento sem laço de for--------------------------------------

s=100000 #número de elementos do training set
X=np.random.randn(s,2) #cria um training set com s pares de números aleatórios

#"vetor coluna" de bias inicializados com 1:
b=np.ones(s)[:,None]

#unindo o vetor de bias ao das caracteristicas:
X=np.hstack((X,b)) 

#vetor e[i] dos outputs esperados para cada par de entrada X[0][i],x[1][i]: (soma de dois números)
e=X.T[0]+X.T[1]
 
#taxa de aprendizagem: (obtida por tentativa e erro, é a que leva o vetor peso mais rapidamente ao ponto de mínimo sem divergir)
n=0.0065

#vetor de pesos a ser "ajustado" pelo treinamento, inicializados com 0:
w=np.zeros(3) 

#nessa etapa, os pesos devem se ajustar de modo que o gradiente do erro total tenda a zero:

#vetor gradiente a ser subtraído de w
grad = (((2*n*(np.dot(X,w)-e))[:,None])*X).sum(axis=0)

#variando o vetor grad na direção oposta ao gradiente, para levar E ao ponto de mínimo:
w -= grad
print('vetor w:', w)


# In[73]:

#ou seja, se o mesmo training set não for passado varias vezes, não se obtem convergência!

