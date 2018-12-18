
# coding: utf-8

# In[2]:

import numpy as np


# In[4]:

#Inicialmente, definimos o número de unidades de cada camada, bem como o tamanho do training-set:

s=10000 #número de elementos do training-set
l1=1  #número de unidades da camada 1, ou seja, input layer
l2=2  #número de unidades da camada 2, ou seja, a primeira hidden layer
l3=1  #número de unidades da camada 3, ou seja, a segunda hidden layer

#A seguir definimos o training set:

X=np.random.rand(s)[None,:]  #valores de entrada para o treinamento. Fazemos [None,:] para "tornar" o vetor uma "matriz de uma única linha", de modo a podermos realizar o "produto matricial" adiante
e=np.tanh(X)   #vetor dos valores esperados associados a X: nesse caso, a própria função tanh para teste do MLP 

#Nesta etapa, inicializamos os vetores peso e o vetor de bias:

W12=np.ones(l2*l1).reshape(l2,l1)  #vetor peso que conecta a camada 1 à 2
W23=np.ones((l2*l3)+1).reshape(l2+1,l3) #vetor peso que conecta a camada 2 à 3. O +1 corresponde ao peso que multiplicará o vetor de bias
b=np.ones(X.size) #vetor de bias

n=0.00005  #taxa de aprendizagem (obtida por tentativa e erro)
num_iter=5000  #número de rodadas que o training set alimentará o MLP


# In[5]:

#Finalmente, realizamos o treinamento:
    
for j in range(num_iter):
    
    #primeiramente, calculamos o output gerado pelo perceptron em cada rodada:
    
    S2=np.dot(W12,X)  #S2 é o vetor que contém os valores de entrada para cada unidade da camada 2. np.dot equivale  ao produto matricial
    F2=np.tanh(S2)   #Aplicamos a função de ativação element-wise: tangente hiperbolica
    F2=np.vstack((F2,b)) #unimos o vetor de bias a F2 a cada rodada.
    Y=(F2*W23).sum(axis=0)  #calculamos então o output Y

    #em seguida, calculamos os gradientes e atualizamos os pesos:
    
    gradW23=(2*(Y-e)*F2).sum(axis=1)[:,None] #gradiente de W23. Fazemos [:,None] para tornar possível o broadcast com W23.
    W23=W23-n*gradW23 #atualizamos o vetor W23, pois o mesmo será utilizado no calculo do gradiente de W12

    gradW12=2*(((((Y-e)*X)*W23[0:2,:])*(1 - (np.tanh(S2))**2)).sum(axis=1))[:,None] #gradiente de W12. Fazemos o slice em W23, pois para o cálculo de W12 só nos interessa as duas primeiras linhas de W23, já que a última linha corresponde ao vetor de bias.
    W12=W12-n*gradW12 #atualizamos W12
    
    print('\n \n rodada' ,j,":")
    print('\n gradW12: \n',gradW12)
    print('\n gradW23: \n',gradW23)
    print('\n W12: \n',W12)
    print('\n W23: \n', W23)

