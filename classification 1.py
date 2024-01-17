import pandas as pd 
import numpy as np 
import matplotlib.pyplot as mp  
import scipy . optimize as opt
print("mazen")
path = "C:\\1.PC\\Career\\ML\\cods\\Classification\\1\\data.txt"
data = pd.read_csv(path,header=None,names=['ex1','ex2','Accepted'])

positive = data [data['Accepted'].isin([1])]
negative = data [data['Accepted'].isin([0])]

fig,ax = mp.subplots(figsize=(8,5))
ax.scatter(positive['ex1'],positive['ex2'],s=50,c='b',marker='o',label='Accepted')
ax.scatter(negative['ex1'],negative['ex2'],s=50,c='r',marker='x',label='NotAccepted')
ax.legend()
data.insert(0,'Ones',1)
cols=data.shape[1] #100*4 then data.shape[0]=100 
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
x=np.array(x)
y=np.array(y)
theta = np.zeros(3)

def segmoid(z) :
    return 1/1+np.exp(-z)

def cost (theta , x , y):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)
    first = np.multiply (-y,np.log(segmoid(x*theta.T)))
    second = np.multiply((1-y),np.log(1-segmoid(x*theta.T)))
    return np.sum(first -second) / (len(x))

def gradiant  (theta , x , y):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)
    parameters=int(theta.ravel().shape[1])
    error=segmoid(x*theta.T)-y
    grad=np.zerors(parameters)
    for i in range(parameters):
        term=np.multiply(error,x[:i])
        grad[i]=np.sum(term)/len(x)
    return grad 

def predicition (theta , X):
    probability = segmoid(X*theta.T)
    return [1 if x>=0.5 else 0 for x in probability]

       
thiscost=cost(theta,x,y)
result = opt.fmin_tnc(func=cost, x0=theta,fprime=gradiant,args=(x,y))
costafteroptimize = cost(result[0], x, y)
theta_min = result[0]
predictions= predicition(theta_min, x)
correct =[1 if (((a==1)and(b==1))or((a==0)and(b==0))) else 0 for (a,b) in zip(predictions ,y)]
accuracy = (sum(map(int,correct)) % len(correct))
print('accuracy={0}%'.format(accuracy))