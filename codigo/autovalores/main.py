import numpy as np
import matplotlib.pyplot as plt


a=np.genfromtxt("TF.dat")
b=np.genfromtxt("Freq.dat")

aux=[]
ls=[]


for i in range(a.size):
    if(a[i]>40):
        aux.append(b[i])

    elif(len(aux)>=1):
        ls.append(aux.copy())
        aux=[]

medias=[]

for i in ls:
    medias.append(np.mean(np.array(i)))

print(medias)

plt.figure(1)
plt.plot(b,a)
plt.show()
