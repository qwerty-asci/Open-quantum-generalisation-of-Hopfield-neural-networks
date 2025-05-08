import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.linalg as scl


class Operador():

    __I=np.eye(2)
    __sz=np.zeros((2,2),dtype="complex")
    __sx=np.zeros((2,2),dtype="complex")
    __sy=np.zeros((2,2),dtype="complex")
    __smas=np.zeros((2,2),dtype="complex")
    __smenos=np.zeros((2,2),dtype="complex")

    __sz[0,0]=1.0
    __sz[1,1]=-1.0

    __sx[1,0]=1.0
    __sx[0,1]=1.0

    __sy[1,0]=1.0j
    __sy[0,1]=-1.0j

    __smas=(__sx+1.0j*__sy)/2.0
    __smenos=(__sx-1.0j*__sy)/2.0

    def init(self):
        pass


    @staticmethod
    def I():
        return Operador.__I
    @staticmethod
    def sz():
        return Operador.__sz
    @staticmethod
    def sx():
        return Operador.__sx
    @staticmethod
    def sy():
        return Operador.__sy
    @staticmethod
    def smas():
        return Operador.__smas
    @staticmethod
    def smenos():
        return Operador.__smenos
    @staticmethod
    def producto_tensorial(operadores):
        """
            Metodo para devolver el producto tensorial de un array de operadores
        """
        aux=None
        aux1=None

        #Comprobamos el numero de operadores en que hay en la lista de forma que si hay solo uno devolvemos ese operador
        if(len(operadores)>1):

            #Recorremos la lista de operadores hasta len(operadores)-1 por que como vamos multiplicando de dos en dos en cada iteracion hacemos uso de dos operadores por lo que seria m(1)@m(2) en i=0 y en i=n-1 m(n-1)@m(n) de esta forma nos ahorramos problemas con el indexado
            for pasos in range(len(operadores)-1):

                #si estamos en la primera iteracion ambos operadores tienen la misma dimension
                if(pasos==0):

                    #creamos un auxiliar que es donde vamos a guardar el operador resultante
                    aux=np.zeros((operadores[0].shape[0]*operadores[1].shape[0],operadores[0].shape[1]*operadores[1].shape[1]),dtype="complex")
                    m=0
                    n=0
                    for i in range(aux.shape[0]):
                        k=0
                        l=0

                        #Contamos cada vez que recorremos todas las filas del operador de menos dimension lo cual implica que hemos recorrido todos los valores de la fila y debemos cambiar a la siguiente
                        if(i!=0 and i%operadores[1].shape[0]==0):
                            m+=1
                            n=0
                        elif(i==0):
                            pass
                        else:
                            n+=1
                        for j in range(aux.shape[0]):

                            #mismo razonamiento con las columnas
                            if(j!=0 and j%operadores[1].shape[0]==0):
                                k+=1
                                l=0
                            elif(j==0):
                                pass
                            else:
                                l+=1
                            aux[i,j]=operadores[0][m,k]*operadores[1][n,l]
                            # print(n," ",k," ",n," ",l," (",i,",",j,")")
                else:
                    #igual que antes pero cambiando las dimensiones del segundo operador
                    aux1=np.zeros((aux.shape[0]*operadores[pasos+1].shape[0],aux.shape[1]*operadores[pasos+1].shape[1]),dtype="complex")
                    m=0
                    n=0
                    for i in range(aux1.shape[0]):
                        k=0
                        l=0
                        if(i!=0 and i%operadores[pasos+1].shape[0]==0):
                            m+=1
                            n=0
                        elif(i==0):
                            pass
                        else:
                            n+=1
                        for j in range(aux1.shape[0]):
                            if(j!=0 and j%operadores[pasos+1].shape[0]==0):
                                k+=1
                                l=0
                            elif(j==0):
                                pass
                            else:
                                l+=1
                            aux1[i,j]=aux[m,k]*operadores[pasos+1][n,l]
                            # print(n," ",k," ",n," ",l," (",i,",",j,")")
                    aux=aux1.copy()
            return aux
        else:
            return operadores[0]

    @staticmethod
    def hermitico(operador):
        """
            Metodo para devolver el operador hermítico
        """
        aux=np.conj(operador.T).copy()

        return aux
    @staticmethod
    def funcion_sobre_operador(matriz,funcion):
        """
    
        Parameters
        ----------
        matriz : array(n,n,dtype=np.complex128)
            Operador que queremos calcular
        funcion : funcion
            Funcion que se va a aplicar sobre el operador
    
        Returns : Matriz calculada
        -------
        TYPE
            Metodo para calcular la accion de una función sobre un operador hermítico
    
        """
        b=np.linalg.eigh(matriz)    
        return b[1]@np.diagflat(funcion(b[0]),0)@b[1].T
    
    @staticmethod
    def diagonalizar(array):
        """
    
        Parameters
        ----------
        array : array(n,n,dtype=np.complex128)
            Operador que que vamos a descomponer en vectores left y right
    
        Returns : autovalores del operador, left and right eigenvectors
        -------
        TYPE
            Metodo para calcular la accion de una función sobre un operador hermítico
    
        """
            
        #Autovalores
        autovalores,Rvectores=scl.eig(array,right=True,overwrite_a=True)
    
            
            # #Normalizamos
            # Lvectores[:,i]/=np.sqrt(aux)
            # Rvectores[:,i]/=np.sqrt(aux)
        
        return autovalores,None,Rvectores


class Onda():

    def __init__(self,n_qubit,semilla,pesos):
        #El orden de la base es: {|ijk...m>}={|111...1>,|111...2>,|111..21>,|111..22>,...,|222..22>}
        self.rng=np.random.default_rng(semilla)
        self.n=n_qubit

        #Inicializamos el vector de onda del sistema de qbits
        self.onda=np.zeros(2**n_qubit,dtype="complex")[np.newaxis].T
        for i in range(self.onda.size):
            self.onda[i]=self.rng.uniform()+self.rng.uniform()*1j
        #Normalizamos la función  de onda
        self.normalizar()


    def normalizar(self):
        a=np.conj(self.onda.T)@self.onda
        if(a[0,0]!=0.0):
            self.onda=self.onda/np.sqrt(np.real(a[0,0]))
        else:
            raise
            self.onda=np.zeros(2**self.n,dtype="complex")[np.newaxis].T


    def norma(self):
        """
            Función que nos devuelve la norma al cuadrado
        """
        return np.real(np.conj(self.onda.T)@self.onda)[0,0]

    def valor_esperado(self,O):
        """
            Función que nos devuelve el valor esperado de un operador
        """
        return np.conj(self.onda.T)@O@self.onda
    
    @staticmethod
    def valor_esperado_matriz_densidad(matriz,operador):
        """
            Función que nos devuelve el valor esperado de un operador
        """
        return np.trace(operador@matriz)

    def accion(self,O):

        """
            Función de onda resultante al aplicarle un operador
        """

        #Estado resultante tras la accion de un operador sobre la funcion de onda
        # print("\n"*4)
        # print(self.onda)
        self.onda=O@self.onda
        # print(O)
        # print(self.onda)
        # print(Operador.hermitico(O)@O)
        self.normalizar()

    def matriz_densidad(self):
        return self.onda@np.conj(self.onda.T)

class Red():
    def __init__(self,qubits,semilla,patrones,omega,beta,gamma):

        """
            rng: Generador de numeros aleatorios del sistema
            omega: nos da la fuerza que tiene el Hamiltoniano cuántico
            n: numero de qubits del sistema
            transicion: nos da la fuerza que tienen los operadores de transicion
            beta: Factor de boltzman del sistema beta=1/T
            red: Nos da los patrones que se han generado en el sistema
        """

        #generamos el estado inicial de la red
        self.rng=np.random.default_rng(semilla)
        self.omega=omega
        self.n=qubits
        self.transicion=gamma
        self.beta=beta

        ############################ Generacion de los patrones de la red #################################

        if(patrones>1):

            #recorremos por todos los qubits generando un patron inicial aleatorio
            reda=np.zeros((patrones,qubits)) #Matriz que tiene como filas el número de patrones y columnas el numero de qbits
            for i in range(qubits):
                for j in range(patrones):
                    if(self.rng.uniform()<0.5):
                        reda[j,i]=1
                    else:
                        reda[j,i]=-1

            #descartamos los patrones que sean identicos, para ello empezamos asignando el primer patron al sistema
            self.red=reda[0]

            #Flag que nos va a decir si el patron esta repetido ya o no
            F=True

            #Recorremos sobre todos los patrones excepto el primero que ya está almacenado
            for i in range(1,patrones):

                #Recorremos sobre todas las filas que tengamos en red
                for j in range(self.red.shape[0]):
                    #Con el operador all() nos devolverá true en caso de que todos los terminos de la red sean iguales
                    if(all(self.red[j]==reda[i])):
                        F=False
                if(F):
                    self.red=np.row_stack((self.red,reda[i]))
                else:
                    F=True


            #matriz de pesos
            self.P=np.zeros((qubits,qubits))

            for i in range(qubits):
                for j in range(qubits):
                    #Con shape[0] nos estamos quedando con el numero de filas, es decir, el numero de patrones distintos que se han generado
                    for n in range(self.red.shape[0]):
                        self.P[i,j]+=self.red[n,i]*self.red[n,j]/qubits
        else:

            #Situacion en la que solo tenemos que generar un patron
            self.red=np.zeros(qubits)

            #recorremos por todos los qubits generando un patron inicial aleatorio
            for i in range(qubits):
                if(self.rng.uniform()<0.5):
                    self.red[i]=1
                else:
                    self.red[i]=-1

            # self.red[0]=1.0
            # self.red[1]=1.0
            # self.red[2]=1.0
            # self.red[3]=1.0
            # self.red[4]=1.0


            #matriz de pesos
            self.P=np.zeros((qubits,qubits))

            for i in range(qubits):
                for j in range(qubits):
                    self.P[i,j]+=self.red[i]*self.red[j]/qubits


        ############################ Fin de la generación de los patrones de la red #################################


        #Almacenamos los estados que se han guardado en la red
        np.savetxt("inicial.dat",self.red,delimiter=" ")


        #Hamiltoniano cuantico del sistema
        #Generamos una lista con todos los operadores que van a intervenir en el Hamiltoniano quantico aumentando su dimension a la correspondiente al numero de spines
        #Calculamos el Hamiltoniano cuantico empleando los sx correspondiente a cada spin en el espacio cuantico correspondiente
        #Como hay que hacer el calculo para todos los operadores correspondiente a cada uno de los spines se hace todo junto de una
        self.__H=np.zeros((2**qubits,2**qubits),dtype="complex")


        #Calculamos los operadores de salto de cada uno de los spines de la red
        #Calculamos los spines en la componente z de cada qubit
        self.__mas=[]
        self.__menos=[]
        self.__sz=[]
        self.__sy=[]
        self.__saltomas=[]
        self.__saltomenos=[]

        for i in range(self.n):
            #Empezamos con el primer qbit donde sx X I X I X I al hacer un ciclo i pasa a ser no nulo por lo que nos permite
            #reescribir pasar el producto a I X sx X I X I... al poner qbits -i-1 el -1 hace referencia a que uno de los operadores
            #tiene que ser la matriz de Pauli sz
            self.__H+=self.omega*Operador.producto_tensorial([Operador.I()]*i+[Operador.sx()]+[Operador.I()]*(qubits-i-1))
            self.__mas.append(Operador.producto_tensorial([Operador.I()]*i+[Operador.smas()]+[Operador.I()]*(qubits-i-1)))
            self.__menos.append(Operador.producto_tensorial([Operador.I()]*i+[Operador.smenos()]+[Operador.I()]*(qubits-i-1)))
            self.__sz.append(Operador.producto_tensorial([Operador.I()]*i+[Operador.sz()]+[Operador.I()]*(qubits-i-1)))
            self.__sy.append(Operador.producto_tensorial([Operador.I()]*i+[Operador.sy()]+[Operador.I()]*(qubits-i-1)))

        self.__Heff=self.__H.copy()
        #Calculamos los operadores de salto del sistema y el Hamiltoniano efectivo del sistema
        for i in range(self.n):
            #Creamos los operadores de salto
            self.__saltomas.append(self.gamma(i,1)@self.__mas[i])
            self.__saltomenos.append(self.gamma(i,-1)@self.__menos[i])
            self.__Heff-=0.5j*(Operador.hermitico(self.__saltomas[i])@self.__saltomas[i]+Operador.hermitico(self.__saltomenos[i])@self.__saltomenos[i])

    def J(self):
        return self.P

    def H(self):
        return self.__H

    def Heff(self):
        return self.__Heff

    def sz(self):
        return self.__sz

    def sy(self):
        return self.__sy

    def saltomas(self):
        return self.__saltomas

    def saltomenos(self):
        return self.__saltomenos

    def patrones(self):
        #Devolvemos el numero de patrones que hay almacenados en el sistema
        if(self.red.ndim==2):
            return self.red.shape[0]
        else:
            return 1

    def gamma(self,iq,signo):
        E=self.dE(iq+1)

        #Como las matrices son diagonales al aplicar una funcion sobre ellas actuará sobre los elementos de la diagonal nada mas
        for i in range(E.shape[0]):
            E[i,i]=np.exp(signo*self.beta*E[i,i]/2.0)/np.sqrt(2.0*np.cosh(self.beta*E[i,i]))
        return E*np.sqrt(self.transicion)




    def dE(self,iq):
        #Calculamos la variacion de la energia ante el intercambio del spin i
        Flag=False

        M=np.zeros((2**self.n,2**self.n),dtype="complex")
        for i in range(self.n):
            #Sumamos sobre todos los elementos de la matriz peso correspondiente excepto al que le corresponde dicho spin
            if((i+1)!=iq and self.P[iq-1,i]!=0.0):
                M+=self.P[iq-1,i]*self.__sz[i]
                Flag=True

        return M

    def overlap(self,rho):

        #Caso con mas de un patron almacenado
        if(len(self.red.shape)>1):

            #Iniciamos un array
            m=np.zeros(self.red.shape[0])

            #Por cada patron vamos calculando el overlapp
            for i in range(self.red.shape[0]):
                for j in range(self.n):
                    m[i]+=self.red[i,j]*np.real(Onda.valor_esperado_matriz_densidad(self.__sz[j],rho))
        else:
            m=0.0
            for j in range(self.n):
                m+=self.red[j]*np.real(Onda.valor_esperado_matriz_densidad(self.__sz[j],rho))

        return m/self.n

    def overlapy(self,rho):

        #Caso con mas de un patron almacenado
        if(len(self.red.shape)>1):

            #Iniciamos un array
            m=np.zeros(self.red.shape[0])

            #Por cada patron vamos calculando el overlapp
            for i in range(self.red.shape[0]):
                for j in range(self.n):
                    m[i]+=self.red[i,j]*np.real(Onda.valor_esperado_matriz_densidad(self.__sy[j],rho))
        else:
            m=0.0
            for j in range(self.n):
                m+=self.red[j]*np.real(Onda.valor_esperado_matriz_densidad(self.__sy[j],rho))

        return m/self.n



if __name__=="__main__":


    semilla=86412
    qubits=2
    patrones=1
    transicion=1.0
    omega=[1.0,2.0,3.0,5.0,10.0]
    beta=500
    
    while(qubits<14):
        print(qubits)
        for j in omega:
            R=Red(qubits,semilla,patrones,j,beta,transicion)
    
            
            #Creamos el superoperador de Liouville
            Lv=np.zeros((2**(2*qubits),2**(2*qubits)),dtype=np.complex128)
            
            #Añadimos los operadores
            Lv+=-1.0j*(Operador.producto_tensorial([R.Heff(),np.eye(2**qubits,dtype=np.complex128)])-Operador.producto_tensorial([np.eye(2**qubits,dtype=np.complex128),np.conj(R.Heff())]))

            
            #Añadimos la parte debida a los operadores de salto
            for i in range(qubits):
                Lv+=Operador.producto_tensorial([R.saltomas()[i],np.conj(R.saltomas()[i])])+Operador.producto_tensorial([R.saltomenos()[i],np.conj(R.saltomenos()[i])])
        
            autovalores=np.linalg.eigvals(Lv.copy())
            
            np.savetxt(str(qubits)+"autovalores_qubits_Omega_"+str(j)+"_beta_"+str(beta)+".dat",autovalores)
        qubits+=1
    
