import numpy as np
import matplotlib.pyplot as plt
import time


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

    def overlap(self,O):

        #Caso con mas de un patron almacenado
        if(len(self.red.shape)>1):

            #Iniciamos un array
            m=np.zeros(self.red.shape[0])

            #Por cada patron vamos calculando el overlapp
            for i in range(self.red.shape[0]):
                for j in range(self.n):
                    m[i]+=self.red[i,j]*np.real(O.valor_esperado(self.__sz[j])[0,0])
        else:
            m=0.0
            for j in range(self.n):
                m+=self.red[j]*np.real(O.valor_esperado(self.__sz[j])[0,0])

        return m/self.n

    def overlapy(self,O):

        #Caso con mas de un patron almacenado
        if(len(self.red.shape)>1):

            #Iniciamos un array
            m=np.zeros(self.red.shape[0])

            #Por cada patron vamos calculando el overlapp
            for i in range(self.red.shape[0]):
                for j in range(self.n):
                    m[i]+=self.red[i,j]*np.real(O.valor_esperado(self.__sy[j])[0,0])
        else:
            m=0.0
            for j in range(self.n):
                m+=self.red[j]*np.real(O.valor_esperado(self.__sy[j])[0,0])

        return m/self.n



if __name__=="__main__":

    semilla=23423423
    qubits=6
    patrones=1
    transicion=1.0
    omega=1.2
    beta=500
    T=20
    steps=10000
    h=1.0/steps
    repeticiones=1500

    lista_de_qubits=[]
    lista_de_saltos=[]

    #Almacenamos el numero de qbits que hay
    for i in range(qubits):
        lista_de_qubits.append(i)
    for j in range(2):
        lista_de_saltos.append(j)

    #Genero la red e inicializo la función de onda del sistema
    R=Red(qubits,semilla,patrones,omega,beta,transicion)
    O=Onda(qubits,semilla,[10.0,4.0*np.pi])

    estado_inicial=O.onda
    # O.onda=np.zeros(2**qubits,dtype="complex")[np.newaxis].T
    # O.onda[2**qubits-2,0]=1.0+0.j
    # estado_inicial=O.onda

    rng=np.random.default_rng(semilla)

    a=np.array([1.0j,1.0])
    b=np.array([-1.0j,1.0])
    c=np.array([1.0])
    d=0.0
    guardado=np.genfromtxt("inicial.dat")
    #creamos los vectores para guardar el overlap de la memoria guardamos cada 4 pasos
    try:
        mz=np.zeros(int(int(T/h)/4),dtype=np.float64)
        my=np.zeros(int(int(T/h)/4),dtype=np.float64)
        if(int(T/h)%4!=0):
            raise
    except:
        print("El numero de pasos no es divisible por 4")

    signo1=0.0
    signo2=0.0

    columnas=1


    #Operadores para la evolucion temporal
    V0=np.eye(2**qubits)-1.0j*h*R.Heff()
    V02=Operador.hermitico(V0)@V0
    cont=0

    #Creamos el array donde vamos a guardar todos los datos
    m_totalz=np.zeros((int(int(T/h)/4),columnas),dtype=np.float64)
    m_totaly=np.zeros((int(int(T/h)/4),columnas),dtype=np.float64)
    
    m_totalz_T=np.zeros((int(int(T/h)/4),columnas),dtype=np.float64)
    m_totaly_T=np.zeros((int(int(T/h)/4),columnas),dtype=np.float64)
    m_totalz_abs=np.zeros((int(int(T/h)/4),columnas),dtype=np.float64)
    m_totaly_abs=np.zeros((int(int(T/h)/4),columnas),dtype=np.float64)

    m_totalz_menos=np.zeros((int(int(T/h)/4),columnas),dtype=np.float64)
    m_totaly_menos=np.zeros((int(int(T/h)/4),columnas),dtype=np.float64)

    saltomassaltomas=[]
    saltomenossaltomenos=[]
    for i in range(qubits):
        saltomassaltomas.append(h*Operador.hermitico(R.saltomas()[i])@R.saltomas()[i])
        saltomenossaltomenos.append(h*Operador.hermitico(R.saltomenos()[i])@R.saltomenos()[i])
    contador=0
    with open("overlap.dat","w") as Fichero:
        for q in range(columnas):

            O.onda=estado_inicial
            c=np.array([1.0])
            d=0.0
            if(q<qubits):
                for i in range(qubits):
                    if(guardado[i]==1.0 and i>=q):
                        d=np.ravel(np.einsum('i,j',c,b))
                    elif(i>=2*q):
                        d=np.ravel(np.einsum('i,j',c,a))
                    else:
                        d=np.ravel(np.einsum('i,j',c,rng.uniform(0.0,1.0,2)))
                    c=d.copy()
            else:
                for i in range(qubits):
                    if(guardado[i]==1.0 and i>=contador):
                        d=np.ravel(np.einsum('i,j',c,a))
                    elif(i>=contador):
                        d=np.ravel(np.einsum('i,j',c,b))
                    else:
                        d=np.ravel(np.einsum('i,j',c,rng.uniform(0.0,1.0,2)))
                    c=d.copy()
                contador+=1

            #creamos los vectores para guardar el overlap de la memoria guardamos cada 4 pasos
            try:
                mz=np.zeros(int(int(T/h)/4),dtype=np.float64)
                my=np.zeros(int(int(T/h)/4),dtype=np.float64)
                if(int(T/h)%4!=0):
                    raise
            except:
                print("El numero de pasos no es divisible por 4")

            O.onda=c[np.newaxis].T
            #Normalizamos la función  de onda
            O.normalizar()
            estado_inicial=O.onda
            signo1=0
            signo2=0


            Fichero.write("Inicial: "+str(abs(R.overlap(O)))+" "+str(abs(R.overlapy(O)))+"          ")
            t1=time.time()
            #Generamos el campo vectorial para cada situacion
            for l in range(repeticiones):

                #creamos los vectores para guardar el overlap de la memoria guardamos cada 4 pasos
                try:
                    mz=np.zeros(int(int(T/h)/4),dtype=np.float64)
                    my=np.zeros(int(int(T/h)/4),dtype=np.float64)
                    if(int(T/h)%4!=0):
                        raise
                except:
                    print("El numero de pasos no es divisible por 4")

                print(l+1)
                O.onda=estado_inicial
                #Inicializamos el sistema en el estado |111....1>


                t=0.0

                #p[0] me da la probabilidad de no transicion de un estado a otro
                #p[i] me da un array con las probabilidad de salto del qbit i
                p=[0.0]
                for i in range(qubits):
                    p.append(np.zeros(2, dtype=np.float64))
                p_t=0.0




                #Iniciamos el bucle de la simulacion
                for k in range(int(T/h)):


                    #Guardamos los valores esperados de la magnetizacion cada vez que estemos en un multiplo de 4
                    if(k%4==0):
                        mz[int(k/4)]+=(R.overlap(O))
                        my[int(k/4)]+=(R.overlapy(O))
                        m_totalz_abs[int(k/4),q]+=np.abs(mz[int(k/4)])/repeticiones
                        m_totaly_abs[int(k/4),q]+=np.abs(my[int(k/4)])/repeticiones


                    #Empezamos calculando todas las probabilidades
                    p[0]=np.real(O.valor_esperado(V02))[0,0]

                    #Para ahorrarnos cuentas innecesarias comprobamos si el sistema evoluciona segun esta dinamica o no

                    if(rng.uniform(0.0,1.0)<p[0]):
                        #El sistema evoluciona según el Hamiltoniano efectivo
                        O.accion(V0)
                    else:
                        #La dinámica en este paso vendrá dada por los saltos por lo que elegimos vamos eligiendo qbits al azar hasta encontrar uno que nos de la transicion

                        #Variable que nos va a indicar si se ha dado ya la transicion o no
                        Flag=True

                        p_t=1.0-p[0]#nos restringimos a las probabilidades de transicion

                        rng.shuffle(lista_de_qubits)

                        for i in lista_de_qubits:
                            if(Flag):
                                #Recorremos desde i+1 por que hemos metido la probabilidad de no salto en p[0]
                                p[i+1][0]=np.real(O.valor_esperado(saltomassaltomas[i]))[0,0]
                                p[i+1][1]=np.real(O.valor_esperado(saltomenossaltomenos[i]))[0,0]

                                #Comprobamos si se da la transicion o no
                                if(rng.uniform(0.0,1.0)<(p[i+1][0]+p[i+1][1])/p_t):

                                    #Comprobamos que transicion es la que se da para ello elegimos una transicion al azar
                                    rng.shuffle(lista_de_saltos)
                                    if(rng.uniform(0.0,1.0)<p[i+1][lista_de_saltos[0]]/(p[i+1][0]+p[i+1][1])):

                                        #Se da la transicion asi que comprobamos cual es la que ha sido seleccionada
                                        if(lista_de_saltos[0]==0):
                                            O.accion(np.sqrt(h)*R.saltomas()[i])
                                        else:
                                            O.accion(np.sqrt(h)*R.saltomenos()[i])
                                    else:
                                        if(lista_de_saltos[1]==0):
                                            O.accion(np.sqrt(h)*R.saltomas()[i])
                                        else:
                                            O.accion(np.sqrt(h)*R.saltomenos()[i])



                                    Flag=False
                                else:
                                    #descartamos estas probabilidades del espacio de posibles transiciones
                                    p_t-=p[i+1][0]+p[i+1][1]

                if(mz[-1]>0):
                    m_totalz[:,q]+=mz.copy()
                    m_totaly[:,q]+=my.copy()
                    signo1+=1
                else:
                    m_totalz_menos[:,q]+=mz.copy()
                    m_totaly_menos[:,q]+=my.copy()
                    signo2+=1
                m_totalz_T[:,q]+=mz.copy()
                m_totaly_T[:,q]+=my.copy()

            t2=time.time()
            print((t2-t1)/repeticiones)
            #Dividimos debido a que nos interesa el promedio
            # mz=mz/repeticiones
            # my=my/repeticiones
            #
            # if(mz[-1]>0):
            #     m_totalz[:,q]=mz.copy()
            #     m_totaly[:,q]=mz.copy()
            # else:
            #     m_totalz_menos[:,q]=mz.copy()
            #     m_totaly_menos[:,q]=mz.copy()

            if(signo1!=0):
                m_totalz[:,q]=m_totalz[:,q]/signo1
                m_totaly[:,q]=m_totaly[:,q]/signo1
            if(signo2!=0):
                m_totalz_menos[:,q]=m_totalz_menos[:,q]/signo2
                m_totaly_menos[:,q]=m_totaly_menos[:,q]/signo2
            
            m_totalz_T[:,q]=m_totalz_T[:,q]/(signo1+signo2)
            m_totaly_T[:,q]=m_totaly_T[:,q]/(signo1+signo2)



            Fichero.write("Final: "+str(mz[-1])+" "+str(my[-1])+"\n")
    
    
    np.savetxt("m_totalz.dat",m_totalz,delimiter="     ")
    np.savetxt("m_totaly.dat",m_totaly,delimiter="     ")
    np.savetxt("m_totalz_T.dat",m_totalz_T,delimiter="     ")
    np.savetxt("m_totaly_T.dat",m_totaly_T,delimiter="     ")
    np.savetxt("m_totalz_menos.dat",m_totalz_menos,delimiter="     ")
    np.savetxt("m_totaly_menos.dat",m_totaly_menos,delimiter="     ")
    np.savetxt("m_totalz_abs.dat",m_totalz_abs,delimiter="     ")
    np.savetxt("m_totaly_abs.dat",m_totaly_abs,delimiter="     ")





