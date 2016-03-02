import matplotlib.pyplot as plt
import numpy as np
import math

# np.loadtxt
def lire(chemin):
	fichier=open(chemin)
	res = []
	for ligne in fichier:
		res.append(float(ligne))
	fichier.close()
	return res

def mesureAbs(x,teta,y,N):
	vecteur = y - np.dot(x.T,teta)
	return np.sum(np.absolute(vecteur))/N

def mesureNormal1(x,teta,y,N):
	vecteur = y - np.dot(x.T,teta)
	return math.sqrt(np.dot(vecteur.T, vecteur))/N

def mesureNormal2(x,teta,y,N):
	vecteur = y - np.dot(x.T,teta)
	return np.dot(vecteur.T, vecteur)/N

def mesureNormal2sigmo(x,teta,y,N):
	vecteur = y - sigmoidMat(np.dot(x.T,teta))
	return np.dot(vecteur.T, vecteur)/N

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoidMat(x):
  return 1 / (np.add(np.zeros(len(x)), np.exp(np.negative(x))))

def mesureLinf(x,teta,y):
	vecteur = y - np.dot(x.T,teta)
	return np.amax(np.absolute(vecteur))

def moindreCarre(x,y):
	p1 = np.linalg.inv(np.dot(x,x.T))
	p2 = np.dot(x,y)
	return  np.dot(p1,p2)

def calcAlpha(t,valDef=1.):
	A = valDef	
	B = A
	C = 4000.
	alpha = A / (B + C * float(t))
	return alpha

def descenteGradient(teta,x,y,N,epsilon=0.001):
	t=1

	list_temps = []
	list_mesureNormal2 = []

	list_temps.append(t)
	list_mesureNormal2.append(mesureNormal2(x,teta,y,N))
	tetaActuel = teta
	alpha = calcAlpha(t)
	p = y - np.dot(x.T,tetaActuel) # parenthese
	inte=np.dot(x,p)
	tetaPlusPlus = tetaActuel + inte * (alpha / float(N))
	while (math.fabs(mesureNormal2(x,tetaActuel,y,N) - mesureNormal2(x,tetaPlusPlus,y,N)) > epsilon):
		tetaActuel=tetaPlusPlus
		t=t+1
		list_temps.append(t)
		list_mesureNormal2.append(mesureNormal2(x,tetaActuel,y,N))
		alpha = calcAlpha(t)
		p = y - np.dot(x.T,tetaActuel) # parenthese
		inte=np.dot(x,p)
		tetaPlusPlus = tetaActuel + inte * (alpha / float(N))
	list_temps.append(t+1)
	list_mesureNormal2.append(mesureNormal2(x,tetaPlusPlus,y,N))
	plt.close('all')
	plt.plot(list_temps,list_mesureNormal2)
	plt.show()


	return tetaPlusPlus

def descenteGradientRisque(teta,x,y,N,epsilon=0.001):
	t=1

	list_temps = []
	list_mesureNormal2 = []

	list_temps.append(t)
	list_mesureNormal2.append(mesureNormal2sigmo(x,teta,y,N))
	tetaActuel = teta
	alpha = calcAlpha(t)
	p = y - sigmoide(np.dot(x.T,tetaActuel)) # parenthese
	inte=np.dot(x,p)
	tetaPlusPlus = tetaActuel + inte * (alpha / float(N))
	while (math.fabs(mesureNormal2sigmo(x,tetaActuel,y,N) - mesureNormal2sigmo(x,tetaPlusPlus,y,N)) > epsilon):
		tetaActuel=tetaPlusPlus
		t=t+1
		list_temps.append(t)
		list_mesureNormal2.append(mesureNormal2sigmo(x,tetaActuel,y,N))
		alpha = calcAlpha(t)
		p = y - sigmoide(np.dot(x.T,tetaActuel)) # parenthese
		inte=np.dot(x,p)
		tetaPlusPlus = tetaActuel + inte * (alpha / float(N))
	list_temps.append(t+1)
	list_mesureNormal2.append(mesureNormal2sigmo(x,tetaPlusPlus,y,N))
	plt.close('all')
	plt.plot(list_temps,list_mesureNormal2)
	plt.show()


	return tetaPlusPlus

def creaMatricePuissanceM(x,N,M):
	res = np.zeros((M +1 ,N))
	for i in range(0,M+1):
		res[i,:] = np.power(x,i)
	return res


def matToFonc(m,x):
	res=0
	for i in range(0,len(m)):
		res = res + m[i]* x**i
	return res


# Recuperation donnees

classe_1=np.loadtxt('taillepoids_f.txt', usecols=(0,))
classe_0=np.loadtxt('taillepoids_h.txt', usecols=(0,))
zeros= np.zeros(len(classe_0))
ones = np.ones(len(classe_1))
teta = np.array([2,3], float)
x = np.append(classe_1,classe_0)
y = np.append(ones, zeros)
grad = descenteGradientRisque(teta, x, y, len(x))


print grad


# Affichage des donnees

plt.close('all')
plt.ylim(-1, 2)
plt.plot(classe_0,zeros,marker='o',color="blue")
plt.plot(classe_1,ones,marker='v',color="red")
plt.title('Tp5')
plt.show()


