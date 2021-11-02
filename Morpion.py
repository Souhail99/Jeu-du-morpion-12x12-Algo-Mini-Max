# -*- coding: utf-8 -*-
"""
Created on Thu May 20 18:56:20 2021

@author: Souha
"""


import numpy as np
import random
import time
from numpy.core.fromnumeric import shape
from numpy.lib.stride_tricks import as_strided

global kernels4, kernels13, kernels31, alltehkernels

kernels4 = []
kernels4.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
kernels4.append(np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]))

kernels13 = []
kernels13.append(np.array([[1, 1, 1, 1]]))
kernels31 = []
kernels31.append(np.array([[1], [1], [1], [1]]))
alltehkernels = []
alltehkernels.append(kernels4)
alltehkernels.append(kernels13)
alltehkernels.append(kernels31)

# Partie Initialisation


#construit un jeu vide
def GrilleVide():
    a=np.full((12, 12), '.')
 
    return a


#permet d'afficher un jeu
def Affiche(jeu):
    for i in range(12):
        for j in range(12):
            print("|",jeu[i,j], end = "|")
        print(" ")
        
    

def joueur(jeu):
    J1=0
    J2=0
    for i in range(12):
        for j in range(12):
            if jeu[i,j]=='X':
                J1+=1
            if jeu[i,j]=='O':
                J2+=1
    if J1==J2:
        return 'X'
    return('O')




#on envoie coups qui est un tuple 
def jouer(jeu,a,b):
    if jeu[a,b]!='.':
        print("case occupee")
        b=int(input("quel coup jouer en y ? "))-1
        a=int(input("quel coup jouer en x ? "))-1
        coups=coupPossible(jeu)
        vef=False
        while(vef!=True):
            for i in range(len(coups)-2):
                if(a==coups[i] and b==coups[i+1]):
                    vef=True
                    break
            if(vef==False):
                b=int(input("quel coup jouer en y ? "))-1
                a=int(input("quel coup jouer en x ? "))-1
        vef=False
        jeu[a,b]=joueur(jeu)
        return jeu
    else:
        jeu[a,b]=joueur(jeu)
        return jeu

def coupPossible(jeu):
    res=[]
    for i in range(12):
        for j in range(12):
            if jeu[i,j]=='.':
                res+=[i,j]
    return res

def coupPossible2(jeu):
    res=[]
    i,j=np.where(jeu=='.')
    res.extend([list(i),list(j)])
    return res



def transformation(jeu):
    temp=np.zeros((12,12));
    for i in range(12):
        for j in range(12):
            if(jeu[i,j]=='.'):
                temp[i,j]=int(0)
            if(jeu[i,j]=='X'):
                temp[i,j]=int(1)
            if(jeu[i,j]=='O'):
                temp[i,j]=int(-1)
    return temp
                
def transformation2(jeu):
    temp=np.zeros((12,12));
    for i in range(12):
        for j in range(12):
            if(jeu[i,j]=='.'):
                temp[i,j]=int(0)
            if(jeu[i,j]=='O'):
                temp[i,j]=int(1)
            if(jeu[i,j]=='X'):
                temp[i,j]=int(-1)
    return temp

# @functools.lru_cache(maxsize=None)
def NouveauGagner(input):
    global alltehkernels
    res = False
    for k in alltehkernels:
        res = res or convo(input, k)
    return res


def convo(input, kernels):
    expanded_input = as_strided(  
        input,
        shape=(
            input.shape[0] - kernels[0].shape[0] + 1,  
            input.shape[1] - kernels[0].shape[1] + 1,
            kernels[0].shape[0],
            kernels[0].shape[1],
        ),
        strides=(
            input.strides[0],
            input.strides[1],
            input.strides[0],  
            input.strides[1],
        ),
        writeable=False, 
    )
    res = False
    for kern in kernels:
        feature_map = np.einsum(
            'xyij,ij->xy',
            expanded_input,
            kern,
        )
        #res = res or (np.any(feature_map==4) 
        res = res or np.any(feature_map==4)
    return res




# Partie MiniMax ET Evaluation

def MiniMax(jeu,profondeur,alpha,beta,tour):
    coups=coupPossible2(jeu)
    if (NouveauGagner(transformation(jeu))==True or NouveauGagner(transformation2(jeu))==True or len(coups)==0):
       if(NouveauGagner(transformation(jeu))==True):
           return[None,+np.inf]
       if(NouveauGagner(transformation2(jeu))==True):
           return[None,-np.inf]
    elif(profondeur==0):
        return [None,evaluation3(jeu)]
    
    if(tour=='X'):
        value=-np.inf
        x=coups[0][random.randint(0,len(coups[0])-1)]
        y=coups[1][coups[0].index(x)]
        for i in range(0,len(coups[0])):
            newgrille=jeu.copy()
            jouer(newgrille,coups[0][i],coups[1][i])
            score=MiniMax(newgrille,profondeur-1,alpha,beta,tour='O')[1]
            if(score>value):
                value=score
                x,y=evaluation4(newgrille)
            alpha=max(value,alpha)
            if(alpha>=beta):
                break
        return [x,y,value]
    
    if(tour=='O'):
        value=np.inf
        x=coups[0][random.randint(0,len(coups[0])-1)]
        y=coups[1][coups[0].index(x)]
        for i in range(0,len(coups)):
            newgrille=jeu.copy()
            jouer(newgrille,coups[0][i],coups[1][i])
            score=MiniMax(newgrille,profondeur-1,alpha,beta,tour='O')[1]
            if(score<value):
                value=score
                #x,y=coups[0][i],coups[1][i]
                x,y=evaluation5(newgrille)
            beta=min(value,alpha)
            if(alpha>=beta):
                break
        return [x,y,value]
    


def evaluation4(jeu):
    listecoups=[]
    val=2 
    beta=[]
    i,j=np.where(jeu=='X')
    k,l=np.where(jeu=='O')
    m,n=np.where(jeu=='.')
    for z in range(len(m)):    
        val=2
        for x in range(len(i)):
            if((m[z]<=i[x]+4 and m[z]>= i[x]-4) and (n[z]<=j[x]+4 and n[z]>= j[x]-4)):
                    if([jeu[i[x],b] for b in range(0,4) if abs(b-m[z])<=3].count('X') == 3):
                        val+=1000000000000000000000
                    if([jeu[b,i[x]] for b in range(0,4) if abs(b-m[z])<=3].count('X') == 3):
                        val+=1000000000000000000000
                    val+=val**[jeu[i[x],b] for b in range(i[x],12) if abs(b-m[z])<=3].count('X')
                    val+=val**[jeu[b,j[x]] for b in range(j[x],12) if abs(b-m[z])<=3].count('X')
                    val+=val**[jeu[j[x],j[x]] for b in range(j[x],12) if abs(b-m[z])<=3].count('X')
        for y in range(len(k)):
            if((m[z]<=k[y]+2 and m[z]>= k[y]-1) and (n[z]<=l[y]+2 and n[z]>= l[y]-2)):
                    val=val-val**[jeu[i[x],b] for b in range(i[x],12) if abs(b-m[z])<=3].count('O')
                    val=val-val**[jeu[b,j[x]] for b in range(j[x],12) if abs(b-m[z])<=3].count('O')
                    val=val-val**[jeu[j[x],j[x]] for b in range(j[x],12) if abs(b-m[z])<=3].count('O')
        val+=val^3
        listecoups.append([(m[z],n[z]),val])
        
    beta = sorted(listecoups, key = lambda z : z[1],reverse=True)
    return beta[0][0][0],beta[0][0][1]





def evaluation3(jeu):
    if(NouveauGagner(transformation(jeu))==True):
        return np.inf
    if(NouveauGagner(transformation2(jeu))==True):
        return -np.inf
    listecoups=[]
    val=1 
    beta=[]
    i,j=np.where(jeu=='X')
    k,l=np.where(jeu=='O')
    m,n=np.where(jeu=='.')
    for z in range(len(m)):    
        val=2
        for x in range(len(i)):
            if((i[x]<=m[z]-4 and i[x]>= 4-m[z]) and (j[x]<=n[z]-4 and j[x]>= 4-n[z])):
                    val+=val^5
        for y in range(len(k)):
            if((k[y]<=m[z]-1 and k[y]>= 1-m[z]) and (l[y]<=n[z]-1 and l[y]>= 1-n[z])):
                    val-=val^6
        val+=val^3
        listecoups.append([(m[z],n[z]),val])
        
    beta = sorted(listecoups, key = lambda z : z[1],reverse=True)
    return beta[1][1]

def evaluation5(jeu):
    listecoups=[]
    val=2 
    beta=[]
    i,j=np.where(jeu=='X')
    k,l=np.where(jeu=='O')
    m,n=np.where(jeu=='.')
    for z in range(len(m)):    
        val=2
        for x in range(len(i)):
            if((i[x]<=m[z]-4 and i[x]>= 4-m[z]) and (j[x]<=n[z]-4 and j[x]>= 4-n[z])):
                    val+=val^5
        for y in range(len(k)):
            if((k[y]<=m[z]-1 and k[y]>= 1-m[z]) and (l[y]<=n[z]-1 and l[y]>= 1-n[z])):
                    val-=val^6
        val+=val^3
        listecoups.append([(m[z],n[z]),val])
        
    beta = sorted(listecoups, key = lambda z : z[1],reverse=True)
    return beta[0][0][0],beta[0][0][1]  


def jouerPartie(depart,humain):
    jeu=depart
    print("******* Bonne Partie *****")
    firstime=False
    firstime2=True
    while (NouveauGagner(transformation(jeu))==True or len(coupPossible(jeu))!=0):
        if(NouveauGagner(transformation(jeu))==True or NouveauGagner(transformation2(jeu))==True):
            break
        if(firstime==False and humain=='O'):
            debut = time.time()
            tab=MiniMax(jeu,3,-np.Infinity,np.Infinity,'O')
            fin = time.time()
            print("Temps :",fin-debut,"secondes entre l'ouverture de la fonction et la fin.")
            x=tab[0]
            y=tab[1]
            jeu=jouer(jeu,x,y)
            print("Voici les coordonnées conseillé :")
            print("En y :",(y+1))
            print("En x :",(x+1))
            Affiche(jeu)
            firstime2=False
        if(firstime2==True):
            Affiche(depart)
        #si c'est au joueur
        if joueur(jeu)==humain :
            print("----- à vous -----") 
            b=int(input("Quel coup jouer en y ? "))-1
            a=int(input("Quel coup jouer en x ? "))-1
            coups=coupPossible(jeu)
            vef=False
            while(vef!=True):
                for i in range(len(coups)-2):
                    if(a==coups[i] and b==coups[i+1]):
                        vef=True
                        break
                if(vef==False):
                    b=int(input("Quel coup jouer en y ? "))-1
                    a=int(input("Quel coup jouer en x ? "))-1
            vef=False
            firstime=True
            firstime2=False
            #jouer coup
            jeu=jouer(jeu,a,b)
            Affiche(jeu)
            
        # si c'est au programme
        
        else:
            print("\n")
            print("----- à moi -----")
            debut = time.time()
            tab=MiniMax(jeu,3,-np.Infinity,np.Infinity,humain)
            fin = time.time()
            print("Temps :",fin-debut,"secondes entre l'ouverture de la fonction et la fin.")
            x=tab[0]
            y=tab[1]
            jeu=jouer(jeu,x,y)
            print("Voici les coordonnées conseillé :")
            print("En y :",(y+1))
            print("En x :",(x+1))
            Affiche(jeu)
            print("\n")
    
    if(NouveauGagner(transformation(jeu))==True):
       print("L'humain a gagné")
    if(NouveauGagner(transformation2(jeu))==True):
       print("L'IA a gagné")
    #Affiche(jeu)
    


