"""
Práctica 3: Sistema musculoesquelético

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Carlos Azael Ramirez Rodriguez
Número de control: 22212267
Correo institucional: L22212267@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Instalar librerias en consola
#!pip install control
#!pip install slycot

# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd


# Datos de la simulación
x0,t0,tend,dt,w,h = 0,0,10,1E-3,10,5
n = round((tend-t0)/dt)+1
t = np.linspace(t0,tend,n)
u = np.zeros(n); u[round(1/dt):round(2/dt)] = 1

def musc(alpha,Cs,Cp,R):
    num = [Cs*R,1-alpha]
    den = [R*(Cs+Cp),1]
    sys = ctrl.tf(num,den)
    return sys

# Funcion de transferencia: Control
alpha,Cs,Cp,R = 0.25, 10E-6, 100E-6, 100
syscontrol = musc(alpha,Cs,Cp,R)
print(f"Función de transferencia del control: {syscontrol}")



# Funcion de transferencia: Caso
alpha,Cs,Cp,R = 0.25, 10E-6, 100E-6, 10E3
syscaso = musc(alpha,Cs,Cp,R)
print(f"Función de transferencia del caso: {syscaso}")



# Respuestas en lazo abierto
clr1 = np.array([100,13,95])/255
clr2 = np.array([217,22,86])/255
clr3 = np.array([235,91,0])/255
clr4 = np.array([255,178,0])/255
clr5 = np.array([42,98,154])/255
clr6 = np.array([247,82,112])/255

_,Fs1 = ctrl.forced_response(syscontrol,t,u,x0)
_,Fs2 = ctrl.forced_response(syscaso,t,u,x0)

fg1 = plt.figure()
plt.plot(t,u,'-',linewidth=1,color= clr1, label= 'Fs(t)')
plt.plot(t,Fs1,'-',linewidth=1,color= clr2, label= 'Fs(t): Control')
plt.plot(t,Fs2,'-',linewidth=1,color= clr3, label= 'Fs(t): Caso')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.1); plt.yticks(np.arange(-0.1,1.1,0.1))
plt.ylabel('Fi(t) [V]')
plt.xlabel('t [s]')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', ncol=1)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('sistema musculoesquelético python.png',dpi=600,bbox_inches='tight')
fg1.savefig('sistema musculoesquelético python.pdf')

#Controlador PI

def controlador(kP,kI,sys):
    Cr = 1E-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = ctrl.tf(numPI,denPI)
    X = ctrl.series(PI,sys)
    sysPI = ctrl.feedback(X,1,sign=-1)
    return sysPI

tratamiento = controlador(0.0209824064736628,2061260.50194967,syscaso)

# Respuestas en lazo cerrado
_,Fs3 = ctrl.forced_response(tratamiento,t,Fs1,x0)


fg2 = plt.figure()
plt.plot(t,u,'-',linewidth=1,color= clr1, label= 'Fs(t)')
plt.plot(t,Fs1,'-',linewidth=1,color= clr2, label= 'Fs(t): Control')
plt.plot(t,Fs2,'-',linewidth=1,color= clr3, label= 'Fs(t): Caso')
plt.plot(t,Fs3,':',linewidth=2,color= clr4, label= 'Fs(t): Tratamiento')

plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.1); plt.yticks(np.arange(-0.1,1.1,0.1))
plt.ylabel('Fi(t) [V]')
plt.xlabel('t [s]')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', ncol=1)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('sistema musculoesqueletico python PI.png',dpi=600,bbox_inches='tight')
fg2.savefig('sistema musculoesqueletico python PI.pdf')


    

