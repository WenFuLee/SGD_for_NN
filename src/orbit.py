from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def func(x,y):
  return (y**2-x**2)

def list_func(xs,ys):
  return [func(x,y) for x,y in zip(xs,ys)]

def func_grad(x,y):
  return (-2*x, 2*y)

def plot_func(xt,yt,c='r'):
  fig = plt.figure()
  ax = fig.gca(projection='3d',
        elev=35., azim=-30)
  X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
  Z = func(X,Y) 
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
    cmap=cm.coolwarm, linewidth=0.1, alpha=0.3)
  ax.set_zlim(-50, 50)
  ax.scatter(xt, yt, func(xt,yt),c=c, marker='o' )
  ax.set_title("x=%.5f, y=%.5f, f(x,y)=%.5f"%(xt,yt,func(xt,yt))) 
  plt.show()
  plt.close()

def plot_func_line(xts,yts,c='r'):
  fig = plt.figure()
  ax = fig.gca(projection='3d',
        elev=35., azim=-30)
  X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
  Z = func(X,Y) 
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
    cmap=cm.coolwarm, linewidth=0.1, alpha=0.3)
  ax.set_zlim(-50, 50)
  zts = list_func(xts,yts)
  ax.scatter(xts[0], yts[0], zts[0], c="g", marker='o' )
  ax.scatter(xts[1:], yts[1:], zts[1:],c=c, marker='o' )
  ax.plot(xts, yts, zs=zts, c=c)
  #ax.set_title("x=%.5f, y=%.5f, f(x,y)=%.5f"%(xt,yt,func(xt,yt))) 
  plt.show()
  plt.close()

def run_grad(plot = True):
  xt = 0.001 
  yt = 4 
  eta = 0.3 
  #plot_func(xt,yt,'r')
  xts = [xt]
  yts = [yt]
  for i in range(20):
    gx, gy = func_grad(xt, yt)
    xt = xt - eta*gx
    yt = yt - eta*gy
    if xt < -5 or yt < -5 or xt > 5 or yt > 5:
      break
    #plot_func(xt,yt,'r')
    xts.append(xt)
    yts.append(yt)
  if plot:
    plot_func_line(xts,yts,'r')
  return (xts, yts)

def run_adagrad(plot = True):
  xt = 0.001
  yt = 4 
  eta = 1.0 
  Gxt = 0
  Gyt = 0
  #plot_func(xt,yt,'b')
  xts = [xt]
  yts = [yt]
  for i in range(20):
    gxt,gyt = func_grad(xt, yt)
    Gxt += gxt**2
    Gyt += gyt**2
    xt = xt - eta*(1./(Gxt**0.5))*gxt
    yt = yt - eta*(1./(Gyt**0.5))*gyt
    if xt < -5 or yt < -5 or xt > 5 or yt > 5:
      break
    #plot_func(xt,yt,'b')
    xts.append(xt)
    yts.append(yt)
  if plot:
    plot_func_line(xts,yts,'b')
  return (xts, yts)

def run_adam(plot = True):
  xt = 0.001
  yt = 4
  eta = 1.0
  mxt = 0
  myt = 0
  beta = 0.9
  Gxt = 0
  Gyt = 0
  lda = 0.999
  xts = [xt]
  yts = [yt]
  for i in range(20):
    gxt,gyt = func_grad(xt, yt)
    mxt = beta * mxt + (1-beta) * gxt
    myt = beta * myt + (1-beta) * gyt
    Gxt = lda * Gxt + (1-lda) * gxt**2
    Gyt = lda * Gyt + (1-lda) * gyt**2
    a = eta * ((1-lda)**0.5) / (1-beta)
    xt = xt - a * mxt / (Gxt**0.5)
    yt = yt - a * myt / (Gyt**0.5)
    if xt < -5 or yt < -5 or xt > 5 or yt > 5:
      break
    xts.append(xt)
    yts.append(yt)
  if plot:
    plot_func_line(xts,yts,'y')
  return (xts, yts)

def plot_all():
  grad_xts, grad_yts = run_grad(False)
  adag_xts, adag_yts = run_adagrad(False)
  adam_xts, adam_yts = run_adam(False)
  grad_zts = list_func(grad_xts, grad_yts)
  adag_zts = list_func(adag_xts, adag_yts)
  adam_zts = list_func(adam_xts, adam_yts)
  fig = plt.figure()
  ax = fig.gca(projection='3d',
        elev=35., azim=-30)
  X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
  Z = func(X,Y) 
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
    cmap=cm.coolwarm, linewidth=0.1, alpha=0.3)
  ax.set_zlim(-50, 50)
  ax.scatter(grad_xts[0], grad_yts[0], grad_zts[0], c="g", marker='o' ) # origin
  ax.scatter(grad_xts[1:], grad_yts[1:], grad_zts[1:],c="r", marker='o' ) # SGD
  ax.plot(grad_xts, grad_yts, zs=grad_zts, c="r") # SGD
  ax.scatter(adag_xts[1:], adag_yts[1:], adag_zts[1:], c="b", marker='o' ) # Adagrad
  ax.plot(adag_xts, adag_yts, zs=adag_zts, c="b") # Adagrad
  ax.scatter(adam_xts[1:], adam_yts[1:], adam_zts[1:], c="y", marker='o' ) # Adam
  ax.plot(adam_xts, adam_yts, zs=adam_zts, c="y") # Adam
  SGD_patch = mpatches.Patch(color='r', label='SGD')
  Adagrad_patch = mpatches.Patch(color='b', label='Adagrad')
  Adam_patch = mpatches.Patch(color='y', label='Adam')
  plt.legend(handles=[SGD_patch, Adagrad_patch, Adam_patch])
  #ax.set_title("x=%.5f, y=%.5f, f(x,y)=%.5f"%(xt,yt,func(xt,yt))) 
  plt.show()
  plt.close()
