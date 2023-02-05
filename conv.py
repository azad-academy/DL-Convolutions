'''
Author: J. Rafid Siddiqui
Azad-Academy
jrs@azaditech.com
https://www.azaditech.com

'''
#==================================================================



import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

class Convolution:

    def __init__(self,X,F,G,mode='full',kernel_window_width=4,style='lines',show_area=False):

        self.fig = plt.figure()
        self.paused = False
        self.X = X
        self.F = F
        self.G = G
        self.Y = np.empty((3,len(X)))
        self.fig.canvas.mpl_connect('button_press_event', self.pause)
        self.fig.canvas.manager.set_window_title('Convolution') 
        self.ax = self.fig.axes
        self.p = [None]*3
        self.pf = [None]*3
        self.show_area = show_area
        self.style = style
        self.animation = None
        self.kernel_window_width = kernel_window_width
        self.cur_pos = None 
        self.mode=mode
        self.styles = ['lines','bars']

    def convolve(self):
        
        c = ['b-','r--','g:']
        lbls = ['$f(x)$','$g(x)$','$f(x)*g(x)$']

        self.cur_pos = np.max(self.X)-self.kernel_window_width
        
        self.Y[0] = self.F(self.X)
        self.Y[1] = self.G(self.X)
        self.Y[2] = signal.convolve(self.Y[0],self.Y[1],'same')
        self.Y[2] = self.Y[2]/np.max(self.Y[2])#int(self.X.shape[0]/2)

        if(self.mode=='window'):
            self.Y[2]=np.zeros(self.X.shape)
        
        for i in range(3):

            if(self.show_area):
                self.pf[i] = plt.fill_between(self.X, self.Y[i],step='mid',alpha=0.4)
            if(self.style=='lines'):
                self.p[i] = plt.plot(self.X, self.Y[i],c[i],linewidth=2,label=lbls[i])
                
            else:
                if(i==0):
                    self.p[i] = plt.bar(self.X, self.Y[i],label=lbls[i],alpha=0.4)
                else:
                    self.p[i] = plt.plot(self.X, self.Y[i],c[i],linewidth=2,label=lbls[i])
                

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center',ncol=3)
        plt.tight_layout()
        self.fig.canvas.manager.set_window_title('Convolution')
        
        return self.Y[2]


    def animate(self,f=200,fps=30):
        
        self.animation = FuncAnimation(self.fig, self.update, frames=f, blit=False, interval=1000/fps, repeat=False)
        self.fig.canvas.draw()

    def pause(self):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused ^= True

    

    def update(self,t):

        if(self.mode=='window'):
            w_min = self.cur_pos
            w_max = self.cur_pos + self.kernel_window_width
            X = np.linspace(w_min,w_max,self.X.shape[0])
            
            Yg = self.G(X+0.05*t)
            
            Yf = self.F(X)
            self.Y[1] = self.G(self.X+0.05*t)
            self.Y[2] = np.convolve(Yf,Yg,'same')
            self.Y[2] = self.Y[2]/int(self.X.shape[0]/2)
                    
            self.p[1][0].set_data(self.X, self.Y[1])
            self.p[2][0].set_data(X, self.Y[2])
            
            self.fig.gca().collections.clear()
            self.fig.gca().fill_between(X, self.Y[2],step='mid',alpha=0.4,facecolor='g')
            
            self.cur_pos -=  0.05

        elif(self.mode=='full'):

            X = self.X + 0.05*t
        
            self.Y[1] = self.G(X)
            if(self.mode=='both'):
                self.Y[0] = self.F(X)    
                plt.xlim([np.min(X),np.max(X)])
            else:
                self.Y[0] = self.F(self.X)
            self.Y[2] = signal.convolve(self.Y[0],self.Y[1],'same')
            self.Y[2] = self.Y[2]/np.max(self.Y[2])
            
            self.p[1][0].set_data(self.X, self.Y[1]) 
            self.p[2][0].set_data(self.X, self.Y[2]) 

        return self.p  

        