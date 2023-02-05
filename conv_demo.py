'''
Author: J. Rafid Siddiqui
Azad-Academy
jrs@azaditech.com
https://www.azaditech.com

'''
#==================================================================

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import math
import matplotlib
import matplotlib.gridspec as gridspec
from ipywidgets import *
from IPython.display import display, clear_output, HTML
from PIL import Image, ImageDraw

import time
import threading
from matplotlib.animation import FuncAnimation

from conv2d import *
from conv import *
#============================================================================================
# GUI Controls and Display
#============================================================================================

conv2 = Convolution2D()


next_btn = widgets.Button(
    value=True,
    description='Convolve Patch',
    disabled=False,
    tooltip='Press to take one convolution step',
    style=ButtonStyle(font_weight='bold',font_color='white'),
    button_style='success',
    icon='fa-step-forward'
    )

conv_btn = widgets.Button(
    value=True,
    description='Convolve Image',
    disabled=False,
    tooltip='Press to run the convolution',
    style=ButtonStyle(font_weight='bold'),
    button_style='info',
    icon='fa-play'
    )

pause_btn = widgets.Button(
    value=True,
    description='Pause/Resume',
    disabled=True,
    tooltip='Pause the convolution process',
    style=ButtonStyle(button_color='#a6a116',font_weight='bold'),
    button_style='warning',
    icon='fa-pause'
    )

reset_btn = widgets.Button(
    value=True,
    description='Reset',
    disabled=True,
    tooltip='Press to Reset',
    style=ButtonStyle(font_weight='bold'),
    button_style='danger',
    icon='fa-stop'
    )


conv_click = lambda x: conv_image()
next_click = lambda x: conv_step()
reset_click = lambda x: conv_reset()
pause_click = lambda x: conv_pause()

conv_btn.on_click(conv_click)
next_btn.on_click(next_click)
reset_btn.on_click(reset_click)
pause_btn.on_click(pause_click)

form_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between'
)

form_items = [HBox([pause_btn,conv_btn,next_btn,reset_btn])]
form = Box(form_items, layout=Layout(
    display='flex',
    flex_flow='column',
    border='solid 2px',
    align_items='center',
    width='60%'
))

anim = None
pause = False
def conv_pause():

    global pause
    pause = not pause
    if(pause):
        conv2.anim.event_source.stop()
    else:
        conv2.anim.event_source.start()
    
    conv_btn.disabled = False
    print("Paused")


def update_fig():
    img = Image.fromarray(conv2.img_arr)
    img = img.convert("RGBA")
    img_with_rect = Image.new('RGBA', img.size, (255, 255, 255, 0))
    drawing = ImageDraw.Draw(img_with_rect)
    rect = [(conv2.cur_pos[1], conv2.cur_pos[0]), (conv2.cur_pos[1]+conv2.patch_size, conv2.cur_pos[0]+conv2.patch_size)]
    drawing.rectangle(rect, fill =(255,0,0,128), outline ="white")
    img_with_rect = Image.alpha_composite(img, img_with_rect)
    conv2.img_axes[0].set_data(img_with_rect)

    cmaps = ['','Reds','Greens','Blues']
    for i in range(1,len(cmaps)):
        
        conv2.img_axes[i].set_data(conv2.cur_patch[:,:,i-1])
        conv2.img_axes[i].set_clim(0,np.max(conv2.cur_patch[:,:,i-1]))

    
    for j in range(4):
        rgb_patch = np.zeros((conv2.patch_size,conv2.patch_size,3),dtype=int)
        for i in range(3):
            conv_patch = conv2.convolve(conv2.cur_patch[:,:,i],conv2.filter[j])
            #conv_patch = signal.convolve2d(conv2.cur_patch[:,:,i],conv2.filter[j],'same')
            #conv_patch = ((conv_patch-np.min(conv_patch))/(np.max(conv_patch)-np.min(conv_patch)))*255
            conv_patch = np.clip(conv_patch,a_min=0,a_max=255)
            conv2.feature_axes[i,j].set_data(conv_patch)
            conv2.features[j+i*j] = conv_patch
            rgb_patch[:,:,i] = conv_patch
        
        conv2.conv_imgs[j,conv2.cur_pos[0]:conv2.cur_pos[0]+conv2.patch_size, conv2.cur_pos[1]:conv2.cur_pos[1]+conv2.patch_size] = rgb_patch[:,:,0] + rgb_patch[:,:,1] + rgb_patch[:,:,2]
        conv2.conv_axes[j].set_data(conv2.conv_imgs[j])
    

def conv_image():
    
    global pause

    pause_btn.disabled = False
    conv_btn.disabled = True
    reset_btn.disabled = True
    pause = False
    
    conv2.anim = FuncAnimation(conv2.fig, conv_step,frames=34,interval=30,repeat=False)
    #while (not pause and (conv2.cur_pos[0]+conv2.patch_size) <= conv2.img_arr.shape[0] ) :
    #    time.sleep(0.1)
    #    conv_step()
        
    
    conv2.fig.canvas.draw()
    
    
    
    
def conv_step(framenr=0):
    global conv2

    reset_btn.disabled=False
    
    if( (conv2.cur_pos[1]+conv2.patch_size) >= conv2.img_arr.shape[1] ):
        conv2.cur_pos[1] = 0
        conv2.cur_pos[0] += conv2.patch_size    
    else:
        conv2.cur_pos[1] += conv2.patch_size

    if( (conv2.cur_pos[0]+conv2.patch_size) > conv2.img_arr.shape[0] ):
        conv2.cur_pos = np.zeros(2,dtype=int)
    
    conv2.cur_patch = conv2.img_arr[conv2.cur_pos[0]:conv2.cur_pos[0]+conv2.patch_size, conv2.cur_pos[1]:conv2.cur_pos[1]+conv2.patch_size, :]

    update_fig()  
    conv2.fig.canvas.draw()
    conv2.fig.canvas.flush_events() 

    return conv2.img_axes[0]


def conv_reset():
    
    initialize()
    conv2.fig.canvas.draw()  
    conv_btn.disabled = False   
    reset_btn.disabled = True
    pause_btn.disabled = True


def create_conv2_axes(fig):

    gs = gridspec.GridSpec(2, 2, figure=fig)

    gs00 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[0,0])
    gs01 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[0,1])
    gs1 = gridspec.GridSpecFromSubplotSpec(4,5, subplot_spec=gs[1,:])
    gs11 = gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec=gs1[1:4,1:5])

    img_axes = np.empty((4),dtype=object)
    channel_axes = np.empty((3),dtype=object)
    kernel_axes = np.empty((4),dtype=object)
    kernel_axes2 = np.empty((5),dtype=object)
    feature_axes = np.empty((3,4),dtype=object)
    conv_axes = np.empty((4),dtype=object)

    img_axes[0] = fig.add_subplot(gs00[0:4,0:3])
    img_axes[0].axis('off')
    img_axes[0] = img_axes[0].imshow(conv2.img)

    #c_names = ['','Red Channel','Green Channel','Blue Channel']
    #c = ['','r','g','b']

    cmaps = ['','Reds','Greens','Blues']
    for i in range(1,4):
        img_axes[i] = fig.add_subplot(gs00[i,3:4])
        img_axes[i].axis('off')
        img_axes[i] = img_axes[i].imshow(np.random.randint(0,255,size=(conv2.patch_size,conv2.patch_size)),cmap=cmaps[i])    

    for i in range(4):
        kernel_axes[i] = fig.add_subplot(gs01[0,i])
        kernel_axes[i].set_title(f'$\mathcal{{K}}_{i+1}$',fontweight='bold',fontsize=12)
        kernel_axes[i].axis('off')

    for i in range(1,4):
        for j in range(4):
            feature_axes[i-1][j] = fig.add_subplot(gs01[i, j])
            feature_axes[i-1][j].axis('off')
            feature_axes[i-1][j] = feature_axes[i-1][j].imshow(np.random.randint(0,255,size=(conv2.img_arr.shape[0],conv2.img_arr.shape[1])),cmap=plt.cm.get_cmap(cmaps[i]).reversed())

    
    
        for j in range(5):
            kernel_axes2[j] = fig.add_subplot(gs1[0,j])
            kernel_axes2[j].axis('off')
            if(j==0):
                kernel_axes2[0].text(.9,.15,'⊛',fontweight='bold',fontsize=30)
            elif(j>0):
                kernel_axes2[j].set_title(f'$\mathcal{{K}}_{j}$',fontweight='bold',fontsize=12)

        for i in range(1,4):
            channel_axes[i-1] =  fig.add_subplot(gs1[i,0])
            channel_axes[i-1].axis('off')
            channel_axes[i-1] = channel_axes[i-1].imshow(np.random.randint(0,255,size=(conv2.img_arr.shape[0],conv2.img_arr.shape[1])),cmap=plt.cm.get_cmap(cmaps[i]))    
            
    
    for j in range(4):
        conv_axes[j] = fig.add_subplot(gs11[j])
        conv_axes[j].axis('off')
        conv_axes[j] = conv_axes[j].imshow(np.random.randint(0,255,size=(conv2.img_arr.shape[0],conv2.img_arr.shape[1])),cmap=plt.cm.get_cmap('afmhot'))    
      
    
    ax = fig.add_subplot(gs00[0,3])
    ax.text(.9,.15,'⊛',fontweight='bold',fontsize=30)
    ax.axis('off')
    ax.tick_params(labelbottom=False, labelleft=False)


    return img_axes,channel_axes,kernel_axes,kernel_axes2,conv_axes,feature_axes

    

def plot_matrix(data,ax,cols,box_dims=[.1,.1,.8,.8]):

    tbl = ax.table(cellText=data, cellLoc="center", rowLoc='center', bbox=box_dims, cellColours=cols)
    for cell in tbl._cells:
        tbl._cells[cell].set_alpha(.7)
    tbl.set_fontsize(10)
    ax.set_xticks([])
    ax.set_yticks([])


def initialize():

    conv2.img_arr = np.array(conv2.img)
    h = conv2.img_arr.shape[0]
    w = conv2.img_arr.shape[1]
    

    pad_w = math.ceil(w % conv2.patch_size) 
    pad_h = math.ceil(h % conv2.patch_size) 
    conv2.img_arr = np.pad(conv2.img_arr, pad_width=[(0, pad_h),(0, pad_w),(0, 0)], mode='constant', constant_values=255)
    conv2.conv_imgs = np.zeros((4, h, w),dtype=int)
    
    cmaps = ['Reds','Greens','Blues']
    cols = np.full((3,3),'c',dtype=str)

    for i in range(3):
        conv2.channel_axes[i].set_data(conv2.img_arr[:,:,i])    

    for i in range(4):
        plot_matrix(conv2.filter[i],conv2.kernel_axes[i],cols)
    
    for i in range(1,5):
        plot_matrix(conv2.filter[i-1],conv2.kernel_axes2[i],cols,box_dims=[.25,.15,.5,.8])

    conv2.cur_pos = np.zeros(2,dtype=int)
    conv2.cur_patch = conv2.img_arr[0:conv2.patch_size, 0:conv2.patch_size, :]

    update_fig()

def show_convolutions2D(img_path):

    global conv2 
    plt.close('all')
    conv2.fig = plt.figure(num='Convolution',figsize=(10,8))
    conv2.img = Image.open(img_path)
    conv2.img_arr = np.array(conv2.img)
    conv2.img_axes,conv2.channel_axes,conv2.kernel_axes,conv2.kernel_axes2,conv2.conv_axes,conv2.feature_axes = create_conv2_axes(conv2.fig)
           
    initialize()     

    conv2.fig.canvas.toolbar_visible = False
    conv2.fig.canvas.header_visible = False
    conv2.fig.canvas.footer_visible = False
    conv2.fig.canvas.manager.set_window_title('Convolution')
    conv2.fig.tight_layout()   

    display(form,conv2.fig.canvas)
    
    