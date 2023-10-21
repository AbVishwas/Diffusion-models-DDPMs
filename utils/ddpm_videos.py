from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation





class animationDDPM:
    def __init__(self, images, output_file_name, fps, x, y):

        self.images = images
        self.output = output_file_name
        self.fps = fps
        self.figure, self.ax = plt.subplots(1,1 , figsize = (12,12))
        self.clb = None
        self.x = x
        self.y = y

        self.x_grid, self.y_grid = np.meshgrid(self.x, self.y)

        print(f"shape of images:{images[0].shape}")
        print(f"grid shape{self.x_grid.shape, self.y_grid.shape}")

    def init_animation(self):
        self.ax.clear()
        self.clb = self.ax.contourf(self.x_grid, self.y_grid, np.array(self.images[0][:,:,0] ))
        self.ax.set_title('denoising images t = 0' , fontdict= {"size":40})
        #self.ax.set_xlabel(' #{}'.format(0, fontdict= {"size":40}))
        
        return [self.clb]
    
    def update_animation(self, i):

        self.ax.clear()
        self.clb = self.ax.contourf(self.x_grid, self.y_grid ,np.array(self.images[i][:,:,0]))   # cmap ='inferno'   , levels = 200
        self.ax.set_title('denoising images t = 0' , fontdict= {"size":40})
        #self.ax.set_xlabel('U #{}'.format(frame, fontdict= {"size":40}))
        
        return [self.clb]

    def create_video(self):

        animation = FuncAnimation(self.figure, self.update_animation, frames = len(self.images), init_func= self.init_animation)#, blit = True )
        print(f"output path is :{self.output}")
        animation.save(self.output, writer = 'ffmpeg', fps = self.fps)
