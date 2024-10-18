import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # Colormap support

from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from sklearn.decomposition import PCA
from typing import Optional
from utils import KabschAlign

class TrajectoryVisualizer:
    def __init__(self, coordinates: np.ndarray, xyz: np.ndarray, ff_type: str):
        self.coordinates = coordinates
        self.xyz = xyz
        self.ref = None  # Reference structure for alignment
        self.ff_type = ff_type

        # Initialize figures and axes
        self.fig, self.ax = plt.subplots(figsize=(15,15))
        self.fig2 = plt.figure(figsize=(15,15))
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        self.annot = None
        self.scatter = None

        self._setup_plots()

    def _setup_plots(self):
        # Configure PCA plot
        self.ax.grid(True)
        self.ax.set_title(f'PCA ({self.ff_type})')

        # Configure 3D conformation plot
        self.ax2.set_xlim(-7, 7)
        self.ax2.set_ylim(-7, 7)
        self.ax2.set_zlim(-7, 7)

        # Plot PCA result
        self.scatter = self.ax.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                                       edgecolors='k', s=30)
        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(10, 10),
                                      textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def update_annot(self, ind):
        # Update annotation based on the selected point
        pos = self.scatter.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        arr = ind["ind"]
        text = f"x: {pos[0]:.2f}, y: {pos[1]:.2f}, {ind}"
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.6)

        # Plot conformation for selected points with color gradient
        cmap = cm.get_cmap('cool')  # Choose a colormap (e.g., 'viridis')

        # Plot conformation for selected points
        for i in arr:
            if self.ref is None:
                self.ref = self.xyz[i].copy()
            conf = self.xyz[i].copy()
            aligned, _ = KabschAlign(self.ref, conf)  
            # self.ax2.plot(aligned[:, 0], aligned[:, 1], aligned[:, 2], color='b')

            # Create color gradient for each point
            n_atoms = aligned.shape[0]
            colors = [cmap(j / n_atoms) for j in range(n_atoms)]

            for j in range(n_atoms - 1):
                self.ax2.plot(
                    aligned[j:j + 2, 0], aligned[j:j + 2, 1], aligned[j:j + 2, 2], 
                    color=colors[j], linewidth=2
                )

            self.ax2.scatter(aligned[:, 0][0], aligned[:, 1][0], aligned[:, 2][0], color='b', s=50)
            self.ax2.scatter(aligned[:, 0][-1], aligned[:, 1][-1], aligned[:, 2][-1], color='r', s=50)
            self.ax2.scatter(aligned[:, 0].mean(), aligned[:, 1].mean(), aligned[:, 2].mean(), 
                             color='g', s=100, marker='*')
            break # break after plotting one conformation
        self.fig2.canvas.draw_idle()

    def _on_click(self, event):
        # Handle mouse click events
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.scatter.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            elif vis:
                self.annot.set_visible(False)
                self.fig.canvas.draw_idle()

    def _on_key(self, event):
        # Handle key press events to reset the reference structure
        self.ref = None
        self.ax2.cla()
        self.ax2.set_xlim(-7, 7)
        self.ax2.set_ylim(-7, 7)
        self.ax2.set_zlim(-7, 7)
        self.fig2.canvas.draw_idle()

    def show(self):
        # Display the plots
        plt.show()
