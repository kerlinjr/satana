# Run annotation pipelines

# Requires the following packages:
# - opencv-python-headless
# - matplotlib


Copy code
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider

# Initialize variables
current_axis = None
rect = None
angle_slider = None

def draw_rectangle(event):
    global rect, current_axis
    if event.inaxes != current_axis:
        return
    if event.button != 1 or rect is not None:
        return
    # Draw the rectangle
    rect = patches.Rectangle((event.xdata, event.ydata), 1, 1, linewidth=1, edgecolor='r', facecolor='none', angle=0)
    current_axis.add_patch(rect)
    fig.canvas.draw()

def update_rectangle(event):
    global rect
    if rect is None:
        return
    width = event.xdata - rect.get_x()
    height = event.ydata - rect.get_y()
    rect.set_width(width)
    rect.set_height(height)
    fig.canvas.draw()

def finalize_rectangle(event):
    global rect
    if event.button != 1:
        return
    rect = None

def update_angle(val):
    global rect
    if rect:
        rect.angle = val
        fig.canvas.draw()

# Load an image using OpenCV
image_path = 'your_image.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a figure and a set of subplots
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
current_axis = ax

# Display the image
ax.imshow(image)

# Connect the mouse events to handlers
fig.canvas.mpl_connect('button_press_event', draw_rectangle)
fig.canvas.mpl_connect('motion_notify_event', update_rectangle)
fig.canvas.mpl_connect('button_release_event', finalize_rectangle)

# Add a slider for rotation
angle_axis = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
angle_slider = Slider(angle_axis, 'Angle', 0.0, 360.0, valinit=0)
angle_slider.on_changed(update_angle)

plt.show()