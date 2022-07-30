import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
from PIL import Image, ImageTk
import os

# load images
datadir = "..\\repos\pytorch-CycleGAN-and-pix2pix\\results\emotions_e03\\test_latest\images"
imgs = os.listdir(datadir)
fakebs = list(filter(lambda img: "fake_B" in img,imgs))
num_imgs = len(fakebs)

# load indexes
filesdir = "indexes"
bechdir = os.path.join(filesdir, "benchmark.npy")
removedir = os.path.join(filesdir, "toremove.npy")
lastind = os.path.join(filesdir, "lastind.npy")

try:
    benchmarks = np.load(bechdir).tolist()
except OSError:
    benchmarks = []
try:
    to_remove = np.load(removedir).tolist()
except OSError:
    to_remove = []
try:
    base_ind = 240#np.load(lastind)
except OSError:
    base_ind = 0

# Define the window layout
size = (300, 300)
layout = [
    [sg.Text(text="Current image", key="-TEXT-")],
    [sg.Image(size=size, key='-IMAGE-')],
    [sg.Button("Bench (b)", key="-BENCH-"), sg.Button('Remove (r)', key="-REM-"), sg.Button('Skip (s)', key="-SKIP-")],
]

# Create the form and show it without the plot
window = sg.Window(
    "Image selection",
    layout,
    finalize=True,
    element_justification="center",
    return_keyboard_events=True,
    font="Helvetica 18",
)

while base_ind < num_imgs:
   
    name = fakebs[base_ind]
    index = int(name.split("_")[1])
    print(base_ind, index)
    window['-TEXT-'].update(f"Image {index}, step {base_ind} of {num_imgs}")

    # Resize PNG file to size (300, 300)
    im = Image.open(os.path.join(datadir,name))
    im = im.resize(size, resample=Image.Resampling.BICUBIC)

    image = ImageTk.PhotoImage(image=im)
    window['-IMAGE-'].update(data=image)

    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    if event in ("-BENCH-", "b", "B"):
        if index not in benchmarks:
            benchmarks.append(index)
        print("benchmarked",index)
    elif event in ('-REM-', "r", "R"):
        if index not in to_remove:
            to_remove.append(index)
        print("removed",index)

    base_ind += 1
    
#save updates
np.save(bechdir,benchmarks)
np.save(removedir,to_remove)
np.save(lastind,base_ind)

window.close()