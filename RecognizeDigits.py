import numpy as np
from os import path
import pickle
import tkinter as tk
from PIL import Image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, denom):
    for i in range(layerLen[-1]):
        denom += np.exp(x[i])
    return np.exp(x)/denom


def funcz(W, a, b):
    return np.add(np.dot(W, a), b)


def go_forward(input):
    Alayer = [[] for i in range(layers)]
    Alayer[0] = np.array(input).flatten()
    z = [0 for i in range(layers-1)]
    denom = 0
    for i in range(layers-1):
        z[i] = funcz(Wmatrix[i], Alayer[i], bvec[i])
        Alayer[i+1] = sigmoid(z[i])
    Alayer[-1] = softmax(z[-1], denom)
    return np.argmax(Alayer[-1])


def clearCanvas():
    canv.delete('all')
    canv.create_rectangle(-10, -10, 400, 400, fill='black')


def paint(event):
    global brush_size, brush_color
    x1 = event.x - brush_size
    x2 = event.x + brush_size
    y1 = event.y - brush_size
    y2 = event.y + brush_size
    canv.create_oval(x1, y1, x2, y2, fill=brush_color, outline=brush_color)


def recognize():
    global answer_label, root
    root.update()
    localdirps = path.join(dir, 'digit.ps')
    localdirjpg = path.join(dir, 'digit.jpg')
    canv.postscript(file=localdirps, colormode='color')

    Image.open(localdirps).save(localdirjpg)
    image = Image.open(localdirjpg)
    width, height = image.size
    image = image.crop((2, 5, width-2, height-5))
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = image.convert('L')

    lum = np.array([[image.getpixel((j, i)) for j in range(28)]
                   for i in range(28)])
    lum = lum.flatten()/255

    answer = go_forward(lum)
    answer_label.config(text=str(answer))


dir = path.dirname(path.abspath(__file__))
foldername = 'DigitNeuralNetworkData'
dir = path.join(dir, foldername)
if (path.isdir(dir)):
    with open(path.join(dir, 'NetworkStructure.txt'), 'r') as f:
        lam = float(f.readline())
        layerLen = list(eval(f.readline()))
    with open(path.join(dir, 'Biases.dat'), 'rb') as f:
        bvec = pickle.load(f)
    with open(path.join(dir, 'Weights.dat'), 'rb') as f:
        Wmatrix = pickle.load(f)
layers = len(layerLen)

brush_size = 8
brush_color = "white"

root = tk.Tk()

root.title("Digits recognition")
root.geometry("300x450")
root.resizable(False, False)

answer_label = tk.Label(root, text="", font=35)
answer_label.pack(side=tk.BOTTOM, pady=(0, 40))

recognize_button = tk.Button(
    root, text="Recognize", command=recognize)
recognize_button.pack(side=tk.BOTTOM, pady=(0, 10))

clear_button = tk.Button(root, text="Clear", width=5, command=clearCanvas)
clear_button.pack(side=tk.BOTTOM, pady=(0, 10))

canv = tk.Canvas(root, width=300, height=300, bg="black")
canv.pack(side=tk.TOP)
canv.create_rectangle(-10, -10, 400, 400, fill='black')

root.update()

canv.bind("<B1-Motion>", paint)

root.mainloop()
