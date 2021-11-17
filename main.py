import tkinter
from prepare_dataset import prepare_dataset
from nn_aksara import nn_aksara
from tkinter import *
from PIL import Image,ImageTk
import cv2
import numpy as np
from tkinter import filedialog as fd

#prepare_dataset.prepare("./dataset/") 
root = Tk()
root.title("Pengenalan Aksara Jawa dengan Backpropagation")


img_preprocessed = ImageTk.PhotoImage(image=Image.fromarray(np.ones((200,200),np.uint8)*200))
img_original = ImageTk.PhotoImage(image=Image.fromarray(np.ones((200,200),np.uint8)*200))

label_preprocessed_img_text = Label(root,text="Preprocessed Image").grid(row=0,column=1)
label_preprocessed_img = Label(root,image=img_preprocessed)
label_preprocessed_img.grid(row=1,column=1,rowspan=2)

label_original_img_text = Label(root,text="Original Image").grid(row=0,column=0)
label_original_img = Label(root,image=img_original)
label_original_img.grid(row=1,column=0,rowspan=2)

def start_training():
    nn_aksara.training()

btn_training = Button(root, text="Start Training", fg="black", command=start_training).grid(row=3,column=0)

def select_image_predict():
    path_new = fd.askopenfilename(title="Pilih gambar yang akan diprediksi", initialdir="/")
    img_original_new_cv = cv2.imread(path_new)
    
    img_original_new_cv_200 = cv2.resize(img_original_new_cv,(200,200),cv2.INTER_NEAREST)
    img_original_new = ImageTk.PhotoImage(image=Image.fromarray(img_original_new_cv_200))
    label_original_img.configure(image=img_original_new)
    label_original_img.image=img_original_new

    img_preprocessed_200 = cv2.resize(prepare_dataset.preprocess(img_original_new_cv),(200,200),interpolation=cv2.INTER_NEAREST)
    img_preprocessed_new = ImageTk.PhotoImage(image=Image.fromarray(img_preprocessed_200))
    label_preprocessed_img.configure(image=img_preprocessed_new)
    label_preprocessed_img.image=img_preprocessed_new

    predicted_text = nn_aksara.prediction(img_original_new_cv)
    label_predicted.configure(text=predicted_text)
    label_predicted.text=predicted_text

btn_select_file = Button(root, text="Pilih Gambar dan Prediksi",fg="black",command=select_image_predict).grid(row=3,column=1)

label_predicted_text = Label(root,text="Prediksi Aksara : ").grid(row=4,column=0)
label_predicted = Label(root,text="")
label_predicted.grid(row=4,column=1)

root.mainloop()
