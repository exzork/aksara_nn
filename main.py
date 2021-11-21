import imutils
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
    epoch = epoch_text.get("1.0",END)
    nn_aksara.training(epoch=int(epoch))

label_epoch = Label(root,text="Epoch : ",anchor='e').grid(sticky=E, row=3,column=0)
epoch_text = Text(root,height=1,width=10)
epoch_text.grid(sticky=W,row=3,column=1)
btn_training = Button(root, text="Start Training", fg="black", command=start_training,width=50).grid(row=4,column=0,columnspan=2)

def select_image_predict():
    path_new = fd.askopenfilename(title="Pilih gambar yang akan diprediksi", initialdir="./")
    img_original_new_cv = cv2.imread(path_new)

    if img_original_new_cv is None:
        return
    
    img_original_new_cv_200 = imutils.resize(image=img_original_new_cv,width=200)
    img_original_new_cv_200 = cv2.cvtColor(img_original_new_cv_200,cv2.COLOR_BGR2RGB)
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

btn_select_file = Button(root, text="Pilih Gambar dan Prediksi",fg="black",command=select_image_predict,width=50).grid(row=5,column=0,columnspan=2)

label_predicted_text = Label(root,text="Prediksi Aksara : ",anchor="e").grid(sticky=E,row=6,column=0)
label_predicted = Label(root,text="",anchor="w")
label_predicted.grid(sticky=W, row=6,column=1)

root.mainloop()
