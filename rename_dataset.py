import os

dataset_dir = "./dataset/"

labels = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]

for label in labels:
    files = os.listdir(dataset_dir+label)
    for i in range(0,len(files)):
        os.rename(dataset_dir+label+"/"+files[i],dataset_dir+label+"/{}.jpg".format(i))