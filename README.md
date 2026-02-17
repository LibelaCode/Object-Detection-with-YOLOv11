# Object-Detection-with-YOLOv11
This repo is made to log information and note down the process of setting up and using the YOLO image detection model from ultralytics. It also serves as a backup for my own projects and an example format for how a working training setup looks like. <br>
Instruction docs located in this file below. 
<br>
<br>
**Requirements**: 
<ul>
  <li> Have pip installed </li>
  <li> Have pytorch installed </li>
  <li> Have python installed (anything above version 3 should work) </li>
  <li> Have OIDv4 toolkit: Link to repo (https://github.com/EscVM/OIDv4_ToolKit) </li>
</ul>

<p> The folders I have uploaded has everything (except the OIDv4 toolkit and it's respective CSV files) used to train YOLOv11n inside </p>
<p> This is also written assuming that the dataset has been downloaded via OIDv4 toolkit as per the requirements. </p>

<br>

<h2> Warning: </h2>  
<ul>
<li> DO NOT USE OIDv4_ToolKit FOLDER AS REAL TRAINING FOLDER. OPEN IT SOMEWHERE ELSE PREFERABLY MORE CLEAR. </li>
<li> Use "/" for everything filepath instead of "\" or else it wont work </li>
<li> There is a general incompatability error with using labels from OpenImages. A script must be ran before adding the labels to the main dataset folder so that it can convert class ID in the label file. More info below. </li>
</ul>

<br>

<h2> General Folder Structure: </h2>

YOLO requires a strict folder structure to train and validate. <br>
For the purpose of clarity, it is recommended that you name the folder where the data (images, labels) is located to "dataset". The structure inside this "dataset" folder should not change and should follow the format below:

~~~
C:/Folder/Folder/dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
~~~

It is not affected by custom directories but everything starting from "dataset/" should follow the file structure as shown in the figure. <br>
<br>
**Warning**: folder names are case sensitive on Linux so use all lowercase (very important)


<br>


<h2> Importing images from the OID folder to the dataset: </h2>

You can find the OID folder in 
~~~
(drive name):\Folders...\OIDv4_ToolKit
~~~
The default location is 
~~~
C:\Users\name of PC admin\OIDv4_ToolKit
~~~

Once you are in you should see a folder called OID with 2 folders inside: <br> 

csv_folder with 2 files: 

<ul> 
  <li> "class-descriptions-boxable.csv" which contains the class IDs for OpenImages classes. Use it to track the class ID of the class label you are trying to convert in the "convert_openimages_to_yolo" script </li>
  <li> "train-annotations-bbox.csv" which is a 1GB file that is used for idk because it's used in the "convert_openimages_to_yolo" script as well just keep it as is </li>
</ul>


Dataset folder with "train" folder inside:

<ul> 
  <li> This is where the images are saved sorted by class name </li>
  <li> Inside the class folders you'll find the images and another folder called "Label" </li>
  <li> "Label" contains the text files with the annotation info that must have their class IDs converted with "convert_openimages_to_yolo" script before adding to actual training folder. This will become important later in the label conversion section </li>
</ul>

<br>

<h2> Dataset Rules: </h2> 

**VERY IMPORTANT**: **File name (.jpg) and label (.txt) name HAVE to match. If they don't you will crash mid training**

Example of not working:
~~~
apple_01.jpg
apple_001.txt
~~~
Example of working:
~~~
apple_01.jpg
apple_01.txt
~~~
File order/position/index/sorting inside the folder itself does not matter, it only matters that file names match. <br>

To assist with this issue there is a **check_filename** script where you can check 2 filepaths and see if the raw string data (names) inside the folder matches with each other.

<br>
<h2> What is data.yaml? </h2>

data.yaml is the config file used to change the path of the dataset and training/validation directories and is used in the training command. It has:
~~~
path: #This should be the filepath for the dataset itself. 

train: #add your training directory from the dataset here
val:   #add your validation directory from the dataset here

nc: 3 #This is to count the number of classes and their class labels
names: 
  0: class 1 
  1: class 2
  2: class 3
  ...
~~~
To clarify for NC: YOLO uses a numerical number system per class so if we designate 0: class 1, this means the item related to class 1 now have the label id 0. This will be important in the label conversion portion of this README. <br>
Another clarification: train and validation paths should be one folder because all classes are added into the same folder after conversion. 

<br>

<h2> Label Conversion: </h2>

**VERY IMPORTANT**: **OIDv4 label annotation is not compatible with YOLO so you should run the "convert_openimages_to_yolo" script first and path it to a seperate folder (so that you can check the .txt annotation file to verify the class ID) before you add them to the main folder** <br>

The incompatability in question is that YOLO expects class ID like "/m/02fh7f" instead of "Eraser".etc <br>
This is because YOLO doesn't support class IDs with letters (ie.eraser) so you have to change the class name to the actual class ID or a designated number <br>
This causes an error while training that looks like this:
~~~
yaml.scanner.ScannerError: while parsing a quoted scalar in "<unicode string>", line 1, column 7 did not find expected hexdecimal number in "<unicode string>", line 1, column 12
~~~
To fix this error, use the **"convert_openimages_to_yolo"** script and follow the instructions inside as well. It is important that the labels are converted first and then split into train and val to prevent headaches. 

</br>

<h2> What is the most efficient way to do label conversion on a large dataset? </h2>
Once you have downloaded data using the OIDv4 Toolkit, you may have noticed that the images and the labels are in a folder called train. If you can't find it here's the default directory: 

~~~
C:\Users\"your name"\OIDv4_ToolKit\OID\Dataset\train\(the dataset for the training item)
~~~

Inside this folder you will see all your images and a folder called "labels". It is exactly what it means and contains the annotation labels for each image you have downloaded. <br>

Afterwards you should follow these steps:
<ul>
  <li> Get the images and labels into 2 folders inside a seperate directory. For the sake of clarity let's say we have the folder structure "item" with "images" and "labels" inside. </li>
  
~~~
├── item/
    ├── images/
        ├── your images here
    ├── labels/
        ├── your labels here
~~~

  <li> After copying over the images and labels sort them by name so that when you split them for training and validation it's far easier. </li>
  <li> Before you convert the label data, open the "class-descriptions-boxable.csv" file in the OIDv4 directory (C:\Users\"your name"\OIDv4_ToolKit\OID\csv_folder) and find the class for the item you are training. 
  <li> Once you have the class ID, read the instructions and change the variables inside the "convert_openimages_to_yolo" script to match the .yaml file. </li>
  <li> After conversion, you may notice a bug inside the label text files. It can be resolved using the solution in the next section. </li>
  <li> Run the "check_filename" script to see if there are any issues, then re-sort by name to make sure both directories look identical. </li>
  <li> Do a 80/20 split for both images and labels with 80% going into training and 20% going into validation. Recall back to the "General Folder Structure" structure if this seems confusing. </li>
  <li> For safe measure you can split first into a new directory and then re-run "check_filename" to fully confirm everything is ready to be sent into the main dataset folder. </li>

~~~
├── item/
    ├── images/
        ├── images-train
        ├── images-val
    ├── labels/
        ├── labels-train
        ├── labels-val
~~~
  
</ul>
TLDR; Convert labels first, then split. 

<br>

<h4> Cases that can happen: </h4> 
In the event that the labels are already in the dataset folders without conversion. Just delete the labels and run the script once for train and another time for validation. Don't forget to change directories and make sure the files have been successfully converted. <br>

In the event that the data conversion adds 2 identical labels with different class name onto one object like this:

~~~
Apple         0.64    284.29372   188.16   481.50664  
1             0.64    284.29372   188.16   481.50664  
~~~

delete the text file and rerun the script. It should keep the new labels and delete the old incorrect one. 



Correct Labels follow this format: 

~~~

<class_id> <x_center> <y_center>  <width>  <height>

Apple         0.64    284.29372   188.16   481.50664  (WRONG)

/m/014j1m     0.64    284.29372   188.16   481.50664  (CORRECT)

~~~

*multiple lines in one label is also correct (it means it has multiple annotations/bounding boxes) just make sure it has correct class ID. 

<br> 
<h2> How to train? </h2>

Use this as a example run:

~~~
yolo train model=models/yolo11n.pt data=data.yaml epochs=1 imgsz=640 batch=8
~~~
Once that works try:
~~~
yolo train model=models/yolo11n.pt data=data.yaml epochs=100 imgsz=640 batch=16 patience=20
~~~
To get more accurate training results:
~~~
yolo train model=models/yolo11n.pt data=data.yaml epochs=20 imgsz=640 batch=16 patience=40
~~~
if out of memory reduce batch to 8

look for: 
<ul>
  <li> box_loss = should steadily go down </li>
  <li> cls_loss = should drop quickly for single class </li>
  <li> mAP50    = should increase over time </li>
</ul>

<br> 
<h2> Training metrics: </h2>

Epochs:
<ul> 
  <li> Effect: More epochs = more learning = higher accuracy (until overfitting) </li>
  <li> Too few  --> underfitting </li>
  <li> Too many --> overfitting / wasted time</li>
  
</ul>

<h4> How many epochs is okay? </h4>
<ul>
  <li> 100 epochs     = OK </li>
  <li> 200–300 epochs = better convergence </li>
  <li> 500+           = only if strong augmentation </li>
</ul>


Patience: 
<ul> 
  <li> Controls Early Stopping rate. Change to 0 to completely remove early stopping. Usually used to save time and keep the best weights. </li>
  <li> patience=50 means if validation mAP does not improve for 50 consecutive epochs, training stops automatically. </li>
  <li> Small patience (10–20) : stops early, safer but may undertrain </li>
  <li> Large patience (50–100): allows longer fine-tuning, better for small datasets </li>
</ul>

<br>

<h2> Training time and CPU/GPU based training: </h2>

**You can skip this part if you don't own an NVIDIA GPU.** <br> 

Training YOLO can be done exclusively on a CPU but it also supports CUDA based GPU training which significantly speeds up the training process. <br><br>
Training on GPUs allows iteration on augumentations and hyperparameters and most modern models such as YOLO are built for GPU based training. <br> <br>
The only limitations being VRAM and the CUDA requirement which means that for smaller devices with no NVDIA graphics card, CPU based training is the only option. The GPU variant of pytorch also must be installed to support CUDA based training. <br> 

The example CMD command for installing pytorch can be found here: 
~~~
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 <br> 
~~~
The up to date command can be found on pytorch's official website: https://pytorch.org <br> 

There is a script on the website as well that allows to check if pytorch is installed. <br> 

<br> 
<h2> Where to check training results? </h2>

~~~
runs/detect/train/
├── weights/
│   ├── best.pt   <-- use this
│   └── last.pt
├── results.png
└── confusion_matrix.png
└── other images and data
~~~
<br> 
<h2> How to use model? </h2>

The trained YOLO model is used together with the OpenCV library and a commercial webcam, although a laptop webcam can also be used. <br> 
The file used to run the model is in the zip file along with all the other resources required. 

<h2> Known Issues! </h2>

Visual noise can affect the detection rate by about effectively 30 to 40%. Bright lights, reflections and object obstructions all affect the model to an extent. To effectively remedy this issue testing should be done in a clear environment (such as a white wall or a clean floor), although it will not represent real life situations. Aside from that more training should be done to continue remedying this issue. The model will be updated every run to make sure users will get the latest option available.  






