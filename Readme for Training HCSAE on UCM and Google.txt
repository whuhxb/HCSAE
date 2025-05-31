1. To train HCSAE on UCM dataset, you should firstly git clone the code from webset using this command:

git clone https://github.com/whuhxb/HCSAE.git

2. Then, you could download the UCM dataset in zip from the given link, and put the data of "data_for_training" into the folder HCSAE_UCM to train the UCM dataset. 
The data in the folder "data_already_run" provides the trained weights and other related results for the UCM dataset, you can also test the result from the second CSAE layer with UCMPoolTest.m file.

HCSAE is an unsupervised two-layer "SAE-Convlution-Pooling" learning framework. To run the first layer of "SAE-Convolution-Pooling", you could run the command:

UCMSingleLayer.m

To run the second layer of "SAE-Convolution-Pooling", you could run the command:

UCMSecondLayer.m

In the unsupervised two-layer "SAE-Convlution-Pooling" architecture, the softmax classifier is only appended after the pooling layer and utilized to test the classification
 performance, in each layer the SAE feature learning is fully unsupervised trained without any label.

3. For the Google dataset, you can conduct the similar operation to the UCM dataset. Firstly, download and unzip the Google data from the given link, then put the data of 
"data_for_training" into the folder HCSAE_Google to train the Google dataset. The data in the folder "data_already_run" contains the trained weights and other related results 
for the Google dataset, you can also directly test the result from the second CSAE layer with the GooglePoolTest.m file.

To run the first layer of HCSAE for Google dataset, you could run the command:

GoogleSingleLayer.m

To run the second layer of HCSAE for Google dataset, you could run the command:

GoogleSecondLayer.m

To run the second layer of HCSAE for Google dataset, you could run the command:
GoogleSecondLayer.m

4. To test the trained results for UCM and Google dataset, download the HCSAE_data.zip from Google Drive: https://drive.google.com/file/d/1l04T-f_pCfe1iHRRUJqtZJhwdaBccsMu/view?usp=drive_link, and run the UCMPoolTest.m or GooglePoolTest.m in the "data_already_run" folder.
