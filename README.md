# Merge-Saving-Prediction
This repository includes a machine learning engine and training to predict the amount of time saving resulted from merging video tasks
The work was done by Shangrui Wu and Chavit Denninnart and the result is going to be submitted to IGSC 2020 conference.

Contact: wushangruide713@gmail.com

### Usage:
1. Clone this repo into your computer  
    git clone https://github.com/hpcclab/Merge-Saving-Prediction.git

2. Follow the official instructions to build [scikit-learn](https://scikit-learn.org/stable/install.html).  
The code has been tested successfully on Ubuntu 16.04 with scikit-learn 0.19.2.

3. GBDT.py [file_path] [table_name], the result will be saved in the current address.  
    file_path:         File input path, test data save in xls/xlsx format  
    table_name:        The name of the table
  
*Noticed: The data format needs to be consistent with that shown in the sample.xlsx
  
Quick Start: We provide a sample as an example, run the script as below,
        python GBDT.py sample.xlsx sample
