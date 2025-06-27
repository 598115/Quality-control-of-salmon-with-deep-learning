# Quality control of salmon with deep learning

# NOTE THIS VERSION HAS NO TRAINING DATA PICTURES INCLUDED AS LERØY DOES NOT WANT TO PUBLISH THEM.

## Requirements

Python version 3.12.3

We recommend using `Anaconda` to manage the enviroment needed to run the project. The enviroment needed can be imported to Anaconda with the `DL-Leroy.yml` file
If u do not wish to use `Anaconda`, all the dependencies are listed in the `DL-Leroy.yml` file if its opened with a text editor.

## Project example
This image shows what our project does with a real result as an example.
#### 1. Image is classified as bad or good. Only good images are of high enough quality to be allowed to the segmentation step
#### 2. A segmentation model produces a predicted segmentation mask meant to detect any defects in the fish which has also been divided into zones
#### 3. The results are stored 

![Project example](https://imgur.com/2YqJ2UY.png)

## Project usage guide

The project has 3 main parts:
1. The `binary_classifier` folder which contains all code belonging to the binary classifier model including the `binary_model.ipynb` script which can be used to train and evaluate the model
2. The `Segmentation` folder which contains all code belonging to the segmentation model. This includes all util classes needed to process the training data and training results. The `train_model.ipynb` script within this folder is the script to run all code needed to train and evaluate the segmentation model.
3. The `run_pipeline.ipynb` script found in the root of the project is the script used to run the projects pipeline after the models are trained. Together with the saved models and code in the `pipeline_utils` folder, it makes classifications on selected images and saves the results in the `results.csv` file.

##### NOTE: The trained models are too large to save on github (even with github LFS). To run the pipeline the models will have to be trained locally. Training and validation data is provided in the project files for this. Minimum 8GB memory is needed for training and a dedicated GPU is recommended. The project has been designed for usage with a dedicated GPU and we can not gurarantee how it performs with CPU only. Once the training scripts are run, the pipeline (`run_pipeline`) should be useable. See steps under for further training details.

### Running the pipeline

1. ##### Selecting an image to classify.
   
   ![Cell 1 of `run_pipeline.ipynb`. Selecting an image](https://imgur.com/Jk2kj6k.png)
   
   When running the cell you will get a popout window to select a local image, alternatively you can insert a local path to the image as a parameter to skip the popout window. You can for example use pictures found in the `binary_classifier/data/test` directory of the project
3. #### Binary classifier step.
   When running this cell the picture will either be approved or denied by the binary classification model. If it is approved, you can run the next cell, if not the denied result will be logged in `results.csv` and the rest of the cells will be skipped and you have to select another picture.
   
   ![Cell 2 of `run_pipeline.ipynb`. Binary classifier prediction](https://imgur.com/grzVqCW.png)
4. #### Segmentation step.
   When running this cell, the picture selected in step 1 will be predicted on by the locally saved segmentation model. When the prediction is done the results will be showed next to the original image.
   
   ![Cell 3 of `run_pipeline.ipynb`. Segmentation spot and zone prediction](https://imgur.com/bw96Csp.png)
6. #### Saving segmentation results.
   This cell saves the the results produced from the segmentation model in step/cell 3 to the `results.csv` file
   
   ![Cell 4 of `run_pipeline.ipynb`. Saving the results](https://imgur.com/Ht1zZSS.png)

### Training the binary model

#### To train the binary model, creating a locally saved model to use for running the pipeline, run all cells of `binary_classifier/binary_model.ipynb`.
This will:
1. Load the images saved in `binary_classifier/data` (which already has augmentation applied) into dataloader objects
2. Initialize the model architechture with imagenet weights, initialize criterion function and optimizer functions.
3. Train the model and save it in `binary_classifier/saved_models` as `binary_classifier_model.pth`
4. Evaluate the model on the test set with a confusion matrix and accuracy, recall and precision scores (`Optional`).

### Training the segmentation model

#### To train the segmentation model, creating a locally saved model to use for running the pipeline, run all cells of `Segmentation/train_model.ipynb`.
This will:
1. Preprocess the training data from label studio, preparing it for training usage (cells 2-4) (`Optional since already applied to repo data`)
2. Load data into dataloaders (cell 5)
3. Train the model, saving the model from the best epoch locally to `Segmentation/saved_models` as `saved_segm_model.pth` (cell 6)
4. Plot training loss (cell 7) (`Optional`)
5. Calculate average dice score of the model (cell 8) (`Optional`)
6. View predictions on a chosen image from a dataloader using batch and image index (Cell 9) (`Optional but dependant on step5/cell8 running first`)
7. View combined prediction of the models zone and spot outputs. Calculates which zone the spots are in with measuring overlapping. (cells 9-10) (`Optinal but dependent on step 5-6/cell8-9 being run first`) Note: This step will show a placeholder image instead of the original image. This is due to to the function being designed for the main `run_pipeline.ipynb` script. The combined prediction together with the original image can be viewed there.

