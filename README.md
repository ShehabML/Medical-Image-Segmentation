
# Medical Image Segmentation using a U-Net architecture

![image](https://github.com/ShehabML/Medical-Image-Segmentation/assets/99077516/3e1662a1-b8b4-440a-b9e1-030248f5f6b7)


In this project i try to implement a U-Net architecture from scratch to apply Image Segmentation on medical tools during surgical operations.   
This project is mainly for learning and applying the U-Net Architecture as stated by this   
[U-Net research paper](https://www.researchgate.net/publication/276923248_U-Net_Convolutional_Networks_for_Biomedical_Image_Segmentation).


## Installation

In order to use this project and try it yourself all you have to do is install the required libraries and then clone the rep, after that you can just run the notebook.  

```bash
  git clone https://github.com/ShehabML/Medical-Image-Segmentation/tree/main
```

As for the required libraries you need to install :     
- numpy   
- v2     
- glob    
- tqdm    
- sklearn 
- lbumentations  
- matplotlib  
- tensorflow >= 2.9.0


## Usage/Examples
 To use the network with your own dataset you can skip to the implementation of the network itself as I start by implementing the data in Colab via Kaggle drive 

If you want to start immedietly with the same data set as i used start py impelmenting the kaggle data set via pandas:

    df = pd.read_csv('dataset.csv')
    
from here you can apply the same functions and run the notebook smoothly ( i hope ).

for the training stage you can use this function:

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        shuffle=False
    )

![image](https://github.com/ShehabML/Medical-Image-Segmentation/assets/99077516/76a85c09-eb1e-4f26-9f43-2e9cea66a5bd)


as for the testing and evaluation use the following:

    y_pred = model.predict(x)[0]
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.int32)

            """ Saving the prediction """
            save_image_path = f"results/{name}"
            save_results(image, mask, y_pred, save_image_path)

            """ Flatten the array """
            y = y.flatten()
            y_pred = y_pred.flatten()

![image](https://github.com/ShehabML/Medical-Image-Segmentation/assets/99077516/c953499e-19ad-4bb3-8785-bd001ffe1306)
            

## Acknowledgements

 - [Dataset](https://www.kaggle.com/datasets/aithammadiabdellatif/binarysegmentation-endovis-17?select=BinarySegmentation)

 - [U-Net Architecture Research Paper](https://www.researchgate.net/publication/276923248_U-Net_Convolutional_Networks_for_Biomedical_Image_Segmentation)

 - [Eng. Ahmed Ibrahim U-Net explained](https://www.youtube.com/playlist?list=PLyhJeMedQd9RBOFDHynaDSu8BlRMCMB-y)


