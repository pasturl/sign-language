# sign-language
## DONE
### Retraining an Image Classifier using argentinian dataset
* Based in https://github.com/hthuwal/sign-language-gesture-recognition
* Dataset argentinian https://facundoq.github.io/datasets/lsa64/
* Notebook example in TF2 https://www.tensorflow.org/hub/tutorials/tf2_image_retraining

## TODO 
1. Use opencv to get data frames from webcam and make predictions in real time (https://gogul.dev/software/hand-gesture-recognition-p1)
2. Add layer to process individual prediction. Basic approach with moving average of predection. More complex approach with LSTM
3. Analyze model performance and do fine tuning
4. Use dataset with 200 signs https://chalearnlap.cvc.uab.cat/challenge/43/description/
5. Use more complex model architecture https://github.com/jackyjsy/CVPR21Chal-SLR

## Nice to have
* Create virtual environment (requeriments or docker)
* Create functions and parametrize

# References

## Datasets
* Argentinian with 64 signs https://facundoq.github.io/datasets/lsa64/
* English with 226 sign labels and 36,302 isolated sign video samples that are performed by 43 different signers in total https://chalearnlap.cvc.uab.cat/challenge/43/description/
* Spanish 91 different gestures, each gesture has 40 samples https://github.com/zparcheta/spanish-sign-language-db

## Models
* Pretrained CNN model + RNN https://github.com/hthuwal/sign-language-gesture-recognition
* Skeleton keypoints + Multiple CNN-3D

## Video processing
* Hand recognition https://gogul.dev/software/hand-gesture-recognition-p1
