# ML Engineer Tech Test

This is a simple tech test asking you to write some Python program with a purpose to verify your learning capability and Python skills. 
Please note that we do expect you to have sufficient Python skills but not on the specific tech stack required. The expectation
is that if you don't know about something, learn how to use it by reading and trying to solve the problem. There are 
plenty of tutorials and examples online, and you can Google as much as you like to complete the task. 

Do get prepared on explaining what you have done, especially when third party code or tutorial code have been used.

## Overall requirement
1. Once the solution is finished, please store it in a public Git repository on GitHub (this is free to create) and share the link with us

## Task 1
Write a python script to:
1. Read the input from `gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv` and load it in a Pandas Dataframe.
2. Split the dataset in 3 splits: train, validation, test with ratio of 60 (train) / 20 (validation) / 20 (test)
3. Perform any feature engineering you might find useful to enable training. It's not required that you create new features.
4. Train an ML model using XGB to predict whether a pet will be adopted or not `Adopted` is the target feature. You will need to use the validation to assess early stopping. You won't need to hypertune any parameter, the default parameters will be sufficient, with the exception of the number of trees which gets tuned by the early stopping mechanism.
5. The script needs to log to the user the performances of the model in the test set in terms of F1 Score, Accuracy, Recall.

Save the model into `artifacts/model` and make sure the folder is <b>not</b> git ignored.

## Task 2
Write a python script to:
1. Load the data from `gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv`
2. Uses the model you trained in the previous step to score all the rows in the CSV, excluding of course the header.
3. Save the output into `output/results.csv` and make sure all files in the `output/` directory is git ignored.
4. Add a unit test to the prediction function.

The output needs to follow the following format:
```
Type,Age,Breed1,Gender,Color1,Color2,MaturitySize,FurLength,Vaccinated,Sterilized,Health,Fee,PhotoAmt,Adopted,Adopted_prediction
Cat,3,Tabby,Male,Black,White,Small,Short,No,No,Healthy,100,1,Yes, No
```
