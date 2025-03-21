# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/9419b281-738c-4032-a757-070eef27b457)

## DESIGN STEPS

## STEP 1:
Import the required libraries and load the dataset.
## STEP 2:
Encode categorical values and normalize numerical data.
## STEP 3:
Divide the dataset into training and testing sets.
## STEP 4:
Create a multi-layer neural network with activation functions.
## STEP 5:
Use an optimizer and loss function to train the model on the dataset.
## STEP 6:
Test the model and generate a confusion matrix.
## STEP 7:
Use the trained model to classify a new sample.
## STEP 8:
Show the confusion matrix, classification report, and predictions.


## PROGRAM

### Name: Kurapati Vishnu Vardhan Reddy
### Register Number: 212223040103
```
class PeopleClassifier(nn.Module):
     def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x

        

```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)


```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```



## Dataset Information

![image](https://github.com/user-attachments/assets/838ebb47-fb2f-4acf-bdd8-c78eb9d9a3ea)

## OUTPUT



### Confusion Matrix

![image](https://github.com/user-attachments/assets/dfdbb520-5435-4a1a-938b-bed15cdd7adc)

### Classification Report

![image](https://github.com/user-attachments/assets/7468ee34-7f17-4809-aa51-3a89fd2a793e)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/36f5ee64-5874-4c66-8aa3-397f68bf2717)

## RESULT
Include your result here
