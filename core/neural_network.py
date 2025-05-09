import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

class SimpleNeuron(nn.Module):
    def __init__(self, input_size):
        super(SimpleNeuron, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

def train_model(model, data, labels, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def save_model(model, path="neural_network.pth"):
    torch.save(model.state_dict(), path)

def load_model(model, path="neural_network.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict(model, new_data):
    inputs = torch.tensor(new_data, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(inputs)
    return prediction.numpy().tolist()


# main.py

# from prediction_engine import PredictionEngine

# Only include this if you're still using numeric prediction somewhere
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# import json

# Optional: if you still need the numeric model
# class SimpleNeuron(nn.Module):
#     def __init__(self, input_size):
#         super(SimpleNeuron, self).__init__()
#         self.linear = nn.Linear(input_size, 1)

#     def forward(self, x):
#         return self.linear(x)

# def train_model(model, data, labels, epochs=100, lr=0.01):
#     criterion = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#     for epoch in range(epochs):
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     return model

# def save_model(model, path="neural_network.pth"):
#     torch.save(model.state_dict(), path)

# def load_model(model, path="neural_network.pth"):
#     model.load_state_dict(torch.load(path))
#     model.eval()
#     return model

# def predict(model, new_data):
#     inputs = torch.tensor(new_data, dtype=torch.float32)
#     with torch.no_grad():
#         prediction = model(inputs)
#     return prediction.numpy().tolist()


# ✅ Use the PredictionEngine for NLP/text-based predictions
# engine = PredictionEngine()
# query = "Can you remind me to call mom at 5 PM?"

# result = engine.predict_all(query)

# for key, value in result.items():
#     print(f"{key}: {value}")
