import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#from predict_digit_by_image import predict_digit_by_image


#works with cuda if running python 3.10...cuda not working with torch on python 3.12 still not supported as of 9/20/24
# Check if CUDA is available, set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a basic neural network
#added another layer, more neurons to help improve real world accuracy
#as of 9.20.24, test accuracy is 96%...real world seems to be about 50%
class DigitIdentifierNN(nn.Module):
    def __init__(self):
        super(DigitIdentifierNN, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 256) #fully connected layer 1 input neurons to hidden neurons
        self.fc2 = nn.Linear(256, 128)  # fully connected layer 2 hidden to hidden
        self.fc3 = nn.Linear(128, 64) #added this layer 3 to add complexity and increase accuracy in real world situations
        self.fc4 = nn.Linear(64, 10) #fully connected l4 hidden to output...10 output neurons for 10 possible outputs 0-9

        #self.fc1 = nn.Linear(28 * 28, 128)  # fully connected layer 1 input neurons to hidden neurons
        #self.fc2 = nn.Linear(128, 64) #fully connected layer 2 hidden to hidden
        #self.fc3 = nn.Linear(64, 10)


        self.dropout = nn.Dropout(0.1) #randomly turn off subset of neurons during training, this case 10%

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) #dropout
        x = torch.relu(self.fc2(x))
        x = self.dropout(x) #dropout
        x = torch.relu(self.fc3(x))
        #x = self.dropout(x)
        x = self.fc4(x)
        #x = self.fc3(x)
        return x

# Set up transformations for the dataset
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#more robust transformations--Affine has to do with preserving straight lines
#turned off randomperspective, seemed to really decrease accuracy of the model
transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    #transforms.RandomPerspective(distortion_scale=0.5),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# Download and load the MNIST training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Download and load the MNIST test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Instantiate the model and move it to the GPU if available
model = DigitIdentifierNN().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) #Adam = Adaptive Moment Estimation

# Training loop--more epochs take more time but decrease loss/improve accuracy
epochs = 20
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        # Move data to the same device as the model (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

torch.save(model.state_dict(), 'DigitIdentifierNN.pth')
print("Training complete.")

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        # Move data to the same device as the model (GPU or CPU)
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")
print(correct, total)


'''
trainedmodel = DigitIdentifierNN()
trainedmodel.load_state_dict(torch.load('DigitIdentifierNN.pth'))
trainedmodel.eval()

image_path0 = 'paintnum0.png'
predicted_digit = predict_digit_by_image(image_path0, trainedmodel)
print(f'This looks like the number {predicted_digit}')
image_path1 = 'paintnum1.png'
predicted_digit = predict_digit_by_image(image_path1, trainedmodel)
print(f'This looks like the number {predicted_digit}')
image_path2 = 'paintnum2.png'
predicted_digit = predict_digit_by_image(image_path2, trainedmodel)
print(f'This looks like the number {predicted_digit}')
image_path3 = 'paintnum3.png'
predicted_digit = predict_digit_by_image(image_path3, trainedmodel)
print(f'This looks like the number {predicted_digit}')
image_path4 = 'paintnum4.png'
predicted_digit = predict_digit_by_image(image_path4, trainedmodel)
print(f'This looks like the number {predicted_digit}')
image_path5 = 'paintnum5.png'
predicted_digit = predict_digit_by_image(image_path5, trainedmodel)
print(f'This looks like the number {predicted_digit}')
image_path6 = 'paintnum6.png'
predicted_digit = predict_digit_by_image(image_path6, trainedmodel)
print(f'This looks like the number {predicted_digit}')
image_path7 = 'paintnum7.png'
predicted_digit = predict_digit_by_image(image_path7, trainedmodel)
print(f'This looks like the number {predicted_digit}')
image_path8 = 'paintnum8.png'
predicted_digit = predict_digit_by_image(image_path8, trainedmodel)
print(f'This looks like the number {predicted_digit}')
image_path9 = 'paintnum9.png'
predicted_digit = predict_digit_by_image(image_path9, trainedmodel)
print(f'This looks like the number {predicted_digit}')
'''