import torch
from PIL import Image
import torchvision.transforms as transforms
#from DigitIdentifierNN import model as DINNM
from DigitIdentifierNN import DigitIdentifierNN, device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def predict_digit_by_image(path, model):
    # load image
    img = Image.open(path).convert('L')  # to grayscale
    img = img.resize((28, 28))  # resize 28 pixels

    # define the transformation, same as the one used in training
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        #transforms.RandomRotation(20),
        #transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        # transforms.RandomPerspective(distortion_scale=0.5),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    #img = transform(img)
    # apply transformation and add a batch dimension
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval() #set model to evaluation mode

    # forward pass thru model to get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)  # get the predicted class

    return predicted.item()


trainedmodel = DigitIdentifierNN().to(device)
trainedmodel.load_state_dict(torch.load('DigitIdentifierNN.pth', weights_only=True))
trainedmodel.eval()

print('iteration 1')
image_path0 = 'paintnum0.png'
predicted_digit = predict_digit_by_image(image_path0, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 0')
image_path1 = 'paintnum1.png'
predicted_digit = predict_digit_by_image(image_path1, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 1')
image_path2 = 'paintnum2.png'
predicted_digit = predict_digit_by_image(image_path2, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 2')
image_path3 = 'paintnum3.png'
predicted_digit = predict_digit_by_image(image_path3, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 3')
image_path4 = 'paintnum4.png'
predicted_digit = predict_digit_by_image(image_path4, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 4')
image_path5 = 'paintnum5.png'
predicted_digit = predict_digit_by_image(image_path5, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 5')
image_path6 = 'paintnum6.png'
predicted_digit = predict_digit_by_image(image_path6, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 6')
image_path7 = 'paintnum7.png'
predicted_digit = predict_digit_by_image(image_path7, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 7')
image_path8 = 'paintnum8.png'
predicted_digit = predict_digit_by_image(image_path8, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 8')
image_path9 = 'paintnum9.png'
predicted_digit = predict_digit_by_image(image_path9, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 9')

print('iteration 2')
image_path0 = 'paintnum0.png'
predicted_digit = predict_digit_by_image(image_path0, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 0')
image_path1 = 'paintnum1.png'
predicted_digit = predict_digit_by_image(image_path1, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 1')
image_path2 = 'paintnum2.png'
predicted_digit = predict_digit_by_image(image_path2, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 2')
image_path3 = 'paintnum3.png'
predicted_digit = predict_digit_by_image(image_path3, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 3')
image_path4 = 'paintnum4.png'
predicted_digit = predict_digit_by_image(image_path4, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 4')
image_path5 = 'paintnum5.png'
predicted_digit = predict_digit_by_image(image_path5, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 5')
image_path6 = 'paintnum6.png'
predicted_digit = predict_digit_by_image(image_path6, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 6')
image_path7 = 'paintnum7.png'
predicted_digit = predict_digit_by_image(image_path7, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 7')
image_path8 = 'paintnum8.png'
predicted_digit = predict_digit_by_image(image_path8, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 8')
image_path9 = 'paintnum9.png'
predicted_digit = predict_digit_by_image(image_path9, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 9')

print('iteration 3')
image_path0 = 'paintnum0.png'
predicted_digit = predict_digit_by_image(image_path0, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 0')
image_path1 = 'paintnum1.png'
predicted_digit = predict_digit_by_image(image_path1, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 1')
image_path2 = 'paintnum2.png'
predicted_digit = predict_digit_by_image(image_path2, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 2')
image_path3 = 'paintnum3.png'
predicted_digit = predict_digit_by_image(image_path3, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 3')
image_path4 = 'paintnum4.png'
predicted_digit = predict_digit_by_image(image_path4, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 4')
image_path5 = 'paintnum5.png'
predicted_digit = predict_digit_by_image(image_path5, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 5')
image_path6 = 'paintnum6.png'
predicted_digit = predict_digit_by_image(image_path6, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 6')
image_path7 = 'paintnum7.png'
predicted_digit = predict_digit_by_image(image_path7, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 7')
image_path8 = 'paintnum8.png'
predicted_digit = predict_digit_by_image(image_path8, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 8')
image_path9 = 'paintnum9.png'
predicted_digit = predict_digit_by_image(image_path9, trainedmodel)
print(f'This looks like the number {predicted_digit}, actual is 9')

'''
image_path = 'paintnum4.png'
model = DigitIdentifierNN().to(device)
predicted_digit = predict_digit_by_image(image_path, model)
print(f'This looks like the number {predicted_digit}')
'''