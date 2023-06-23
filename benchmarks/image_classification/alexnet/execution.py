# Download an example image from the pytorch website
import urllib
from PIL import Image
from torchvision import transforms
import torch
import alexnet
import fullySimulatedAlexnet
import torch.nn.functional as F 

url, filename = ('https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg', 'resources/dog.jpg')
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

with open('resources/class_names_ImageNet.txt') as labels:
        classes = [i.strip() for i in labels.readlines()]

print("\nFully Simulated ALEXNET\n")
fully_simulated_alexnet_model = fullySimulatedAlexnet.fully_simulated_alexnet_model(pretrained=True)
with torch.no_grad():
    fully_simulated_output = fully_simulated_alexnet_model(input_batch)

fully_simulated_sorted, fully_simulated_indices = torch.sort(fully_simulated_output, descending=True)
fully_simulated_percentage = F.softmax(fully_simulated_output, dim=1)[0] * 100.0

fully_simulated_results = [(classes[i], fully_simulated_percentage[i].item()) for i in fully_simulated_indices[0][:5]]
for i in range(5):
    print('{}: {:.4f}%'.format(fully_simulated_results[i][0], fully_simulated_results[i][1]))

print("\nSimulated ALEXNET\n")
alex_model = alexnet.alexnet_model(pretrained=True)
with torch.no_grad():
    output = alex_model(input_batch)

# Format result for human understanding
sorted, indices = torch.sort(output, descending=True)
percentage = F.softmax(output, dim=1)[0] * 100.0

results = [(classes[i], percentage[i].item()) for i in indices[0][:5]]
for i in range(5):
    print('{}: {:.4f}%'.format(results[i][0], results[i][1]))