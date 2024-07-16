def denormalize(image, mean, std): # function to denormalize images
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(image.device)
    return image * std + mean

def normalize(image, mean, std): # function to normalize images
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(image.device)
    return (image - mean) / std
import torch.nn.functional as F

def fgsm2(image, dataGrad, mean, std):

    height = 32 # dimensions of images
    center = 8 # start of center piece
    centerEnd = 24 # end of center piece
    epsilonCenter = 2/255 # epsilon at center location
    epsilonOuter = 8/255 #epsilon for the rest of the area
    denormImage = denormalize(image, mean, std) # denormalize the image before processing it

    changedImage = denormImage.clone().detach().to(device) # clone the image and detach it

    changedImage[:, :, center:centerEnd, center:centerEnd] += epsilonCenter * torch.sign(dataGrad[:, :, center:centerEnd, center:centerEnd]) # add epsilon to the center piece of the image
    # add epsilon to the outer pieces of the image
    changedImage[:, :, :center, :] += epsilonOuter * torch.sign(dataGrad[:, :, :center, :])
    changedImage[:, :, centerEnd:, :] += epsilonOuter * torch.sign(dataGrad[:, :, centerEnd:, :])
    changedImage[:, :, :, :center] += epsilonOuter * torch.sign(dataGrad[:, :, :, :center])
    changedImage[:, :, :, centerEnd:] += epsilonOuter * torch.sign(dataGrad[:, :, :, centerEnd:])

    changedImage = torch.clamp(changedImage, 0, 1) # clamp the image
    changedImage = normalize(changedImage, mean, std) # normalize the image

    return changedImage

def test(model, test_loader, mean, std):
    model.eval() # Set the model in evaluation mode
    alteredCorrect = 0 # for trackign accuracy
    total = 0 # for trackign accuracy
    for images, labels in test_loader: # Iterate over the test data
        images, labels = images.to(device), labels.to(device) # Move the images and labels to the device
        images.requires_grad = True # require the gradients for the inputs

        out = model(images) # Forward pass the data through the model
        loss = F.cross_entropy(out, labels) # Calculate the loss
        loss.backward() # Backward pass to calculate the gradients
        model.zero_grad() # Zero all existing gradients

        changedImages = fgsm2(images, images.grad.data, mean, std) # Generate adversarial examples using FGSM

        out = model(changedImages) # Predict class labels for the perturbed images
        pred = out.argmax(dim=1, keepdim=True) # Get the predicted class labels

        alteredCorrect += pred.eq(labels.view_as(pred)).sum().item() # Update total and correct predictions
        total += labels.size(0) # add to total

    final_acc = alteredCorrect/total #calculate final accuracy
    print(f'Accuracy on adversarial examples: {final_acc * 100:.2f}%')

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

test(net, test_loader, mean, std)
