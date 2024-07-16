def denormalize(image, mean, std): # function to denormalize images
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(image.device)
    return image * std + mean

def normalize(image, mean, std): # function to normalize images
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(image.device)
    return (image - mean) / std
def test_adversarial(model, device, test_loader, epsilon, mean, std):
    model.eval() # Set the model in evaluation mode
    # Initialize variables to track accuracy
    correct = 0
    total = 0

    # Iterate through the test dataset
    for images, labels in test_loader:
        # Move tensors to the configured device (GPU if available)
        images, labels = images.to(device), labels.to(device)
        # Indicate that we require the gradients for the inputs
        images.requires_grad = True
        # Forward pass the data through the model
        outputs = model(images)
        # Calculate the loss using cross entropy
        loss = nn.CrossEntropyLoss()(outputs, labels)
        # Zero all existing gradients
        model.zero_grad()
        # Backward pass
        loss.backward()
        # Collect the gradient of the loss with respect to the input image
        data_grad = images.grad.data
        # Generate adversarial examples using FGSM
        perturbed_images = fgsm_attack(images, epsilon, data_grad, mean, std)
        # Predict class labels for the perturbed images
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        # Update total and correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = correct / total
    print(f'Accuracy on adversarial examples: {accuracy * 100:.2f}%')

def fgsm_attack(image, epsilon, data_grad, mean, std):
    # Denormalize the image
    denorm_image = denormalize(image, mean, std)

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create perturbed image by adjusting each pixel of the input image
    perturbed_image = denorm_image + epsilon * sign_data_grad
    # Clip the perturbed image to ensure it still falls within valid range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    norm_perturbed_image = normalize(perturbed_image, mean, std)

    # Return the perturbed image
    return norm_perturbed_image

# Define epsilon value
epsilon = 8/255
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

# Call the test_adversarial function
test_adversarial(net, device, test_loader, epsilon, mean, std)
