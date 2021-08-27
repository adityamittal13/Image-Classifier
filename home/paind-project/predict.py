if gpu == 'yes': 
    device = 'cuda:0'
else:
    device = 'cpu'
device = device

if gpu and torch.cuda.is_available():
      model.cuda()

def process_image(image):
    transformations = transforms.Compose(transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                                             std = [0.229, 0.224, 0.225]))
    np_image = np.array(image)
    np_image.transpose()
    return np_image

    
def imshow(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()
    
    image = Image.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, topk=5):
    image = Variable(image.unsqueeze(0), volatile=True)

    if cuda:
        image = image.cuda()

    output = model.forward(image.float())
    ps = torch.exp(output)
    prob, idx = ps.topk(topk)
    return [y for y in prob.data[0]], [model.idx_to_class[x] for x in idx.data[0]]

model.to(device)
images, labels = next(iter(trainloaders)
images, labels = images.to(device), labels.to(device)
images2 = process_image(images)
images3 = imshow(images2.view(224, 224))
probs, classes = predict(images3 , model)
plt.countplot(data = classes, x = cat_to_name)
