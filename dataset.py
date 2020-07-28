import torch
import torch.utils.data as data
from torchvision import transforms, utils
import os

from PIL import Image






class RandomCropLongEdge(object):
  """Crops the given PIL Image on the long edge with a random start point.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    size = (min(img.size), min(img.size))
    # Only step forward along this edge if it's the long edge
    i = (0 if size[0] == img.size[0] 
          else np.random.randint(low=0,high=img.size[0] - size[0]))
    j = (0 if size[1] == img.size[1]
          else np.random.randint(low=0,high=img.size[1] - size[1]))
    return transforms.functional.crop(img, i, j, size[0], size[1])

  def __repr__(self):
    return self.__class__.__name__

class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):

      return transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
      return self.__class__.__name__


class Normal_dataset(data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.image = os.listdir(root)
        self.transform = transform

    def __getitem__(self, index):
        path = os.path.join(self.root, self.image[index])

        img = Image.open(path)
        img = img.convert('RGB')
        img_norm = self.transform(img)

        return img_norm

    def __len__(self):
        return(len(self.image))


def get_loaders(root, batch_size, normalization=False, augmentation=True):


    norm_mean = [0.485,0.456,0.406]
    norm_std = [0.229,0.224,0.225]
    image_size = 224

    if augmentation:
        train_transform = [CenterCropLongEdge(), transforms.Resize((image_size, image_size)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomAffine(degrees=[-10,10], scale=[0.9,1.2])]

    else:
        train_transform = [CenterCropLongEdge(), transforms.Resize(image_size)]

    if normalization: 
        train_transform = transforms.Compose(train_transform + [
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)])
    else:
        train_transform = transforms.Compose(train_transform + [
                    transforms.ToTensor()])


    dataset = Normal_dataset(root, train_transform)

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def read_picture(image_path, normalization=False):
    
    #norm_mean = [0.485,0.456,0.406]
    #norm_std = [0.229,0.224,0.225]

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]

    image_size = 256

    train_transform = [CenterCropLongEdge(), transforms.Resize(image_size)]
    if normalization:
        train_transform = transforms.Compose(train_transform + [
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)])
    else:
        train_transform = transforms.Compose(train_transform + [
                    transforms.ToTensor()])

    
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = train_transform(img)

    return img.unsqueeze(0)






if __name__ == '__main__':

    dataset = 'flower'
    data_root = os.path.join('train_data', dataset)
    datalodar = get_loaders(data_root, batch_size=64, normalization=False, augmentation=False)

    for img_batch in datalodar:
        img = img_batch[1, :, :, :]
        img = transforms.ToPILImage()(img.detach().cpu())
        img.save('sample.jpg')

        print(img_batch.size())

        exit()