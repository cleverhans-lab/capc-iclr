from torchsummary import summary
import argparse

from architectures.resnet import ResNet10
from architectures.resnet import ResNet18
from models import VGG

parser = argparse.ArgumentParser()
parser.add_argument('--num-classes', type=int, default=10,
                    help='Number of classes.')
parser.add_argument('--dataset',
                    type=str,
                    # default='svhn',
                    default='cifar10',
                    help='Dataset.')
args = parser.parse_args()

model = ResNet10(name='resnet8', args=args).cuda()
# model = VGG(name='vgg-svhn', args=args).cuda()
# model = VGG(name='vgg-cifar', args=args).cuda()

summary(model, input_size=(3, 32, 32))