import wget
import os
import tarfile

cwd = os.getcwd()
handbag_url = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2handbags.tar.gz'
shoes_url = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz'
maps_url = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz'
cityscapes_url = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz'
facades_url = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

wget.download(handbag_url)
wget.download(shoes_url)
wget.download(maps_url)
wget.download(cityscapes_url)
wget.download(facades_url)

with tarfile.open('./edges2handbags.tar.gz') as tar:
    tar.extractall()
    tar.close()

with tarfile.open('./edges2shoes.tar.gz') as tar:
    tar.extractall()
    tar.close()

with tarfile.open('./maps.tar.gz') as tar:
    tar.extractall()
    tar.close()

with tarfile.open('./cityscapes.tar.gz') as tar:
	tar.extractall()
	tar.close()

with tarfile.open('./facades.tar.gz') as tar:
	tar.extractall()
	tar.close()

os.remove('./edges2handbags.tar.gz')
os.remove('./edges2shoes.tar.gz')
os.remove('./maps.tar.gz')
os.remove('./cityscapes.tar.gz')
os.remove('./facades.tar.gz')
