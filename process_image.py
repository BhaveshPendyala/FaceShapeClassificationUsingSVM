from PIL import Image, ImageFile, ImageOps
import matplotlib.pyplot as plt
import pathlib
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True


def plot_images(image, Caption1):
    plt.close()
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    xlabel = Caption1
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def gray_img(imdir, outdir):
    im = Image.open(imdir).convert('L')
    out_filename = outdir
    im.save(out_filename, 'JPEG', quality=100)


def equalize_img(imdir, outdir):
    im = Image.open(imdir)
    out_filename = outdir
    ImageOps.equalize(im).save(out_filename, 'JPEG', quality=100)


dirs = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

imagedir = "FaceShapeDatasetProcessingInput/training_set"
outdir = "FaceShapeDatasetProcessingOutput/training_set"

sub_dir = [q for q in pathlib.Path(imagedir).iterdir() if q.is_dir()]
for i in range(len(sub_dir)):
    images_dir = [p for p in pathlib.Path(sub_dir[i]).iterdir() if p.is_file()]
    for j in range(len(images_dir)):
        outfile = outdir + "/" + dirs[i] + "/" + os.path.split(str(images_dir[j]))[-1]
        gray_img(images_dir[j], outfile)
        equalize_img(outfile, outfile)

imagedir = "FaceShapeDatasetProcessingInput/testing_set"
outdir = "FaceShapeDatasetProcessingOutput/testing_set"

sub_dir = [q for q in pathlib.Path(imagedir).iterdir() if q.is_dir()]
for i in range(len(sub_dir)):
    images_dir = [p for p in pathlib.Path(sub_dir[i]).iterdir() if p.is_file()]
    for j in range(len(images_dir)):
        outfile = outdir + "/" + dirs[i] + "/" + os.path.split(str(images_dir[j]))[-1]
        gray_img(images_dir[j], outfile)
        equalize_img(outfile, outfile)


