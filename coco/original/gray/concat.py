from PIL import Image
import matplotlib.pyplot as plt
import os


path = "C:/Users/Youjing Yu/PycharmProjects/deep-learning-cgh/coco/val2017/1/gray"
intenimg = os.listdir(path)


# Use splitext() to get filename and extension separately.
def get_name(path):
    filename = os.path.basename(path)
    (file, ext) = os.path.splitext(filename)
    return file


for a in intenimg:
    for b in intenimg:
        namea = get_name(a)
        nameb = get_name(b)
        print(namea)
        print(nameb)
        if namea == nameb[4:]:
            img1 = Image.open(a)
            img2 = Image.open(b)
            result = Image.new(img1.mode, (256 * 2, 256))
            result.paste(img1, box=(0, 0))
            result.paste(img2, box=(256, 0))
            result.save(namea+"concat.jpg")
            plt.imshow(result)
