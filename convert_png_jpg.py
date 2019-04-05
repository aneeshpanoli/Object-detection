from PIL import Image
import os


# given the folder with pngs, replace png with jpeg
# NOT WORKING*************
def convert_png_to_jpg():
    image_files = os.listdir('scrshots')
    for i in image_files:
        print(os.path.splitext(i)[1])
        if os.path.splitext(i)[1] == '.png':
            img_path = os.path.join('scrshots', i)
            im = Image.open(img_path).convert("RGB")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            x, y = im.size
            bg.paste(im, im.size, im)
            bg.save('scrshots' + os.path.splitext(i)[0] + ".jpg")
            os.remove(img_path)


# this is for actual renaming
def rename_files():
    # give the folder where the images to be renamed are
    image_files = os.listdir('scrshots')
    for i in image_files:
        img_path = os.path.join('scrshots', i)
        # get the destination name from the following script
        output = rename_file('ball_')
        os.rename(img_path, output)


# this is to resolve name conflicts
# and return the name to save/rename
def rename_file(class_name):
    # help to start where you end
    loop_num = 1
    cwd = os.getcwd()
    sc_path = os.path.join(cwd, 'scrshots')
    fname = class_name + '{:03}'.format(loop_num) + '.jpg'
    output = os.path.join(sc_path, fname)
    while os.path.exists(output):
        loop_num += 1
        fname = class_name + '{:03}'.format(loop_num) + '.jpg'
        output = os.path.join(sc_path, fname)
    # output = os.path.join('screenShot', fname)
    return output


if __name__ == '__main__':
    # convert_png_to_jpg()
    rename_files()
