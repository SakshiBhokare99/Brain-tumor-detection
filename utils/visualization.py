import matplotlib.pyplot as plt

def show(img,mask):
    plt.subplot(1,2,1)
    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.imshow(mask,cmap="gray")

    plt.show()