import matplotlib.pyplot as plt
import torch

def visualize_filters(model, binarize=False, is_hyper=True, in_feature=None):
    layer1 = None
    if is_hyper and in_feature is not None:
        filters = model.forward_filters(z=in_feature)
        layer1 = filters['0.weight']
        tensor = layer1.data.cpu().numpy()
    else:
        body_model = [i for i in model.children()][0]
        layer1 = body_model[0]
        tensor = layer1.weight.data.cpu().numpy()
    plot_kernels(filters=tensor,binarize=binarize)

def plot_kernels(filters, num_cols=6, binarize=False):
    print("Plotting {} kernels".format(filters.shape))
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    print("Min Weight: {}".format(f_min)) 
    print("Max Weight: {}".format(f_max)) 
    filters = (filters - f_min) / (f_max - f_min)
    if len(filters.shape) > 4:
        print("Filter size > 4! Squeezing Tensor")
        filters = filters.squeeze(0)
    if binarize:
        filters = filters >= 0.5
    # get number of filters
    n_filters = filters.shape[0] // 4
    # calculate number of rows
    num_rows = (n_filters + num_cols - 1) // num_cols
    # create figure and axes
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    axs = axs.flatten()
    # plot filters
    for i in range(n_filters):
        f = filters[i, ...]
        ax = axs[i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(f[0, :, :], cmap='gray')
    # hide any remaining axes
    for j in range(n_filters, len(axs)):
        axs[j].axis('off')
    plt.show()

def visualize_single(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = img.squeeze()
    plt.imshow(img, cmap="gray")
    plt.show()

def debug3d(img,skel):
    fig = plt.figure()

    # Subplot for img[0] as voxels
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    img_binary = img[0].detach().squeeze().cpu().numpy() > 0.5  # Threshold to create a binary array
    ax1.voxels(img_binary, facecolors='blue', edgecolor='k')
    ax1.set_title('Image Voxel Plot')

    # Subplot for skel[0] as voxels
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    skel_binary = skel[0].detach().squeeze().cpu().numpy() > 0.5  # Threshold for binary skeleton
    ax2.voxels(skel_binary, facecolors='red', edgecolor='k')
    ax2.set_title('Skeleton Voxel Plot')

    plt.show()

def viz3dTriplet(img,skel, pred):
    fig = plt.figure()

    # Subplot for img[0] as voxels
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    img_binary = img[0].detach().squeeze().cpu().numpy() > 0.5  # Threshold to create a binary array
    ax1.voxels(img_binary, facecolors='blue', edgecolor='k')
    ax1.set_title('Image Voxel Plot')

    # Subplot for skel[0] as voxels
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    skel_binary = skel[0].detach().squeeze().cpu().numpy() > 0.5  # Threshold for binary skeleton
    ax2.voxels(skel_binary, facecolors='red', edgecolor='k')
    ax2.set_title('Skeleton Voxel Plot')

    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    pred_skel_binary = pred[0].detach().squeeze().cpu().numpy() > 0.5  # Threshold for binary skeleton
    ax2.voxels(pred_skel_binary, facecolors='green', edgecolor='k')
    ax2.set_title('Predicted Skeleton Plot')

    plt.show()

def vizTriplet(img,skel, pred, title_1="Image", title_2="Skel", title_3="Pred"):
    fig = plt.figure()

    # Subplot for img[0] as voxels
    ax1 = fig.add_subplot(1, 3, 1)
    img_binary = img[0].detach().squeeze().cpu().numpy() > 0.5  # Threshold to create a binary array
    ax1.imshow(img_binary, cmap="gray")
    ax1.set_title(title_1)
    ax1.axis('off')


    # Subplot for skel[0] as voxels
    ax2 = fig.add_subplot(1, 3, 2)
    skel_binary = skel[0].detach().squeeze().cpu().numpy() > 0.5  # Threshold for binary skeleton
    ax2.imshow(skel_binary, cmap="gray")
    ax2.set_title(title_2)
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    pred_skel_binary = pred[0].detach().squeeze().cpu().numpy() > 0.5  # Threshold for binary skeleton
    ax3.imshow(pred_skel_binary, cmap="gray")
    ax3.set_title(title_3)
    ax3.axis('off')

    plt.show()