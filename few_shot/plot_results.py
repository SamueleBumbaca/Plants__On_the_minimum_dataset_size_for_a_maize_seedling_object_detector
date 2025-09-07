import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pdf2image import convert_from_path

# Assuming you have already created your subplots and plotted your data
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

pdf_path = '/mnt/e/PhD/Paper2/plots/vitb/1_50shot/SAGIT22_C_4175_2022_05_27-65-111.tif.pdf'
image1 = convert_from_path(pdf_path)[0]
pdf_path = '/mnt/e/PhD/Paper2/plots/vitb/2_50shot/SAGIT22_C_4178_2022_04_29-22-35.tif.pdf'
image2 = convert_from_path(pdf_path)[0]
pdf_path = '/mnt/e/PhD/Paper2/plots/vitb/3_50shot/SAGIT22_C_4178_2022_05_10-49-79.tif.pdf'
image3 = convert_from_path(pdf_path)[0]
# Plot your data
axs[0].imshow(image1)
axs[0].axis('off')  # Hide the axes
axs[1].imshow(image2)
axs[1].axis('off')  # Hide the axes
axs[2].imshow(image3)
axs[2].axis('off')  # Hide the axes

# Create a ScalarMappable for the colorbar
norm = mcolors.Normalize(vmin=0, vmax=1)  # Adjust vmin and vmax as needed
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])

# Adjust subplot parameters to minimize white space
plt.subplots_adjust(left=0.01, right=0.87, bottom=0.01, top=0.99, wspace=0.02)

# Add colorbar to the right of the subplots
cbar_ax = fig.add_axes([0.89, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Score')

plt.show()