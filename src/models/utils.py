import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import os

def time_series_to_plot(time_series_batch, dpi=35, titles=None):
    """Convert a batch of time series to a tensor with a grid of their plots
    
    Args:
        time_series_batch (Tensor): (batch_size, seq_len, dim) tensor of time series
        dpi (int): dpi of a single image
        feature_idx (int): index of the feature that goes in the plots (the first one by default)
        n_images_per_row (int): number of images per row in the plot
        titles (list of strings): list of titles for the plots

    Output:
        single (channels, width, height)-shaped tensor representing an image
    """
    # Iterates over the time series
    fault_type_images = []
    # HINT: pi = 60 is good, size of 16, 3 is also good
    # iteration by fault types:
    for i, series in enumerate(time_series_batch.detach()):
        time_series_images = []
        for j in range(series.size(1)):

            fig = plt.figure(dpi=dpi, figsize=(16, 3))
            ax = fig.add_subplot(1, 1, 1)
            if titles:
                ax.set_title(titles[i])
            ax.plot(series[:, j].numpy())  # plots a single feature of the time series
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            time_series_images.append(data)
            plt.close(fig)
        fault_type_images.append(time_series_images)

    # Swap channel
    fault_type_images = [torch.from_numpy(np.stack(fti)).permute(0, 3, 1, 2) for fti in fault_type_images]
    # Make grid
    grids_image_list = [vutils.make_grid(fti.detach(), nrow=1) for fti in fault_type_images]
    return grids_image_list

def tensor_to_string_list(tensor):
    """Convert a tensor to a list of strings representing its value"""
    scalar_list = tensor.squeeze().numpy().tolist()
    return ["%.5f" % scalar for scalar in scalar_list]

class DatasetGenerator:
    def __init__(self, generator, seq_len=96, noise_dim=100, dataset=None):
        """Class for fake dataset generation
        Args:
            generator (pytorch module): trained generator to use
            seq_len (int): length of the sequences to be generated
            noise_dim (int): input noise dimension for gan generator
            dataset (Dataset): dataset providing normalize and denormalize functions for deltas and series (by default, don't normalize)
        """
        self.generator = generator
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        self.dataset = dataset

    def generate_dataset(self, outfile=None, batch_size=4, delta_list=None, size=1000):
        """Method for generating a dataset
        Args:
            outfile (string): name of the npy file to save the dataset. If None, it is simply returned as pytorch tensor
            batch_size (int): batch size for generation
            seq_len (int): sequence length of the sequences to be generated
            delta_list (list): list of deltas to be used in the case of conditional generation
            size (int): number of time series to generate if delta_list is present, this parameter is ignored
        """
        #If conditional generation is required, then input for generator must contain deltas
        if delta_list:
            noise = torch.randn(len(delta_list), self.seq_len, self.noise_dim) 
            deltas = torch.FloatTensor(delta_list).view(-1, 1, 1).repeat(1, self.seq_len, 1)
            if self.dataset:
                #Deltas are provided in original range, normalization required
                deltas = self.dataset.normalize_deltas(deltas)
            noise = torch.cat((noise, deltas), dim=2)
        else:
            noise = torch.randn(size, self.seq_len, self.noise_dim) 
        
        out_list = []
        for batch in noise.split(batch_size):
            out_list.append(self.generator(batch))
        out_tensor = torch.cat(out_list, dim=0)
         
        #Puts generated sequences in original range
        if self.dataset:
            out_tensor = self.dataset.denormalize(out_tensor)

        if outfile:
            np.save(outfile, out_tensor.detach().numpy())
        else:
            return out_tensor 



def get_latest_model_id(dir_name):
    model_ids = list()
    for d in os.listdir(dir_name):
        if os.path.isdir(os.path.join(dir_name, d)):
            try:
                model_ids.append(int(d.split("_")[0]))
            except ValueError:
                pass
    return max(model_ids) if len(model_ids) else 0


if __name__ == "__main__":
    model = torch.load('checkpoints/cnn_conditioned_alternate1_netG_epoch_85.pth') 
    gen = DatasetGenerator(model)
    print("Shape of example dataset:", gen.generate_dataset(delta_list=[i for i in range(100)]).size())
