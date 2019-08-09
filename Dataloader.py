import torch.utils.data
def generate_LM_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        lengths = data_dict['x_target_length'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()
        
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict