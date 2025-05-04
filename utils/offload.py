import torch
from enum import Enum
from loguru import logger

# [GPU, Layer, Page, Data]
class offloadData:
    # define a state enum DISK, CPU, GPU
    class LOCATION(Enum):
        DISK = 0
        CPU = 1
        GPU = 2
    
    def __init__(self):
        self.state = self.LOCATION.DISK
        self.tensor = None
        self.page_num = 0
        self.numGPU = 0
        self.lenData = 0
        self.numLayers = 0
        self.start_idx = 0  # start index of the page
        
    def assign(self, tensor, start_idx):
        self.tensor = tensor
        self.state = self.LOCATION.CPU
        self.page_num = tensor.size(2)  # Page is now at dim 2
        self.numLayers = tensor.size(1)  # Layer is at dim 1
        self.numGPU = tensor.size(0)
        self.lenData = tensor.size(3) # length of a page
        self.start_idx = start_idx
    
    @property
    def last_idx(self):
        return self.start_idx + self.page_num - 1
    
    def toDisk(self):
        self.state = self.LOCATION.DISK
        self.tensor = None
    
    def toCPU(self):
        if self.state == self.LOCATION.CPU:
            return
        self.state = self.LOCATION.CPU
        # create a tensor for now 
        self.tensor = torch.zeros(self.numGPU, self.numLayers, self.page_num, self.lenData)
    
    def delete(self):
        self.tensor = None
        self.state = self.LOCATION.DISK
        self.page_num = 0
    
    def removeTail(self):
        if self.state == self.LOCATION.CPU:
            self.tensor = self.tensor[:, :, :self.page_num - 1, :]  # Update to reflect the page dimension at dim=2
            self.page_num = self.tensor.size(2)  # Page number is now in dim 2
        else:
            self.page_num = self.page_num - 1

class offloadMetaData:
    def __init__(self, req_idx: int):
        self.data_list = []
        self.page_num = 0
        self.req_idx = req_idx
        self.promised_last_page_idx = 0
        self.received_last_page_idx = 0
    
    def promise(self, promise_pages):
        self.promised_pages = promise_pages
        self.promised_last_page_idx += promise_pages
    
    def append(self, tensor):
        start_idx = self.received_last_page() + 1
        page_num = tensor.size(2)  # Page is at dim 2
        d = offloadData()
        d.assign(tensor, start_idx)

        # if len(self.data_list) > 0:
        #     self.data_list[-1].removeTail()
        #     self.page_num -= 1
        self.data_list.append(d)
        self.page_num += page_num

        for data in self.data_list:
            logger.info(f"[offload_meta_data append] req_idx: {self.req_idx}, data.page_num: {data.page_num}, data.start_idx: {data.start_idx}, data.last_idx: {data.last_idx}")

        # assert self.page_num <= self.target_page_num, f"page_num exceeds target_page_num, page_num: {self.page_num}, target_page_num: {self.target_page_num}"
    
    def delete_tail(self):
        if len(self.data_list) > 0:
            self.data_list[-1].removeTail()
            self.page_num -= 1
            self.received_last_page_idx -= 1
            self.promised_last_page_idx -= 1
        for data in self.data_list:
            logger.info(f"[offload_meta_data delete_tail] req_idx: {self.req_idx}, data.page_num: {data.page_num}, data.start_idx: {data.start_idx}, data.last_idx: {data.last_idx}")

    def reconstruct(self):
        # print("[reconstruct] self.req_idx: ", self.req_idx)
        # print("[reconstruct] data_list: ", len(self.data_list))
        for data in self.data_list:
            # print("data.page_num: ", data.page_num)
            data.toCPU()
        tensor_list = [data.tensor for data in self.data_list]
        # Concatenate along the 'page' dimension (dim=2)
        if len(tensor_list) == 0:
            return None
        # reconstructed_tensor = torch.cat(tensor_list, dim=2).contiguous()
        # d = offloadData()
        # d.assign(reconstructed_tensor, self.data_list[-1].start_idx)
        # self.data_list = [d]
        # return reconstructed_tensor
        return torch.cat(tensor_list, dim=2).contiguous()
    
    def toDisk(self):
        for data in self.data_list:
            data.toDisk()
    
    def delete(self):
        for data in self.data_list:
            data.delete()
        self.data_list = []
        self.page_num = 0
    
    def received_last_page(self):
        last_idx = self.data_list[-1].last_idx if len(self.data_list) > 0 else 0
        logger.info(f"[offload_meta_data] req_idx: {self.req_idx}, len(data_list): {len(self.data_list)}, last_idx: {last_idx}, page_num: {self.page_num}")
        assert last_idx == self.page_num - 1 or last_idx== self.page_num == 0, f"last_idx: {last_idx}, page_num: {self.page_num}"
        return last_idx
    
    def __len__(self):
        return len(self.data_list)
    
    def is_empty(self):
        return len(self.data_list) == 0

    def is_not_empty(self):
        return len(self.data_list) > 0

def generate_tensor(numGPU, numLayers, numPages, dataSize, start_page):
    # Generate data where each page has values equal to its index
    pages = torch.arange(start_page, start_page + numPages).reshape(1, 1, numPages, 1)  # Shape: (1,1,numPages,1)
    # Expand to full shape
    result = pages.expand(numGPU, numLayers, numPages, dataSize)
    return result

if __name__ == "__main__":
    numGPU = 8  # Assume 8 GPUs
    numLayers = 5
    dataSize = 10  # Size of data dimension

    metaData = offloadMetaData()
    
    # Set the target page number
    total_pages = 10 + 5 + 5
    metaData.target_page_num = total_pages

    # Generate tensors with the [GPU, Layer, Page, Data] layout
    tensor1 = generate_tensor(numGPU, numLayers, 10, dataSize, 0)
    tensor2 = generate_tensor(numGPU, numLayers, 5, dataSize, 10)
    tensor3 = generate_tensor(numGPU, numLayers, 5, dataSize, 15)
    
    metaData.append(tensor1)
    metaData.append(tensor2)
    metaData.append(tensor3)
    
    all_data = metaData.reconstruct()
    print(all_data)
