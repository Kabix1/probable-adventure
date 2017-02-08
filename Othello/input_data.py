#####################################################################################################################
# data handler for tensorflow project
#####################################################################################################################

import pickle
import random
import os

class input_data:
    file_path = ""

    def __init__(self, path):
        self.file_path = path

    def next_batch(self, num_items):
        f = open(self.file_path, 'rb')
        test = pickle.load(f)
        element_size = len(pickle.dumps(test))
        data = []
        size = os.stat(self.file_path).st_size
        f.seek(random.randint(0, size/element_size) * element_size)
        for _ in range(num_items):
            try:
                data.append(pickle.load(f))
            except EOFError:
                f.seek(0)
        return list(zip(*data))
