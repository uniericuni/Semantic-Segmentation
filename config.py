
'''
# Global variables (Macros)

# meta parameters table
meta_tbl = {}

# example
meta_tbl['hidden1'] = metaparam('hidden1', 10, 1, 'SAME', 30)

# meta parameters class
# name: layer name scope
# field: shape of filter, numpy array with ints specified [height, width, input channel]
# stride: sampling rate of the layer
# padding: padding method, either 'VALID' or 'SAME', accroding the Tensorflow documentation
# depth: expected output layer depth
class metaparam():
    def __init__(self, name, field, stride, padding, depth):
        self.name = name
        self.field = field  
        self.stride = stide
        self.padding = padding
        self.depth = depth
    
    def isValid(self, width):
        if (width-self.filed+2*self.padding)%self.stride==0:
            return -1
        return (width-self.filed+2*padding)/stride + 1
'''
