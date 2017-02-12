# meta parameters table
meta_tbl = {}

# example
# hidden layer 1 input field
meta_tbl['hidden1'] = metaparam('hidden1', 10, 1, 0, 3)

# meta parameters class 
class metaparam():
    def __init__(self, name, field, stride, padding, depth):
        self.name = name
        self.field = field  
        self.strid = stide
        self.padding = padding
        self.depth = depth
    
    def isValid(self, width):
        if (width-self.filed+2*self.padding)%self.stride==0:
            return -1
        return (width-self.filed+2*padding)/stride + 1
