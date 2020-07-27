class Hubbard():
    def __init__(self, nsites, t, U):
        self.nsites = nsites
        self.t = t
        self.U = U
        self.matrix = None
        self.built = False

    def gen_matrix(self):
        pass
    
    def get_elem(self, i, j):
        pass