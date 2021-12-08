import numpy as np

class Perceptron(object):
    """
    Classifaer based on perseptron.
    Params:
    -------
    eta : float
        Eduacation velocity (between 0.0, 1.0)
    
    n_iter : int
        Number of education epochs
    
    random_state : int
        Random generator start value aimed for initialization random weights
    
    Attibutes:
    ----------
    w_ : array_like
        Weight after adjustment
        
    errors : list
        Number of wrong classifications in each epoch
    
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Adjustment to the train dataset
        Params:
        -------
        X : array_like , in view [n_examples, n_features]
            Train set
        y : array_like , in view [n_examples]
            Target set
        
        Returns:
        --------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] = update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """ Calculate common input """
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """ Returns class mark after one education step """
        return np.where(self.net_input(X) >= 0, 1, -1)