from sklearn.preprocessing import normalize
import numpy as np

class Vectors:
    """
    A class for handling vectors, particularly useful for cosine similarity analyses.

    The `Vectors` class is designed to store and manipulate a set of vectors and
    optionally, any number of additional attributes related to these vectors. 
    It ensures that the vectors are normalized (unit vectors) upon initialization. 
    It also provides methods for calculating the distance (based on cosine 
    similarity) between vectors, for accessing and setting vector elements, as well
    as for combining multiple Vectors instances.

    Attributes:
        data (np.ndarray): A 2D numpy array where each row is an unit vector.
        n_samples (int): The number of vectors (rows) in the data array.
        n_features (int): The dimensionality of the vectors (columns) in the data array.
        Optional attributes (np.ndarray): Additional attributes related to the vectors
                                          in the data array.
        set_of_optional_attributes (set): Number of optional attributes. 

    Methods:
        __init__(self, x, **attributes): Initializes a Vectors object with the given data vectors and 
                             optional attributes.
        add_attributes(self, attributes): Adds optional attributes to the Vectors object.
        distance(self, other=None): Computes the cosine distance between vectors.
        __add__(self, other): Combines two Vectors objects together.
        __getitem__(self, index): Retrieves vector(s) or element(s) at given ind(ices).
        __setattr__(self, name, value): Sets an attribute of the Vectors object.
    """
    
    def __init__(self, x: np.ndarray or Vectors, **attributes):
        """
        Initializes a Vectors object.

        The constructor can take either a 2D numpy array or an existing Vectors
        object as input. In the case of a numpy array, the vectors (rows) are 
        normalized to unit length. Optional attributes related to these vectors
        can also be provided.

        Args:
            x (np.ndarray or Vectors): A 2D numpy array of floating-point 
                                       numbers where each row represents an 
                                       unit vector, or an instance of Vectors.
            **attributes: Keyword arguments representing additional attributes
                          associated with the vectors.

        Raises:
            ValueError: If the input 'x' is not a 2D numpy array of 
                        floating-point numbers or an instance of Vectors.
            ValueError: if self.data is assigned a numpy array with rows 
                        not normalized to unit length.
            ValueError: if the optional attributes are not 1D or 2D numpy arrays.
            ValueError: if the optional attributes do not have the same length as
                        the number of samples.
        """
        if isinstance(x, np.ndarray) and (x.ndim == 2) and 
                        np.issubdtype(x.dtype, np.floating):            
            self.data = normalize(x)
        elif isinstance(x, Vectors):
            self.data = x.data
        else:
            raise ValueError("Invalid input type. "+
            "Expected a 2D numpy array of floating-point "+
            "numbers or an instance of Vectors.")
        self.set_of_optional_attributes = set([])
        self.attributes = attributes

    def add_attributes(self, attributes):
        """
        Adds optional attributes to the Vectors object.

        This method allows for the addition of extra data associated with each 
        vector, such as labels or other metadata.

        Args:
            attributes (dict): A dictionary where keys are attribute names (strings)
                               and values are numpy arrays of attribute values.
                               These numpy arrays can be 1D or 2D.
        
        Returns:
            None

        Raises:
            ValueError: if argument attributes is not a dictionary.
            ValueError: if the keys of the attributes dictionary are not strings.
            ValueError: if the optional attributes values are not 1D or 2D numpy arrays.
            ValueError: if the optional attributes arrays do not have the same length as
                        the number of samples.
        """
        self.attributes = attributes

    def distance(self, other:Vectors=None) -> np.ndarray:
        """
        Calculates the cosine distance between vectors. The distances are in 
        the range [0, 1], with 0 indicating identical vectors, 0.5 indicating 
        orthogonal vectors and 1 indicating vectors is opposite directions.

        If no 'other' Vectors object is provided, it calculates the pairwise 
        cosine distances between all vectors in the object. 
        If an 'other' Vectors object is provided, it computes the cosine 
        distances between each vector in self and each vector in 'other'.

        Args:
            other (Vectors, optional): Another Vectors object. Defaults to None.

        Returns:
            - If 'other' is None, returns a 1D array of cosine distances between 
              vectors in self. Distance are ordered in the 1D array as follows:
              01, 02, 03, 04, ..., 12, 13, 14, ..., 23, 24, ..., 34, ...
            - If 'other' is provided, returns a 2D array of cosine distances 
              between each vector in self and each vector in 'other'. 

        Raises:
            ValueError: If the input 'other' is not a Vectors object or if 
                        the two Vectors objects have a different number of 
                        features (dimensionality).
        """
        if other is None:
            triu_indices = np.triu_indices(self.n_samples, k=1)
            return 0.5*(1-np.sum(self.data[triu_indices[0]]*
                                 self.data[triu_indices[1]],
                                 axis=1))
        elif isinstance(other, Vectors):
            if other.n_features == self.n_features:
                return 0.5*(1-np.dot(self.data, other.data.T))
            else:
                raise ValueError('Vectors whose distances are to be computed '+
                                 'must have the same number of features.')
        else:
            raise ValueError("Invalid input type. "+
            "Expected an instance of Vectors or None.")

    def __add__(self, other:Vectors) -> Vectors:
        """
        Combines two Vectors objects by vertically stacking their data arrays
        and concatenating their optional attributes.

        This operation effectively concatenates the vectors from 'self' and 
        'other', resulting in a new Vectors object that contains all the 
        vectors from both input objects. It also concatenates any optional 
        attributes present in both objects.
        Optional attributes are combined if they have the same name and their
        arrays are compatible for concatenation.
        
        Note: optional attributes are combined only if:
             - both Vectors instances have the same attributes' names, and 
             - the optional attribute arrays have the same number of columns 
               (in the case of 2D arrays).
        
        Args:
            other (Vectors): The Vectors object to add to this one.

        Returns:
            Vectors: A new Vectors object containing the combined vectors from 
                     both input objects.

        Raises:
            ValueError: If:
                        - the input 'other' is not a Vectors object, or
                        - the two Vectors objects have a different number of 
                          features (dimensionality), or
                        - optional attributes are present in one Vectors instance
                          but not in the other, or
                        - the shape of the optional attributes arrays are not
                          compatible for concatenation (in the case of 2D
                          arrays)
        """
        if isinstance(other, Vectors):
            if self.set_of_optional_attributes != other.set_of_optional_attributes:
                raise ValueError('Vectors to be combined must have the same '+
                                 'optional attributes. Following attributes are '+
                                 'not present in either of the Vectors instances '+
                                 'being added: '+
                                 str(self.set_of_optional_attributes.symmetric_difference(
                                     other.set_of_optional_attributes)))
            else:
                d=[]
                for x in self.set_of_optional_attributes:
                    if self.__dict__[x].ndim == other.__dict__[x].ndim:
                        if (self.__dict__[x].ndim==2) and 
                           (self.__dict__[x].shape[1] == 
                            other.__dict__[x].shape[1]):
                            d += [(x, np.concatenate([self.__dict__[x], other.__dict__[x]]))]
                        else:
                            raise ValueError("Optional attribute arrays of either Vectors "+
                                             "instances being added must have the same "+
                                             "number of columns. Array shapes of attribute "+
                                             f"{x} are {self.__dict__[x].shape} and "+
                                             f"{other.__dict__[x].shape}.")
                    else:
                        raise ValueError("Optional attribute arrays of either Vectors "+
                                         "instances being added must have the same "+
                                         "dimensions. Array dimensions of attribute "+
                                         f"{x} are {self.__dict__[x].ndim} and "+
                                         f"{other.__dict__[x].ndim}.")
                d = dict(d)
            if other.n_features == self.n_features:
                return Vectors(np.vstack([self.data, other.data]), **d)
            else:
                raise ValueError('Vectors to be combined (or here "added") must '+
                                 'have the same number of features.')
        else:
            raise ValueError("Invalid input type. Expected an instance of Vectors.")

    def __getitem__(self, index):
        """
        Retrieves vector(s) or specific element(s) from the data array.

        Supports two types of indexing:
        - A 2-tuple (row, column) to retrieve a single element.
        - A 1-tuple (row) to retrieve an entire row (vector) as a new Vectors object.
          Corresponding optional attributes are also retrieved.

        Args:
            index (tuple): The index to access. It can be a 1-tuple or a 2-tuple of 
                           integers (or slices) representing the row and column indices.

        Returns:
            np.ndarray or Vectors: If a 2-tuple is given, returns the value(s) at 
                                   that position.
                                   If a 1-tuple is given, returns a new Vectors 
                                   object representing the row(s).

        Raises:
            ValueError: If the index is not a 1-tuple or a 2-tuple.
        """
        if len(index) == 2:
            return self.data[index[0], index[1]]
        elif len(index) == 1:
            d = dict([(x, self.__dict__[x][index[0]]) 
                for x in self.set_of_optional_attributes])
            return Vectors(self.data[index[0]], **d)
        else:
            raise ValueError("Invalid index. Expected a 1-tuple or 2-tuple")
    
    def __setattr__(self, name, value):
        """
        Sets the attributes of the Vectors object after performing some checks.

        Args:
            name (str): The name of the attribute to set.
            value: The value to set the attribute to.
        """

        if name == 'data':
            if isinstance(x, np.ndarray) and (x.ndim == 2) and 
                        np.issubdtype(x.dtype, np.floating) and
                        np.allclose(np.linalg.norm(x, axis=1), 1):
                self.__dict__[name] = x
                self.n_samples, self.n_features = np.shape(x)
            else:
                raise ValueError("data attribute of Vectors instance "+
                                 "must be a 2D numpy array with each row "+
                                 "being an unit vector.")

        elif name == 'n_samples':
            if value == np.shape(self.data)[0]:
                self.__dict__[name] = value
            else:
                raise ValueError("n_samples attribute of Vectors instance "+
                                 "must be equal to the number of rows in the "+
                                 "data attribute.")

        elif name == 'n_features':
            if value == np.shape(self.data)[1]:
                self.__dict__[name] = value
            else:
                raise ValueError("n_features attribute of Vectors instance "+
                                 "must be equal to the number of columns in "+
                                 "the data attribute.")

        elif name == 'attributes':
            if not isinstance(value, dict):
                raise ValueError("Optional attributes to the Vectors instance "+
                                 "must be a dictionary.")
            for k in value:
                if not isinstance(k, str):
                    raise ValueError("Optional attribute names must be strings.")
                if isinstance(value[k], np.ndarray) and (value[k].ndim in [1,2]) and
                   (len(value[k])==self.n_samples):
                    self.__dict__[k] = value[k]
                else:
                    raise ValueError("Optional attribute values must be 1D or 2D numpy "+
                                     "arrays of the same length as the number of samples.")                    
            self.set_of_optional_attributes.update(value.keys())

        elif name == 'set_of_optional_attributes':
            if isinstance(value, set):
                if all([(isinstance(x, str) and hasattr(self, x)) 
                        for x in value]):
                    self.__dict__[name] = value
                else:
                    raise ValueError("Optional attribute names must be strings and "+
                                     "must be entries of self.__dict__")
            else:
                raise ValueError("set_of_optional_attributes attribute of Vectors "+
                                 "instance must be a set.")
    
        else:
            self.__dict__[name] = value