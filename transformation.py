'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Hannah Soria
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        self.orig_dataset = orig_dataset
        super(Transformation, self).__init__(data)

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        currdata = self.orig_dataset.select_data(headers) # get the data for the headers
        dict = {} # make a new dictionary
        for col_index in range(currdata.shape[1]): # loop through the length of the data
            dict[headers[col_index]]=col_index # add the data to the dict with the correct index

        self.data = data.Data(headers = headers, data = currdata, header2col=dict) # create new data option

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        ones = np.ones((self.data.get_num_samples(),1)) # make np array of ones 
        new_array = np.hstack((self.data.data, ones)) # add ones to the data 
        return new_array

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        n = self.data.get_num_dims() # number of variables 
        t = np.eye(n+1, n+1) # create translation matrix
        for i in range(len(magnitudes)): #loop through the magnitudes
            t[i,n] = magnitudes[i] # set the correct values to the matrix
        return t
        

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        n = self.data.get_num_dims() # number of variables 
        s = np.eye(n+1, n+1) # create scaling matrix
        for i in range(len(magnitudes)): #loop through the magnitudes
            s[i,i] = magnitudes[i] # set the correct values to the matrix
        return s

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        trans_matrix = self.translation_matrix(magnitudes) #matrix instance
        headers = self.data.get_headers() # get the headers
        d = self.get_data_homogeneous() # get homogenous data
        d = d.T # transpose data
        new_matrix = (trans_matrix@d).T # multiply trans and data and transpose again
        new_matrix = new_matrix[:,0:-1] # get rid of homogenous row
        self.data = data.Data(data=new_matrix, headers=headers, header2col=self.data.get_mappings()) #new data object
        return new_matrix

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        scale_matrix = self.scale_matrix(magnitudes) # scale istance
        headers = self.data.get_headers() # get the headers
        d = self.get_data_homogeneous() # get the homogenous data
        d = d.T # transpose data
        new_matrix = (scale_matrix@d).T # multiply scale and data and transpose again
        new_matrix = new_matrix[:,0:-1] # get rid of homogenous row
        self.data = data.Data(data=new_matrix, headers=headers, header2col=self.data.get_mappings()) #new data object
        return new_matrix

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        headers = self.data.get_headers() # get the headers
        d = self.get_data_homogeneous() # get the homogenous data
        d = d.T # transpose data
        new_matrix = (C @ d).T # multiply C and data and transpose again
        new_matrix = new_matrix[:,0:-1] # get rid of homogenous row
        self.data = data.Data(data=new_matrix, headers=headers, header2col=self.data.get_mappings()) #new data object
        return new_matrix

    # Grace Moberg helped me with this function, most pecifically the for loop and correcting setting the data up 
    # to "multiply" scaler and translator
    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        n = self.data.get_num_dims() # get number of variables
        headers = self.data.get_headers() # get the headers to get list of all of the variables
        minimum = self.min(headers) # find the global minimum
        min_val = min(minimum) #minimum in array
        maximum = self.max(headers) # find the global max
        max_val = max(maximum) #maximum in array
        difference = max_val - min_val # find the range

        min_list = [] #list of minimums
        range_list = [] #list of ranges
        for i in range(len(headers)): # for the length of headers
            min_list.append(-min_val) # add min to list
            range_list.append(1/difference) # add difference to list (1/diff) is to mimick multiplication

        tmatrix = self.translation_matrix(min_list) # get the minimum translation matrix. -minimum so that it moves to origin
        smatrix = self.scale_matrix(range_list) # replicate divison
        cmatrix = smatrix@tmatrix # create the matrix to normalize MATRIX WISE SCALE THEN TRANSLATE
        normalize = self.transform(cmatrix) # actually apply the math
        self.data = data.Data(data=normalize, headers=headers, header2col=self.data.get_mappings()) #new data object
        return normalize

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        n = self.data.get_num_dims() # get number of variables
        headers = self.data.get_headers() # get the headers to get list of all of the variables
        minimum = self.min(headers) # find the global minimum
        maximum = self.max(headers) # find the global max
        difference = maximum - minimum # find the range

        min_list = [] # create min list
        range_list = [] # create range list
        for i in range(len(headers)): # for the length of headers
            min_list.append(-minimum[i]) # add the minimum subtracted to list
            range_list.append(1/difference[i]) # add difference to list (1/diff) is to mimick multiplicatio

        tmatrix = self.translation_matrix(min_list) # get the minimum translation matrix. -minimum so that it moves to origin
        smatrix = self.scale_matrix(range_list) # replicate divison
        cmatrix = smatrix@tmatrix # create the matrix to normalize MATRIX WISE SCALE THEN TRANSLATE
        normalize = self.transform(cmatrix) # actually apply the math
        self.data = data.Data(data=normalize, headers=headers, header2col=self.data.get_mappings()) #new data object
        return normalize

    # I had help from a TA to correctly index the matrices
    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        hdata = self.get_data_homogeneous() #get the homogenous data
        rmatrix = np.eye(hdata.shape[1],hdata.shape[1]) # identity matrix as base for rotation matrix of the correct size
        radians = np.radians(degrees) # convert the degrees to radians
        headers = self.data.get_headers()
        if headers.index(header) == 0: # if the header parameter = the x coordinate then build this matrix
            rmatrix[1][1] = np.cos(radians)
            rmatrix[1][2] = -(np.sin(radians))
            rmatrix[2][1] = np.sin(radians)
            rmatrix[2][2] = np.cos(radians)
            # 1   0   0   0
            # 0  cos -sin 0
            # 0  sin  cos 0
            # 0   0   0   1
        elif headers.index(header) == 1: # if the header parameter = the x coordinate then build this matrix
            rmatrix[0][0] = np.cos(radians)
            rmatrix[0][2] = np.sin(radians)
            rmatrix[2][0] = -(np.sin(radians))
            rmatrix[2][2] = np.cos(radians)
        #     cos 0   sin 0
        #     0   1   0   0
        #   -sin  0  cos  0
        #     0   0   0   1
        elif headers.index(header) == 2: # if the header parameter = the z coordinate then build this matrix
            rmatrix[0][0] = np.cos(radians)
            rmatrix[1][0] = np.sin(radians)
            rmatrix[0][1] = -(np.sin(radians))
            rmatrix[1][1] = np.cos(radians)
        #    cos -sin 0   0
        #     sin cos 0   0
        #     0   0   1   0
        #     0   0   0   1
        return rmatrix


    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        rmatrix = self.rotation_matrix_3d(header, degrees) # create the rotation matrix
        rotated_matrix = self.transform(rmatrix) # transform the matrix
        self.data = data.Data(data=rotated_matrix, headers=self.data.get_headers(), header2col=self.data.get_mappings()) #new data object
        return rotated_matrix

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        headers = [ind_var,dep_var,c_var] # independent and dependent variable as headers (x,y)
        rows = [] # for the rows data

        select_data = self.orig_dataset.select_data(headers, rows) #select the data
        x = select_data[:,0] # the Xs are the first column
        y = select_data[:,1] # the Ys are the second column
        z = select_data[:,2]

        color_map = palettable.colorbrewer.sequential.Greys_9
        plt.figure(figsize=(5,6))
        plt.scatter(x, y, c=z, s=75, cmap=color_map.mpl_colormap, edgecolor='black')
        plt.title(title) # set the title as the title passed
        plt.xlabel(ind_var) # set the x label as independent
        plt.ylabel(dep_var) # set the y label as the dependent
        plt.colorbar(label=c_var)
        return x,y,z
    

    #EXTENSIONS

    #matplotlib documentation was consulted
    def scatter_4d(self,ind_var,dep_var,c_var,e_var, title=None):
        headers = [ind_var,dep_var,c_var,e_var] # independent and dependent variable as headers (x,y)
        rows = [] # for the rows data

        select_data = self.orig_dataset.select_data(headers, rows) #select the data
        x = select_data[:,0] # the Xs are the first column
        y = select_data[:,1] # the Ys are the second column
        z = select_data[:,2] 
        e = select_data[:,3]

        color_map = palettable.colorbrewer.sequential.Greens_9
        plt.figure(figsize=(5,6))
        plt.scatter(x, y, c=z, s=(e**2), cmap=color_map.mpl_colormap, edgecolor='black')
        plt.title(title) # set the title as the title passed
        plt.xlabel(ind_var) # set the x label as independent
        plt.ylabel(dep_var) # set the y label as the dependent
        plt.colorbar(label=c_var)
        return x,y,z,e
    
    #matplotlib documentation and overstack was consulted
    def scatter_5d(self,ind_var,dep_var,c_var,e_var,f_var, title=None):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        headers = [ind_var,dep_var,c_var,e_var,f_var] # independent and dependent variable as headers (x,y)
        rows = [] # for the rows data

        select_data = self.orig_dataset.select_data(headers, rows) #select the data
        x = select_data[:,0] # the Xs are the first column
        y = select_data[:,1] # the Ys are the second column
        z = select_data[:,2]
        e = select_data[:,3]
        f = select_data[:,4]

        color_map = palettable.colorbrewer.sequential.Greens_9
        ax.scatter(x, y, c=z, s=(f**2), cmap=color_map.mpl_colormap, edgecolor='black')
        plt.title(title) # set the title as the title passed
        plt.xlabel(ind_var) # set the x label as independent
        plt.ylabel(dep_var) # set the y label as the dependent
        ax.set_zlabel(c_var) # set the z label
        plt.rcParams["figure.figsize"] = (50,20) #set size
        pcm = ax.get_children()[0] # added to fix mapping
        plt.colorbar(pcm,label=e_var, pad = .1) # create color bar
        return x,y,z,e
    
    def normalize_zscore(self):
        headers = self.data.get_headers() # get the headers to get list of all of the variables
        mean = self.mean(headers)# find the mean
        SD = self.std(headers)# find the SD
        mean_list = [] #list for means
        SD_list = [] #list for SD
        for i in range(len(headers)): #for the length of headers
            mean_list.append(-mean[i]) # add the subtracted mean to list
            SD_list.append(1/SD[i]) # add SD to the list
        tmatrix = self.translation_matrix(mean_list) #translate
        smatrix = self.scale_matrix(SD_list) # scale
        cmatrix = smatrix@tmatrix # create the matrix to normalize MATRIX WISE SCALE THEN TRANSLATE
        normalize = self.transform(cmatrix)# actually apply the math

        self.data = data.Data(data=normalize, headers=headers, header2col=self.data.get_mappings()) #new data object
        return normalize