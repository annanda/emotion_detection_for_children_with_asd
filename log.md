# Things I did with the dataset 

The files are in `arff` format, which are not very well supported by Python. 
So I needed to implement some changes in order to work with the features extracted by the AVEC 2018 team. 

1. I eliminated the string attribute (which was just the name of the file for each row) 
2. I changed the attribute name from frameTime to frametime to obtain consistency and to be able to use the merge function of Pandas.

Obs: the whole dataset of features and annotationxws is not included on the version control because it is too big, but I will include an example of each file.  