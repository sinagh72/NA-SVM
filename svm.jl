using LIBSVM
using RDatasets
using Printf
using Statistics

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")

# First four dimension of input data is features
X = Matrix(iris[:, 1:4])'

X[1,:]
# LIBSVM handles multi-class data automatically using a one-against-one strategy
y = iris.Species

# Split the dataset into training set and testing set
Xtrain = X[:, 1:2:end]
Xtest  = X[:, 2:2:end]
ytrain = y[1:2:end]
ytest  = y[2:2:end]

# Train SVM on half of the data using default parameters. See documentation
# of svmtrain for options
model = svmtrain(Xtrain, ytrain)

# Test model on the other half of the data.
ŷ, decision_values = svmpredict(model, Xtest);

# Compute accuracy
@printf "Accuracy: %.2f%%\n" mean(ŷ .== ytest) * 100