# Load X and y variable
using JLD
data = load("nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# added from misc
function distancesSquared(X1,X2)
	(n,d) = size(X1)
	(t,d2) = size(X2)
	@assert(d==d2)
	return X1.^2*ones(d,t) + ones(n,d)*(X2').^2 - 2X1*X2'
end

# Compute number of training examples and number of features
(n,d) = size(X)

# Fit least squares model
include("leastSquares.jl")
model = leastSquares(X,y)

# Report the error on the test set
using Printf
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("TestError = %.2f\n",testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat[:],yhat,"r")
ylim((-300,400))
title("Linear Model")
savefig("figure_1.png")

# Gaussiam Radial Basis Function
using LinearAlgebra
function RBF_kernal(X, X_hat, sigma)
        return exp.(-distancesSquared(X_hat, X)./(2*sigma*sigma))
end

function rbfBasis(Xtrain, y_train, Xtest, sigma, lambda)
        # dimensions
        (n_train,d_train) = size(Xtrain)
        (n_test, d_test) = size(Xtest)
        # basis
        Z_train = RBF_kernal(Xtrain, Xtrain, sigma)
        Z_test = RBF_kernal(Xtrain, Xtest, sigma)
        # solve for weights v
        v = (Z_train'*Z_train+lambda*Matrix{Float64}(I, n_train, n_train)) \ Z_train'*y_train
        # return pred
        return Z_test*v
end

# set hyper parameters
sigma = 1
lambda = 1

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
yhat = rbfBasis(X, y, Xtest, sigma, lambda)
plot(Xtest[:],yhat,"r.")
ylim((-300,400))
title("RBF Model: λ = 1, σ = 1")
gcf()
savefig("figure_2png")

# split data function
using Random
function split_data(X, y, split)
        total_data = size(y)[1]
        indices = randperm(total_data)
        training_indices = indices[1:Int(floor(split*total_data))]
        validation_indices = indices[Int(floor(split*total_data))+1:end]
        X_train, y_train = X[training_indices,:], y[training_indices]
        X_val, y_val = X[validation_indices,:], y[validation_indices]
        return X_train, y_train, X_val, y_val
end

function minimize_hyper_parameters(X, y, iterations)
	# split up the data
	X_train, y_train, X_val, y_val = split_data(X, y, 0.5)
	t = size(X_val,1)

	# iterate through combinations of lambda and sigma
	best_error = Inf
	sigmas, lambdas  = range(0.0001, stop = 50, length = iterations) |> collect, range(0.0001, stop = 50, length = iterations) |> collect
	sigma_best, lambda_best = 0.0001, 0.0001

	# iterate
	for iter_1 = 1:iterations
		for iter_2 = 1:iterations
			sigma_current = sigmas[iter_1]
			lambda_current = lambdas[iter_2]
			yhat = rbfBasis(X_train, y_train, X_val, sigma_current, lambda_current)
			current_error = sum((yhat - y_val).^2)/t
			if current_error < best_error
				sigma_best = sigma_current
				lambda_best = lambda_current
				best_error = current_error
			end
		end
	end
	return best_error, lambda_best, sigma_best
end

# get the best hyper parameters
best_error, lambda_best, sigma_best = minimize_hyper_parameters(X, y, 100)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
yhat = rbfBasis(X, y, Xtest, sigma_best, lambda_best)
plot(Xtest[:],yhat,"r.")
ylim((-300,400))
title("RBF Model: λ = 10e-4, σ = 0.50515")
gcf()
savefig("figure_3.png")
