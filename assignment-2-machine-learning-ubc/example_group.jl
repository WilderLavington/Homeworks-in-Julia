# Load X and y variable
using JLD
data = load("groupData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit multi-class logistic regression classifer
include("logReg.jl")
model = logRegSoftmax(X,y)

# Compute training and validation error
using Statistics
yhat = model.predict(X)
trainError = mean(yhat .!= y)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)

# Count number of parameters in model and number of features used
nModelParams = sum(model.w .!= 0)
nFeaturesUsed = sum(sum(abs.(model.w),dims=2) .!= 0)
@show(trainError)
@show(validError)
@show(nModelParams)
@show(nFeaturesUsed)

# Show the image as a matrix
using PyPlot
imshow(model.w);

# question 2 solution

# includes
include("findMin.jl")
include("misc.jl")

""" Vanilla Softmax """
# function to perform softmax predictions
function softmax_prediction(W_vec, X, f, n, k)
    # initialize predictions
    pred = zeros(n)
    W = reshape(W_vec, (f, k))
    # iteratate trough data set
    for i = 1:n
        # calculate sum
        current_sum = 0
        for j = 1:k
            current_sum += exp.(W[:,j]'*X[i,:])
        end
        p = zeros(k)
        # iterate through classes
        for j = 1:k
            p[j] = exp.(W[:,j]'*X[i,:]) / current_sum
        end
        # find index with highest probability
        pred[i] = Int(findmax(p)[2])
    end
    return pred
end
# softmax objective and gradient
function softmaxObj(w,X,y,k)
	(n,d) = size(X)

	W = reshape(w,d,k)

	XW = X*W
	Z = sum(exp.(XW),dims=2)

	nll = 0
	G = zeros(d,k)
	for i in 1:n
		nll += -XW[i,y[i]] + log(Z[i])

		pVals = exp.(XW[i,:])./Z[i]
		for c in 1:k
			G[:,c] += X[i,:]*(pVals[c] - (y[i] == c))
		end
	end
	return (nll,reshape(G,d*k,1))
end

""" L2 Regularization """
# softmax with L2 regularization
function softmaxObjL2(w,X,y,k,lambda)
	(n,d) = size(X)

	W = reshape(w,d,k)

	XW = X*W
	Z = sum(exp.(XW),dims=2)

	nll = 0
	G = zeros(d,k)
	for i in 1:n
		nll += -XW[i,y[i]] + log(Z[i])

		pVals = exp.(XW[i,:])./Z[i]
		for c in 1:k
			G[:,c] += X[i,:]*(pVals[c] - (y[i] == c))
		end
	end
	return (nll + 0.5*lambda*norm(w,2)^2, reshape(G,d*k,1) .+ lambda.*w)
end
# train a L2 regularized softmax classier
function logRegSoftmaxL2(X, y, f, n, k, lambda)
    # set objective function
    funObj(W_vec) = softmaxObjL2(W_vec,X,y,k,lambda)
    # initialize weights
    w_init =0.005*rand(Int(f*k))
    # train weights
    w = findMin(funObj, w_init, derivativeCheck=true)
    # return function
    return w
end
# train model with L2 regularization
lambda = 10
f = size(X,2)
k = maximum(y)
n = size(X,1)
w_L2 = logRegSoftmaxL2(X, y, f, n, k, lambda)
# look at validation error for lambda = 10
println(mean(softmax_prediction(w_L2, X, f, n, k) .!= y))
println(mean(softmax_prediction(w_L2, Xtest, f, n, k) .!= ytest))
# look at the number of model parameters and features
println(sum(w_L2 .!= 0))
println(sum(sum(abs.(reshape(w_L2,100,5)),dims=2) .!= 0))

""" L1 Regularization """
# train a L1 regularized softmax classier
function logRegSoftmaxL1(X, y, f, n, k, lambda)
    # set objective function (dont mess with gradient objective)
    funObj(W_vec) = softmaxObj(W_vec,X,y,k)
    # initialize weights
    w_init =0.005*rand(Int(f*k))
    # train weights
    w = findMinL1(funObj, w_init, lambda)
    # return function
    return w
end
# train model with L1 regularization
lambda = 10
f = size(X,2)
k = maximum(y)
n = size(X,1)
w_L1 = logRegSoftmaxL1(X, y, f, n, k, lambda)
# look at validation error for lambda = 10
println(mean(softmax_prediction(w_L1, X, f, n, k) .!= y))
println(mean(softmax_prediction(w_L1, Xtest, f, n, k) .!= ytest))
# look at the number of model parameters and features
println(sum(w_L1 .!= 0))
println(sum(sum(abs.(reshape(w_L1,100,5)),dims=2) .!= 0))

""" Proximal Group L1 Regularization """
function groupL1prox(groups, wNew, lambda, alpha)
    for group = 1:groups
        start_ = Int(length(wNew)/groups)*(group-1)+1
        end_ = Int(length(wNew)/groups)*(group)
        w_g = wNew[start_:end_]
        wNew[start_:end_] = (w_g./norm(w_g,2))*max.(norm(w_g,2) .- lambda*alpha,0)
    end
    return wNew
end
function groupL1func(groups, wNew)
    eval = 0
    for group = 1:groups
        start_ = Int(length(wNew)/groups)*(group-1)+1
        end_ = Int(length(wNew)/groups)*(group)
        w_g = wNew[start_:end_]
        eval += sum(w_g.^2).^0.5
    end
    return eval
end
function proxGradGroupL1(funObj, groups, w, lambda;maxIter=200,epsilon=1e-4)
	(f,g) = funObj(w)
	# Initial step size and sufficient decrease parameter
	gamma = 1e-4
	alpha = 1
	for i in 1:maxIter
		# Gradient step on smoooth part
		wNew = w - alpha*g
		# Proximal step on non-smooth part (bto)
        wNew = groupL1prox(groups, wNew, lambda, alpha)
        # update
		(fNew,gNew) = funObj(wNew)
		# Decrease the step-size if we increased the function
		gtd = dot(g,wNew-w)
		# set group l1 funcs
		prox_new = groupL1func(groups, wNew)
		prox_old = groupL1func(groups, w)
		# perform linesearch
		while fNew + lambda*prox_new > f + lambda*prox_old - gamma*alpha*gtd
			@printf("Backtracking\n")
			alpha /= 2
			# Try out the smaller step-size
            wNew = w - alpha*g
    		# Proximal step on non-smooth part (bto)
            wNew = groupL1prox(groups, wNew, lambda, alpha)
            # update
			(fNew,gNew) = funObj(wNew)
			# set group l1 funcs
			prox_new = groupL1func(groups, wNew)
			prox_old = groupL1func(groups, w)
		end
		# Guess the step-size for the next iteration
		y = gNew - g
		alpha *= -dot(y,g)/dot(y,y)
		# Sanity check on the step-size
		if (!isfinitereal(alpha)) | (alpha < 1e-10) | (alpha > 1e10)
			alpha = 1
		end
		# Accept the new parameters/function/gradient
		w = wNew
		f = fNew
		g = gNew
		# Print out some diagnostics
		optCond = norm(w-groupL1prox(groups, w-g, lambda, alpha),Inf)
		@printf("%6d %15.5e %15.5e %15.5e\n",i,alpha,f+lambda*groupL1func(groups, w),optCond)
		# We want to stop if the gradient is really small
		if optCond < epsilon
			@printf("Problem solved up to optimality tolerance\n")
			return w
		end
	end
	@printf("Reached maximum number of iterations\n")
	return w
end
function softmaxClassiferGL1(X, y, f, n, k, lambda)
    # set objective function (dont mess with gradient objective)
    funObj(W_vec) = softmaxObj(W_vec,X,y,k)
    # initialize weights
    w_init =0.00*rand(Int(f*k))
    # train weights
    w = proxGradGroupL1(funObj, f, w_init, lambda)
    # return function
    return w
end
# train model with group L1 regularization
lambda = 10
f = size(X,2)
k = maximum(y)
n = size(X,1)
w_GL1 = softmaxClassiferGL1(X, y, f, n, k, lambda)
# look at validation error for lambda = 10
println(mean(softmax_prediction(w_GL1, X, f, n, k) .!= y))
println(mean(softmax_prediction(w_GL1, Xtest, f, n, k) .!= ytest))
# look at the number of model parameters and features
println(sum(w_GL1 .!= 0))
println(sum(sum(abs.(reshape(w_GL1,100,5)),dims=2) .!= 0))
