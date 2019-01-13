# Load X and y variable
using JLD
data = load("multiData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Data is already roughly standardized, but let's add bias
n = size(X,1)
X = [ones(n,1) X]

# Do the same transformation to the test data
t = size(Xtest,1)
Xtest = [ones(t,1) Xtest]

# Fit one-vs-all logistic regression model
include("logReg.jl")
model = logRegOnevsAll(X,y)

# Compute training and validation error
using Statistics
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@show(trainError)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)
@show(validError)

# Plot results
k = maximum(y)
include("plot2Dclassifier.jl")
plot2Dclassifier(X,y,model,Xtest=Xtest,ytest=ytest,biasIncluded=true,k=5)
title("logRegOnevsAll")
gcf()

# includes
include("findMin.jl")
include("misc.jl")

# softmax objective
function softmax_objective(W_vec, X, y, f, n, k)
    W = reshape(W_vec, (f, k))
    objective = 0
    for i = 1:n
        objective += -W[:,y[i]]'*X[i,:]
        log_sum = 0
        for j = 1:k
            log_sum += exp.(W[:,j]'*X[i,:])
        end
        objective += log.(log_sum)
    end
    return objective
end

# softmax gradient
function softmax_gradient(W_vec, X, y, f, n, k)
    W = reshape(W_vec, (f, k))
    dims = size(W)
    objective = zeros(dims)
    for index_1 = 1:dims[1]
        for index_2 = 1:dims[2]
            objective_current = 0
            for i = 1:n
                # first chunk
                indicator = 0
                if y[i] == index_2
                    indicator = 1
                end
                objective_current -= X[i,index_1]*indicator
                # sum of exps
                sum_of_exp = 0
                for c = 1:k
                    sum_of_exp += exp(W[:,c]'*X[i,:])
                end
                objective_current += exp(W[:,index_2]'*X[i,:])*X[i,index_1] / sum_of_exp
            end
            # add to gradient
            objective[index_1, index_2] = objective_current
        end
    end
    return vec(objective)
end

function softmaxClassifier(X, y, f, n, k)
    # set objective function
    funObj(W_vec) = (softmax_objective(W_vec, X, y, f, n, k), softmax_gradient(W_vec, X, y, f, n, k))
    # initialize weights
    w_init =0.005*rand(Int(f*k))
    # train weights
    w = findMin(funObj, w_init)
    # return function
    return w
end

# predict function
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

# train model
w = softmaxClassifier(X, y, f, n, k)

# validation error
println(mean(softmax_prediction(w, X, f, n, k) .!= y))
println(mean(softmax_prediction(w, Xtest, f, n, k) .!= ytest))
