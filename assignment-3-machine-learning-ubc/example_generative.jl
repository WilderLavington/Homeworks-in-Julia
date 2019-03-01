using JLD, Printf, Statistics
using LinearAlgebra
include("misc.jl") # Includes mode function and GenericModel typedef

# Load X and y variable
data = load("gaussNoise.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a KNN classifier
k = 1
include("knn.jl")
model = knn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)

function gda(X,y,Xtest,ytest)
    # get parameters
    d = length(X[1,:])
    k = maximum(y)
    n = length(y)
    # initialize MLEs
    pi_ = zeros(k)
    mu = zeros(k, d)
    sigma = zeros(d, d, k)
    # compute pi
    for i = 1:k
        pi_[i] = length(findall(x->x==i, y))/n
    end
    # compute mu
    for i = 1:k
        mu[i,:] =  sum(X[findall(x->x==i, y),:], dims = 1)/length(findall(x->x==i, y))
    end
    # compute sigma
    for i = 1:k
        indices = findall(x->x==i, y)
        temp = zeros(d, d)
        for j = 1:length(indices)
            temp += (X[indices[j],:]-mu[i,:])*(X[indices[j],:]-mu[i,:])'
        end
        sigma[:,:,i] = temp/length(findall(x->x==i, y))
    end
    # fit data
	cluster_prob = zeros(length(ytest),k)
    for i = 1:k
		(t,d) = size(Xtest)
		# compute pdf values
		PDFs = zeros(t)
		SigmaInv = sigma[:,:,i]^-1
		logZ = (d/2)log(2pi) + (1/2)logdet(sigma[:,:,i])
		for j in 1:t
			xc = Xtest[j,:] - mu[i,:]
			loglik = -(1/2)dot(xc,SigmaInv*xc) - logZ
			PDFs[j] = loglik
		end
		cluster_prob[:,i] = log(pi_[i]) .+ PDFs
	end
	# pick the most likly label for each data point
	yhat = zeros(size(ytest))
	for i = 1:length(ytest)
		max = findall(x->x==maximum(cluster_prob[i,:]), cluster_prob[i,:])
		yhat[i] = Integer(max[1])
	end
	error = sum([1 for x in 1:length(yhat) if yhat[x] == ytest[x]])/length(yhat)
	return pi_, mu, sigma, yhat, 1-error
end

pi_, mu, sigma, y_hat, error = gda(X,y,Xtest,ytest)
println(error)


function tda(X,y,Xtest,ytest)
    # get parameters
    d = length(X[1,:])
    k = maximum(y)
    n = length(y)
    # compute pi
    pi_ = zeros(k)
    for i = 1:k
        pi_[i] = length(findall(x->x==i, y))/n
    end
	# compute log prob of all data under each label
	log_prob = zeros(length(ytest),k)
    for i = 1:k
		# train density model for each tupe of label
        model = studentT(X[findall(x->x==i, y),:])
		# compute log probabilities of model for new points
		log_prob[:,i] = model.pdf(Xtest)*pi_[i]
    end
    # pick the most likly label for each data point
	yhat = zeros(size(ytest))
	for i = 1:length(ytest)
		max = findall(x->x==maximum(log_prob[i,:]), log_prob[i,:])
		yhat[i] = Integer(max[1])
	end
	# calculate error
	error = sum([1 for x in 1:length(yhat) if yhat[x] == ytest[x]])/length(yhat)
	# return models, error, and predictions
    return yhat, 1-error
end

yhat, error = tda(X,y,Xtest,ytest)
print(error)
