using JLD, PyPlot

# Load multi-modal data
data = load("mixtureData.jld")
X = data["X"]

include("gaussianDensity.jl")
model = gaussianDensity(X)

# Plot data and densities (you can ignore the code below)
plot(X[:,1],X[:,2],".")
gcf()
increment = 100
(xmin,xmax) = xlim()
xDomain = range(xmin,stop=xmax,length=increment)
(ymin,ymax) = ylim()
yDomain = range(ymin,stop=ymax,length=increment)

xValues = repeat(xDomain,1,length(xDomain))
yValues = repeat(yDomain',length(yDomain),1)

z = model.pdf([xValues[:] yValues[:]])

@assert(length(z) == length(xValues),"Size of model function's output is wrong");

zValues = reshape(z,size(xValues))

contour(xValues,yValues,zValues)
gcf()

using LinearAlgebra

function gaussian_PDF(X,mu,Sigma)
    Xc = X - repeat(mu',n)
    SigmaInv = Sigma^-1
    (t,d) = size(X)
    PDFs = zeros(t)
    logZ = (d/2)log(2pi) + (1/2)logdet(Sigma)
    for i in 1:t
        xc = X[i,:] - mu
        loglik = -(1/2)dot(xc,SigmaInv*xc) - logZ
        PDFs[i] = loglik
    end
    return PDFs
end

function gaussianEM(X,k,Xtest,iterations)
    # get variable info
    d = length(X[1,:])
    n = length(X[:,1])
    # initialize mixture variables
    pi_ = rand(k); pi_ = pi_ / sum(pi_)
    mu = 2*(rand(d,k).-0.5)
    Sigma = zeros(d,d,k)
    for i = 1:k; Sigma[:,:,i] += Matrix{Float64}(I, d, d); end
    r = zeros(n,k)
    # apply EM
    for i = 1:iterations
        # first we have to compute the responsability
        for j = 1:k
            r[:,j] = exp.(gaussian_PDF(X, mu[:,j], Sigma[:,:,j])).*pi_[j]
        end
        r = r ./ sum(r, dims = 2)
		println(r[1,:])
		println(r[2,:])
        # next we can compute the updates to our parameters
        for j = 1:k
            # update pi
            pi_[j] = (1/n)*sum(r[:,j])
            # update mu
            mu[:,j] = (1/sum(r[:,j]))*sum(r[:,j].*X,dims=1)
            # update sigma
            temp = 0*Matrix{Float64}(I, d, d)
            for q = 1:n
                temp += r[q,j].*(X[q,:]-mu[:,j])*(X[q,:]-mu[:,j])'
            end
            Sigma[:,:,j] = (1/sum(r[:,j]))*temp
        end
    end
    # now predict on each of the Xtest
    log_prob = zeros(length(Xtest[:,1]),k)
    for c = 1:k
        log_prob[:,c] = gaussian_PDF(Xtest,mu[:,c],Sigma[:,:,c]) .+ log(pi_[c])
    end
    # take max to get label
    yhat = zeros(length(log_prob[:,1]))
	for i = 1:length(log_prob[:,1])
		max = findall(x->x==maximum(log_prob[i,:]), log_prob[i,:])
		yhat[i] = Integer(max[1])
	end
    # return everything
    return pi_, mu, Sigma, yhat
end

pi_, mu, Sigma, yhat = gaussianEM(X,3,X,20)

figure()
plot(X[findall(x->x==1, yhat),1],X[findall(x->x==1, yhat),2],".")
plot(X[findall(x->x==2, yhat),1],X[findall(x->x==2, yhat),2],".")
plot(X[findall(x->x==3, yhat),1],X[findall(x->x==3, yhat),2],".")
savefig("pic3.png")
gcf()
