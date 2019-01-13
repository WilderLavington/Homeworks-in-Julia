# Load X and y variable
using JLD, Statistics, Printf
data = load("outliersData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("leastSquares.jl")
model = leastSquares(X,y)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with least squares: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with least squares: %.3f\n",testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.01:maximum(X)
yhat = model.predict(Xhat)
plot(Xhat,yhat,"g")
title("Least Squares Fit")
savefig("figure_01.png")
gcf()

using MathProgBase, GLPKMathProgInterface

function leastAbsolutes(X,y)
    # add bias to X
    bias = ones(size(y))
    phi = [bias X]
    # initialize matrices
    M = size(phi)[2]
    b = zeros(2*size(y)[1])
    A = zeros(2*size(y)[1], M)
    # generate constraint matrices
    for i = 1:2:2*size(y)[1]
        A[i,1:M] = phi[Int((i+1)/2),:]
        A[i+1,1:M] = -phi[Int((i+1)/2),:]
        b[i] = y[Int((i+1)/2)]
        b[i+1] = -y[Int((i+1)/2)]
        zero_vec = zeros(2*size(y)[1])
        zero_vec[i] = -1
        zero_vec[i+1] = -1
        A = [A zero_vec]
    end
    # generate minimization matrix
    C = zeros(size(y)[1]+M)
    for i = M:size(y)[1]+M
        C[i] = 1
    end
    # apply linsolve
    w = linprog(C, A, -Inf, b, -Inf, Inf, GLPKSolverLP())
    return w.sol[1:M]
end

w = leastAbsolutes(X,y)
bias = ones(size(y))

# Plot model
figure()
plot(X,y,"b.")
Xhat = minimum(X):1/502:maximum(X)
phi = [bias Xhat]
yhat = phi*w
plot(Xhat, yhat,"g")
title("Least Absolutes (Robust Regression) Fit")
savefig("figure_02.png")
gcf()

function leastMax(X,y)
    # add bias to X
    bias = ones(size(y))
    phi = [bias X]
    # initialize matrices
    M = size(phi)[2]
    b = zeros(2*size(y)[1])
    A = zeros(2*size(y)[1], M)
    # generate constraint matrices
    for i = 1:2:2*size(y)[1]
        A[i,1:M] = phi[Int((i+1)/2),:]
        A[i+1,1:M] = -phi[Int((i+1)/2),:]
        b[i] = y[Int((i+1)/2)]
        b[i+1] = -y[Int((i+1)/2)]
    end
    zero_vec = -1*ones(2*size(y)[1])
    A = [A zero_vec]
    # generate minimization matrix
    C = zeros(M+1)
    C[M+1] = 1
    # apply linsolve
    w = linprog(C, A, -Inf, b, -Inf, Inf, GLPKSolverLP())
    return w.sol[1:M]
end

#
w = leastMax(X,y)
bias = ones(size(y))

# Plot model
figure()
plot(X,y,"b.")
Xhat = minimum(X):1/502:maximum(X)
phi = [bias Xhat]
yhat = phi*w
plot(Xhat, yhat,"g")
title("Least Max (Brittle Regression) Fit")
savefig("figure_03.png")
gcf()
