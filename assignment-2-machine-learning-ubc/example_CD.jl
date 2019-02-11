# Load X and y variable
using JLD, Printf, LinearAlgebra
data = load("binaryData.jld")
(X,y) = (data["X"],data["y"])

# Add bias variable, initialize w, set regularization and optimization parameters
(n,d) = size(X)
X = [ones(n,1) X]
d += 1
w = zeros(d,1)
lambda = 1
maxPasses = 500
progTol = 1e-4
verbose = true

## Run and time coordinate descent to minimize L2-regularization logistic loss

# Start timer
time_start = time_ns()
# Compute Lipschitz constant of 'f'
sd = eigen(X'X)
L = maximum(sd.values) + lambda;
# Start running coordinate descent
w_old = copy(w);
for k in 1:maxPasses*d
    # Choose variable to update 'j'
    j = rand(1:d)
    # Compute partial derivative 'g_j'
    r = X*w - y
    g = X'*r + lambda*w
    g_j = g[j];
    # Update variable
    w[j] -= (1/L)*g_j;
    # Check for lack of progress after each "pass"
    # - Turn off computing 'f' and printing progress if timing is crucial
    if mod(k,d) == 0
        r = X*w - y
        f = (1/2)norm(r)^2 + (lambda/2)norm(w)^2
        delta = norm(w-w_old,Inf);
        if verbose
            @printf("Passes = %d, function = %.4e, change = %.4f\n",k/d,f,delta);
        end
        if delta < progTol
            @printf("Parameters changed by less than progTol on pass\n");
            break;
        end
        global w_old = copy(w);
    end
end
# End timer
@printf("Seconds = %f\n",(time_ns()-time_start)/1.0e9)
# modified q

""" Coordinate Decent Using Coordinate-wise decent O(n)"""
function CD_On(X, y, d)
    # Compute Lipschitz constant of 'f'
    sd = eigen(X'X)
    L = maximum(sd.values) + lambda;
    # Start running coordinate descent
    w = zeros(d,1)
    w_old = copy(w);
    # pre-compute something
    v = -X'*y
    z = X*w
    for k in 1:maxPasses*d
        # Choose variable to update 'j'
        j = rand(1:d)
        # Compute partial derivative 'g_j' ~ O(n)
        g_j = v[j] + sum(X[:,j].*z) + lambda*w[j]
        # Update variable ~O(1)
        w_prev = copy(w[j])
        w[j] -= (1/L)*g_j;
        w_new = copy(w[j])
        # now update the residual ~ O(n)
        z = z .- X[:,j].*w_prev .+ X[:,j].*w_new
        # Check for lack of progress after each "pass"
        # - Turn off computing 'f' and printing progress if timing is crucial
        if mod(k,d) == 0
            r = X*w - y
            f = (1/2)norm(r)^2 + (lambda/2)norm(w)^2
            delta = norm(w-w_old,Inf);
            if verbose
                @printf("Passes = %d, function = %.4e, change = %.4f\n",k/d,f,delta);
            end
            if delta < progTol
                @printf("Parameters changed by less than progTol on pass\n");
                break;
            end
            w_old = copy(w);
        end
    end
end
time_start = time_ns()
w_cd1 = CD_On(X, y, d)
@printf("Seconds = %f\n",(time_ns()-time_start)/1.0e9)

""" Coordinate Decent Using Coordinate-wise decent O(n) with optimal stepsize"""
function CD_On_OS(X, y, d)
    # Compute Lipschitz constant of 'f'
    sd = eigen(X'X)
    # Start running coordinate descent
    w = zeros(d,1)
    w_old = copy(w);
    # pre-compute something
    v = -X'*y
    z = X*w
    for k in 1:maxPasses*d
        # Choose variable to update 'j'
        j = rand(1:d)
        # Compute partial derivative 'g_j' ~ O(n)
        g_j = v[j] + sum(X[:,j].*z)
        # calculate optimal update
        optimal_update = (sum(X[:,j].*y) - sum(z) + sum(X[:,j])*w[j]) / (lambda + sum(X[:,j].*X[:,j]))
        # get stepsize that would achieve this
        alpha = (w[j] - optimal_update) / g_j
        # Update variable ~O(1)
        w_prev = copy(w[j])
        w[j] -= alpha*g_j;
        w_new = copy(w[j])
        # now update the residual ~ O(n)
        z = z .- X[:,j].*w_prev .+ X[:,j].*w_new
        # Check for lack of progress after each "pass"
        # - Turn off computing 'f' and printing progress if timing is crucial
        if mod(k,d) == 0
            r = X*w - y
            f = (1/2)norm(r)^2 + (lambda/2)norm(w)^2
            delta = length(w)*norm(w-w_old,Inf);
            if verbose
                @printf("Passes = %d, function = %.4e, change = %.4f\n",k/d,f,delta);
            end
            if delta < progTol
                @printf("Parameters changed by less than progTol on pass\n");
                break;
            end
            w_old = copy(w);
        end
    end
end
time_start = time_ns()
CD_On_OS(X, y, d)
@printf("Seconds = %f\n",(time_ns()-time_start)/1.0e9)
