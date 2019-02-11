# Load X and y variable
using JLD, Printf, LinearAlgebra
data = load("quantum.jld")
(X,y) = (data["X"],data["y"])

# Add bias variable, initialize w, set regularization and optimization parameters
(n,d) = size(X)
lambda = 1

# Initialize
maxPasses = 10
progTol = 1e-4
verbose = true
w = zeros(d,1)
lambda_i = lambda/n # Regularization for individual example in expectation

# Start running stochastic gradient
w_old = copy(w);
for k in 1:maxPasses*n

    # Choose example to update 'i'
    i = rand(1:n)

    # Compute gradient for example 'i'
    r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
    g_i = r_i*X[i,:] + (lambda_i)*w

    # Choose the step-size
    alpha = 1/(lambda_i*k)

    # Take thes stochastic gradient step
    global w -= alpha*g_i

    # Check for lack of progress after each "pass"
    if mod(k,n) == 0
        yXw = y.*(X*w)
        f = sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
        delta = norm(w-w_old,Inf);
        if verbose
            @printf("Passes = %d, function = %.4e, change = %.4f\n",k/n,f,delta);
        end
        if delta < progTol
            @printf("Parameters changed by less than progTol on pass\n");
            break;
        end
        global w_old = copy(w);
    end
end


""" Looking at different stepsize types """
function Stochastic_gradient(X, y, w_init, lambda, alpha_sequence, maxPasses)

    # intitialize things
    w = w_init
    lambda_i = lambda/n # Regularization for individual example in expectation
    iterations = 0

    # Start running stochastic gradient
    w_old = copy(w);

    for k in 1:maxPasses*n

        # Choose example to update 'i'
        i = rand(1:length(y))

        # Compute gradient for example 'i'
        r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
        g_i = r_i*X[i,:] + (lambda_i)*w

        # Take thes stochastic gradient step
        w -= alpha_sequence(k)*g_i

        # Check for lack of progress after each "pass"
        if mod(k,n) == 0
            yXw = y.*(X*w)
            f = sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
            delta = norm(w-w_old,Inf);
            if verbose
                @printf("Passes = %d, function = %.4e, change = %.4f\n",k/n,f,delta);
            end
            if delta < progTol
                @printf("Parameters changed by less than progTol on pass\n");
                break;
            end
            w_old = copy(w);
        end
        iterations += 1
    end
    return w, iterations, sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
end

function test_const(X, y, w_init, lambda, n, maxPasses)
    possible_alphas =  range(1, stop = 50, length = n) |> collect
    run_times = zeros(n,1)
    evals = zeros(n,1)
    for i = 1:n
        alpha = possible_alphas[i]
        alpha_seq(k) = map(k -> 1 / (alpha*k), k)
        w, iterations, eval = Stochastic_gradient(X, y, w_init, lambda, alpha_seq, maxPasses)
        run_times[i] = iterations
        evals[i] = eval
    end
    return run_times, possible_alphas, evals
end

function test_decay(X, y, w_init, lambda, n, maxPasses)
    possible_alphas =  range(1, stop = 50, length = n) |> collect
    run_times = zeros(n,1)
    evals = zeros(n,1)
    for i = 1:n
        alpha = possible_alphas[i]
        alpha_seq(k) = map(k -> 1 / (alpha*k^0.5), k)
        w, iterations, eval = Stochastic_gradient(X, y, w_init, lambda, alpha_seq, maxPasses)
        run_times[i] = iterations
        evals[i] = eval
    end
    return run_times, possible_alphas, evals
end

function test_adapt_decay(X, y, w_init, lambda, n, maxPasses)
    possible_alphas = range(0.01, stop = 1, length = n) |> collect
    run_times = zeros(n,1)
    evals = zeros(n,1)
    for i = 1:n
        alpha = possible_alphas[i]
        alpha_seq(k) = map(k -> 1 / (1+k^(0.5 + alpha)), k)
        w, iterations, eval = Stochastic_gradient(X, y, w_init, lambda, alpha_seq, maxPasses)
        run_times[i] = iterations
        evals[i] = eval
    end
    return run_times, possible_alphas, evals
end

function mesh_adapt_decay(X, y, w_init, lambda, n, maxPasses)
    possible_alphas = range(0.01, stop = 0.5, length = n) |> collect
    alpha_0 = range(0.01, stop = 5, length = n) |> collect
    run_times = zeros(n,n)
    evals = zeros(n,n)
    best_alpha0 = 1
    best_alpha = 1
    best_eval = 100000
    for j = 1:n
        for i = 1:n
            alpha = possible_alphas[i]
            alpha_seq(k) = map(k -> alpha_0[j] / (1+k^(0.5 + alpha)), k)
            w, iterations, eval = Stochastic_gradient(X, y, w_init, lambda, alpha_seq, maxPasses)
            run_times[i,j] = iterations
            evals[i,j] = eval
            if eval < best_eval
                best_alpha0 = alpha_0[j]
                best_alpha = alpha
                best_eval = eval
            end
        end
    end
    return run_times, possible_alphas, evals, best_alpha0, best_alpha
end

# Initialize
maxPasses = 10
progTol = 1e-4
verbose = true
w_init = zeros(d,1)
lambda_i = lambda/n # Regularization for individual example in expectation
n = 25

run_times_1, possible_alphas_1, evals_1  = test_decay(X, y, w_init, lambda, n, maxPasses)

run_times_2, possible_alphas_2, evals_2 = test_const(X, y, w_init, lambda, n, maxPasses)

run_times_3, possible_alphas_3, evals_3 = test_adapt_decay(X, y, w_init, lambda, n, maxPasses)

using PyPlot
figure()
plot(possible_alphas_1, evals_1)
gcf()
figure()
plot(possible_alphas_2, evals_2)
gcf()
figure()
plot(possible_alphas_3, evals_3)
gcf()

run_times_4, possible_alphas_4, evals_4, best_alpha0, best_alpha = mesh_adapt_decay(X, y, w_init, lambda, n, maxPasses)


""" Averageing strategies simple running average """
function Stochastic_gradient_avg(X, y, w_init, lambda, alpha_sequence, maxPasses)

    # intitialize things
    w = w_init
    lambda_i = lambda/n # Regularization for individual example in expectation
    iterations = 0

    # Start running stochastic gradient
    w_old = copy(w);

    for k in 1:maxPasses*n

        # Choose example to update 'i'
        i = rand(1:length(y))

        # Compute gradient for example 'i'
        r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
        g_i = r_i*X[i,:] + (lambda_i)*w

        # Take thes stochastic gradient step
        w = (1/k)*((w-alpha_sequence(k)*g_i) + (k-1)*w)

        # Check for lack of progress after each "pass"
        if mod(k,n) == 0
            yXw = y.*(X*w)
            f = sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
            delta = norm(w-w_old,Inf);
            if verbose
                @printf("Passes = %d, function = %.4e, change = %.4f\n",k/n,f,delta);
            end
            if delta < progTol
                @printf("Parameters changed by less than progTol on pass\n");
                break;
            end
            w_old = copy(w);
        end
        iterations += 1
    end
    return w, iterations, sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
end

""" Averageing strategies simple running average """
function Stochastic_gradient_decay_avg(X, y, w_init, lambda, alpha_sequence, maxPasses, beta)

    # intitialize things
    w = w_init
    lambda_i = lambda/n # Regularization for individual example in expectation
    iterations = 0

    # Start running stochastic gradient
    w_old = copy(w);

    for k in 1:maxPasses*n

        # Choose example to update 'i'
        i = rand(1:length(y))

        # Compute gradient for example 'i'
        r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
        g_i = r_i*X[i,:] + (lambda_i)*w

        # Take thes stochastic gradient step
        w = (1-beta)*(w-alpha_sequence(k)*g_i) + beta*w

        # Check for lack of progress after each "pass"
        if mod(k,n) == 0
            yXw = y.*(X*w)
            f = sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
            delta = norm(w-w_old,Inf);
            if verbose
                @printf("Passes = %d, function = %.4e, change = %.4f\n",k/n,f,delta);
            end
            if delta < progTol
                @printf("Parameters changed by less than progTol on pass\n");
                break;
            end
            w_old = copy(w);
        end
        iterations += 1
    end
    return w, iterations, sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
end

# test some different betas
function test_betas(X, y, w_init, lambda, n, maxPasses, alpha_0, alpha)
    betas = range(0.01, stop = 0.5, length = n) |> collect
    run_times = zeros(n,1)
    evals = zeros(n,1)
    best_beta = 0
    best_eval = 10000000
    for i = 1:n
        beta = betas[i]
        alpha_seq(k) = map(k -> alpha_0 / (1+k^(0.5 + alpha)), k)
        w, iterations, eval = Stochastic_gradient_avg(X, y, w_init, lambda, alpha_seq, maxPasses)
        run_times[i] = iterations
        evals[i] = eval
        if eval < best_eval
            best_beta = beta
            best_eval = eval
        end
    end
    return run_times, betas, evals, best_beta
end

# use best alphas from before
alpha_seq(k) = map(k -> best_alpha0 / (1+k^(0.5 + best_alpha)), k)

# get convergence for Averageing
w, iterations, f = Stochastic_gradient_avg(X, y, w_init, lambda, alpha_seq, maxPasses)

# now lets try different decaying averages
n = 100
run_times_5, possible_betas_5, evals_5, best_beta = test_betas(X, y, w_init, lambda, n, maxPasses, best_alpha0, best_alpha)

# maybe try some plotting
figure()
plot(possible_betas_5, evals_5)
gcf()

""" Creating and testing adagrad """
function AdaGrad(X, y, w_init, lambda, alpha_sequence, maxPasses, delta)
    # intitialize things
    w = w_init
    lambda_i = lambda/n # Regularization for individual example in expectation
    iterations = 0
    # Start running stochastic gradient
    w_old = copy(w);
    # initialize diagnol matrix stored as vector
    D =  zeros(length(w))
    for k in 1:maxPasses*n
        # Choose example to update 'i'
        i = rand(1:length(y))
        # Compute gradient for example 'i'
        r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
        g_i = r_i*X[i,:] + (lambda_i)*w
        # update diagnol matrix
        D = D + g_i.*g_i
        # compute scaling
        scaling = 1 ./ (delta .+ D).^0.5
        # Take thes stochastic gradient step
        w -= alpha_sequence(k)*scaling.*g_i
        # Check for lack of progress after each "pass"
        if mod(k,n) == 0
            yXw = y.*(X*w)
            f = sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
            delta = norm(w-w_old,Inf);
            if verbose
                @printf("Passes = %d, function = %.4e, change = %.4f\n",k/n,f,delta);
            end
            if delta < progTol
                @printf("Parameters changed by less than progTol on pass\n");
                break;
            end
            w_old = copy(w);
        end
        iterations += 1
    end
    return w, iterations, sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
end

function test_delta(X, y, w_init, lambda, alpha_sequence, maxPasses, n)
    deltas = range(0.1, stop = 10, length = n) |> collect
    run_times = zeros(n,1)
    evals = zeros(n,1)
    best_delta = 0
    best_eval = 1000000
    for i = 1:n
        delta = deltas[i]
        w, iterations, eval = AdaGrad(X, y, w_init, lambda, alpha_sequence, maxPasses, delta)
        run_times[i] = iterations
        evals[i] = eval
        if eval < best_eval
            best_delta = delta
            best_eval = eval
        end
    end
    return run_times, deltas, evals, best_delta
end

alpha_seq(k) = map(k -> best_alpha0 / (1+k^(0.5 + best_alpha)), k)
run_times_6, deltas_6, evals_6, best_delta_6 = test_delta(X, y, w_init, lambda, alpha_seq, maxPasses, n)

# maybe try some plotting
figure()
plot(possible_betas_5, evals_5)
gcf()

# using approximately best info, lets see how we do.
alpha_seq(k) = map(k -> 0.11 / (1+k^(best_alpha-.1)), k)
w, iterations, f = AdaGrad(X, y, w_init, lambda, alpha_seq, maxPasses, best_delta_6)

""" SAG algorithm with 1/L stepsize """
function SAG(X, y, w_init, lambda, maxPasses)
    # intitialize things
    w = w_init
    (n,d) = size(X)
    lambda_i = lambda/n # Regularization for individual example in expectation
    iterations = 0
    # calculate L
    L_ = zeros(length(X[1,:]))
    for col = 1: length(X[1,:]);  L_[col] = norm(X[:,col],2)^2; end
    L = (1/4)*maximum(L_)+lambda
    # Start running stochastic gradient
    w_old = copy(w);
    # set initial v_i, and g
    v = zeros(length(w),n)
    g = zeros(length(w))
    for k in 1:maxPasses*n
        # Choose example to update 'i'
        i = rand(1:length(y))
        # Compute gradient for example 'i'
        r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
        g_i = r_i*X[i,:] + (lambda_i)*w
        # update g
        g = g - v[:,i] + g_i
        # update v_i
        v[:,i] = g_i
        # Take thes stochastic gradient step
        w -= g./(L*n)
        # Check for lack of progress after each "pass"
        if mod(k,n) == 0
            yXw = y.*(X*w)
            f = sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
            delta = norm(w-w_old,Inf);
            if verbose
                @printf("Passes = %d, function = %.4e, change = %.4f\n",k/n,f,delta);
            end
            if delta < progTol
                @printf("Parameters changed by less than progTol on pass\n");
                break;
            end
            w_old = copy(w);
        end
        iterations += 1
    end
    return w, iterations, sum(log.(1 .+ exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
end

w, iterations, f = SAG(X, y, w_init, lambda, maxPasses)
