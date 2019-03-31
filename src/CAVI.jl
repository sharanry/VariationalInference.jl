# Using Base modules.
using Random
# Load the distributions library.
using Distributions


mutable struct UGMM_CAVI <: InferenceAlgorithm
    X # Data
    K::Int       # number of iterations
    N::Int
    sigma2
    m
    phi
    s2
end

function UGMM_CAVI(X, num_components::Int, n_samples::Int, sigma2=1)
    K=num_components
    N=n_samples
    rng = RandomDevice()
    phi = rand(Dirichlet(K*N, (rand(rng, Float32,(1))*rand(rng, 1:9))[1]),(K));
    m = rand(rng,floor(minimum(X)):floor(maximum(X)), (1,K))
    m += maximum(X)*rand(rng, Float32, (1,K))
    s2 = rand(rng, Float32, (1, K))
    print("\nInit mean\n")
    println(m)
    print("\nInit s2\n")
    println(s2)
    UGMM_CAVI(X, num_components, n_samples, sigma2, m, phi, s2)
       
end

function get_elbo(alg::UGMM_CAVI)
    t1 = log.(alg.s2) - alg.m./(alg.sigma2)
    t1 = sum(t1)
    t2 = -0.5.*(reshape(alg.X, (alg.N*alg.K, 1)).^2 * ones(1, alg.K) + ones(alg.N*alg.K, 1)*((alg.s2)+(alg.m).^2 ))
    t2 .+= reshape(alg.X, (alg.N*alg.K, 1))*(alg.m)
    t2 .-= log.(alg.phi)
    t2 .+= alg.phi
    t2 = sum(t2)
    return t1 + t2
end

function _update_phi(alg::UGMM_CAVI)
    t1 = reshape(alg.X, (alg.N*alg.K, 1))*alg.m
    t2 = -(0.5*alg.m.^2 + 0.5*alg.s2)
    exponent = t1 .+ t2
    phi = exp.(exponent)
    alg.phi = phi ./ sum(phi,dims=2)
end
function _update_mu(alg::UGMM_CAVI)
    alg.m = sum((alg.phi.*reshape(alg.X, (alg.N*alg.K, 1))),dims=1) .* (1 ./alg.sigma2 .+ sum(alg.phi,dims=1)).^(-1)
    @assert size(alg.m)[2] == alg.K
    #print(self.m)
    alg.s2 = (1 ./alg.sigma2 .+ sum(alg.phi,dims=1)).^(-1)
    @assert size(alg.s2)[2] == alg.K
end

function _cavi_step(alg::UGMM_CAVI)
    _update_phi(alg)
    _update_mu(alg)
end

function fit(alg::UGMM_CAVI; max_iter=100, tol=1e-20, print_interval=50)
    elbo_values = [get_elbo(alg)]
    m_history = [alg.m]
    s2_history = [alg.s2]
    for iter_ in 1:(max_iter+1)
        _cavi_step(alg)
        append!(m_history, [alg.m])
        append!(s2_history, [alg.s2])
        append!(elbo_values, [get_elbo(alg)])

        if iter_ % print_interval == 0
            println(iter_, " Means: ", m_history[iter_], " ELBO: ", elbo_values[end])
        end
        if abs(elbo_values[end-1] - elbo_values[end]) < tol
            i=iter_
            elbo_val = elbo_values[end]
            println("ELBO converged with $elbo_val at iteration $i")
            break
        end
        if iter_ == max_iter
            elbo_val = elbo_values[end]
            print("ELBO ended with ll $elbo_val")
        end
    end    
    elbo_values, m_history ,s2_history
end