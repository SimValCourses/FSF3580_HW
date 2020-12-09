using LinearAlgebra, PyPlot, LaTeXStrings
include("QR_val.jl")
include("alpha_example.jl")

tol = 1e-10;
alpha_v = 10.0.^(-5:1:5);
n = 20;
iterv = zeros(length(alpha_v),1);
err_m = zeros(n,n,length(alpha_v));
eigm = zeros(n,length(alpha_v));
iterv_th = zeros(length(alpha_v),1);
conv_v = zeros(length(alpha_v),1);

for (i, α) in enumerate(alpha_v)
    A = alpha_example(α,n);
    global T, U, iter, err_v = QR_val(A, tol);
    iterv[i] = iter;
    W = eigen(A);
    ee = sort(abs.(W.values));
    eigm[:,i] = ee;
    for (j, λ1) in enumerate(ee)
        for (k, λ2) in enumerate(ee[j+1:n])
            err_m[k,j,i] = λ1/λ2;
        end
    end
    conv_v[i] = maximum(maximum(abs.(err_m[:,:,i])));
    iterv_th[i] = log(tol)/log(conv_v[i]);
end

pygui(true)
figure(1)
semilogx(alpha_v, iterv, color="red", linestyle="-",
        label=L"$\textnormal{Number of iterations}$")
semilogx(alpha_v, iterv_th, color="black", linestyle="--",
        label=L"$\textnormal{Predicted number of iterations}$")
ylabel(L"$\textnormal{Iterations}$")
xlabel(L"$\alpha$")
legend()
