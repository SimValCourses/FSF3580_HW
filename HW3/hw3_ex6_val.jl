# Exercise 6 - HW3
using LinearAlgebra, PyPlot, LaTeXStrings

expv = -16:1:-1;
ϵ_vec = 10.0.^expv;
err_v = zeros(length(ϵ_vec),1);

for (i, ϵ) in enumerate(ϵ_vec)
    A = [π 1; 0 π+ϵ];
    W = eigen(A);
    ee = W.values;
    V = W.vectors;
    D = diagm(exp.(ee));
    F = V*D*inv(V);

    # compute exact solution
    α = exp(π) - π*(exp(π+ϵ)-exp(π))/ϵ;
    β = (exp(π+ϵ)-exp(π))/ϵ;
    F_exact = α*I + β*A;

    err_v[i] = norm(F_exact-F);
end

# Plotting
pygui(true)
figure(1)
loglog(ϵ_vec, err_v, color="blue", linestyle="-")
xlabel(L"$\epsilon$")
ylabel(L"$||f(A) - F||_2$")
