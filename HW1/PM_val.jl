# Power iteration
using LinearAlgebra

function PM(A,v,kmax, e_ref)
    PM_err_v = zeros(nmax,1)
    for k = 1:kmax
        global x, e_pm
        x = A*x;
        x = x/norm(x);
        e_pm = (x'*A*x) ./ (x'*x);
        PM_err_v[k] = abs(e_pm - e_ref);
    end
    return x, e_pm, PM_err_v
end
