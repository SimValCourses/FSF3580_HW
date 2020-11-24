# Rayleigh Quotient iteration
using LinearAlgebra

function RQI(A,v,kmax, e_ref)
    RQ_err_v = zeros(kmax,1);
    for k = 1:kmax
        global x, e_rq
        e_rq = (x'*A*x) ./ (x'*x);
        x = (A-e_rq*I)\x;
        x = x/norm(x);
        RQ_err_v[k] = abs(e_rq - e_ref);
    end
    return x, e_rq, RQ_err_v
end
