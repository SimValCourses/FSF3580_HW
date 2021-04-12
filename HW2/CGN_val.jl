function my_cgn_val(A, b, m, x_exact, A_orig, b_orig)
    r = copy(b);
    p = copy(b);
    x = zeros(length(b),1);
    res_v = zeros(m,1);
    err_v = zeros(m,1);
    res_v_orig = zeros(m,1);
    res_v_2 = zeros(m,1);
    time_v = zeros(m,1);

    for i = 1:m
        t_cgn = @timed begin
            alpha = (r'*r)/(p'*A*p);
            x = x + alpha .*p;
            r_old = deepcopy(r);
            r = r_old - alpha .*(A*p);
            beta = (r'*r)/(r_old'*r_old);
            p = r + beta .*p;
        end
        if i == 1
            time_v[i] = t_cgn.time;
        else
            time_v[i] = time_v[i-1] + t_cgn.time;
        end
        res_v[i]= norm(A*x-b)/norm(b)	         # residual norm
        err_v[i]= norm(x-x_exact)/norm(x_exact); # error norm
        res_v_orig[i]= norm(A_orig*x-b_orig)/norm(b_orig); # error norm
        res_v_2[i] = norm(r);
    end
    return x, res_v, err_v, time_v, res_v_orig, res_v_2
end
