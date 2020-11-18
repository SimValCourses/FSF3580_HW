function my_cg_val(A,b,m)
    r = copy(b);
    p = copy(b);
    x = zeros(length(b),1);
    for i = 1:m
        alpha = (r'*r)/(p'*A*p);
        x = x + alpha .*p;
        r_old = r;
        r = r_old - alpha .*(A*p);
        beta = (r'*r)/(r_old'*r_old);
        p = r + beta .*p;
    end
    return x
end
