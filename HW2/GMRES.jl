function gmres_xe(A,b,m,xe,v)

    n=length(b);
    Q=complex(zeros(n,m+1));
    H=complex(zeros(m+1,m));
    Q[:,1]=b/norm(b);
	resnormv=zeros(m);
	errnormv=zeros(m);
	
	for k=1:m
		global xtilde
		w          = A*Q[:,k];			# compute A*q_k
        h,β,z      = GS(Q,w,k,v);		# orthogonalize w against columns of Q_k
        Q[:,k+1]   = z/β;				# add new column to Q_k --> Q_(k+1)
        H[1:k+1,k] = [h;β];				# build upper Hesserberg matrix
		e1         = zeros(k+1);		# construct RHS
		e1[1]      = norm(b);
		zstar      = H[1:k+1,1:k]\e1;	# solve linear system
		xtilde     = Q[:,1:k]*zstar;	# compute approximate solution
		resnormv[k]= norm(A*xtilde-b)/norm(b)	# compute residual norm
		errnormv[k]= norm(xe-xtilde)/norm(xe)	# compute error norm given exact result xe
	end
	return xtilde,resnormv,errnormv

end
