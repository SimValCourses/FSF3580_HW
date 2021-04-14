using LinearAlgebra

function kpik(A,B,E,LE,m,tol,tolY,infoV)

  #@assert(isdefined(:vecnorm),"Your julia version is too old. vecnorm() not defined")

  infoV
  start = time()

  #### Check if we solve the general lyapunov equation and user did not provide LE
  if E != 1 && LE == 1
    #### If this is the case, calculate LE
    if issparse(E)
      LE = try
            sparse(cholfact(E,perm=1:size(E,1))[:L])
           catch
            error("E must be SPD")
          end
    else
      LE = try
            cholfact(E)[:L]
          catch
            error("E must be SPD")
          end
    end
  end
  rhs=LE\B
  nrmb=norm(rhs)^2
  nrma=norm(A)
  sqrt2=sqrt(2)
  er2=zeros(m,1)

  n,sh=size(rhs)

  Y=[]
  odds=[]
  er2=[]

  if E == 1
    condestE=1
    singE=condestE/norm(Matrix(I,n,n))
elseif (norm(E-sparse(I,n,n))>1e-14)
    condestE=cond(Matrix(E))
    singE=condestE/norm(E)
  else
    singE=1
  end

  if norm(A-A',1)<1e-14
     if issparse(A)
       cfA = cholfact(-A,perm=1:size(A,1))
       LA = sparse(cfA[:L])
       UA = -LA'
     else
       UA = chol(-A)
       LA = -UA'
     end
     infoV && println("A sym. Completed Chol factorization\n")
     k_max =2
   else
     luA = lu(Matrix(A))
     LA = luA.L
     UA = luA.U


     infoV && println("A nonsym. Completed LU factorization\n")
     k_max = m
   end

   s=2*sh
   #rhs1=LE'*(UA\(LA\(LE*rhs)))

   # Orthogonalize [B,A^{-1}B] with an economy-size QR
   rr = [ rhs LE'*(UA\(LA\(LE*rhs))) ]

   # Julia qr decomposition is always "economy size"
   U,beta=qr(rr)
   U = U[1:n,1:s]

   ibeta=inv(beta[1:s,1:s])
   beta = beta[1:sh,1:sh]
   beta2=beta*beta'
   H=zeros((m+1)*s,m*s)
   T=zeros((m+1)*s,m*s)
   L=zeros((m+1)*s,m*s)
   local js, j, rho
   for j=1:m
     jms=(j-1)*s+1
     j1s=(j+1)*s
     js=j*s
     js1=js+1
     jsh=(j-1)*s+sh

     # Expand the basis
     # multiply by A
     Up = zeros(n,s)
     Up[1:n,1:sh] = LE\(A*(LE'\U[:,jms:jsh]))
     # solve with A

     Up[1:n,sh+1:s] = LE'*(UA\(LA\(LE*U[1:n,jsh+1:js])))

     # orthogonalize the new basis block wrt all the previous ones by modified gram
     for l=1:2
        k_min=max(1,j-k_max)
        for kk=k_min:j
            k1=(kk-1)*s+1
            k2=kk*s
            coef= U[1:n,k1:k2]'*Up
            H[k1:k2,jms:js] = H[k1:k2,jms:js]+ coef
            Up = Up - U[:,k1:k2]*coef
        end
      end

      if j<=m
        Up,H[js1:j1s,jms:js] = qr(Up)
        hinv=inv(H[js1:j1s,jms:js])
      end


      ###############################################################
      # Recover the columns of T=U'*A*U (projection of A onto the space) from
      # the colums of H.
      # REMARK: we need T as coefficient matrix of the projected problem.
      Iden=sparse(I, js+s, js+s)

      if (j==1)
        L[1:j*s+sh,(j-1)*sh+1:j*sh] = [H[1:s+sh,1:sh]/ibeta[1:sh,1:sh] Matrix(I,s+sh,sh)/ibeta[1:sh,1:sh]]*ibeta[1:s,sh+1:s]
      else
        L[1:j*s+s,(j-1)*sh+1:j*sh] = L[1:j*s+s,(j-1)*sh+1:j*sh] + H[1:j*s+s,jms:jms-1+sh]*rho
      end

      odds = [odds; jms:(jms-1+sh)]   # store the odd block columns
      evens = 1:js
      flag = trues(size(evens))
      flag[odds] .= false
      evens = evens[flag]
      T[1:js+s,odds]=H[1:js+s,odds]   #odd columns

      T[1:js+sh,evens]=L[1:js+sh,1:j*sh]   #even columns
      L[1:j*s+s,j*sh+1:(j+1)*sh] = ( Iden[1:j*s+s,(js-sh+1):js]- T[1:js+s,1:js]*H[1:js,js-sh+1:js])*hinv[sh+1:s,sh+1:s]
      rho = hinv[1:sh,1:sh]\hinv[1:sh,sh+1:s]

      #################################################################

      # Solve the projected problem by Bartels-Stewart
      # Do "type lyap" from command window if interested
      TT = T[1:js,1:js]
      Y = lyap(TT,Matrix(I,j*s,sh)*beta2*Matrix(j*s,sh)')

      # safeguard to preserve symmetry
      Y = (Y+Y')/2

      # Compute the residual norm. See the article by Valeria

      cc = [H[js1:j1s,js-s+1:js-sh] L[js1:j1s,(j-1)*sh+1:j*sh]]

      nrmx = norm(Y)

      er2=[er2;sqrt2*norm(cc*Y[js-s+1:js,:])/(nrmb+singE*nrma*nrmx)]

      infoV && println("KPIK It: $j -- Current Backwards Error: $(er2[j])")

      (er2[j]<tol) ? break : U = [U Up]

    end

    # Done
    # reduce solution rank if needed
    sY,uY=eig(Y)
    id=sortperm(sY)
    sY=sort(sY)

    sY=flipdim(sY,1)
    uY=uY[:,id[end:-1:1]]
    is = 0
    for ii in 1:size(sY)[1]
      if abs(sY[ii])>tolY
        is = is+1
      end
    end

    Y0 = uY[:,1:is]*diagm(sqrt(sY[1:is]))
    Z = LE'\(U[1:n,1:js]*Y0)
    er2=er2[1:j]
    elapsed = time()-start

    if infoV
      println("its  Back. Error            space dim. CPU Time")
      println("$j    $(er2[j])  $js          $(elapsed)")
    end

    return Z, er2

end
