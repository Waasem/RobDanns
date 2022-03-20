function [ pi2, opt_cost ] = OMT( p0, p1, C )
% p0 = initial distribution
% p1 = final distribution
% C = transference cost
% pi1 or pi2 = transference plan 
flat = @(x)x(:);
Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), flat(reshape(1:n0*n1,n0,n1) ), ones(n0*n1,1) );
Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), flat(reshape(1:n0*n1,n0,n1)' ),ones(n0*n1,1) );
Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];
Aeq=Sigma(length(p0),length(p1)); beq=[p0(:);p1(:)];
% positivity constraint
lb=zeros(size(C(:)));
% solve linear program 
% maxit = 1e4; tol = 1e-9;
options = optimset('Display','none');
pi1 = linprog(C(:),[],[],Aeq,beq,lb,[],[],options);  % as ub and x0 = []
pi2 = reshape(pi1,[length(p0) length(p1)]);
indices = pi2<1.0e-6;
pi2(indices) = 0;  
pi2= sparse(pi2);
opt_cost = pi1'*C(:);
end

