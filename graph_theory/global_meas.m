function [APL,avg_cc, avg_deg, glo_eff, ALE, CPL, kappa1, kappa_abs, Ent, kappa_matrix] = global_meas(A)
% This also gives kappa matrix which global_meas does not give
%   Detailed explanation goes here
APL = charpath(A); % average path length
%cc_1 = mean(clustering_coef_bu(A));
avg_cc = mean(clustering_coef_wu(A));
avg_deg = mean(degrees_und(A));
glo_eff = efficiency_bin(A); % global efficiency
[~,~,~,~,~,ALE] = graphProperties(A);
[CPL,~,~,~,~,~] = graphProperties(A);
kappa = kappa_HCP( A); % Compute the kappa matrix
A_bin = A;
A_bin(A>=1)=1;
kappa_matrix = kappa*A_bin;
kappa1 = mean(sum(kappa_matrix,2));
kappa_abs = mean(abs(sum(kappa_matrix,2)));
%     % Algebraic connectivity
%     G = graph(A) ;        % str
%     L = laplacian(G);
%     [~, D] = eigs(L,2,'sm');
%     s=-sort(-diag(D));
%     algeb_conn = s(1,1);
    % topological entropy
    E = log(max(eig(A)));
    Ent(1,:) = E;
end

