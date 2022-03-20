% function [avg_BC, avg_eigen_cen, ALE, algeb_conn] = centrality_and_addl_meas(A)
function [avg_BC, avg_eigen_cen, ALE] = centrality_and_addl_meas(A)
%   Detailed explanation goes here

avg_BC = mean(betweenness_bin(A));
% avg_BC = mean(betweenness_wei(A));
avg_eigen_cen = mean(eigenvector_centrality_und(A));
[~,~,~,~,~,ALE] = graphProperties(A);

% % Algebraic connectivity
% G = graph(A) ;        % str
% L = laplacian(G);
% [~, D] = eigs(L,2,'sm');
% s=-sort(-diag(D));
% algeb_conn = s(1,1);

end

