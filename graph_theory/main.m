%% Copyright (c) Rowan AI Lab and authors of this code.

% This source code is licensed under the under the Creative Commons Zero v1.0 Universal LICENSE file in the root directory of this source tree.

%% Load matrices, compute Ollivier-Ricci curvature and Entropy 
close all
clear all
clc
%% Load data
% WS-flex graphs: 64-node 54 Graphs (including FC)
sixty_four1 = load('Graphs_plus_adjacencices_64n_53.mat');
sixty_four2 = load('Graphs_plus_adjacencices_64n_54.mat');
n = max(size(sixty_four1.result_plot_1));

%% Other experiments from ER, BA, WS graphs.
% % WS Graphs
% WS_RWN_sixty_four =load('WS_RWN_graphs_n32_100.mat');
% n = max(size(WS_RWN_sixty_four.result_plot_1));

% % ER Graphs
% ER_RWN_sixty_four =load('ER_RWN_graphs_n32_100.mat');
% n = max(size(ER_RWN_sixty_four.result_plot_1));

% % BA Graphs
% BA_RWN_sixty_four =load('BA_RWN_graphs_n32_100.mat');
% n = max(size(BA_RWN_sixty_four.result_plot_1));

% % WS-flex FC Graph
% sixty_four = load('Graphs_plus_adjacency_matrices_64n_FC.mat');
% n = max(size(sixty_four.result_FC));

%% Graph Calculations
% WS-flex Graphs
for i = 1:n
A = sixty_four.result_plot_1{i,7};
[APL,avg_cc, avg_deg,glo_eff, CPL, kappa1, kappa_abs, Ent, kappa_matrix] = global_meas(A);

%degree, p, seed, CC, L from original code
results_64.deg{1,i} = sixty_four.result_plot_1{i,2};
results_64.p{1,i} = sixty_four.result_plot_1{i,3};
results_64.seed{1,i} = sixty_four.result_plot_1{i,4};
results_64.CC{1,i} = sixty_four.result_plot_1{i,5}; %Clustering Coefficient from Python library
results_64.L{1,i} = sixty_four.result_plot_1{i,6}; %Avg Path Length from Python library
%New graph measures
results_64.adj_matrices{1,i} = A;
results_64.APL{1,i} = APL;
results_64.avg_cc{1,i} = avg_cc;
results_64.avg_deg{1,i} = avg_deg;
results_64.glo_eff{1,i} = glo_eff;
results_64.CPL{1,i} = CPL;
%results_64.algeb_conn{1,i} = algeb_conn;
results_64.kappa{1,i} = kappa1;
results_64.kappa_abs{1,i} = kappa_abs;
results_64.entropy{1,i} = Ent;
results_64.kappa_matrix{1,i} = kappa_matrix;
%%
% % Additional Graph calculations for WS-flex graphs
% for i = 1:n
% A = sixty_four1.result_plot_1{i,7};
% [avg_BC, avg_eigen_cen, ALE] = centrality_and_addl_meas(A);
% % deg, p, seed, CC, L, APL, avg_deg, glo_eff, CPL, kappa_abs, entropy, Adj_Matrix
% 
% %degree, p, seed, CC, L from original code
% results_64.deg{1,i} = sixty_four2.stackup_with_adj_and_FC{i,1};
% results_64.p{1,i} = sixty_four2.stackup_with_adj_and_FC{i,2};
% results_64.seed{1,i} = sixty_four2.stackup_with_adj_and_FC{i,3};
% results_64.CC{1,i} = sixty_four2.stackup_with_adj_and_FC{i,4}; %Clustering Coefficient from Python library
% results_64.L{1,i} = sixty_four2.stackup_with_adj_and_FC{i,5}; %Avg Path Length from Python library
% results_64.APL{1,i} = sixty_four2.stackup_with_adj_and_FC{i,6}; %Avg Path Length from matlab
% results_64.avg_deg{1,i} = sixty_four2.stackup_with_adj_and_FC{i,7}; %Avg Degree from matlab
% results_64.glo_eff{1,i} = sixty_four2.stackup_with_adj_and_FC{i,8}; %Global efficiency from matlab
% results_64.CPL{1,i} = sixty_four2.stackup_with_adj_and_FC{i,9}; %Char Path Length from matlab
% results_64.kappa_abs{1,i} = sixty_four2.stackup_with_adj_and_FC{i,10}; %Curvature Length from matlab
% results_64.entropy{1,i} = sixty_four2.stackup_with_adj_and_FC{i,11}; %Entropy from matlab
% 
% %New graph measures
% results_64.avg_BC{1,i} = avg_BC;                % Avg Betweenness Centrality
% results_64.avg_eigen_cen{1,i} = avg_eigen_cen;  % Avg Eigenvalue Centrality
% results_64.ALE{1,i} = ALE;                       % Avg Local Efficiency
% % results_64.algeb_conn{1,i} = algeb_conn;        % Algebraic Connectivity
% results_64.adj_matrices{1,i} = A;
% i
% end

%%
% % WS Graphs
% for i = 1:n
% A = WS_RWN_sixty_four.result_plot_1{i,6};
% [APL,avg_cc, avg_deg,glo_eff, CPL, kappa1, kappa_abs, Ent, kappa_matrix] = global_meas_5Jan21(A);
% 
% %degree, p, seed, CC, L from original code
% WS_RWN_results_32.r{1,i} = WS_RWN_sixty_four.result_plot_1{i,1};
% WS_RWN_results_32.p{1,i} = WS_RWN_sixty_four.result_plot_1{i,2};
% WS_RWN_results_32.k{1,i} = WS_RWN_sixty_four.result_plot_1{i,3};
% WS_RWN_results_32.CC{1,i} = WS_RWN_sixty_four.result_plot_1{i,4}; %Clustering Coefficient from Python library
% WS_RWN_results_32.L{1,i} = WS_RWN_sixty_four.result_plot_1{i,5}; %Avg Path Length from Python library
% %New graph measures
% WS_RWN_results_32.adj_matrices{1,i} = A;
% WS_RWN_results_32.APL{1,i} = APL;
% WS_RWN_results_32.avg_cc{1,i} = avg_cc;
% WS_RWN_results_32.avg_deg{1,i} = avg_deg;
% WS_RWN_results_32.glo_eff{1,i} = glo_eff;
% WS_RWN_results_32.CPL{1,i} = CPL;
% %results_64.algeb_conn{1,i} = algeb_conn;
% WS_RWN_results_32.kappa{1,i} = kappa1;
% WS_RWN_results_32.kappa_abs{1,i} = kappa_abs;
% WS_RWN_results_32.entropy{1,i} = Ent;
% WS_RWN_results_32.kappa_matrix{1,i} = kappa_matrix;
% i
%%
% % % ER Graphs
% for i = 1:n
% A = ER_RWN_sixty_four.result_plot_1{i,5};
% [APL,avg_cc, avg_deg,glo_eff, CPL, kappa1, kappa_abs, Ent, kappa_matrix] = global_meas_5Jan21(A);
% 
% %degree, p, seed, CC, L from original code
% ER_RWN_results_32.r{1,i} = ER_RWN_sixty_four.result_plot_1{i,1};
% ER_RWN_results_32.p{1,i} = ER_RWN_sixty_four.result_plot_1{i,2};
% ER_RWN_results_32.CC{1,i} = ER_RWN_sixty_four.result_plot_1{i,3}; %Clustering Coefficient from Python library
% ER_RWN_results_32.L{1,i} = ER_RWN_sixty_four.result_plot_1{i,4}; %Avg Path Length from Python library
% %New graph measures
% ER_RWN_results_32.adj_matrices{1,i} = A;
% ER_RWN_results_32.APL{1,i} = APL;
% ER_RWN_results_32.avg_cc{1,i} = avg_cc;
% ER_RWN_results_32.avg_deg{1,i} = avg_deg;
% ER_RWN_results_32.glo_eff{1,i} = glo_eff;
% ER_RWN_results_32.CPL{1,i} = CPL;
% %results_64.algeb_conn{1,i} = algeb_conn;
% ER_RWN_results_32.kappa{1,i} = kappa1;
% ER_RWN_results_32.kappa_abs{1,i} = kappa_abs;
% ER_RWN_results_32.entropy{1,i} = Ent;
% ER_RWN_results_32.kappa_matrix{1,i} = kappa_matrix;
% i
%%
% % BA Graphs
% for i = 1:n
% 
% A = BA_RWN_sixty_four.result_plot_1{i,5};
% [APL,avg_cc, avg_deg,glo_eff, CPL, kappa1, kappa_abs, Ent, kappa_matrix] = global_meas_5Jan21(A);
% 
% %degree, p, seed, CC, L from original code
% BA_RWN_results_32.r{1,i} = BA_RWN_sixty_four.result_plot_1{i,1};
% BA_RWN_results_32.m{1,i} = BA_RWN_sixty_four.result_plot_1{i,2};
% BA_RWN_results_32.CC{1,i} = BA_RWN_sixty_four.result_plot_1{i,3}; %Clustering Coefficient from Python library
% BA_RWN_results_32.L{1,i} = BA_RWN_sixty_four.result_plot_1{i,4}; %Avg Path Length from Python library
% %New graph measures
% BA_RWN_results_32.adj_matrices{1,i} = A;
% BA_RWN_results_32.APL{1,i} = APL;
% BA_RWN_results_32.avg_cc{1,i} = avg_cc;
% BA_RWN_results_32.avg_deg{1,i} = avg_deg;
% BA_RWN_results_32.glo_eff{1,i} = glo_eff;
% BA_RWN_results_32.CPL{1,i} = CPL;
% %results_64.algeb_conn{1,i} = algeb_conn;
% BA_RWN_results_32.kappa{1,i} = kappa1;
% BA_RWN_results_32.kappa_abs{1,i} = kappa_abs;
% BA_RWN_results_32.entropy{1,i} = Ent;
% BA_RWN_results_32.kappa_matrix{1,i} = kappa_matrix;
% i
% end

%%
% save('WS_results_32n_100.mat','WS_RWN_results_32')
% save('ER_results_32n_100.mat','ER_RWN_results_32')
% save('BA_results_32n_100.mat','BA_RWN_results_32')
save('Results_all_graph_measures_64n.mat','results_64')
