%% Load data of SD HCP
close all
clear all
clc
%% load mat files
load('SD_HCP.mat')
%load('agg_data_gp2_prior.mat')
%% Make node measures mat files
% gp_1
n = 180; k_ASD = max(size(agg_data_SD_ASD_HCP));
kappa_ASD = zeros(n,n,k_ASD);
A_ASD = zeros(n,n,k_ASD);
A_bin_ASD = zeros(n,n,k_ASD);
kappa_sparse_ASD = zeros(n,n,k_ASD);
nod_kappa_sparse_ASD = zeros(n,k_ASD);
str_ASD = zeros(n,k_ASD);
cc_ASD = zeros(n,k_ASD);
bc_ASD = zeros(n,k_ASD);

% TD
k_TD = max(size(agg_data_SD_TD_HCP));
kappa_TD = zeros(n,n,k_TD);
A_TD = zeros(n,n,k_TD);
A_bin_TD = zeros(n,n,k_TD);
kappa_sparse_TD = zeros(n,n,k_TD);
nod_kappa_sparse_TD = zeros(n,k_TD);
str_TD = zeros(n,k_TD);
cc_TD = zeros(n,k_TD);
bc_TD = zeros(n,k_TD);
%% ASD Loop
for i = 1: k_ASD
    kappa_ASD(:,:,i) = agg_data_SD_ASD_HCP{1,i}.kappa;
    A_ASD(:,:,i) = agg_data_SD_ASD_HCP{1,i}.connectivity;
    A_bin1 = A_ASD(:,:,i);
    A_bin1(A_bin1>1)=1; 
    A_bin_ASD(:,:,i) = A_bin1;
    sp1 = kappa_ASD(:,:,i).*A_bin1 ;
    kappa_sparse_ASD (:,:,i) = sp1;
    nod_kappa_sparse_ASD (:,i) = abs(sum(sp1,2));
    str_ASD(:,i)= agg_data_SD_ASD_HCP{1,i}.strength_weighted;
    cc_ASD(:,i)= agg_data_SD_ASD_HCP{1,i}.cluster_coef_weighted;
    bc_ASD(:,i)= agg_data_SD_ASD_HCP{1,i}.betweenness_centrality_weighted;
end
%% TD Loop
for i = 1: k_TD
    kappa_TD(:,:,i) = agg_data_SD_TD_HCP{1,i}.kappa;
    A_TD(:,:,i) = agg_data_SD_TD_HCP{1,i}.connectivity;
    A_bin1 = A_TD(:,:,i);
    A_bin1(A_bin1>1)=1; 
    A_bin_TD(:,:,i) = A_bin1;
    sp1 = kappa_TD(:,:,i).*A_bin1 ;
    kappa_sparse_TD (:,:,i) = sp1;
    nod_kappa_sparse_TD (:,i) = abs(sum(sp1,2));
    str_TD(:,i)= agg_data_SD_TD_HCP{1,i}.strength_weighted;
    cc_TD(:,i)= agg_data_SD_TD_HCP{1,i}.cluster_coef_weighted;
    bc_TD(:,i)= agg_data_SD_TD_HCP{1,i}.betweenness_centrality_weighted;
end