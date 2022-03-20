function kappa_matrix1 = kappa_HCP( A )
%[ kappa, nod_kappa, kappa_matrix1 ] = kappa_HCP( A )
G = graph(A);
% Compute binary network and shortest path between all nodes
A_bin = A;
A_bin(A_bin>1)=1; 
A_bin = sparse(A_bin);
A_sp = graphallshortestpaths(A_bin,'directed',false);  % shortest path between all nodes
% Graph Curvature between nodes A and B
n = max(size(A));
kappa_matrix = zeros(n,n);
Opt_cost1 = zeros(n,n);
Opt_cost2 = zeros(n,n);
for p = 1:n
    for m = p:n
        if m == p
            opt_cost1 = 0;
            opt_cost2 = 0;
            kappa = 0;
        else
               d_shortest = A_sp(m,p);
            if d_shortest == inf % check if the nodes are not connected
               opt_cost1 = inf;
               opt_cost2 = inf;
               kappa = 0;
            else
                opt_cost1 =d_shortest;
               
                % Now diffuse mass (half) to neighboring nodes of both node1
                % and node2 based on edge weights. Then compute OMT between
                % both distributions.
                
                % For First Node
                 N1 = neighbors(G,m);
                    % calculate how to diffuse mass
                edge_wt1 = zeros(1,max(size(N1)));
                for i=1:max(size(N1))
                 edge_wt1(:,i)=G.Edges.Weight(findedge(G,m,N1(i)));
                end
                edge_wt1 = edge_wt1./(sum(edge_wt1));
                % specify mass to be diffused to neigbors 0.5 here
                edge_wt1= edge_wt1.*0.5; 
                p0=[0.5 edge_wt1];  % distribution p0 on node 1 and its neighbors
                N0=[m N1'];         % node numbers i.e., node 1 and all the neighboring nodes
                % For Second Node
                N2 = neighbors(G,p); % neighbors of the second node
                edge_wt2=zeros(1,max(size(N2)));
                for f=1:max(size(N2))
                 edge_wt2(:,f) = G.Edges.Weight(findedge(G,p,N2(f)));
                end
                edge_wt2 = edge_wt2./(sum(edge_wt2));
                % specify mass to be diffused to neigbors 0.5 here
                edge_wt2 = edge_wt2.*0.5; 
                p1=[0.5 edge_wt2];   % distribution p1 on node 2 and its neighbors
                N_1=[p N2'];         % node numbers i.e., node 2 and all the neighboring nodes
                % Compute C between N0 and N1

                C2= A_sp(N0,N_1);
                [ ~, opt_cost2 ] = OMT( p0, p1, C2 );
                kappa = (opt_cost1 - opt_cost2)/opt_cost1;
            end
        end
        kappa_matrix(m,p)=kappa;
        Opt_cost1(m,p)= opt_cost1;
        Opt_cost2(m,p)= opt_cost2;
    end
end
kappa_matrix1=kappa_matrix+kappa_matrix';
% nod_kappa= sum(kappa_matrix1);
% kappa=sum(sum(kappa_matrix1));
end