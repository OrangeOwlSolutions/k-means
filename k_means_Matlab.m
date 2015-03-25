function [cluster_centers, Mask] = kmeans_customized(image_under_test1, K)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                           %
%   kmeans image segmentation                               %
%                                                           %
%   Input:                                                  %
%          image_under_test1    : Grey scale image          %
%          K                    : Number of classes         %
%   Output:                                                 %
%          cluster_centers      : array of cluster centers  %
%          Mask                 : image classification Mask %
%                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Preliminary operations on the image
image_under_test1 = double(image_under_test1);
image_under_test0 = image_under_test1;              % --- Make members_of_cluster_kk copy
image_under_test1 = image_under_test1(:);           % --- Vectorize image_under_test1
NumElems          = length(image_under_test1);

% --- Forces the minimum of the image to be 1
mi                = min(image_under_test1);      
image_under_test1 = image_under_test1 - mi + 1;     

% --- Create image histogram
m                 = max(image_under_test1) + 1;
histogram         = zeros(1,m);                 % --- Histogram initialization
for kk=1:NumElems
  histogram(image_under_test1(kk)) = histogram(image_under_test1(kk)) + 1;
end
ind               = find(histogram);                    % --- Indices of the non-zero entries of histogram
Num_non_zero_hist = length(ind);                % --- Number of non-zero entries of histogram

%%%%%%%%%%%%%%
% ITERATIONS %
%%%%%%%%%%%%%%
cluster_centers   = (1:K) * m / (K + 1);        % --- Cluster center initialization

membership=zeros(1,m);
while(true)
  
  cluster_centers_old = cluster_centers;        % --- Save old version of cluster centers
 
  % --- Explores only non-zero elements of histogram. For each histogram
  % element, characterized by its index, finds the closest cluster center
  % and assigns membership accordingly
  for kk = 1 : Num_non_zero_hist
      c                  = abs(ind(kk) - cluster_centers);
      cc                 = find(c == min(c));
      membership(ind(kk)) = cc(1);
  end
  
  % --- Recalculates the cluster centers
  for kk=1:K,
      % --- Finds the members of cluster kk
      members_of_cluster_kk = find(membership == kk);                                        
      % --- Calculates the mass center of cluster kk
      cluster_centers(kk) = sum(members_of_cluster_kk .* histogram(members_of_cluster_kk)) / sum(histogram(members_of_cluster_kk));
  end
  
  % --- Stopping rule
  if(cluster_centers == cluster_centers_old) break; end;
  
end

%%%%%%%%%%%%%%%%%%%%
% MASK CALCULATION %
%%%%%%%%%%%%%%%%%%%%
ImageSize = size(image_under_test0);
Mask      = zeros(ImageSize);
for kk=1:ImageSize(1),
    for ll=1:ImageSize(2),
        % --- Finds the distance between the pixel and all the determined
        % cluster centers
        c = abs(image_under_test0(kk,ll) - cluster_centers);
        % --- Finds the cluster center with the least distance
        cluster_with_least_distance_to_pixel = find(c == min(c));  
        Mask(kk,ll) = cluster_with_least_distance_to_pixel(1);
    end
end

% --- Restores the original image values
cluster_centers = cluster_centers + mi - 1;

