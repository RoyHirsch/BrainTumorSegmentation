function  = sort_gmm(gm)
% excart relevant data
means = gm.mu;
variance = gm.Sigma;
prior = gm.ComponentProportion;
gm_sort = gm;

% find the right order of the GMM elements
[val,ind] = intersect(means,sort(means))
sorted_means = means(ind);
sorted_variance = variance(ind);
sorted_prior = prior(ind);

% re-argange the model
gm_sort.mu = sorted_means;
gm_sort.Sigma = sorted_variance;
gm_sort.ComponentProportion = sorted_prior;

end