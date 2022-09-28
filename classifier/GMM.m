classdef GMM < handle
    %GMM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        alpha
        mu
        sigma
        K
        dim
    end
    
    methods
        function obj = GMM(K,dim)
            %GMM Construct an instance of this class
            %   Detailed explanation goes here
            obj.K = K;
            obj.dim = dim;
            obj.sigma = zeros(dim,dim,K);
            obj.alpha = ones(1,K) / K;
        end
        
        function fit(obj,data)
            [size1,size2] = size(data);
            [obj.mu, r]= k_means_plus(data,obj.K);
            for i = 1:obj.K
                obj.sigma(:,:,i) = eye(obj.dim)*(r(i)^2);
            end
            for i = 1:250
                % E-step
                N_pdf = zeros(size1,obj.K);
                for j = 1:obj.K
                    try
                        if max(obj.sigma(:,:,j),[],'all') < 1e-50
                            obj.sigma(:,:,j) =sqrt(obj.sigma(:,:,j));
                        end
                        N_pdf(:,j) = mvnpdf(data,obj.mu(j,:),obj.sigma(:,:,j));
                    catch
                        obj.sigma(:,:,j) = eye(size2,size2);
                        N_pdf(:,j) = mvnpdf(data,obj.mu(j,:),obj.sigma(:,:,j));
                    end
                end
                N_sum = sum(N_pdf.* obj.alpha, 2);
                alpha_N = N_pdf .* obj.alpha;
                w = alpha_N ./N_sum;
                if sum(isnan(w),'all')>=1
                    error
                end
                % M-step
                n = sum(w, 1);
                obj.alpha = n/size1;
                for j = 1:obj.K
                    if n(j)~=0
                        obj.mu(j,:) = sum(w(:,j).*data,1)/n(j);
                    else
                         obj.mu(j,:) = sum(w(:,j).*data,1);
                    end
                end
                for j = 1:obj.K
                    data_ = data - obj.mu(j,:);
                    if n(j)~=0
                        obj.sigma(:,:,j) = (data_')*diag(w(:,j))*data_/n(j);
                        if sum(obj.sigma(:,:,j),'all') == 0 || rank(obj.sigma(:,:,j)) ~=size2
                            obj.sigma(:,:,j) = eye(size2,size2);
                        end
                    else
                        obj.sigma(:,:,j) = eye(size2,size2);
                    end
                end
            end
        end
        function y = prod(obj,x)
            y = zeros(size(x,1),1);
            for i = 1:obj.K
                    if obj.alpha(i)~=0
                        y = y + mvnpdf(x,obj.mu(i,:),obj.sigma(:,:,i)) * obj.alpha(i);
                    end
            end
        end
    end
end
function [y,r ]= k_means_plus(data,k)
shape = size(data);
mu = mean(data,1);
y = randn(k,shape(2)) + mu;
%for ii = 1:1

for center_idx = 1:k
    mini_dist = zeros(1,shape(1));
    for idx = 1:shape(1)
       d_vector  = data(idx,:) - y;
       d = vecnorm(d_vector,2,2);
       mini_dist(idx) = min(d);
    end
    [~,max_idx] = max(mini_dist);
    y(center_idx,:) = data(max_idx,:);
end

%{
for times = 1:0
    class = zeros(shape(1),shape(2),k);
    num_class = zeros(1,k);
    for idx = 1:shape(1)
            d_vector = data(idx,:) - y;
            d = vecnorm(d_vector,2,2);
            [~,min_idx] = min(d);
            num_class(min_idx) = num_class(min_idx) + 1;
            class( num_class(min_idx),:,min_idx) = data(idx,:);
    end
    for i = 1:k
        if num_class(i) ~= 0
            y(i,:) = sum(class(:,:,i),1) / num_class(i);
        else
            num = randi(shape(1));
            choose_idx = randi(shape(1),1,num);
            y(i,:) = mean(data(choose_idx,:),1);
        end
    end
end
%}
%end
class = zeros(shape(1),shape(2),k);
num_class = zeros(1,k);
for idx = 1:shape(1)
        d_vector = data(idx,:) - y;
        d = vecnorm(d_vector,2,2);
        [~,min_idx] = min(d);
        num_class(min_idx) = num_class(min_idx) + 1;
        class( num_class(min_idx),:,min_idx) = data(idx,:);
end
r = zeros(1,k);
for idx = 1:k
    if num_class(idx) ~= 0
        r(idx) = mean(vecnorm(y(idx,:) - class(1:num_class(idx),:,idx),2,2));
    else
        r(idx) = 0;
    end
end
end


