classdef RBFNN_QUICK < handle
    %RBFNN_QUICK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        TF
        mu
        d
        w
        beta
        K
        phi
        x
        y
        times
    end
    
    methods
        function obj = RBFNN_QUICK()
        end
        function fit(obj,x,y,w,K,times)
            obj.x = x;
            obj.y = (y+1)/2;
            if isempty(obj.TF)
                obj.TF = minimax_tf(obj.x);
            end
            switch nargin 
                case 3
                    if isempty(obj.K)
                        obj.K = 5;
                    end
                    if isempty(obj.w)
                        obj.w = ones(size(x,1),1)/size(x,1);
                    end
                    if isempty(obj.times)
                        obj.times = 100;
                    end
                case 4
                    if isempty(obj.K)
                        obj.K = 5;
                    end
                    obj.w = w;
                    if isempty(obj.times)
                        obj.times = 100;
                    end
                case 5
                    obj.w = w;
                    obj.K = K;
                    if isempty(obj.times)
                        obj.times = 100;
                    end
                otherwise
                    obj.w = w;
                    obj.K = K;
                    obj.times = times;
            end

            x_ = obj.TF.tf(obj.x);
            [obj.mu, obj.d] = k_means_plus(x_,obj.K);
            obj.phi = obj.transform(x_);
            obj.beta = randn(obj.K+1,1);
            
            for i = 1:obj.times
                delta = obj.grdient(obj.beta);
                f = @(h) obj.err(sigmoid(obj.phi * (obj.beta+h*delta)));
                stepsize = PSO(f);
                obj.beta = obj.beta +stepsize* delta;
                %disp(obj.binary_loss(sigmoid(obj.phi * (obj.beta))))
            end
        end

         function y = grdient(obj,beta)
            pred_y = sigmoid(obj.phi * beta);
            D = diag(pred_y.*(1 - pred_y) .* obj.w);
            H = obj.phi'*D*obj.phi;
            diff = sum((obj.y - pred_y) .* obj.w.*obj.phi,1);
            [U, D, V] = svd(H);
            z = U'*(diff');
            y = zeros(obj.K+1,1);
            D = diag(D);
            idx= D~=0;
            y(idx) = z(idx)./D(idx);
            y = V*y;
         end
        function y = transform(obj,x)
            shape = size(x);
            y = zeros(obj.K,shape(1));
            for i = 1:shape(1)
                d_vector = obj.mu - x(i,:);
                d = vecnorm(d_vector,2,2);
                y(:,i) = d;
            end
            y = y';
            y =Inverse_multiquadric(y,obj.d);
            y = [y,ones(size(y,1),1)];
        end
        function out=copy(obj)
        out = RBFNN_QUICK();
        if ~isempty(obj.TF)
            out.TF =obj.TF.copy();
        end
        out.mu =obj.mu;
        out.d =obj.d;
        out.w =obj.w;
        out.beta =obj.beta;
        out.K =obj.K;
        out.phi =obj.phi;
        out.x =obj.x;
        out.y =obj.y;
        out.times = obj.times;
        end
        function y = predict(obj,x)
            x_ = obj.TF.tf(x);
            phi = obj.transform(x_);
            y = sigmoid(phi * obj.beta);
            y = round(y);
            y = 2*y-1;
        end
        function out = binary_loss(obj,pred_y)
        out = -obj.y.*log(pred_y + 1e-15) - (1 - obj.y).*log((1 - pred_y) + 1e-15);
        s = out.*obj.w;
        out = sum(s);
        end
        function out = err(obj,pred_y)
                pred_y = round(pred_y);
                out = (obj.w' * (pred_y~=obj.y));
        end
    end
end
function y = sigmoid(x)
y = 1./(1 + exp(-x));
end
function y = Gaussian(y,d)
    y =exp( -(y.^2)./(2*d.^2 + 1e-8));
end
function y = Inverse_multiquadric(y,d)
y =(y.^2)./(2*d.^2 + 1e-15);
y = 1./sqrt(1 + y);
end

function y = PSO(f)
    pool_size = 10;
    max_iter = 5;
    phi1 = 0.5;
    phi2 = 0.5;
    pool =  [linspace(0,0.1,pool_size/2),linspace(0.1,1,pool_size/2)];
    v_pool = zeros(size(pool));
    each_argmin = pool;
    each_minimum = inf * ones(size(pool));
    energy_pool = inf * ones(size(pool));
    for times = 1:max_iter
        for idx = 1:pool_size
           energy_pool(idx) = f(pool(idx));
           if energy_pool(idx) <= each_minimum(idx)
                each_minimum(idx) = energy_pool(idx);
                each_argmin(idx) = pool(idx);
           end
        end
        [~ , times_argmin_idx] = min(energy_pool);
        times_argmin = pool(times_argmin_idx);
        for idx = 1:pool_size
            v_pool(idx) = v_pool(idx) + phi1 * (times_argmin - pool(idx)) + phi2 * (each_argmin(idx) - pool(idx));
            pool(idx) = pool(idx) + v_pool(idx);
        end
    end
    for idx = 1:pool_size
       energy_pool(idx) = f(pool(idx));
       if energy_pool(idx) <= each_minimum(idx)
            each_minimum(idx) = energy_pool(idx);
            each_argmin(idx) = pool(idx);
       end
    end
    [~ , times_argmin_idx] = min(energy_pool);
    y = pool(times_argmin_idx);
    if y == 0
        y = rand()*0.01;
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

