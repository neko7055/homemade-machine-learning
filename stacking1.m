classdef stacking1 < handle
    
    properties
        beta
        n
        x
        y
        learner
    end
    
    methods
        function obj = stacking1(learner)
            obj.learner = learner.copy();
            obj.n = length(learner);
        end

        function fit(obj,x,y,times)
            if nargin == 3
                times = 5000;
            end
            obj.x = zeros(size(x,1),obj.n);
            for i = 1:obj.n
                obj.x(:,i) = obj.learner{i}.prob_predict(x);
            end
            obj.x = [obj.x,ones(size(x,1),1)];
            obj.y = (y+1)/2;
            obj.beta = randn(size(obj.x,2),1);


            for i = 1:times
                delta = obj.grdient(obj.beta);
                %f = @(h) obj.binary_loss(sigmoid(obj.x * (obj.beta+h*delta)));
                %stepsize = PSO(f);
                obj.beta = obj.beta +0.01* delta;
                disp(obj.binary_loss(sigmoid(obj.x * (obj.beta))))
            end
        end
        function y = grdient(obj,beta)
            pred_y = sigmoid(obj.x * beta);
            D = diag(pred_y.*(1 - pred_y));
            H = obj.x'*D*obj.x;
            diff = sum((obj.y - pred_y).*obj.x,1);
            [U, D, V] = svd(H);
            z = U'*(diff');
            y = zeros(length(diff),1);
            D = diag(D);
            idx= D~=0;
            y(idx) = z(idx)./D(idx);
            y = V*y;
        end
        function out = copy(obj)
                out = stacking1();
                out.beta =obj.beta;
                out.x =obj.x;
                out.y =obj.y;
                out.n=obj.n;
                out.learner=obj.learner;
        end
        function y = predict(obj,x)
            x_ = zeros(size(x,1),obj.n);
            for i = 1:obj.n
                x_(:,i) = obj.learner{i}.prob_predict(x);
            end
            X = [x_,ones(size(x,1),1)];
            y = sigmoid(X*obj.beta);
            y = round(y);
            y = 2*y-1;
        end
        function out = binary_loss(obj,pred_y)
        out = -obj.y.*log(pred_y) - (1 - obj.y).*log(1 - pred_y);
        s = out;
        out = sum(s(~isnan(s)));
        end
        function out = err(obj,pred_y)
                out = mean((pred_y - obj.y).^2);
        end
    end
end

function y = sigmoid(x)
y = 1./(1 + exp(-x));
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

