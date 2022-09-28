classdef SVM < handle
    
    properties
        beta
        TF
        w
        x
        y
        times
    end
    
    methods
        function obj = SVM()
        end

        function fit(obj,x,y,w,times)
            switch nargin 
                case 3
                    if isempty(obj.w)
                        obj.w = ones(size(x,1),1)/size(x,1);
                    end
                    if isempty(obj.times)
                        obj.times = 100;
                    end
                case 4
                    obj.w = w;
                    if isempty(obj.times)
                        obj.times = 100;
                    end
                otherwise
                    obj.w = w;
                    obj.times = times;
            end
            if isempty(obj.TF)
                obj.TF = minimax_tf(x);
            end
            x = obj.TF.tf(x);
            obj.x = [x,ones(size(x,1),1)];
            obj.y = y;
            obj.beta = randn(size(obj.x,2),1);
            H = eye(size(obj.x,2));
            for i = 1:obj.times
                if i == 1
                    pred_y = obj.x * obj.beta;
                    grad = -(sum(((pred_y.*obj.y)<1).*obj.y .* obj.w.*obj.x,1) )';
                else
                    grad = grad_next;
                end
                
                if any(isnan(H))
                    delta_x = -grad;
                    H = eye(size(obj.x,2));
                else
                    TMP = H*grad;
                    delta_x = -TMP;
                end
                stepsize_f = @(a) loss((obj.x * (obj.beta + a*delta_x)),obj.y,obj.w);
                stepsize = PSO(stepsize_f);
                obj.beta = obj.beta + stepsize*delta_x;

                pred_y = obj.x * obj.beta;
                grad_next = -(sum(((pred_y.*obj.y)<1).*obj.y .* obj.w.*obj.x,1) )'; %- 1e-4 *obj.beta'
                delta_grad = grad_next - grad;
                H = energy_Hessian(delta_grad,delta_x,H);
                %disp(loss(pred_y,obj.y,obj.w))
                %disp(mean(sign(pred_y)~=obj.y))
            end
        end
        function out = copy(obj)
                out = SVM();
                out.beta =obj.beta;
                if ~isempty(obj.TF)
                    out.TF =obj.TF.copy();
                end
                out.w =obj.w; 
                out.x =obj.x;
                out.y =obj.y;
                out.times = obj.times;
        end
        function y = predict(obj,x)
            x = obj.TF.tf(x);
            X = [x,ones(size(x,1),1)];
            y = X*obj.beta;
            y = y>0;
            y = 2*y-1;
        end
    end
end

function out = loss(pred_y,y,w)
if nargin==2
    out = mean(max(1-y.*pred_y,0));
else
    out = sum(max(1-y.*pred_y,0).*w);
end
end

function H_next = energy_Hessian(delta_grad,delta_x,H)
n = length(delta_x);
I = eye(n);
tmp_yxt = delta_grad * (delta_x');
tmp_ytx = delta_grad' * (delta_x);
tmp_xxt = delta_x * (delta_x');
TMP2 = (I - tmp_yxt/tmp_ytx);
H_next = TMP2'*H*TMP2 + tmp_xxt/tmp_ytx;
end

function y = PSO(f)
    pool_size = 1000;
    max_iter = 10;
    phi1 = 0.7;
    phi2 = 0.3;
    pool = [linspace(0,0.1,pool_size/2),linspace(0.1,1,pool_size/2)];
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
        y = 1e-8;
    end
end