classdef RBF < handle
    
    properties
        mu
        r
        TF
        w
        times
    end
    
    methods
        function obj = RBF()
        end

        function fit(obj,x,y,w,times,stepsize)
            switch nargin 
                case 3
                    if isempty(obj.w)
                        obj.w = ones(size(x,1),1)/size(x,1);
                    end
                    if isempty(obj.times)
                        obj.times = 5000;
                    end
                    stepsize = 0.09;
                case 4
                    obj.w = w;
                    if isempty(obj.times)
                        obj.times = 5000;
                    end
                    stepsize = 0.09;
                case 5
                    obj.w = w;
                    stepsize = 0.09;
                    if isempty(obj.times)
                        obj.times = 5000;
                    end
                otherwise
                    obj.w = w;
                    obj.times = times;
            end

            if isempty(obj.TF)
                obj.TF = minimax_tf(x);
            end
            x = obj.TF.tf(x);
            obj.mu = mean(x,1);
            obj.r = 1/mean(vecnorm(x-obj.mu,2,2));
            y = (y+1)/2;
            for i = 1:obj.times
                d_v = x-obj.mu;
                d =v_norm(d_v);
                pred_y = exp(- d * obj.r);
                gred = (y - pred_y).*pred_y .* obj.w;
                gred_mu =  sum(gred.*(d_v)*obj.r,1);
                gred_r =  sum(gred.*(-d),1);
                obj.mu = obj.mu +stepsize* gred_mu;
                obj.r = obj.r +stepsize* gred_r;
            end
        end
        function out = copy(obj)
                out =RBF();
                out.mu =obj.mu;
                if ~isempty(obj.TF)
                    out.TF =obj.TF.copy();
                end
                out.w =obj.w; 
                out.r =obj.r;
                out.times = obj.times;
        end
        function y = predict(obj,x)
            x = obj.TF.tf(x);
            y =  exp(- v_norm(x-obj.mu) * obj.r);
            y = round(y);
            y = 2*y-1;
        end
    end
end

function out = v_norm(x)
out = sum(x.^2,2);
end
