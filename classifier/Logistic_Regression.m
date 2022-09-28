classdef Logistic_Regression < handle
    
    properties
        beta
        TF
        w
        x
        y
        times
    end
    
    methods
        function obj = Logistic_Regression()
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
                    obj.times = times;
                otherwise
                    obj.w = w;
                    obj.times = times;
            end

            if isempty(obj.TF)
                obj.TF = minimax_tf(x);
            end
            x = obj.TF.tf(x);
            obj.x = [x,ones(size(x,1),1)];
            obj.y = (y+1)/2;
            obj.beta = randn(size(obj.x,2),1);
            for i = 1:obj.times
                pred_y = sigmoid(obj.x * obj.beta);
                gred =  sum((obj.y - pred_y).*pred_y.*(1 - pred_y) .* obj.w.*obj.x,1);
                obj.beta = obj.beta +stepsize* gred';
            end
        end
        function out = copy(obj)
                out = Logistic_Regression();
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
            y = sigmoid(X*obj.beta);
            y = round(y);
            y = 2*y-1;
        end
    end
end

function y = sigmoid(x)
y = 1./(1 + exp(-x));
end