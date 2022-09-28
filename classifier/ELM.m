classdef ELM < handle
    %KNN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        TF
        x
        y
        k
        random_w
        weight
        w
    end
    
    methods
        function obj = ELM()
        end
        
        function fit(obj,x,y,w,max_k)
            switch nargin 
                case 3
                    if isempty(obj.w)
                        obj.w = ones(size(x,1),1)/size(x,1);
                    end
                    if isempty(obj.k)
                        obj.k = 10;
                    end
                case 4
                    obj.w = w;
                    if isempty(obj.k)
                        obj.k = 10;
                    end
                otherwise
                    obj.w = w;
                    obj.k = max_k;
            end
            SIZE = size(x);
            if isempty(obj.TF)
                obj.TF = minimax_tf(x);
            end
            obj.x = [obj.TF.tf(x),ones(SIZE(1),1)];
            obj.y = y;
            obj.random_w = randn(SIZE(2)+1, obj.k);
            o =[sigmoid(obj.x*obj.random_w),ones(SIZE(1),1)];
            W = diag(obj.w);
            obj.weight = pinv(o'*W*o)*o'*W*y;
            
        end
        function out=copy(obj)
        out = ELM();
        if ~isempty(obj.TF)
            out.TF =obj.TF.copy();
        end
        out.x =obj.x;
        out.y =obj.y;
        out.k =obj.k;
        out.random_w = obj.random_w;
        out.weight=obj.weight;
        out.w = obj.w;
        end
        function pred_y = predict(obj,x)
            SIZE = size(x);
            x = [obj.TF.tf(x),ones(SIZE(1),1)];
            pred_y = [sigmoid(x*obj.random_w),ones(SIZE(1),1)]* obj.weight;
            pred_y = double(pred_y > 0);
            pred_y = 2*pred_y - 1;
        end
    end
end

function y = sigmoid(x)
y = 1./(1 + exp(-x));
end

